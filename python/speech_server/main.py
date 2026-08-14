from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from speech_server.audio import float32_to_s16le, s16le_to_float32
from speech_server.config import (
    CUDA_DEVICE,
    KOKORO_ONNX,
    KOKORO_VOICES,
    MIN_SILENCE_MS,
    MODELS_DIR,
    SILERO_ONNX,
    TTS_LANG,
    TTS_VOICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from speech_server.protocol import now_s
from speech_server.stt import STT
from speech_server.tts import TTS, iter_pcm_frames
from speech_server.vad import SileroOnnx, SpeechSegment, StreamingVAD

log = logging.getLogger("speech_server")

STATE: dict[str, Any] = {
    "ready": False,
    "vad": None,
    "stt": None,
    "tts": None,
    "error": None,
    "stt_lock": None,
}


async def ensure_stt() -> STT:
    if STATE["stt"] is not None:
        return STATE["stt"]
    async with STATE["stt_lock"]:
        if STATE["stt"] is None:
            log.info("loading STT model=%s device=%s", WHISPER_MODEL, WHISPER_DEVICE)
            STATE["stt"] = await asyncio.to_thread(STT)
            log.info("STT ready")
        return STATE["stt"]


def _require_models() -> None:
    missing = [p for p in (SILERO_ONNX, KOKORO_ONNX, KOKORO_VOICES) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"missing models in {MODELS_DIR}: {names}. "
            "Run: uv run python -m speech_server.download_models"
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        _require_models()
        STATE["stt_lock"] = asyncio.Lock()
        log.info("loading VAD CUDA %s", SILERO_ONNX)
        STATE["vad"] = SileroOnnx()
        log.info("vad providers=%s", STATE["vad"].session.get_providers())
        STATE["tts"] = TTS()
        log.info("tts providers=%s", STATE["tts"].kokoro.sess.get_providers())
        STATE["ready"] = True
        STATE["error"] = None
        log.info("VAD+TTS ready (Whisper loads in background)")

        async def _bg_stt() -> None:
            try:
                await ensure_stt()
            except Exception as exc:
                STATE["error"] = str(exc)
                log.exception("background STT load failed")

        asyncio.create_task(_bg_stt())
    except Exception as exc:
        STATE["ready"] = False
        STATE["error"] = str(exc)
        log.exception("startup failed")
        raise
    yield
    STATE["ready"] = False


app = FastAPI(title="speech-server", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": bool(STATE["ready"]),
            "vad": STATE["vad"] is not None,
            "stt": STATE["stt"] is not None,
            "tts": STATE["tts"] is not None,
            "whisper_model": WHISPER_MODEL,
            "whisper_device": WHISPER_DEVICE,
            "whisper_compute": WHISPER_COMPUTE_TYPE,
            "cuda_device": CUDA_DEVICE,
            "vad_providers": list(STATE["vad"].session.get_providers()) if STATE["vad"] else [],
            "tts_providers": list(STATE["tts"].kokoro.sess.get_providers()) if STATE["tts"] else [],
            "tts_voice": TTS_VOICE,
            "tts_lang": TTS_LANG,
            "models_dir": str(MODELS_DIR),
            "error": STATE["error"],
        }
    )


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


@app.websocket("/ws/speech-in")
async def speech_in(ws: WebSocket) -> None:
    await ws.accept()
    if not STATE["ready"]:
        await _send_json(ws, {"type": "error", "message": "models not ready"})
        await ws.close()
        return
    try:
        stt = await ensure_stt()
    except Exception as exc:
        log.exception("stt load failed")
        await _send_json(ws, {"type": "error", "message": f"stt load: {exc}"})
        await ws.close()
        return
    vad = StreamingVAD(engine=STATE["vad"])
    vad.reset()
    queue: asyncio.Queue[SpeechSegment | None] = asyncio.Queue()

    async def transcribe_loop() -> None:
        while True:
            seg = await queue.get()
            if seg is None:
                return
            t0 = time.perf_counter()
            try:
                text, lang = await asyncio.to_thread(stt.transcribe, seg.audio)
            except Exception as exc:
                log.exception("stt failed")
                await _send_json(ws, {"type": "error", "message": f"stt: {exc}"})
                continue
            stt_ms = int((time.perf_counter() - t0) * 1000)
            log.info("stt_ms=%d chars=%d lang=%s", stt_ms, len(text), lang)
            try:
                await _send_json(
                    ws,
                    {
                        "type": "transcript",
                        "t": now_s(),
                        "text": text,
                        "language": lang,
                        "stt_ms": stt_ms,
                    },
                )
            except Exception:
                return

    async def emit(events: list[tuple[str, SpeechSegment | None]], vad_ms: float = 0.0) -> None:
        for kind, seg in events:
            if kind == "speech_start":
                log.info("speech_start vad_ms=%.2f", vad_ms)
                await _send_json(ws, {"type": "speech_start", "t": now_s()})
            elif kind == "speech_end" and seg is not None:
                log.info("speech_end duration_ms=%d vad_ms=%.2f", seg.duration_ms, vad_ms)
                try:
                    await _send_json(
                        ws,
                        {
                            "type": "speech_end",
                            "t": now_s(),
                            "duration_ms": seg.duration_ms,
                        },
                    )
                except Exception:
                    pass
                await queue.put(seg)

    worker = asyncio.create_task(transcribe_loop())
    recv: asyncio.Task = asyncio.create_task(ws.receive())
    try:
        while True:
            timeout = MIN_SILENCE_MS / 1000 if vad._in_speech else None
            done, _ = await asyncio.wait({recv}, timeout=timeout)
            if recv not in done:
                await emit(list(vad.flush()))
                continue
            try:
                message = recv.result()
            except WebSocketDisconnect:
                break
            except Exception:
                log.exception("speech-in receive failed")
                break
            if message["type"] == "websocket.disconnect":
                break
            recv = asyncio.create_task(ws.receive())
            data = message.get("bytes")
            if data is None:
                continue
            samples = s16le_to_float32(data)
            t0 = time.perf_counter()
            events = list(vad.feed(samples))
            await emit(events, (time.perf_counter() - t0) * 1000)
    except WebSocketDisconnect:
        pass
    finally:
        if not recv.done():
            recv.cancel()
            try:
                await recv
            except (asyncio.CancelledError, Exception):
                pass
        for _, seg in vad.flush():
            if seg is not None:
                await queue.put(seg)
        await queue.put(None)
        try:
            await worker
        except (asyncio.CancelledError, Exception):
            pass
        vad.reset()


@app.websocket("/ws/tts")
async def tts_ws(ws: WebSocket) -> None:
    await ws.accept()
    if not STATE["ready"]:
        await _send_json(ws, {"type": "error", "message": "models not ready"})
        await ws.close()
        return
    tts: TTS = STATE["tts"]
    seq = 0
    lock = asyncio.Lock()

    async def speak(cmd: dict[str, Any], my_seq: int) -> None:
        speak_id = str(cmd.get("id") or "0")
        text = str(cmd.get("text") or "").strip()
        voice = str(cmd.get("voice") or TTS_VOICE)
        lang = str(cmd.get("lang") or TTS_LANG)
        if not text:
            await _send_json(ws, {"type": "error", "message": "empty speak text"})
            return
        t0 = time.perf_counter()
        first = True
        async with lock:
            if seq != my_seq:
                return
            await _send_json(
                ws,
                {
                    "type": "audio_start",
                    "id": speak_id,
                    "sample_rate": 24000,
                    "channels": 1,
                    "format": "s16le",
                },
            )
            try:
                async for samples in tts.stream(text, voice=voice, lang=lang):
                    if seq != my_seq:
                        log.info("tts cancel id=%s", speak_id)
                        return
                    for frame in iter_pcm_frames(samples):
                        if seq != my_seq:
                            log.info("tts cancel id=%s", speak_id)
                            return
                        if first:
                            first = False
                            log.info(
                                "tts_first_chunk_ms=%d id=%s",
                                int((time.perf_counter() - t0) * 1000),
                                speak_id,
                            )
                        await ws.send_bytes(float32_to_s16le(frame))
            except Exception as exc:
                log.exception("tts failed")
                await _send_json(ws, {"type": "error", "message": f"tts: {exc}"})
                return
            if seq != my_seq:
                return
            synth_ms = int((time.perf_counter() - t0) * 1000)
            log.info("tts audio_end id=%s synth_ms=%d", speak_id, synth_ms)
            await _send_json(ws, {"type": "audio_end", "id": speak_id, "synth_ms": synth_ms})

    tasks: set[asyncio.Task[None]] = set()
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            raw = message.get("text")
            if raw is None:
                continue
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(ws, {"type": "error", "message": "invalid json"})
                continue
            kind = cmd.get("type")
            if kind == "speak":
                task = asyncio.create_task(speak(cmd, seq))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            elif kind == "cancel":
                seq += 1
                log.info("tts cancel requested id=%s", cmd.get("id"))
            else:
                await _send_json(ws, {"type": "error", "message": f"unknown type {kind}"})
    except WebSocketDisconnect:
        pass
    finally:
        seq += 1
        for task in list(tasks):
            task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("speech_server.main:app", host="0.0.0.0", port=8765)
