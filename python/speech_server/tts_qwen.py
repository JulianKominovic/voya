from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator

import numpy as np

from speech_server.config import (
    CUDA_DEVICE,
    SAMPLE_RATE_TTS,
    TTS_LANG,
    TTS_MODEL,
    TTS_TEMPERATURE,
    TTS_VOICE,
)

log = logging.getLogger(__name__)

DEFAULT_SPEAKER = "Serena"
CHUNK_SIZE = 8  # codec steps; ~0.67 s of audio at 12 Hz
# Missed EOS otherwise fills the talker StaticCache (default 512 ≈ 36 s of noise).
MAX_NEW_TOKENS = 192  # ~16 s at 12 Hz


def _max_new_tokens(text: str) -> int:
    return max(48, min(MAX_NEW_TOKENS, len(text) * 3 + 36))


_LANG = {
    "es": "Spanish",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "auto": "Auto",
}


def _pcm(chunk: np.ndarray, sr: int) -> np.ndarray:
    samples = np.asarray(chunk, dtype=np.float32).ravel()
    if samples.size == 0:
        return samples
    if int(sr) == SAMPLE_RATE_TTS:
        return samples
    import librosa

    return librosa.resample(
        y=samples, orig_sr=int(sr), target_sr=SAMPLE_RATE_TTS
    ).astype(np.float32)


class QwenTTS:
    def __init__(self) -> None:
        try:
            import torch
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as exc:
            raise RuntimeError(
                "TTS_ENGINE=qwen needs the qwen extra. On the 5080: uv sync --extra qwen"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen TTS requires CUDA (RTX 5080). torch.cuda.is_available() is False."
            )
        self.device = f"cuda:{CUDA_DEVICE}"
        log.info("loading qwen CUDA %s model=%s", self.device, TTS_MODEL)
        self.model = FasterQwen3TTS.from_pretrained(
            TTS_MODEL,
            device=self.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        # Sampling (not greedy) so the model emits EOS; conservative temperature
        # so it doesn't moan. The predictor graph bakes temperature at capture.
        self.model.predictor_graph.temperature = TTS_TEMPERATURE
        self.model.warmup()
        base = self.model.model
        speakers = list(base.get_supported_speakers())
        self._speakers = {s.lower(): s for s in speakers}
        languages = list(base.get_supported_languages())
        self._languages = {s.lower(): s for s in languages}
        self.default_voice = self._speaker(TTS_VOICE)
        self.default_lang = self._language(TTS_LANG)
        self._lock = asyncio.Lock()
        log.info(
            "qwen ready speakers=%s default_voice=%s lang=%s temperature=%.2f",
            speakers,
            self.default_voice,
            self.default_lang,
            TTS_TEMPERATURE,
        )

    def _speaker(self, voice: str) -> str:
        key = voice.strip().lower()
        if key in self._speakers:
            return self._speakers[key]
        return self._speakers.get(DEFAULT_SPEAKER.lower(), DEFAULT_SPEAKER)

    def _language(self, lang: str) -> str:
        mapped = _LANG.get(lang.strip().lower(), lang.strip())
        key = mapped.lower()
        if key in self._languages:
            return self._languages[key]
        fallback = self._languages.get("spanish", "Spanish")
        log.warning("unknown lang %s, using %s", lang, fallback)
        return fallback

    async def stream(
        self,
        text: str,
        voice: str | None = None,
        lang: str | None = None,
    ) -> AsyncIterator[np.ndarray]:
        speaker = self._speaker(voice or self.default_voice)
        language = self._language(lang or self.default_lang)
        loop = asyncio.get_running_loop()
        out: asyncio.Queue[np.ndarray | BaseException | None] = asyncio.Queue()
        cancelled = threading.Event()
        max_tokens = _max_new_tokens(text)

        def _produce() -> None:
            steps = 0
            peak = 0.0
            try:
                import torch

                torch.cuda.synchronize()
                for chunk, sr, timing in self.model.generate_custom_voice_streaming(
                    text=text,
                    language=language,
                    speaker=speaker,
                    temperature=TTS_TEMPERATURE,
                    chunk_size=CHUNK_SIZE,
                    max_new_tokens=max_tokens,
                ):
                    if cancelled.is_set():
                        log.info("qwen cancel steps=%d peak=%.3f", steps, peak)
                        return
                    steps = int(timing.get("total_steps_so_far") or steps)
                    samples = _pcm(chunk, sr)
                    if not samples.size:
                        continue
                    if not np.isfinite(samples).all():
                        raise RuntimeError("qwen non-finite pcm")
                    peak = max(peak, float(np.max(np.abs(samples))))
                    asyncio.run_coroutine_threadsafe(out.put(samples), loop).result()
                if steps >= max_tokens - CHUNK_SIZE:
                    log.warning(
                        "qwen token cap steps=%d max=%d chars=%d peak=%.3f",
                        steps,
                        max_tokens,
                        len(text),
                        peak,
                    )
                else:
                    log.info(
                        "qwen done steps=%d max=%d chars=%d peak=%.3f",
                        steps,
                        max_tokens,
                        len(text),
                        peak,
                    )
                asyncio.run_coroutine_threadsafe(out.put(None), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(out.put(exc), loop).result()

        async with self._lock:
            infer = asyncio.create_task(asyncio.to_thread(_produce))
            try:
                while True:
                    item = await out.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
                await infer
            except BaseException:
                cancelled.set()
                if not infer.done():
                    try:
                        await asyncio.shield(infer)
                    except Exception:
                        pass
                raise
