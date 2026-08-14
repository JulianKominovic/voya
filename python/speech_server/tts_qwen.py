from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from speech_server.config import (
    CUDA_DEVICE,
    SAMPLE_RATE_TTS,
    TTS_LANG,
    TTS_MODEL,
    TTS_VOICE,
)

log = logging.getLogger(__name__)

DEFAULT_SPEAKER = "Serena"
CHUNK_SIZE = 8  # codec steps; ~0.67 s of audio at 12 Hz

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
        # CUDA graph for the 15 codebooks bakes sampling at capture time.
        self.model.predictor_graph.do_sample = False
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
            "qwen ready speakers=%s default_voice=%s lang=%s greedy",
            speakers,
            self.default_voice,
            self.default_lang,
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

        def _produce() -> None:
            try:
                for chunk, sr, _timing in self.model.generate_custom_voice_streaming(
                    text=text,
                    language=language,
                    speaker=speaker,
                    do_sample=False,
                    chunk_size=CHUNK_SIZE,
                ):
                    samples = _pcm(chunk, sr)
                    if samples.size:
                        asyncio.run_coroutine_threadsafe(out.put(samples), loop).result()
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
                if not infer.done():
                    try:
                        await asyncio.shield(infer)
                    except Exception:
                        pass
                raise
