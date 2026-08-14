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


class QwenTTS:
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TTS_ENGINE=qwen needs the qwen extra. On the 5080: uv sync --extra qwen"
            ) from exc
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(f"qwen-tts import failed: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen TTS requires CUDA (RTX 5080). torch.cuda.is_available() is False."
            )
        self.device = f"cuda:{CUDA_DEVICE}"
        log.info("loading qwen CUDA %s model=%s", self.device, TTS_MODEL)
        self.model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        speakers = list(self.model.get_supported_speakers())
        self._speakers = {s.lower(): s for s in speakers}
        languages = list(self.model.get_supported_languages())
        self._languages = {s.lower(): s for s in languages}
        self.default_voice = self._speaker(TTS_VOICE)
        self.default_lang = self._language(TTS_LANG)
        self._lock = asyncio.Lock()
        log.info(
            "qwen ready speakers=%s default_voice=%s lang=%s",
            speakers,
            self.default_voice,
            self.default_lang,
        )

    def _speaker(self, voice: str) -> str:
        key = voice.strip().lower()
        if key in self._speakers:
            return self._speakers[key]
        fallback = self._speakers.get(DEFAULT_SPEAKER.lower(), DEFAULT_SPEAKER)
        log.warning("unknown voice %s, falling back to %s", voice, fallback)
        return fallback

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

        def _create() -> np.ndarray:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
            )
            samples = np.asarray(wavs[0], dtype=np.float32).ravel()
            if int(sr) != SAMPLE_RATE_TTS:
                import librosa

                samples = librosa.resample(
                    y=samples, orig_sr=int(sr), target_sr=SAMPLE_RATE_TTS
                ).astype(np.float32)
            return samples

        async with self._lock:
            infer = asyncio.create_task(asyncio.to_thread(_create))
            try:
                samples = await infer
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(infer)
                except Exception:
                    pass
                raise
        yield samples
