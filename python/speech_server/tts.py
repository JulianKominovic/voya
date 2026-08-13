from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from speech_server.config import KOKORO_ONNX, KOKORO_VOICES, TTS_LANG, TTS_VOICE
from speech_server.gpu import cuda_providers

log = logging.getLogger(__name__)

TTS_FRAME = 2400  # 100 ms @ 24 kHz


class TTS:
    def __init__(self) -> None:
        log.info("loading kokoro CUDA %s", KOKORO_ONNX)
        session = ort.InferenceSession(
            str(KOKORO_ONNX),
            providers=cuda_providers(),
        )
        used = session.get_providers()
        if "CUDAExecutionProvider" not in used:
            raise RuntimeError(f"Kokoro did not bind CUDA, providers={used}")
        self.kokoro = Kokoro.from_session(session, str(KOKORO_VOICES))
        self.default_voice = TTS_VOICE
        self.default_lang = TTS_LANG
        self._g2p: dict[str, object] = {}
        try:
            from misaki.espeak import EspeakG2P

            self._g2p[self.default_lang] = EspeakG2P(language=self.default_lang)
            log.info("misaki EspeakG2P ready lang=%s", self.default_lang)
        except Exception as exc:
            log.warning("misaki G2P unavailable, kokoro tokenizer will phonemize: %s", exc)
        voices = self.kokoro.get_voices()
        if self.default_voice not in voices:
            log.warning("voice %s not in voices.bin; available=%s", self.default_voice, voices[:8])

    def _phonemes(self, text: str, lang: str) -> str | None:
        g2p = self._g2p.get(lang)
        if g2p is None:
            try:
                from misaki.espeak import EspeakG2P

                g2p = EspeakG2P(language=lang)
                self._g2p[lang] = g2p
            except Exception:
                return None
        phonemes, _ = g2p(text)  # type: ignore[operator]
        return phonemes or None

    async def stream(
        self,
        text: str,
        voice: str | None = None,
        lang: str | None = None,
    ) -> AsyncIterator[np.ndarray]:
        voice = voice or self.default_voice
        lang = lang or self.default_lang
        if voice not in self.kokoro.voices:
            log.warning("unknown voice %s, falling back to %s", voice, self.default_voice)
            voice = self.default_voice
        try:
            phonemes = self._phonemes(text, lang)
        except Exception as exc:
            log.warning("g2p failed (%s), kokoro tokenizer: %s", lang, exc)
            phonemes = None
        if phonemes:
            gen = self.kokoro.create_stream(phonemes, voice=voice, lang=lang, is_phonemes=True)
        else:
            gen = self.kokoro.create_stream(text, voice=voice, lang=lang)
        async for samples, _sr in gen:
            yield np.asarray(samples, dtype=np.float32)


def iter_pcm_frames(samples: np.ndarray, frame: int = TTS_FRAME) -> list[np.ndarray]:
    x = np.asarray(samples, dtype=np.float32).ravel()
    if x.size == 0:
        return []
    return [x[i : i + frame] for i in range(0, x.size, frame)]
