from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from speech_server.config import (
    CUDA_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)
from speech_server.gpu import require_whisper_cuda

log = logging.getLogger(__name__)


class STT:
    def __init__(self) -> None:
        require_whisper_cuda(WHISPER_DEVICE)
        log.info(
            "loading whisper model=%s device=%s:%s compute=%s",
            WHISPER_MODEL,
            WHISPER_DEVICE,
            CUDA_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            device_index=CUDA_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        self.language = WHISPER_LANGUAGE or None

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        if audio.size == 0:
            return "", self.language or ""
        wav = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        segments, info = self.model.transcribe(
            wav,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        lang = info.language or (self.language or "")
        return text, lang
