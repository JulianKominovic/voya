from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import onnxruntime as ort

from speech_server.config import (
    MIN_SILENCE_MS,
    MIN_SPEECH_MS,
    PREROLL_MS,
    SAMPLE_RATE_IN,
    SILERO_ONNX,
    VAD_THRESHOLD,
    VAD_WINDOW,
)
from speech_server.gpu import cuda_providers


@dataclass
class SpeechSegment:
    audio: np.ndarray
    duration_ms: int


class SileroOnnx:
    """Silero VAD v5 ONNX: 512 samples @ 16 kHz, LSTM state + 64-sample context."""

    def __init__(self, model_path: str | None = None) -> None:
        path = str(model_path or SILERO_ONNX)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            path,
            sess_options=opts,
            providers=cuda_providers(),
        )
        used = self.session.get_providers()
        if "CUDAExecutionProvider" not in used:
            raise RuntimeError(f"Silero did not bind CUDA, providers={used}")
        self._in_names = [i.name for i in self.session.get_inputs()]
        self._out_names = [o.name for o in self.session.get_outputs()]
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, window: np.ndarray) -> float:
        if window.shape != (VAD_WINDOW,):
            raise ValueError(f"expected {VAD_WINDOW} samples, got {window.shape}")
        x = np.concatenate([self._context, window.reshape(1, -1)], axis=1).astype(
            np.float32
        )
        feeds: dict[str, np.ndarray] = {}
        for name in self._in_names:
            if name == "input":
                feeds[name] = x
            elif name in ("state", "h", "c"):
                feeds[name] = self._state
            elif name == "sr":
                feeds[name] = np.array(SAMPLE_RATE_IN, dtype=np.int64)
        outs = self.session.run(self._out_names, feeds)
        by_name = dict(zip(self._out_names, outs, strict=True))
        prob = float(np.squeeze(by_name.get("output", outs[0])))
        new_state = by_name.get("stateN", by_name.get("state", None))
        if new_state is None and len(outs) > 1:
            new_state = outs[1]
        if new_state is not None:
            self._state = np.asarray(new_state, dtype=np.float32)
        self._context = x[:, -64:]
        return prob


class StreamingVAD:
    def __init__(
        self,
        engine: SileroOnnx | None = None,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = MIN_SPEECH_MS,
        min_silence_ms: int = MIN_SILENCE_MS,
        preroll_ms: int = PREROLL_MS,
    ) -> None:
        self.engine = engine or SileroOnnx()
        self.threshold = threshold
        self.min_speech_windows = max(1, round(min_speech_ms / 1000 * SAMPLE_RATE_IN / VAD_WINDOW))
        self.min_silence_windows = max(1, round(min_silence_ms / 1000 * SAMPLE_RATE_IN / VAD_WINDOW))
        self.preroll_windows = max(1, round(preroll_ms / 1000 * SAMPLE_RATE_IN / VAD_WINDOW))
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=self.preroll_windows)
        self._utterance: list[np.ndarray] = []
        self._in_speech = False
        self._voice_run = 0
        self._silence_run = 0
        self._speech_windows = 0

    def reset(self) -> None:
        self.engine.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._utterance.clear()
        self._in_speech = False
        self._voice_run = 0
        self._silence_run = 0
        self._speech_windows = 0

    def feed(self, samples: np.ndarray) -> Iterator[tuple[str, SpeechSegment | None]]:
        if samples.size == 0:
            return
        buf = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32)])
        offset = 0
        while offset + VAD_WINDOW <= buf.size:
            window = buf[offset : offset + VAD_WINDOW].copy()
            offset += VAD_WINDOW
            yield from self._step(window)
        self._pending = buf[offset:]

    def _step(self, window: np.ndarray) -> Iterator[tuple[str, SpeechSegment | None]]:
        prob = self.engine.probability(window)
        voiced = prob >= self.threshold
        if not self._in_speech:
            self._preroll.append(window)
            if voiced:
                self._voice_run += 1
                if self._voice_run >= self.min_speech_windows:
                    self._in_speech = True
                    self._utterance = list(self._preroll)
                    self._speech_windows = len(self._utterance)
                    self._silence_run = 0
                    self._preroll.clear()
                    yield ("speech_start", None)
            else:
                self._voice_run = 0
            return

        self._utterance.append(window)
        self._speech_windows += 1
        if voiced:
            self._silence_run = 0
            return
        self._silence_run += 1
        if self._silence_run < self.min_silence_windows:
            return
        audio = np.concatenate(self._utterance) if self._utterance else window
        duration_ms = int(round(audio.size / SAMPLE_RATE_IN * 1000))
        self.engine.reset()
        self._utterance.clear()
        self._preroll.clear()
        self._in_speech = False
        self._voice_run = 0
        self._silence_run = 0
        self._speech_windows = 0
        yield ("speech_end", SpeechSegment(audio=audio, duration_ms=duration_ms))
