from __future__ import annotations

import numpy as np


def s16le_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(data, dtype="<i2")
    return ints.astype(np.float32) / 32768.0


def float32_to_s16le(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    return ints.tobytes()


class SampleFifo:
    """Growable float32 FIFO for reassembling PCM into fixed windows."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._buf.size)

    def push(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.float32).ravel()
        if x.size == 0:
            return
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()

    def pop(self, n: int) -> np.ndarray | None:
        if self._buf.size < n:
            return None
        out = self._buf[:n].copy()
        self._buf = self._buf[n:]
        return out

    def clear(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
