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
