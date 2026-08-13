from __future__ import annotations

import logging
import os

import onnxruntime as ort

log = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"


def cuda_providers() -> list[tuple[str, dict[str, int]]]:
    """ONNX Runtime must run on CUDA. No CPU fallback."""
    available = ort.get_available_providers()
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            "CUDA is required (speech_server runs on the RTX 5080). "
            f"onnxruntime providers={available}. "
            "Install onnxruntime-gpu (not the CPU package onnxruntime) "
            "and check the NVIDIA driver."
        )
    os.environ["ONNX_PROVIDER"] = CUDA_PROVIDER
    device_id = int(os.environ.get("CUDA_DEVICE", "0"))
    log.info("onnx CUDA device_id=%s providers=%s", device_id, available)
    return [(CUDA_PROVIDER, {"device_id": device_id})]


def require_whisper_cuda(device: str) -> None:
    if device != "cuda":
        raise RuntimeError(
            f"WHISPER_DEVICE={device!r}; only cuda is supported on the 5080."
        )
