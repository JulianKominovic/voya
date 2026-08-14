from __future__ import annotations

import ctypes
import logging
import os
import sysconfig
from pathlib import Path

import onnxruntime as ort

log = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"


def _preload_cuda_runtime() -> None:
    """Load the pip-provided CUDA/cuDNN libs before onnxruntime dlopens its provider.

    onnxruntime-gpu does not search site-packages/nvidia/*/lib on Linux, so the
    CUDA 13 runtime + cuDNN 9 must be registered in the process namespace first.
    Loading by SONAME satisfies the provider lib's DT_NEEDED entries at session time.
    """
    base = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not base.exists():
        return
    order = ("libcudart", "libcublasLt", "libcublas", "libcurand", "libcudnn")
    for prefix in order:
        for so in sorted(base.glob(f"*/lib/{prefix}.so*")):
            try:
                ctypes.CDLL(str(so))
            except OSError as exc:
                log.warning("could not preload %s: %s", so, exc)


def cuda_providers() -> list[tuple[str, dict[str, int]]]:
    """ONNX Runtime must run on CUDA. No CPU fallback."""
    _preload_cuda_runtime()
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
