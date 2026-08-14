from __future__ import annotations

import ctypes
import logging
import os
import sysconfig
from pathlib import Path

import onnxruntime as ort

log = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"

# ORT 1.27+ wants CUDA 13. CTranslate2 wheels still DT_NEED libcublas.so.12.
_ORT_LIBS = (
    "libcudart.so.13",
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcurand.so.10",
    "libcudnn.so.9",
)
_CT2_LIBS = (
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
)


def _nvidia_root() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / "nvidia"


def _preload_libs(names: tuple[str, ...], *, required: bool = False) -> None:
    """Load pip NVIDIA .so files by SONAME so dlopen finds them later.

    onnxruntime-gpu / CTranslate2 do not search site-packages/nvidia/*/lib.
    """
    base = _nvidia_root()
    if not base.exists():
        if required:
            raise RuntimeError(
                "CUDA pip libs missing under site-packages/nvidia. Run: uv sync"
            )
        return
    missing: list[str] = []
    for name in names:
        matches = sorted(base.glob(f"*/lib/{name}"))
        if not matches:
            matches = sorted(
                so
                for so in base.glob(f"*/lib/{name}*")
                if ".so" in so.name
            )
        if not matches:
            missing.append(name)
            continue
        so = matches[0]
        try:
            ctypes.CDLL(str(so))
        except OSError as exc:
            log.warning("could not preload %s: %s", so, exc)
            missing.append(name)
    if required and missing:
        raise RuntimeError(
            "CTranslate2 needs CUDA 12 cuBLAS (libcublas.so.12). "
            f"missing={missing}. On the 5080: uv sync "
            "(nvidia-cublas-cu12 + nvidia-cuda-runtime-cu12). "
            "Do not symlink .so.12 -> .so.13."
        )


def cuda_providers() -> list[tuple[str, dict[str, int]]]:
    """ONNX Runtime must run on CUDA. No CPU fallback."""
    _preload_libs(_ORT_LIBS)
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
    _preload_libs(_CT2_LIBS, required=True)
