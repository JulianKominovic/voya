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

_RTLD = getattr(ctypes, "RTLD_GLOBAL", 0)


def _nvidia_root() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / "nvidia"


def _dlopen(so: Path) -> None:
    ctypes.CDLL(str(so), mode=_RTLD)


def _find_so(base: Path, name: str) -> Path | None:
    matches = sorted(base.glob(f"*/lib/{name}"))
    if not matches:
        matches = sorted(
            so
            for so in base.glob(f"*/lib/{name}*")
            if ".so" in so.name
        )
    return matches[0] if matches else None


def _link_cudnn_for_ort(so9: Path) -> None:
    """ORT CUDA EP dlopens libcudnn.so from $ORIGIN (capi/). Same cu13 lib, not a 12↔13 alias."""
    capi = Path(ort.__file__).resolve().parent / "capi"
    if not capi.is_dir():
        return
    target = so9.resolve()
    for name in ("libcudnn.so", "libcudnn.so.9"):
        dest = capi / name
        try:
            if dest.exists() or dest.is_symlink():
                if dest.resolve() == target:
                    continue
                dest.unlink()
            dest.symlink_to(target)
            log.info("ort capi %s -> %s", dest.name, target)
        except OSError as exc:
            log.warning("could not link %s: %s", dest, exc)


def _preload_libs(
    names: tuple[str, ...],
    *,
    required: bool = False,
    what: str = "CUDA libs",
) -> dict[str, Path]:
    """Load pip NVIDIA .so files by SONAME so dlopen finds them later.

    onnxruntime-gpu / CTranslate2 do not search site-packages/nvidia/*/lib.
    """
    base = _nvidia_root()
    found: dict[str, Path] = {}
    if not base.exists():
        if required:
            raise RuntimeError(
                "CUDA pip libs missing under site-packages/nvidia. Run: uv sync"
            )
        return found
    missing: list[str] = []
    for name in names:
        so = _find_so(base, name)
        if so is None:
            missing.append(name)
            continue
        try:
            _dlopen(so)
        except OSError as exc:
            log.warning("could not preload %s: %s", so, exc)
            missing.append(name)
            continue
        found[name] = so
    if required and missing:
        hint = ""
        if any("cudnn" in m for m in missing):
            hint = (
                " nvidia-cudnn-cu12 shares nvidia/cudnn/lib with cu13; "
                "a uv sync that removed cu12 can delete libcudnn. On the 5080: "
                "uv sync --extra qwen --reinstall-package nvidia-cudnn-cu13"
            )
        elif what.startswith("CTranslate2"):
            hint = (
                " On the 5080: uv sync (nvidia-cublas-cu12 + nvidia-cuda-runtime-cu12). "
                "Do not symlink .so.12 -> .so.13."
            )
        raise RuntimeError(f"{what} missing={missing}.{hint}")
    return found


def cuda_providers() -> list[tuple[str, dict[str, int]]]:
    """ONNX Runtime must run on CUDA. No CPU fallback."""
    found = _preload_libs(_ORT_LIBS, required=True, what="ORT CUDA 13")
    so9 = found.get("libcudnn.so.9")
    if so9 is not None:
        _link_cudnn_for_ort(so9)
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
    _preload_libs(_CT2_LIBS, required=True, what="CTranslate2 CUDA 12")
