"""Download Silero VAD and Kokoro ONNX weights into python/models/."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

from speech_server.config import KOKORO_ONNX, KOKORO_VOICES, MODELS_DIR, SILERO_ONNX

# silero-vad v5.1.2 (512-sample windows). Do not use master — v6 changed the ONNX I/O.
SILERO_URL = (
    "https://github.com/snakers4/silero-vad/raw/"
    "6478567951ae5c9979ad7b234185b5515f4be7a1/src/silero_vad/data/silero_vad.onnx"
)
KOKORO_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


def _hook(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block * block_size, total)
    pct = done * 100 // total
    mb = done / (1024 * 1024)
    total_mb = total / (1024 * 1024)
    sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
    sys.stdout.flush()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urlretrieve(url, tmp, reporthook=_hook)
        print()
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"saved {dest}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fetch(SILERO_URL, SILERO_ONNX)
    fetch(KOKORO_URL, KOKORO_ONNX)
    fetch(VOICES_URL, KOKORO_VOICES)
    print("done.")


if __name__ == "__main__":
    main()
