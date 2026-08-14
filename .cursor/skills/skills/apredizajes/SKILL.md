---
name: apredizajes
description: This file holds the lessons collected from the agents. Here we write errors that were made and how they were fixed, or approaches the language model took that were not appropriate. This file is maintained and updated by the agent.
---

# Lessons

This file holds the lessons collected from the agents. Here we write errors that were made and how they were fixed, or approaches the language model took that were not appropriate. This file is maintained and updated by the agent.

## Errors

### Python defaulted to CPU

#### The voice MVP started Whisper on `cpu`/`int8`/`small` and Silero/Kokoro with `CPUExecutionProvider`. Python runs on the RTX 5080: all audio (VAD, STT, TTS) must go through CUDA, no CPU fallback.

#### Fix

Defaults `WHISPER_MODEL=large-v3-turbo`, `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`. ONNX with `onnxruntime-gpu` and `CUDAExecutionProvider` (Silero + Kokoro via `Kokoro.from_session`). If there's no CUDA, the process aborts. Don't install the CPU package `onnxruntime` (conflict with `onnxruntime-gpu`; override in `pyproject.toml`).

### Whisper died on libcublas.so.12 under CUDA 13

#### faster-whisper → CTranslate2 still `DT_NEEDED`s `libcublas.so.12`. The venv only had CUDA 13 pip libs (`nvidia-cublas`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`), so STT crashed after VAD/TTS were already on CUDA 13.

#### Fix

Keep CUDA 13 for ORT. Also install `nvidia-cublas-cu12` + `nvidia-cuda-runtime-cu12` and preload `.so.12` before `WhisperModel`. Do not install `nvidia-cudnn-cu12` (same `libcudnn.so.9` path as cu13). Do not symlink `.so.12` → `.so.13`. Do not put both toolkits on `LD_LIBRARY_PATH`.

### Qwen extra deleted libcudnn (Silero/ORT)

#### `uv sync --extra qwen` pulled torch's `nvidia-cudnn-cu12` into the same `site-packages/nvidia/cudnn/lib` as `nvidia-cudnn-cu13`. A later `uv sync` without the extra uninstalled cu12 and deleted `libcudnn`. VAD then died: ORT `dlopen("libcudnn.so")` → file not found. Qwen also needs `--extra qwen`; plain `uv sync` removes `qwen-tts`.

#### Fix

Override `nvidia-cudnn-cu12; sys_platform == 'never'` in `pyproject.toml` (torch uses cu13's `libcudnn.so.9`). After a broken sync on the 5080: `uv sync --extra qwen --reinstall-package nvidia-cudnn-cu13`. `gpu.py` links `libcudnn.so` into `onnxruntime/capi` (`$ORIGIN`) so ORT finds it.

### TTS speak queued vs seq-cancel

#### Node sends one `speak` per LLM sentence and waits for that many `audio_end` before unmuting the mic (`pendingSpeak`). Python bumped `seq` on every `speak`, which cancelled the previous utterance without `audio_end`. After a 2–4 sentence reply, `pendingSpeak` stuck > 0, the mic stayed muted, and the next turn never started.

#### Fix

Python: increment `seq` only on `cancel` (and disconnect). New `speak` tasks share the current `seq` and queue on the existing lock. Node: decrement `pendingSpeak` only when the `id` is in `ttsWaits`, so a late `audio_end`/`error` after abort cannot desync the next turn.
