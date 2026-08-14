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
