---
name: apredizajes
description: En este archivo se encuentran los apredizajes recaudados de los agentes. Aca escribimos errores que se han cometido y cómo se corrigieron o approaches que tomó el modelo de lenguaje pero no fueron los adecuados. Este archivo es mantenido y actualizado por el agente.
---

# Aprendizajes

En este archivo se encuentran los apredizajes recaudados de los agentes. Aca escribimos errores que se han cometido y cómo se corrigieron o approaches que tomó el modelo de lenguaje pero no fueron los adecuados. Este archivo es mantenido y actualizado por el agente.

## Errores

### Python defaulted to CPU

#### El MVP de voz arrancó Whisper en `cpu`/`int8`/`small` y Silero/Kokoro con `CPUExecutionProvider`. Python corre en la RTX 5080: todo el audio (VAD, STT, TTS) tiene que ir por CUDA, sin fallback a CPU.

#### Corrección

Defaults `WHISPER_MODEL=large-v3-turbo`, `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`. ONNX con `onnxruntime-gpu` y `CUDAExecutionProvider` (Silero + Kokoro vía `Kokoro.from_session`). Si no hay CUDA, el proceso aborta. No instalar el paquete CPU `onnxruntime` (conflicto con `onnxruntime-gpu`; override en `pyproject.toml`).

### Kokoro create_stream vs cancel

#### `Kokoro.create_stream` lanza `process_batches` con `asyncio.create_task` y no lo cancela cuando el consumer se corta. `cancel` / un speak nuevo suelta el lock de la conexión mientras `InferenceSession.run` sigue. ORT no es thread-safe: crash o audio basura.

#### Corrección

No usar `create_stream`. `asyncio.to_thread(kokoro.create)` detrás de un `asyncio.Lock` en `TTS`, con `asyncio.shield` para que un `task.cancel()` en disconnect no libere el lock antes de que termine `create`.
