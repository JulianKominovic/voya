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

### Extra Qwen borró libcudnn (Silero/ORT)

#### `uv sync --extra qwen` metió el `nvidia-cudnn-cu12` de torch en el mismo `site-packages/nvidia/cudnn/lib` que `nvidia-cudnn-cu13`. Un `uv sync` sin extra desinstaló cu12 y se llevó `libcudnn`. El VAD murió: ORT `dlopen("libcudnn.so")` → no existe. Qwen además necesita `--extra qwen`; un `uv sync` solo saca `qwen-tts`.

#### Corrección

Override `nvidia-cudnn-cu12; sys_platform == 'never'` en `pyproject.toml` (torch usa el `libcudnn.so.9` de cu13). Si el sync ya rompió la 5080: `uv sync --extra qwen --reinstall-package nvidia-cudnn-cu13`. `gpu.py` linkea `libcudnn.so` en `onnxruntime/capi` (`$ORIGIN`) para que ORT lo encuentre. No overridear `sox`: qwen-tts lo importa al cargar (`speech_vq.py`). Gradio sí puede quedar overrideado.

### TTS speak encolado vs seq-cancel

#### Node manda un `speak` por oración del LLM y espera ese número de `audio_end` antes de reenviar el mic (`pendingSpeak`). Python incrementaba `seq` en cada `speak`, lo que cancelaba el utterance anterior sin `audio_end`. Tras una respuesta de 2–4 oraciones, `pendingSpeak` quedaba > 0, el mic mudo, y el turno siguiente no arrancaba.

#### Corrección

Python: `seq += 1` solo en `cancel` (y disconnect). Los `speak` nuevos comparten el `seq` actual y se encolan en el lock. Node: decrementar `pendingSpeak` solo si el `id` está en `ttsWaits`, para que un `audio_end`/`error` tardío después de un abort no desfase el turno siguiente.
