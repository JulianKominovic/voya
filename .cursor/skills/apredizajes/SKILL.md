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

### TTS speak queued vs seq-cancel

#### Node manda un `speak` por oración del LLM y espera ese número de `audio_end` antes de reenviar el mic (`pendingSpeak`). Python incrementaba `seq` en cada `speak`, lo que cancelaba el utterance anterior sin `audio_end`. Tras una respuesta de 2–4 oraciones, `pendingSpeak` quedaba > 0, el mic mudo, y el turno siguiente no arrancaba.

#### Corrección

Python: `seq += 1` solo en `cancel` (y disconnect). Los `speak` nuevos comparten el `seq` actual y se encolan en el lock. Node ya no usa `pendingSpeak` para mutear el mic: el mic se reenvía siempre (si no, no hay barge-in). Un `speak` in-flight; tras `cancel` se tira el PCM hasta el próximo `audio_start` de la generación actual.

### Mutear el mic durante TTS mata barge-in

#### El gate `pendingSpeak === 0` en el proxy de PCM hacía que Silero no viera habla mientras Kokoro sonaba. `speech_start` no disparaba y no se podía interrumpir.

#### Corrección

Forward siempre. Debounce de barge-in = `MIN_SPEECH_MS` en el VAD (default 350). Drop de eco en Node: transcripts cortos (<400 ms) que empezaron durante TTS o en los ~300 ms posteriores.

### Qwen TTS no emitió EOS → 36s de ruido

#### Sesión `7ae364fd`: el STT del turno 3 estaba bien (`El número es 10.`). Qwen arrancó `Listo, anotado.` (15 chars) y nunca emitió el EOS del codec. El talker de `faster-qwen3-tts` tiene `max_seq_len=512` por default (~36 s a 12 Hz). Node vio first_pcm y ningún `audio_end` en 34 s hasta el cancel. El usuario escuchó ruido; Whisper no se rompió.

#### Corrección

Capar `max_new_tokens` según el largo del texto (máx 192 ≈ 16 s). `torch.cuda.synchronize()` antes de generar (Whisper/ORT comparten la 5080 con CUDA graphs). En cancel, dejar de pedir chunks en vez de drenar el cap. Loguear `steps`/`peak`; warning si pega el tope. No tratar un EOS perdido como "el STT mandó basura".

### OpenRouter: Provider returned error en el ask agentic

#### Sesión `e78daf75`, turno `aad2`: el orchestrator (`deepseek/deepseek-v4-flash-0731`) armó el `ask` bien. El agentic (`inclusionai/ling-3.0-flash`) falló en ~500 ms con `Provider returned error`. `errMsg` solo mostraba ese string. Repro: Ling **sin** tools → 200 Novita; Ling **con** tools → 429, Novita y DeepInfra en `upstream_provider_shared_pool` (el pool compartido de OpenRouter se acaba en function calling). El orchestrator reintentó `ask` y habló el fallo.

#### Corrección

`errMsg` parsea `statusCode` + `body` del SDK (`429 Novita rate-limited…`). Si `AGENTIC_MODEL` ≠ orchestrator, `streamChat` reintenta una vez con `ORCHESTRATOR_MODEL`.

### OpenRouter: no endpoint found that support tool use

#### El orchestrator manda `tools` siempre. El default `deepseek/deepseek-chat` (y `sort: latency`) caía en providers sin function calling → 404, y eso se hablaba por TTS.

#### Corrección

Default `deepseek/deepseek-chat-v3.1`. Con tools, `provider.requireParameters: true` para que OpenRouter solo elija endpoints que acepten `tools`.

### pnpm exec desde el root de voya

#### `pnpm exec tsc` / `pnpm exec tsx` desde `/Users/julian/dev/voya` (no hay `package.json`) hace que pnpm suba de carpeta y ejecute en otro proyecto de la máquina. Falla o typechequea el repo equivocado. `ERR_PNPM_RECURSIVE_EXEC_FIRST_FAIL`.

#### Corrección

Siempre `working_directory: node/` o `pnpm --dir node …`. El package de Node está en `node/`.

### TTS leyó el system prompt / protocolo de tools

#### El orchestrator (deepseek-v4-flash) metió en `content` el protocolo de `pending_question` y renglones del system (`No inventes el estado de la cola`) más cadena de pensamiento (`Debo…`, `El usuario…`). Node hablaba cada oración en streaming (`onSentence` → TTS) antes de saber si había tool calls. El agentic además usó `ask_user` para esperar al minion, y esa pregunta interna se habló.

#### Corrección

No hablar el `content` de un round si hay tools y ninguna es `ask` (la narración corta sí va con `ask`). Filtrar oraciones que son substring del system, `Instructions:`, o CoT (`Debo` / `El usuario` / `Cuando el usuario` / `Parece que`). Prompt: no leer system/tools; `ask_user` no es para esperar minions.

### Agentic no ve el report del minion → loop de tools

#### Sesión `cd87b954`: el minion `developer` corrió `shell` y `report(done)` con el macOS bien. El mailbox del agentic es serial: ese `minion_done` quedó encolado detrás del `ask` en curso. El agentic no lo vio, spameó `talk_to`/`list_minions` hasta `error: demasiadas herramientas`, y cada `talk_to` encolaba otro `runMinion` (el minion volvía a arrancar y a reportar done). Los dones se procesaron después, con el `ask` ya fallido.

#### Corrección

Si el agentic está en un `ask`, `report(done|status)` se inyecta en el inbox del turno actual (no un job nuevo). `talk_to` no relanza el minion si `busy`. Si llega un done antes de `answer` o al tope de rounds, se devuelve ese texto. Prompt: no pinguear minions con `talk_to`/`list_minions`.

