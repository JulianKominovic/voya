# AGENTS.md

Instrucciones para agentes. Humanos: [README.md](README.md). Producto: [IDEA.md](IDEA.md).

## Qué es

MVP de voz localhost: Chrome → Node `:8787` → Python `:8765` (Silero + Whisper + Kokoro). Echo TTS. **Sin LLM ni tools.**

Norte: agente de código por voz ([IDEA.md](IDEA.md)). No adelantar LLM, tools, React, AudioWorklet ni Pipecat salvo que lo pidan.

## Límites

- **Python = inferencia tonta.** No orquesta, no LLM, no tools. VAD + STT en el mismo proceso.
- **Python = solo CUDA** (RTX 5080). Defaults: Whisper `large-v3-turbo` / `cuda` / `float16`. ONNX: `onnxruntime-gpu` + `CUDAExecutionProvider`. Sin CUDA el proceso aborta. No instalar el paquete CPU `onnxruntime` (conflicto; hay override en `python/pyproject.toml`). No agregar fallback a CPU.
- **Node = único server de cara al cliente.** Proxy de PCM. No corre VAD/STT/TTS.
- Frontend habla solo con Node. Python no es público.
- Chrome, no Safari. Auriculares: el parlante se cuela al mic.

## Correr

```bash
# 5080
cd python && uv sync && uv run python -m speech_server.download_models
uv run uvicorn speech_server.main:app --host 0.0.0.0 --port 8765

# Mac (o misma máquina)
cd node && npm i && npm start
# otra máquina: SPEECH_URL=ws://<ip-5080>:8765/ws/speech-in TTS_URL=ws://<ip-5080>:8765/ws/tts npm start
```

- Node: [http://127.0.0.1:8787](http://127.0.0.1:8787). Python: `GET /health` — `vad_providers` / `tts_providers` deben incluir `CUDAExecutionProvider`.
- `python/models/` no se commitea. Env: `python/speech_server/config.py`.

## Contratos

- Mic → Python: PCM s16le mono **16 kHz**. TTS → browser: PCM s16le mono **24 kHz**.
- WS: binario = PCM, texto = JSON.
- speech-in → Node: `speech_start`, `speech_end`, `transcript`, `error`.
- Node → tts: `speak`, `cancel`. tts → Node: `audio_start`, PCM, `audio_end`.
- Tipos: `python/speech_server/protocol.py`.

## Layout

- `python/speech_server/` — FastAPI, `/health`, `/ws/speech-in`, `/ws/tts`
- `node/src/server.mjs` — estáticos + un WS de cliente + proxy
- `node/public/` — HTML/JS vanilla (`ScriptProcessor`)

## Estilo

- Python 3.11+, `from __future__ import annotations`, config por env.
- Node ESM, stdlib + `ws`. Frontend sin bundler.
- Mínimo: no deps nuevas si stdlib alcanza; el diff más corto gana.
- Ponytail (`.cursor/skills/skills/ponytail/`) si piden lazy / yagni / ponytail.
- Errores de agentes: actualizar `.cursor/skills/skills/apredizajes/SKILL.md`.

## Tests

No hay suite. No agregar pytest/jest por si acaso. Lógica no trivial: un `assert` / `__main__` si hace falta.
