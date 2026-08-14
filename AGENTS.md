# AGENTS.md

Instructions for agents. Humans: [README.md](README.md). Product: [IDEA.md](IDEA.md).

## What it is

localhost voice MVP: Chrome → Node `:8787` → Python `:8765` (Silero + Whisper + Kokoro). Echo TTS. **No LLM or tools.**

North: voice coding agent ([IDEA.md](IDEA.md)). Don't get ahead with LLM, tools, React, AudioWorklet, or Pipecat unless asked.

## Limits

- **Python = dumb inference.** No orchestration, no LLM, no tools. VAD + STT in the same process.
- **Python = CUDA only** (RTX 5080). Defaults: Whisper `large-v3-turbo` / `cuda` / `float16`. ONNX: `onnxruntime-gpu` + `CUDAExecutionProvider`. Without CUDA the process aborts. Don't install the CPU package `onnxruntime` (conflict; there's an override in `python/pyproject.toml`). Don't add a CPU fallback.
- **Node = the only client-facing server.** PCM proxy. It doesn't run VAD/STT/TTS.
- The frontend talks only to Node. Python isn't public.
- Chrome, not Safari. Headphones: the speaker leaks into the mic.

## Run

```bash
# 5080
cd python && uv sync && uv run python -m speech_server.download_models
uv run uvicorn speech_server.main:app --host 0.0.0.0 --port 8765

# Mac (or same machine)
cd node && npm i && npm start
# another machine: SPEECH_URL=ws://<ip-5080>:8765/ws/speech-in TTS_URL=ws://<ip-5080>:8765/ws/tts npm start
```

- Node: [http://127.0.0.1:8787](http://127.0.0.1:8787). Python: `GET /health` — `vad_providers` / `tts_providers` must include `CUDAExecutionProvider`.
- `python/models/` isn't committed. Env: `python/speech_server/config.py`.

## Contracts

- Mic → Python: PCM s16le mono **16 kHz**. TTS → browser: PCM s16le mono **24 kHz**.
- WS: binary = PCM, text = JSON.
- speech-in → Node: `speech_start`, `speech_end`, `transcript`, `error`.
- Node → tts: `speak`, `cancel`. tts → Node: `audio_start`, PCM, `audio_end`.
- Types: `python/speech_server/protocol.py`.

## Layout

- `python/speech_server/` — FastAPI, `/health`, `/ws/speech-in`, `/ws/tts`
- `node/src/server.ts` — static files + one client WS + proxy
- `node/src/app.ts` — Chrome UI (`ScriptProcessor`); `tsc` emits `public/app.js`

## Style

- Python 3.11+, `from __future__ import annotations`, config via env.
- Node ESM + TypeScript (`tsx`). Frontend without bundler (`tsc` → `public/app.js`).
- Minimal: no new deps if stdlib suffices; the shortest diff wins.
- Ponytail (`.cursor/skills/skills/ponytail/`) if they ask for lazy / yagni / ponytail.
- Agent errors: update `.cursor/skills/skills/apredizajes/SKILL.md`.

## Tests

No suite. Don't add pytest/jest just in case. Non-trivial logic: an `assert` / `__main__` if needed.
