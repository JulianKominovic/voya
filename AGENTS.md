# AGENTS.md

Instructions for agents. Humans: [README.md](README.md). Product: [IDEA.md](IDEA.md). Orchestrator: [FLOW.md](FLOW.md).

## What it is

localhost voice: Chrome → Node `:8787` (turns, barge-in, LLMs) → Python `:8765` (Silero + Whisper + Kokoro). Echo TTS is debug.

North: voice coding agent ([IDEA.md](IDEA.md)). Don't add file/shell tools, React, AudioWorklet, or Pipecat unless asked.

## Limits

- **Python = dumb inference.** No orchestration, no LLM, no tools. VAD + STT in the same process.
- **Python = CUDA only** (RTX 5080). Defaults: Whisper `large-v3-turbo` / `cuda` / `float16`. ONNX: `onnxruntime-gpu` + `CUDAExecutionProvider`. Without CUDA the process aborts. Don't install the CPU package `onnxruntime` (conflict; there's an override in `python/pyproject.toml`). Don't add a CPU fallback.
- **Node = the only client-facing server.** PCM proxy + voice state machine + orchestrator/agentic/minions. It doesn't run VAD/STT/TTS.
- The frontend talks only to Node. Python isn't public.
- Chrome, not Safari. Headphones: the speaker leaks into the mic.
- Mic is **always forwarded** (no TTS mute gate). Gating kills barge-in.

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
- `python/models/` isn't committed. Env: `python/speech_server/config.py`. Node env: `node/.env` (`OPENROUTER_API_KEY`, `ORCHESTRATOR_MODEL`, `AGENTIC_MODEL`).
- Session jsonl: `node/logs/` (gitignored).

## Contracts

- Mic → Python: PCM s16le mono **16 kHz**. TTS → browser: PCM s16le mono **24 kHz**.
- WS: binary = PCM, text = JSON.
- speech-in → Node: `speech_start`, `speech_end`, `transcript`, `error`.
- Node → tts: `speak`, `cancel`. tts → Node: `audio_start`, PCM, `audio_end`. Cancelled utterances never get `audio_end`.
- Node → client: `state`, `question_asked`, `question_resolved`, plus the speech/tts events.
- Types: `python/speech_server/protocol.py`. Sequencing: [FLOW.md](FLOW.md).

## Layout

- `python/speech_server/` — FastAPI, `/health`, `/ws/speech-in`, `/ws/tts`
- `node/src/server.ts` — static files + client WS + Python reconnect
- `node/src/session.ts` — voice states, `generation_id`, barge-in, TTS one-in-flight
- `node/src/agents.ts` — orchestrator + agentic mailbox + minions
- `node/src/questions.ts` — FIFO `ask_user`
- `node/src/memory.ts` — LLM window + jsonl log
- `node/src/app.ts` — Chrome UI (`ScriptProcessor`); `tsc` emits `public/app.js`

## Style

- Python 3.11+, `from __future__ import annotations`, config via env.
- Node ESM + TypeScript (`tsx`). Frontend without bundler (`tsc` → `public/app.js`).
- Minimal: no new deps if stdlib suffices; the shortest diff wins.
- Ponytail (`.cursor/skills/ponytail/`) if they ask for lazy / yagni / ponytail.
- Agent errors: update `.cursor/skills/apredizajes/SKILL.md`.

## Tests

No suite. Don't add pytest/jest just in case. Non-trivial logic: an `assert` / `__main__` if needed.
