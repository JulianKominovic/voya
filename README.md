# Voya — Voice MVP

Speak into the mic, watch the transcript, hear the same text through Kokoro. No LLM or tools.

Python is **CUDA-only** (RTX 5080). Node + Chrome can live on the Mac.

```
HTML (Chrome)  →  Node :8787  →  Python :8765  (5080, CUDA)
  PCM 16 kHz         proxy          Silero + Whisper
  PCM 24 kHz         echo TTS       Kokoro stream
```

Use **headphones**. Node stops forwarding the mic while TTS is playing, but the speaker still leaks into the mic.

Chrome, not Safari (Safari fights `AudioContext.sampleRate`).

## Requirements

**5080 (Python)**

- Recent NVIDIA driver (Blackwell / sm_120)
- [uv](https://docs.astral.sh/uv/)
- `espeak-ng` (Spanish G2P): `sudo apt install espeak-ng` or equivalent
- `onnxruntime-gpu` (not the CPU package `onnxruntime`)

**Mac (Node + mic)** — or the same machine if the mic is there

- Node 18+, Chrome

## Run

On the 5080:

```bash
cd python
uv sync
uv run python -m speech_server.download_models
uv run uvicorn speech_server.main:app --host 0.0.0.0 --port 8765
```

Starts on CUDA or fails. No CPU fallback.

On the Mac (or local):

```bash
cd node
npm i
SPEECH_URL=ws://<ip-5080>:8765/ws/speech-in \
TTS_URL=ws://<ip-5080>:8765/ws/tts \
  npm start
```

If Node runs on the same machine as Python, `npm start` is enough (defaults to `127.0.0.1:8765`).

Chrome → [http://127.0.0.1:8787](http://127.0.0.1:8787)

1. **Synthesize** a phrase — validates Kokoro CUDA + 24 kHz playback.
2. Headphones → **Speak**.

First run downloads Whisper `large-v3-turbo` (Hugging Face cache) and Kokoro + Silero to `python/models/`.

`GET http://<ip-5080>:8765/health` — `vad_providers` / `tts_providers` must include `CUDAExecutionProvider`. `stt` turns true once Whisper finishes loading.

## Ports and env

| | default | env |
|---|---|---|
| Python WS | `0.0.0.0:8765` | — |
| Node HTTP/WS | `0.0.0.0:8787` | `PORT` |
| speech-in | `ws://127.0.0.1:8765/ws/speech-in` | `SPEECH_URL` |
| tts | `ws://127.0.0.1:8765/ws/tts` | `TTS_URL` |
| Whisper | `large-v3-turbo` / `cuda` / `float16` / `es` | `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_LANGUAGE` |
| GPU | device `0` | `CUDA_DEVICE` |
| VAD | threshold `0.5`, min speech 250 ms, silence 700 ms, pre-roll 300 ms | `VAD_THRESHOLD`, `MIN_SPEECH_MS`, `MIN_SILENCE_MS`, `PREROLL_MS` |
| TTS | voice `ef_dora`, lang `es` | `TTS_VOICE`, `TTS_LANG` |
| models | `python/models` | `MODELS_DIR` |

A `WHISPER_DEVICE` other than `cuda` aborts the process.

## Audio

- Mic → Python: PCM s16le mono **16 kHz**. Chunks ~20–40 ms; Silero regroups into 512 samples (32 ms).
- TTS → browser: PCM s16le mono **24 kHz**. WebSocket: binary = PCM, text = JSON.

## Layout

- `python/speech_server/` — one FastAPI, `GET /health`, `WS /ws/speech-in`, `WS /ws/tts`
- `node/src/server.mjs` — static files + one client WS + proxy to Python
- `node/public/` — minimal HTML (ScriptProcessor, not AudioWorklet)
