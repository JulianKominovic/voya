# Voya — MVP de voz

Hablar por el mic, ver la transcripción, oír el mismo texto por Kokoro. Sin LLM ni tools.

Python es **solo CUDA** (RTX 5080). Node + Chrome pueden vivir en la Mac.

```
HTML (Chrome)  →  Node :8787  →  Python :8765  (5080, CUDA)
  PCM 16 kHz         proxy          Silero + Whisper
  PCM 24 kHz         echo TTS       Kokoro stream
```

Usá **auriculares**. Node deja de forwardear el mic mientras hay TTS, pero el parlante igual se cuela al mic.

Chrome, no Safari (Safari pelea `AudioContext.sampleRate`).

## Requisitos

**5080 (Python)**

- NVIDIA driver reciente (Blackwell / sm_120)
- [uv](https://docs.astral.sh/uv/)
- `espeak-ng` (G2P español): `sudo apt install espeak-ng` o equivalente
- `onnxruntime-gpu` (no el paquete CPU `onnxruntime`)

**Mac (Node + mic)** — o la misma máquina si el mic está ahí

- Node 18+, Chrome

## Correr

En la 5080:

```bash
cd python
uv sync
uv run python -m speech_server.download_models
uv run uvicorn speech_server.main:app --host 0.0.0.0 --port 8765
```

Arranca en CUDA o falla. No hay fallback a CPU.

En la Mac (o local):

```bash
cd node
npm i
SPEECH_URL=ws://<ip-5080>:8765/ws/speech-in \
TTS_URL=ws://<ip-5080>:8765/ws/tts \
  npm start
```

Si Node corre en la misma máquina que Python, `npm start` alcanza (defaults a `127.0.0.1:8765`).

Chrome → [http://127.0.0.1:8787](http://127.0.0.1:8787)

1. **Sintetizar** una frase — valida Kokoro CUDA + playback 24 kHz.
2. Auriculares → **Hablar**.

Primera corrida baja Whisper `large-v3-turbo` (cache Hugging Face) y Kokoro + Silero a `python/models/`.

`GET http://<ip-5080>:8765/health` — `vad_providers` / `tts_providers` tienen que incluir `CUDAExecutionProvider`. `stt` pasa a true cuando termina de cargar Whisper.

## Puertos y env

| | default | env |
|---|---|---|
| Python WS | `0.0.0.0:8765` | — |
| Node HTTP/WS | `0.0.0.0:8787` | `PORT` |
| speech-in | `ws://127.0.0.1:8765/ws/speech-in` | `SPEECH_URL` |
| tts | `ws://127.0.0.1:8765/ws/tts` | `TTS_URL` |
| Whisper | `large-v3-turbo` / `cuda` / `float16` / `es` | `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_LANGUAGE` |
| GPU | device `0` | `CUDA_DEVICE` |
| VAD | umbral `0.5`, min habla 250 ms, silencio 700 ms, pre-roll 300 ms | `VAD_THRESHOLD`, `MIN_SPEECH_MS`, `MIN_SILENCE_MS`, `PREROLL_MS` |
| TTS | voz `ef_dora`, lang `es` | `TTS_VOICE`, `TTS_LANG` |
| modelos | `python/models` | `MODELS_DIR` |

`WHISPER_DEVICE` distinto de `cuda` aborta el proceso.

## Audio

- Mic → Python: PCM s16le mono **16 kHz**. Chunks ~20–40 ms; Silero reagrupa a 512 samples (32 ms).
- TTS → browser: PCM s16le mono **24 kHz**. WebSocket: binario = PCM, texto = JSON.

## Layout

- `python/speech_server/` — un FastAPI, `GET /health`, `WS /ws/speech-in`, `WS /ws/tts`
- `node/src/server.mjs` — estáticos + un WS de cliente + proxy a Python
- `node/public/` — HTML mínimo (ScriptProcessor, no AudioWorklet)
