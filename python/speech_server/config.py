from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

SILERO_ONNX = MODELS_DIR / "silero_vad.onnx"
KOKORO_ONNX = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "es")

VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "250"))
MIN_SILENCE_MS = int(os.environ.get("MIN_SILENCE_MS", "700"))
PREROLL_MS = int(os.environ.get("PREROLL_MS", "300"))

TTS_VOICE = os.environ.get("TTS_VOICE", "ef_dora")
TTS_LANG = os.environ.get("TTS_LANG", "es")

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_TTS = 24000
VAD_WINDOW = 512
