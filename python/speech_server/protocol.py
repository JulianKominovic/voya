from __future__ import annotations

from typing import Any, Literal, TypedDict


class SpeechStart(TypedDict):
    type: Literal["speech_start"]
    t: float


class SpeechEnd(TypedDict):
    type: Literal["speech_end"]
    t: float
    duration_ms: int


class Transcript(TypedDict):
    type: Literal["transcript"]
    t: float
    text: str
    language: str
    stt_ms: int


class ErrorMsg(TypedDict):
    type: Literal["error"]
    message: str


class SpeakCmd(TypedDict):
    type: Literal["speak"]
    id: str
    text: str
    voice: str
    lang: str


class CancelCmd(TypedDict):
    type: Literal["cancel"]
    id: str


class AudioStart(TypedDict):
    type: Literal["audio_start"]
    id: str
    sample_rate: int
    channels: int
    format: Literal["s16le"]


class AudioEnd(TypedDict):
    type: Literal["audio_end"]
    id: str
    synth_ms: int


def now_s() -> float:
    import time

    return time.time()


def error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}
