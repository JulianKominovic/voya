# IDEA — Conversational coding agent (voice)

Product and architecture document. Original source: [Grok conversation](https://grok.com/share/c2hhcmQtMw_0a536557-7394-4f73-8ea1-a1fa0ee818b4). Pipeline reference: [ElBruno.Realtime](https://github.com/elbruno/ElBruno.Realtime).

## What we want

An app to program **100% in conversation**. It's not a chat next to the code: it's talking.

- The AI **narrates** what it's doing while it works.
- It **discusses** technical decisions (trade-offs, alternatives) before or while applying changes.
- The user **doesn't want to read**. The main channel is voice: talk and be talked back to.
- The system must **do** (read/edit files, run commands, tools), not just chat.

Current tools (Cursor, Claude Code, Aider, etc.) get close to the agentic style, but none deliver that end-to-end oral experience.

## Constraints

| Constraint | Implication |
| --- | --- |
| Zero reading / all oral | Voice UX, not editor UX. Audible feedback while processing. |
| Avoid expensive realtime APIs | No OpenAI Realtime / Gemini Live as main path (5×–20× more expensive). |
| Hardware | RTX 5080 16 GB for local audio. LLM in the cloud (DeepSeek or another cheap one). |
| Tool calling | Mandatory: files, shell, codebase. Moshi/PersonaPlex alone don't cut it. |
| Perceived latency | Goal: ~1–2 s from when the user stops speaking until the AI starts responding. |

## Architecture decision

**Classic pipeline, not end-to-end speech-to-speech.** Split: audio inference in Python (RTX 5080), brain in Node.

```
frontend → Node (only client-facing server: turns, LLM, tools)
               │ PCM live (proxy, no VAD)
               ▼
          Python/5080  speech-in (Silero + Whisper)
               │ transcript
               ▼
          Node → LLM/tools → sentences
               ▼
          Python/5080  tts (Kokoro) → Node → speaker
```

Why this path and not a single model (Moshi, PersonaPlex, Mini-Omni, Qwen-Omni):

- Those models give full-duplex and low latency, but **have no native tool calling** and reason worse for code.
- The pipeline keeps the "brain" in a text LLM with tools (DeepSeek, Claude, Qwen, etc.) and audio stays local → cost is almost only the LLM.
- The intermediate text is an advantage: it can be logged, debugged, and fed to tools.

ElBruno.Realtime **already proved** this pipeline feels genuinely conversational. It's the reference to copy in spirit (orchestration + pluggable providers), not necessarily the runtime (.NET).

## Why ElBruno.Realtime convinces

Repo: [elbruno/ElBruno.Realtime](https://github.com/elbruno/ElBruno.Realtime)

A .NET framework for real-time local-first audio conversation, with the same pipeline we chose:

```
Microphone (PCM)
    → Silero VAD (~2 MB ONNX)
    → Whisper STT (Whisper.net, GGML)
    → any IChatClient (Ollama / OpenAI / Azure)
    → pluggable TTS (Kokoro, QwenTTS, VibeVoice)
    → Speaker
```

What matters from their results:

1. **Smart VAD** — doesn't wait for a "ready"; cuts the turn at ~250 ms of minimum speech and ~300 ms of silence. That's what makes a batch STT *feel* realtime.
2. **End-to-end streaming** — `ConverseAsync` emits events: transcript ready → LLM text chunks → TTS audio chunks. The user hears the response as soon as there's first audio, not when everything finishes.
3. **Interchangeable providers** — STT, TTS, VAD and LLM were swapped independently. Fits "local STT/TTS + cheap remote LLM".
4. **Text at every stage** — unlike PersonaPlex/Moshi (audio in → audio out, no text), here there's transcription and textual response. Essential for a coding agent.
5. **They themselves discarded PersonaPlex for interactive use** — the ONNX backbone is ~13.3 GB and slower. The component pipeline is what they use in realtime scenarios (console, SignalR, Blazor).

Their useful samples as UX north: mic conversation in console, API + SignalR, Blazor UI with timeline / audio level / latency.

**What not to copy blindly**

- .NET / Blazor runtime: Python only for GPU audio; Node for the agent. The idea is the *pipeline*, not the language.
- Whisper `tiny.en` / `base.en`: we need **Spanish** (and probably English code). Go with `large-v3-turbo` / faster-whisper on GPU.
- Local LLM like phi4-mini: we want a cheap cloud model **with strong tools**.
- QwenTTS ~5.5 GB: on the 5080, Kokoro 82M is lighter and good enough; Qwen/VibeVoice as a quality upgrade.

## Research (from the Grok conversation)

### OpenRouter

| Capability | Streaming | Notes |
| --- | --- | --- |
| TTS `/api/v1/audio/speech` | Yes (bytes, PCM or MP3) | PCM for low latency. |
| STT `/api/v1/audio/transcriptions` | No (batch) | ~60 s timeout, multipart up to 25 MB. |

Usable as cloud fallback, not as main path. OpenRouter's bottleneck is STT.

### Costs (order of magnitude, cheap pipeline)

Estimate for 1 h of fluid conversation (user ~15–20 min + AI talking): **~$0.15–$0.40** if STT/TTS are local; if everything were cheap cloud, Whisper turbo STT ~$0.04–$0.10/h, Kokoro ~$0.05–$0.10/h of speech, LLM variable.

Native realtime (OpenAI / Gemini Live): 5×–20× more expensive. Deliberately avoided.

### Local stack on RTX 5080 (16 GB)

| Piece | Model | VRAM | Latency |
| --- | --- | --- | --- |
| VAD | Silero | minimal | 300–600 ms end of speech |
| STT | faster-whisper large-v3-turbo (int8) or Parakeet TDT | 2–6 GB | 200–500 ms / chunk |
| TTS | **Kokoro 82M** | < 1 GB | first chunk 50–150 ms (RTF 0.01–0.03) |
| LLM | DeepSeek (or another) via API | 0 local | 400–800 ms TTFT |

Plenty of VRAM for STT + TTS together. Typical perceived latency: **1.0–1.8 s**. Lowered with TTS streaming + short LLM responses.

### Local speech-to-speech (discarded as brain)

| Model | Full-duplex | On 5080 | Tools | For code |
| --- | --- | --- | --- | --- |
| Moshi (Kyutai) | Yes | Yes, quantized | Not native | Weak |
| PersonaPlex 7B | Yes | Tight / tight-quant | No | Weak |
| Mini-Omni / Mini-Omni2 | Partial | Yes | No | Weak |
| Qwen-Omni | Streaming | Forced | Evolving | Better, immature |

Hugging Face `audio-to-audio` is almost all vocoder / codec / denoise. The real conversational ones in that list: **PersonaPlex** and the Kyutai/Moshi line.

If Moshi is ever used, it would be **voice only**, with an LLM with tools behind it. Not worth the extra complexity today.

## What makes it feel "conversational" without pure realtime

Copied from the research + from what ElBruno already implements:

1. **VAD** — at ~0.6–0.8 s of silence, send the clip. Don't wait for a button (though push-to-talk remains a valid, more predictable mode).
2. **Immediate feedback** — sound or a short phrase ("ok, let me see…") while STT + LLM run.
3. **TTS streaming** — start playing the first chunk, not the whole WAV.
4. **Short responses** — 2–4 sentences, narrate what it's doing, then the next block. TTS starts earlier.
5. **Conversational system prompt** — always explain what and why; discuss important options before touching code.

## Architecture: Python on the 5080 + Node orchestrator

Python is **dumb inference**. Node is **the only client-facing server** and all the logic.

```
  frontend (mic / speaker)
        │ single WebSocket
        ▼
  ┌─────────────────────────────────────────┐
  │  Node                                   │
  │  turns, barge-in, LLM, tools, what to say│
  │  PCM live proxy ↔ Python (no VAD)       │
  └───────┬─────────────────────▲───────────┘
          │ PCM chunks          │ speech_start / transcript
          │ sentence text       │ audio_chunk
          ▼                     │
  ┌─────────────────────────────────────────┐
  │  PC RTX 5080 — Python                   │
  │  WS speech-in: Silero + Whisper (together) │
  │  WS tts: Kokoro streaming               │
  └─────────────────────────────────────────┘
```

Node **does** see PCM, but only as a pipe: it forwards bytes. It doesn't wait for you to finish speaking or build a WAV. Silero stays in Python, on the live stream.

### The mic doesn't need to talk directly to Python

A localhost/LAN PCM proxy adds **~1–2 ms**, irrelevant against 200 ms of STT + 400–800 ms of LLM.

What does hurt (20–80 ms) is something else: **VAD in Node** that buffers the turn and *then* sends the clip. We don't do that.

The frontend talks only to Node. Python isn't public. Simpler, and doesn't change the latency you feel.

### What goes in Python (5080)

Two WebSocket servers, **no brain**:

| Service | Process | Does | Doesn't do |
| --- | --- | --- | --- |
| **speech-in** | Single | Receive live PCM, Silero, trim the turn, transcribe, emit events | Decide whether the turn goes to the LLM, tools, policies |
| **tts** | Another (or the same process if it simplifies GPU) | Text → PCM streaming | Decide what to say, filter code, turns |

Python **doesn't** orchestrate. It may have Silero's minimal state machine (threshold, min speech, min silence): that's signal, not agent.

**Silero goes in Python, in the same process as STT. Not in Node.** Not because it needs the GPU (CPU, <1 ms/chunk), but so that when silence is cut, Whisper starts with the buffer already in RAM.

Don't split VAD and STT into two microservices.

### What goes in Node

All the sequencing:

- Single WS to the frontend (PCM in, PCM out, state).
- Live proxy to speech-in / tts.
- Turn policy: `speech_end` + transcript → send to LLM or discard?
- Barge-in: `speech_start` while TTS is playing → cancel TTS and the LLM.
- Cloud LLM + tool loop (read / edit / shell / search).
- What gets voiced: assistant text only, never diffs or tool JSON.
- Sentence splitter toward TTS.
- Session, workspace, confirmations.

Node **doesn't** run Silero, Whisper, or Kokoro.

### Everything in Python then?

Viable (Pipecat, FastAPI, single process). I don't choose it because of the mic.

| | Node + Python split | All Python |
| --- | --- | --- |
| Voice latency | Same if VAD+STT are together on GPU | Same |
| Where 80% of the code lives | Agent/tools in Node (where the experience is) | Agent/tools in Python |
| Operations | Two runtimes, internal WS contract | One process, one deploy |
| GPU audio ecosystem | Python stays small and boring | Mixes inference + product |
| Risk | Coordinating two servers | Rewriting the agent in a less comfortable language |

The split is worth it because **the agent is the product** and Python just wraps GPU. Unifying into Python to avoid the audio proxy is optimizing 2 ms and paying the cost in the biggest part of the repo.

Reconsider all-Python only if maintaining two processes truly hurts, or if the orchestrator ends up being 200 lines instead of an agent.

### Event contracts (Node ↔ Python)

**speech-in → Node:** `speech_start`, `speech_end`, `partial_transcript?`, `transcript` (final), `error`.

**Node → tts:** `speak` (sentence), `cancel` (barge-in / new turn).

**tts → Node:** `audio_chunk`, `utterance_end` (Node passes them to the client).

Assistant TTS is a **pipeline**, not an LLM tool.

### Differences vs ElBruno

| ElBruno.Realtime | This app |
| --- | --- |
| One .NET process does VAD+STT+LLM+TTS | Python = GPU audio; Node = agent |
| Generic voice assistant | Coding agent that talks |
| Optional local LLM (phi4-mini) | Cloud LLM with tools (DeepSeek / similar) |
| Whisper tiny.en | Whisper turbo multilingual (ES + EN code) |
| QwenTTS default | Kokoro first |
| Conversational turn | Turn + tool loop + narration |

## Tentative stack

- **5080 / Python:** FastAPI (or equivalent) + WebSocket. Silero + faster-whisper large-v3-turbo + Kokoro 82M.
- **Node:** orchestrator, LLM, tools, turns. Frontend later.
- **Cloud:** DeepSeek (or another cheap one with function calling) via OpenRouter or direct API.
- **Cloud fallback:** OpenRouter STT/TTS if the 5080 is down.

## Open / to decide

- One Python process (speech-in + tts, less GPU overhead) vs two processes (crash isolation).
- Mic/speaker on the 5080 PC vs on another machine (LAN). PCM still goes through Node toward Python.
- Push-to-talk vs continuous VAD vs both.
- How to narrate tools without reading diffs: oral summary, confirmation before big edits.
- Agent workspace / sandbox (cwd, git, permissions).
- Voice language: Spanish for conversation, English for code identifiers.

## Natural next step

1. On the 5080: `speech-in` WS (VAD+STT) and `tts` WS, without LLM.
2. Minimal Node: `transcript` → LLM stream (no tools) → sentences → TTS. Measure end-to-end latency.
3. If it feels conversational, hook up tools.
