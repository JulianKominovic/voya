# IDEA — Agente de programación conversacional (voz)

Documento de producto y arquitectura. Fuente original: [conversación Grok](https://grok.com/share/c2hhcmQtMw_0a536557-7394-4f73-8ea1-a1fa0ee818b4). Referencia de pipeline: [ElBruno.Realtime](https://github.com/elbruno/ElBruno.Realtime).

## Qué queremos

Una app para programar **100% en diálogo**. No es un chat al lado del código: es hablar.

- La IA **narra** qué está haciendo mientras trabaja.
- **Discute** las decisiones técnicas (trade-offs, alternativas) antes o mientras aplica cambios.
- El usuario **no quiere leer**. El canal principal es voz: hablar y que le hablen de vuelta.
- El sistema tiene que **hacer** (leer/editar archivos, correr comandos, tools), no solo charlar.

Las herramientas actuales (Cursor, Claude Code, Aider, etc.) se acercan al estilo agentic, pero ninguna entrega esa experiencia oral de punta a punta.

## Restricciones

| Restricción | Implicación |
| --- | --- |
| Cero lectura / todo oral | UX de voz, no de editor. Feedback audible mientras procesa. |
| Evitar APIs realtime caras | No OpenAI Realtime / Gemini Live como path principal (5×–20× más caros). |
| Hardware | RTX 5080 16 GB para audio local. LLM en la nube (DeepSeek u otro barato). |
| Tool calling | Obligatorio: archivos, shell, codebase. Moshi/PersonaPlex solos no alcanzan. |
| Latencia percibida | Objetivo: ~1–2 s desde que termina de hablar hasta que la IA empieza a responder. |

## Decisión de arquitectura

**Pipeline clásico, no speech-to-speech end-to-end.** Split: inferencia de audio en Python (RTX 5080), cerebro en Node.

```
frontend → Node (único server de cara al cliente: turnos, LLM, tools)
              │ PCM live (proxy, no VAD)
              ▼
         Python/5080  speech-in (Silero + Whisper)
              │ transcript
              ▼
         Node → LLM/tools → oraciones
              ▼
         Python/5080  tts (Kokoro) → Node → speaker
```

Por qué este camino y no un modelo único (Moshi, PersonaPlex, Mini-Omni, Qwen-Omni):

- Esos modelos dan full-duplex y baja latencia, pero **no tienen tool calling nativo** y razonan peor para código.
- El pipeline deja el “cerebro” en un LLM de texto con tools (DeepSeek, Claude, Qwen, etc.) y el audio queda local → costo casi solo del LLM.
- El texto intermedio es una ventaja: se puede loguear, debuggear, y alimentar tools.

ElBruno.Realtime **ya demostró** que este pipeline se siente conversacional de verdad. Es la referencia a copiar en espíritu (orquestación + providers enchufables), no necesariamente el runtime (.NET).

## Por qué ElBruno.Realtime convence

Repo: [elbruno/ElBruno.Realtime](https://github.com/elbruno/ElBruno.Realtime)

Framework .NET de conversación de audio en tiempo real, local-first, con el mismo pipeline que elegimos:

```
Microphone (PCM)
    → Silero VAD (~2 MB ONNX)
    → Whisper STT (Whisper.net, GGML)
    → cualquier IChatClient (Ollama / OpenAI / Azure)
    → TTS pluggable (Kokoro, QwenTTS, VibeVoice)
    → Speaker
```

Lo que importa de sus resultados:

1. **VAD inteligente** — no espera un “listo”; corta el turno con ~250 ms de habla mínima y ~300 ms de silencio. Eso es lo que hace que un STT batch *se sienta* realtime.
2. **Streaming de punta a punta** — `ConverseAsync` emite eventos: transcripción lista → chunks de texto del LLM → chunks de audio TTS. El usuario oye la respuesta apenas hay primer audio, no cuando termina todo.
3. **Providers intercambiables** — STT, TTS, VAD y LLM se swapearon de forma independiente. Encaja con “STT/TTS local + LLM remoto barato”.
4. **Texto en cada etapa** — a diferencia de PersonaPlex/Moshi (audio in → audio out, sin texto), acá hay transcripción y respuesta textual. Imprescindible para un agente de código.
5. **Ellos mismos descartaron PersonaPlex para lo interactivo** — el backbone ONNX son ~13.3 GB y es más lento. El pipeline por componentes es el que usan en los escenarios realtime (consola, SignalR, Blazor).

Sus samples útiles como norte de UX: conversación por micrófono en consola, API + SignalR, UI Blazor con timeline / nivel de audio / latencia.

**Qué no copiar ciegamente**

- Runtime .NET / Blazor: Python solo para GPU audio; Node para el agente. La idea es el *pipeline*, no el lenguaje.
- Whisper `tiny.en` / `base.en`: nosotros necesitamos **español** (y probablemente código en inglés). Ir a `large-v3-turbo` / faster-whisper en GPU.
- LLM local tipo phi4-mini: nosotros queremos un modelo barato en nube **con tools fuertes**.
- QwenTTS ~5.5 GB: en 5080 Kokoro 82M es más liviano y suficientemente bueno; Qwen/VibeVoice como upgrade de calidad.

## Investigación (de la conversación Grok)

### OpenRouter

| Capacidad | Streaming | Notas |
| --- | --- | --- |
| TTS `/api/v1/audio/speech` | Sí (bytes, PCM o MP3) | PCM para baja latencia. |
| STT `/api/v1/audio/transcriptions` | No (batch) | Timeout ~60 s, multipart hasta 25 MB. |

Usable como fallback cloud, no como path principal. El cuello de botella de OpenRouter es el STT.

### Costos (orden de magnitud, pipeline barato)

Estimación de 1 h de conversación fluida (usuario ~15–20 min + IA hablando): **~$0.15–$0.40** si STT/TTS son locales; si todo fuera cloud barato, STT Whisper turbo ~$0.04–$0.10/h, Kokoro ~$0.05–$0.10/h de habla, LLM variable.

Realtime nativo (OpenAI / Gemini Live): 5×–20× más caro. Se esquiva a propósito.

### Stack local en RTX 5080 (16 GB)

| Pieza | Modelo | VRAM | Latencia |
| --- | --- | --- | --- |
| VAD | Silero | mínima | 300–600 ms fin de habla |
| STT | faster-whisper large-v3-turbo (int8) o Parakeet TDT | 2–6 GB | 200–500 ms / chunk |
| TTS | **Kokoro 82M** | < 1 GB | first chunk 50–150 ms (RTF 0.01–0.03) |
| LLM | DeepSeek (u otro) vía API | 0 local | 400–800 ms TTFT |

VRAM de sobra para STT + TTS juntos. Latencia percibida típica: **1.0–1.8 s**. Se baja con TTS streaming + respuestas cortas del LLM.

### Speech-to-speech local (descartado como cerebro)

| Modelo | Full-duplex | En 5080 | Tools | Para código |
| --- | --- | --- | --- | --- |
| Moshi (Kyutai) | Sí | Sí, quantizado | No nativo | Débil |
| PersonaPlex 7B | Sí | Justo / justo-quant | No | Débil |
| Mini-Omni / Mini-Omni2 | Parcial | Sí | No | Débil |
| Qwen-Omni | Streaming | Forzado | En evolución | Mejor, inmaduro |

Hugging Face `audio-to-audio` es casi todo vocoder / codec / denoise. Los conversacionales reales de esa lista: **PersonaPlex** y línea Kyutai/Moshi.

Si algún día se usa Moshi, sería **solo cara de voz**, con un LLM con tools detrás. Hoy no vale la complejidad extra.

## Cómo se siente “conversacional” sin realtime puro

Copiado de la investigación + de lo que ElBruno ya implementa:

1. **VAD** — al ~0.6–0.8 s de silencio, mandar el clip. No esperar un botón (aunque push-to-talk sigue siendo un modo válido, más predecible).
2. **Feedback inmediato** — sonido o frase corta (“ok, déjame ver…”) mientras corre STT + LLM.
3. **TTS streaming** — empezar a reproducir el primer chunk, no el WAV entero.
4. **Respuestas cortas** — 2–4 oraciones, narrar lo que hace, luego el siguiente bloque. El TTS arranca antes.
5. **System prompt de diálogo** — siempre explicar qué hace y por qué; discutir opciones importantes antes de tocar código.

## Arquitectura: Python en la 5080 + Node orquestador

Python es **inferencia tonta**. Node es **el único server de cara al cliente** y toda la lógica.

```
  frontend (mic / speaker)
        │ un solo WebSocket
        ▼
  ┌─────────────────────────────────────────┐
  │  Node                                   │
  │  turnos, barge-in, LLM, tools, qué decir│
  │  proxy PCM live ↔ Python (no corre VAD) │
  └───────┬─────────────────────▲───────────┘
          │ PCM chunks          │ speech_start / transcript
          │ texto oraciones     │ audio_chunk
          ▼                     │
  ┌─────────────────────────────────────────┐
  │  PC RTX 5080 — Python                   │
  │  WS speech-in: Silero + Whisper (junto) │
  │  WS tts: Kokoro streaming               │
  └─────────────────────────────────────────┘
```

Node **sí** ve PCM, pero solo como caño: reenvía bytes. No espera a que termines de hablar ni arma un WAV. Silero sigue en Python, sobre el stream en vivo.

### No hace falta que el mic hable directo con Python

Un proxy localhost/LAN de PCM suma **~1–2 ms**, irrelevante contra 200 ms de STT + 400–800 ms del LLM.

Lo que sí duele (20–80 ms) es otra cosa: **VAD en Node** que buffera el turno y *después* manda el clip. Eso no lo hacemos.

El frontend habla solo con Node. Python no es público. Más simple, y no cambia la latencia que sentís.

### Qué va en Python (5080)

Dos webservers WebSocket, **sin cerebro**:

| Servicio | Proceso | Hace | No hace |
| --- | --- | --- | --- |
| **speech-in** | Uno solo | Recibir PCM live, Silero, recortar el turno, transcribir, emitir eventos | Decidir si el turno va al LLM, tools, políticas |
| **tts** | Otro (o el mismo proceso si simplifica GPU) | Texto → PCM streaming | Elegir qué decir, filtrar código, turnos |

Python **no** orquesta. Puede tener el state machine mínimo de Silero (umbral, min speech, min silencio): eso es señal, no agente.

**Silero va en Python, en el mismo proceso que el STT. No en Node.** No porque necesite la GPU (CPU, <1 ms/chunk), sino para que al cortar silencio Whisper arranque con el buffer ya en RAM.

No separar VAD y STT en dos microservicios.

### Qué va en Node

Toda la secuencialidad:

- Único WS hacia el frontend (PCM in, PCM out, estado).
- Proxy live hacia speech-in / tts.
- Política de turnos: `speech_end` + transcript → ¿mando al LLM o descarto?
- Barge-in: `speech_start` mientras hay TTS → cancelar TTS y el LLM.
- LLM nube + tool loop (read / edit / shell / search).
- Qué se vocaliza: solo texto de asistente, nunca diffs ni JSON de tools.
- Splitter de oraciones hacia TTS.
- Sesión, workspace, confirmaciones.

Node **no** corre Silero ni Whisper ni Kokoro.

### ¿Todo en Python entonces?

Viable (Pipecat, FastAPI, un solo proceso). No lo elijo por el mic.

| | Split Node + Python | Todo Python |
| --- | --- | --- |
| Latencia de voz | Igual si VAD+STT están juntos en GPU | Igual |
| Dónde está el 80% del código | Agente/tools en Node (donde hay experiencia) | Agente/tools en Python |
| Operación | Dos runtimes, contrato WS interno | Un proceso, un deploy |
| Ecosistema GPU audio | Python se queda chico y aburrido | Mezcla inferencia + producto |
| Riesgo | Coordinar dos servers | Reescribir el agente en un lenguaje menos cómodo |

El split vale porque **el agente es el producto** y Python solo envuelve GPU. Unificar a Python para evitar el proxy de audio es optimizar 2 ms y pagar el costo en la parte más grande del repo.

Reconsiderar todo-Python solo si duele de verdad mantener dos procesos, o si el orquestador termina siendo 200 líneas y no un agente.

### Contratos de eventos (Node ↔ Python)

**speech-in → Node:** `speech_start`, `speech_end`, `partial_transcript?`, `transcript` (final), `error`.

**Node → tts:** `speak` (oración), `cancel` (barge-in / nuevo turno).

**tts → Node:** `audio_chunk`, `utterance_end` (Node se los pasa al cliente).

TTS del asistente es **pipeline**, no una tool del LLM.

### Diferencias vs ElBruno

| ElBruno.Realtime | Esta app |
| --- | --- |
| Un proceso .NET hace VAD+STT+LLM+TTS | Python = audio GPU; Node = agente |
| Asistente de voz genérico | Agente de coding que habla |
| LLM local opcional (phi4-mini) | LLM nube con tools (DeepSeek / similar) |
| Whisper tiny.en | Whisper turbo multilingual (ES + código EN) |
| QwenTTS default | Kokoro primero |
| Turno conversacional | Turno + loop de tools + narración |

## Stack tentativo

- **5080 / Python:** FastAPI (o equivalente) + WebSocket. Silero + faster-whisper large-v3-turbo + Kokoro 82M.
- **Node:** orquestador, LLM, tools, turnos. Frontend después.
- **Nube:** DeepSeek (u otro barato con function calling) vía OpenRouter o API directa.
- **Fallback cloud:** OpenRouter STT/TTS si la 5080 no está.

## Abierto / a decidir

- Un proceso Python (speech-in + tts, menos overhead GPU) vs dos procesos (crash isolation).
- Mic/speaker en la PC de la 5080 vs en otra máquina (LAN). El PCM igual pasa por Node hacia Python.
- Push-to-talk vs VAD continuo vs ambos.
- Cómo narrar tools sin leer diffs: resumen oral, confirmación antes de edits grandes.
- Workspace / sandbox del agente (cwd, git, permisos).
- Idioma de voz: español para charla, inglés para identificadores de código.

## Próximo paso natural

1. En la 5080: WS `speech-in` (VAD+STT) y WS `tts`, sin LLM.
2. Node mínimo: `transcript` → LLM stream (sin tools) → oraciones → TTS. Medir latencia de punta a punta.
3. Si se siente conversacional, enganchar tools.
