# Plan 4 — `interrupt_response` configurable por sesión

## Contexto y motivador

Este plan nace de una sesión de relevamiento entre los dos repos del workspace:

- **`speech-to-speech/`** (s2s): pipeline de voz VAD → STT → LLM → TTS de Hugging Face, usado como referencia de qué se puede lograr.
- **El resto del repo (Voya)**: voz local (Python/5080: Silero + Whisper + Kokoro) + cerebro en Node (orquestador, agentic, minions, question queue). Arquitectura por decisión: Python = dumb inference, Node = único servidor y toda la orquestación.

Motivación de la charla: detectar los gaps entre ambos y **tomar de s2s las cosas buenas que sirvan para Voya**, adaptándolas a su arquitectura (Python nunca orquesta; lo que se porte tiene que ser señal de inferencia o protocolo, no lógica de agente). La conversación identificó como gap #4 el control del barge-in:

- s2s expone en su protocolo Realtime `turn_detection.interrupt_response` por sesión, configurable en runtime vía `session.update`. Cuando está desactivado, el usuario puede hablar durante la respuesta del asistente: el audio se transcribe, la respuesta sigue reproduciéndose, y el transcript se procesa al terminar (ver la sección "Interruption Handling" del README del engine de s2s: "user speech during a response is transcribed but the response keeps playing").
- Voya tiene el barge-in **siempre on**: cualquier `speech_start` hace gen++, aborta TTS y el LLM (`session.ts:245-275`). No hay knob. Para un asistente que explica decisiones de código (el caso de uso de Voya), a veces se quiere que el usuario no pueda cortar una explicación en curso.

Intención: portar el concepto de `interrupt_response` como flag por sesión en Node, con el mismo comportamiento de s2s: con interrupt off, la respuesta sigue sonando, el habla del usuario se transcribe, y el texto entra como turno siguiente al terminar la respuesta. La decisión queda en Node (orquestación), Python no cambia.

## Objetivo

Poder desactivar el barge-in cuando el asistente está explicando algo importante: toggle por sesión (cliente o env) que hace que el habla del usuario ya no corte la respuesta en curso, sin perder el mensaje.

## Diseño

1. **Node (`config.ts`)**: `INTERRUPT_ON_SPEECH = true` (default, env).

2. **`session.ts`**:
   - Campo `interruptOnSpeech` en `Session`.
   - En `onSpeechStart()`: si `!interruptOnSpeech` → **no** hacer `gen++`, no llamar `abortVoicePath()`. En su lugar, seguir trackeando el audio (los eventos `speech_end`/`transcript` ya entran al `pendingStt`); el transcript final se guarda como `deferredUserText`.
   - Al terminar el turno en curso (`maybeFinishTurn` → `goIdleOrListening`): si hay `deferredUserText`, disparar `beginUserTurn` normal (el texto no se pierde, entra como turno siguiente).
   - El check actual `pending.gen !== this.gen` sigue funcionando porque gen no cambió al no hacer barge-in.
   - Echo-tail y merge: desactivar `wantMerge` mientras `interruptOnSpeech=false` y hay turno en curso (si no, el merge se mezcla con el turno diferido).

3. **`server.ts`**: mensaje nuevo del cliente `interrupt {enabled: boolean}` → `session.interruptOnSpeech = ...` + `emitState()` (para que la UI muestre el toggle).

4. **Frontend (`app.ts`)**: botón/toggle "no interrumpir" + estado en el event `state`.

## Edge cases

- Usuario habla durante la explicación y supera la cola → el transcript se encola y el LLM lo procesa al terminar (comportamiento exacto de s2s).
- Con interrupt off y TTS largo, la UI debe advertirlo (mic "escuchando pero no interrumpiendo").
- Con interrupt on (default): comportamiento actual, sin cambios.

## Verificación

- Explicar algo largo con interrupt off, hablar encima → la respuesta sigue completa y el turno nuevo arranca al terminar.
- Toggle on → barge-in normal (el gen++ aborta TTS y LLM como hoy).

## Doc updates

- `FLOW.md`: sección "Barge-in" — nueva condición: el barge-in solo dispara si `interruptOnSpeech` está activo; descripción del turno diferido.
- `AGENTS.md`: mensaje `interrupt` en el contrato cliente ↔ Node.
