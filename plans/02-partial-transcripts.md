# Plan 2 — Partial transcripts (live transcription)

## Contexto y motivador

Este plan nace de una sesión de relevamiento entre los dos repos del workspace:

- **`speech-to-speech/`** (s2s): pipeline de voz VAD → STT → LLM → TTS de Hugging Face, usado como referencia de qué se puede lograr.
- **El resto del repo (Voya)**: voz local (Python/5080: Silero + Whisper + Kokoro) + cerebro en Node (orquestador, agentic, minions, question queue). Arquitectura por decisión: Python = dumb inference, Node = único servidor y toda la orquestación.

Motivación de la charla: detectar los gaps entre ambos y **tomar de s2s las cosas buenas que sirvan para Voya**, adaptándolas a su arquitectura (Python nunca orquesta; lo que se porte tiene que ser señal de inferencia o protocolo, no lógica de agente). La conversación identificó como gap #2 los transcripts parciales:

- s2s tiene **live transcription**: durante el habla emite `transcription.delta` (hipótesis acumulativas con commit por prefijo en límites de palabra, sin retracciones), y el `completed` final es autoritativo.
- Voya hoy no muestra ni procesa nada hasta `speech_end` → Whisper → `transcript` final (`python/speech_server/main.py:195`). FLOW.md lo declara explícitamente: "no partial transcripts for now".

Intención: portar la transcripción parcial a Voya como **señal de UI en vivo** (sensación de realtime), con la regla dura de s2s: el parcial es hipótesis inestable, el final lo reemplaza y es el único que alimenta al orquestador. Es inferencia pura (Whisper sobre audio creciente) — cabe en la parte "dumb" de Python sin violar la arquitectura.

## Objetivo

Emitir transcripción parcial durante el habla para la UI en vivo, sin tocar la lógica de orquestación: los parciales jamás llegan al LLM, al orquestador ni a la memoria.

## Diseño

1. **Protocolo** (`python/speech_server/protocol.py`): nuevo mensaje `partial_transcript {t, text}` con el texto acumulado de la hipótesis actual (no deltas).

2. **Python (`stt.py` + `main.py`)**:
   - En `STT`, método `transcribe_partial(audio)` reutilizando el mismo modelo (misma llamada que `transcribe`; no aplica `condition_on_previous_text` porque el audio es creciente).
   - En `main.py` (speech-in): mientras `vad._in_speech`, cada `LIVE_INTERVAL_MS` (500 ms) correr `stt.transcribe_partial` sobre el buffer actual vía `to_thread`, con dos gates:
     - **Gate de inicio**: solo empezar después de `LIVE_START_DELAY_MS` (p. ej. 1.5 s) de habla — evita quemar GPU en balbuceos cortos y no duplica trabajo en turnos de una palabra.
     - **Gate de crecimiento**: no re-transcribir si el buffer creció < ~300 ms de audio desde la última pasada.
   - En `speech_end`, cancelar el loop parcial y emitir solo el `transcript` final (autoritativo).
   - **Protección de concurrencia**: el parcial y el final comparten GPU. faster-whisper no tiene cancelación, así que nunca correr parcial y final en paralelo: en `speech_end` el loop parcial se frena y espera a que el final termine. Si el parcial atrasa el final, se descarta su resultado.

3. **Node (`node/src/session.ts`)**: `onPartialTranscript` → reenviar al cliente con tag `gen`/`turn_id`; jamás alimentar el LLM con parciales. Si llega un parcial de gen viejo, dropear (mismo guard que `onTranscript`).

4. **Frontend (`node/src/app.ts`)**: renderizar el parcial en gris en el transcript del turno; al llegar el `transcript` final, reemplazarlo (semántica exacta de s2s: el final es autoritativo, reemplaza cualquier parcial del mismo `item_id`).

## Config

- `LIVE_TRANSCRIPTION` (off por default), `LIVE_INTERVAL_MS`, `LIVE_START_DELAY_MS` en `config.py`.
- Toggle por sesión en Node (env) si se quiere control por sesión.

## Edge cases

- Pausas largas con Silero aún en speech (min_silence no alcanzado) siguen emitiendo parciales viejos — está bien, se reemplazan.
- Turno corto (< `LIVE_START_DELAY_MS`) nunca emite parcial.
- Barge-in: los parciales del gen viejo se dropean por gen (mecanismo existente).

## Verificación

- Hablar lento con la UI abierta → ver el parcial en gris crecer y el final reemplazarlo.
- Confirmar en el log de Node que nunca hay un parcial con el prefijo del turno siguiente (los parciales no cruzan turnos).
- Medir que `stt_ms` del transcript final no empeore con el loop parcial activo.

## Doc updates

- `FLOW.md`: nuevo mensaje `partial_transcript` en el contrato Node ↔ Python.
- `AGENTS.md`: actualizar la línea "no partial transcripts for now".
