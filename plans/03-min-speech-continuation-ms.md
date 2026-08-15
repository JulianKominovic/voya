# Plan 3 — `min_speech_continuation_ms` (histéresis de VAD)

## Contexto y motivador

Este plan nace de una sesión de relevamiento entre los dos repos del workspace:

- **`speech-to-speech/`** (s2s): pipeline de voz VAD → STT → LLM → TTS de Hugging Face, usado como referencia de qué se puede lograr.
- **El resto del repo (Voya)**: voz local (Python/5080: Silero + Whisper + Kokoro) + cerebro en Node (orquestador, agentic, minions, question queue). Arquitectura por decisión: Python = dumb inference, Node = único servidor y toda la orquestación.

Motivación de la charla: detectar los gaps entre ambos y **tomar de s2s las cosas buenas que sirvan para Voya**, adaptándolas a su arquitectura (Python nunca orquesta; lo que se porte tiene que ser señal de inferencia o protocolo, no lógica de agente). La conversación identificó como gap #3 la histéresis del VAD:

- s2s define `min_speech_continuation_ms` (default 192 ms) como **histéresis de duración**: el umbral de habla mínimo para *continuar* un turno que acaba de cerrarse (reabrible) es menor que el de un inicio nuevo (`min_speech_ms` 384 ms). El pairing recomendado es 384/192. Sin esto, una persona que retoma rápido ("sí—[pausa]—claro") no se registra como continuación sino como turno nuevo, o se pierde porque no alcanzó el umbral.
- Voya usa un único umbral fijo: `MIN_SPEECH_MS = 350` en `config.py`, aplicado a todo inicio de habla por igual en `StreamingVAD._step` (`python/speech_server/vad.py:121-137`).

Intención: portar la histéresis a Voya en su forma más simple — una segunda ventana de habla menor que se aplica cuando el audio nuevo arranca dentro de una ventana corta posterior al cierre del último segmento. Es señal de inferencia pura, vive entera en Python.

## Objetivo

Que un turno recién cerrado que se retoma rápido sea aceptado con menos umbral de habla que un inicio nuevo. Es además la base del Plan 1 (Fase B): el reopen especulativo necesita que el VAD acepte pronto el habla que continúa un turno abierto.

## Diseño

1. **`config.py`**: `MIN_SPEECH_CONT_MS = 192` (default, el pairing recomendado de s2s es 384/192).

2. **`vad.py` (`StreamingVAD._step`)**:
   - Guardar `last_end_ms` (timestamp monotónico del último `_close`).
   - En `_step`, al evaluar inicio de habla: si `now - last_end_ms < REOPEN_WINDOW_MS` (ventana corta, p. ej. 800 ms), usar `min_speech_cont_windows = max(1, round(192 ms / window))` en vez de `min_speech_windows`.
   - Reiniciar `last_end_ms` en `reset()`.

3. **Interacción con el eco (importante, documentar en FLOW.md)**: Python no sabe cuándo el TTS terminó de sonar. Con una ventana de 800 ms y el echo-tail de Node (`ECHO_MIN_MS` 400, `ECHO_TAIL_MS` 300 en `config.ts`) ya activo, el riesgo de start espurio post-TTS es bajo. **Si en la práctica aparece eco**: la corrección es que Node mande un marker al WS speech-in cuando arranca TTS para que Python desactive la histéresis durante ~1 s. Dejar anotado como follow-up, no implementar en v1.

## Config

`MIN_SPEECH_CONT_MS` (env, default 192). El resto de umbrales no cambia.

## Edge cases

- El umbral menor solo aplica dentro de la ventana de 800 ms post-cierre; cualquier habla fuera de la ventana usa el umbral normal.
- `_close` actualiza `last_end_ms`; un `reset()` (fin de conexión) lo borra.
- No interactúa con `wantMerge` de Node (Plan 1 Fase B lo reemplaza por revisiones; mientras tanto, ambos conviven: Python acepta antes el habla, Node decide el merge con su timing actual).

## Verificación

- Hablar "sí — [pausa ~0.5 s] — claro" → la segunda parte se registra como speech con 192 ms de habla en vez de esperar 350 ms.
- Test de assert en `vad.py` con arrays sintéticos (silence → speech corto → silence → speech corto dentro de la ventana vs. fuera de la ventana).

## Doc updates

- `FLOW.md`: tabla de VAD en la sección "Mic / speaker rules" con los dos umbrales.
- `AGENTS.md`: mención del pairing 384/192 si se documentan defaults.
