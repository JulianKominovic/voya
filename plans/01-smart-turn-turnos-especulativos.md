# Plan 1 — Smart Turn + turnos especulativos

## Contexto y motivador

Este plan nace de una sesión de relevamiento entre los dos repos del workspace:

- **`speech-to-speech/`** (s2s): el repo de Hugging Face, pipeline de voz VAD → STT → LLM → TTS, muy pulido en turn-taking. Se usa como referencia de "qué se puede lograr".
- **El resto del repo (Voya)**: voz local (Python/5080: Silero + Whisper + Kokoro) + cerebro en Node (orquestador, agentic, minions, question queue). Arquitectura por decisión: Python = dumb inference, Node = único servidor y toda la orquestación.

Motivación de la charla: detectar los gaps entre ambos y **tomar de s2s las cosas buenas que sirvan para Voya**, adaptándolas a su arquitectura (Python nunca orquesta; lo que se porte tiene que ser señal de inferencia o protocolo, no lógica de agente). La conversación identificó como gap #1 el turn-taking:

- s2s tiene **Smart Turn v3.2**: un modelo ONNX que valida el fin de turno de Silero mirando acústica + prosodia, y distingue "terminé de hablar" de "pausé a mitad de frase".
- s2s tiene **turnos especulativos con revisiones**: arranca STT + LLM antes de comprometer el turno (latencia escondida); si el usuario retoma, el turno se reabre como revisión N+1, el audio acumulado se re-emite y el trabajo anterior se descarta antes de llegar al usuario.
- Voya hoy solo tiene un merge heurístico de 1.5 s (`wantMerge` en `session.ts`) que aborta el LLM y concatena textos, sin re-emisión de audio ni trabajo especulativo.

Intención: portar Smart Turn y el modelo de revisiones a Voya de forma fiel al espíritu — inferencia en Python, decisión y gates de commit en Node — en dos fases para reducir riesgo.

## Objetivo

Que Python distinga "terminé de hablar" de "hice una pausa", y que Node arranque el LLM antes de comprometer el turno, descartando el trabajo si el usuario retoma. Es el cambio que más reduce latencia percibida y cortes a mitad de frase.

## Fase A — Smart Turn endpointing (solo Python)

### 1. Nuevo `python/speech_server/smart_turn.py`

Port minimal de `speech-to-speech/src/speech_to_speech/VAD/smart_turn.py`:

- Clase `SmartTurnAnalyzer`: carga `smart-turn-v3.2-cpu.onnx` con onnxruntime en `CPUExecutionProvider` (onnxruntime-gpu ya incluye CPU EP; no contradice la regla CUDA-only — es exactamente lo que hace s2s, que fuerza CPU para este modelo).
- **Punto delicado — extracción de features**: s2s usa `transformers.WhisperFeatureExtractor`, y Voya no tiene transformers (dependencia pesada, contraria a lo minimal). Implementar el log-mel filterbank en numpy puro (~50 líneas: STFT con n_fft 400 / hop 160, 80 bandas mel triangulares, log). Antes de integrar, verificar contra el output del extractor real con un WAV fijo (tolerancia ~1e-3).
- Entrada: hasta 8 s de audio mono 16 kHz (resamplear si hace falta, pad a 8 s). Salida: probabilidad de "turno completo".
- Parámetros: `SMART_TURN_THRESHOLD` (0.5), warmup al cargar.

### 2. Descarga del modelo

- `SMART_TURN_ONNX` en `config.py` (`models/smart-turn-v3.2-cpu.onnx`).
- Agregar la descarga a `download_models.py` vía huggingface_hub (ya es dependencia): repo `pipecat-ai/smart-turn-v3`, archivo `smart-turn-v3.2-cpu.onnx`.

### 3. Integración en `python/speech_server/vad.py`

Extender `StreamingVAD`:

- En `_close()`, antes de emitir `speech_end`, correr el analyzer sobre el audio del segmento.
- Si `complete` (p > threshold): comportamiento actual, sin cambio.
- Si `incomplete`: **no cerrar todavía** — mantener el buffer y esperar hasta `SMART_TURN_INCOMPLETE_DELAY_MS` (600 ms) de silencio adicional; si el usuario retoma, el segmento sigue creciendo y se evalúa de nuevo; si no retoma, cerrar igual (tope `SMART_TURN_MAX_WAIT_MS` = 2 s).
- Si el analyzer falla: fallback al comportamiento actual con log (igual que s2s).
- Evento intermedio opcional `speech_hold` (Python esperando confirmación de turno) para que Node no marque un estado raro.

### 4. Config

`SMART_TURN` (on/off, default on), `SMART_TURN_THRESHOLD`, `SMART_TURN_INCOMPLETE_DELAY_MS`, `SMART_TURN_MAX_WAIT_MS` en `config.py`.

### Verificación Fase A

- assert/`__main__` en `smart_turn.py` con un WAV de habla completa vs. una pausa a mitad.
- Prueba en vivo: pausas de >1 s midiendo si `speech_end` se posterga.

## Fase B — Turnos especulativos + protocolo de revisiones (Python + Node)

### 1. Protocolo (`python/speech_server/protocol.py`)

- `speech_end` gana `turn_id` + `revision`.
- Nuevos mensajes:
  - `speech_reopen {turn_id, revision}` — el usuario retomó dentro de la gracia; Node aborta el trabajo especulativo de esa revisión.
  - `speech_commit {turn_id, revision}` — expiró la gracia; el output especulativo puede hablarse.
- El audio acumulado se re-emite (VAD+STT viven en el mismo proceso: STT corre sobre el segmento mergeado completo y sale otro `speech_end` con `revision+1`).

### 2. Python (`vad.py` + `main.py`)

- `StreamingVAD` mantiene `turn_id` por segmento; al cerrar un turno "incomplete", emitir `speech_end` provisional y arrancar un timer de commit (`SPECULATIVE_REOPEN_MS` = 800 ms, espejo del `speculative_reopen_ms` de s2s).
- Si el habla retoma antes del commit → `speech_reopen`, mergear buffer, re-STT, `speech_end rev+1`.
- Si expira → `speech_commit`.
- En `main.py`, el loop de transcripción propaga `turn_id`/`revision` en el `transcript`.

### 3. Node (`node/src/session.ts`)

- Reemplazar el merge heurístico actual (`wantMerge` con `MERGE_MS` 1.5 s) por el protocolo de revisiones: la política de decisión queda igual, pero la señal ya no es timing heurístico sino eventos de Python.
- `onSpeechEnd` provisional: arrancar el LLM **especulativo** (mismo `runOrchestrator`) pero marcar el turno como `uncommitted`; `speakAssistant` acumula en vez de hablar.
- `onSpeechReopen`: `abortVoicePath()` (gen++, igual que barge-in) y descartar el texto acumulado.
- `onSpeechCommit`: recién ahí empezar a hablar el texto ya generado (si el LLM aún no terminó, se habla a medida que sale, con el gate desactivado).
- Mantener `MERGE_MS` como fallback solo si Smart Turn está off (`SMART_TURN=false`).
- **Importante**: el transcript del turno provisional reabierto NO debe entrar a memoria (`memory.push` de userText) hasta commit — si no, la memoria se llena de versiones descartadas.

### 4. Frontend (`node/src/app.ts`)

`speech_reopen`/`speech_commit` como eventos de estado (opcional; el behavior core es invisible al usuario).

### Edge cases Fase B

- Commit llega mientras TTS especulativo ya empezó → el audio en cola se libera en orden.
- Barge-in real (speech_start con gen++) durante un turno especulativo → el commit/reopen viejo se descarta por gen (mismo mecanismo actual).
- Tope `unanswered_reopen_ms` (7 s, como s2s): un turno provisional huérfano no queda reabrible para siempre.

### Verificación Fase B

- Escenario manual "decir una frase, pausar ~1 s, seguir": debe verse `speech_reopen` en los logs de Node y NO duplicarse el texto en memoria.
- "Decir frase completa": el LLM arranca antes del `speech_commit` (verificable comparando `llm start` vs `speech_commit` en el log).

## Doc updates

- `FLOW.md`: nuevo contrato `speech_reopen`/`speech_commit`, reglas de commit especulativo.
- `AGENTS.md`: nota de que Smart Turn corre en CPU EP (excepción a la regla CUDA, igual que en s2s).
