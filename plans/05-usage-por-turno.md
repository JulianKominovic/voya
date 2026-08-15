# Plan 5 — Reportes de usage por turno

## Contexto y motivador

Este plan nace de una sesión de relevamiento entre los dos repos del workspace:

- **`speech-to-speech/`** (s2s): pipeline de voz VAD → STT → LLM → TTS de Hugging Face, usado como referencia de qué se puede lograr.
- **El resto del repo (Voya)**: voz local (Python/5080: Silero + Whisper + Kokoro) + cerebro en Node (orquestador, agentic, minions, question queue). Arquitectura por decisión: Python = dumb inference, Node = único servidor y toda la orquestación.

Motivación de la charla: detectar los gaps entre ambos y **tomar de s2s las cosas buenas que sirvan para Voya**, adaptándolas a su arquitectura (Python nunca orquesta; lo que se porte tiene que ser señal de inferencia o protocolo, no lógica de agente). La conversación identificó como gap #5 la operación y el costo:

- s2s preserva y reporta el **token usage a través de las cancelaciones**: aunque una respuesta se corte por barge-in, el `response.done` incluye el usage real del completion parcial (ver "Interruption Handling" del README del engine de s2s: "preserving provider-reported usage for billing").
- Voya solo lleva un contador de chars con tope de sesión (`llmChars` / `SESSION_MAX_LLM_CHARS` en `agents.ts:377` y `config.ts:51`), una heurística, y no expone cuánto costó cada turno. Los tiempos ya se loguean (`llm first_token_ms`, `stt_ms`, `tts synth_ms`) pero dispersos y sin tokens.

Intención: portar la disciplina de medición de s2s — medir de verdad lo que cuesta cada turno (tokens por modelo y proveedor, tiempos por etapa) y reportarlo a la UI, al log y al tope de sesión. Es puramente observabilidad en Node, sin tocar Python.

## Objetivo

Saber cuánto cuesta cada turno (tokens por modelo, proveedor, tiempos de cada etapa) sin adivinar: reporte por turno a la UI, persistencia en el jsonl, y tope de sesión basado en tokens reales en vez de chars.

## Diseño

1. **`agents.ts` (`streamChatOnce`)**: capturar `usage` del stream de OpenRouter — el último chunk trae `usage: {prompt_tokens, completion_tokens, total_tokens}`; extraerlo con un helper análogo a `chunkMeta` — y devolverlo en el resultado de `streamChat`. Con fallback de modelo, anotar qué modelo/proveedor facturó (`model_used`).

2. **`session.ts`**:
   - Nuevo evento a cliente: `usage {gen, turn_id, model, provider, prompt_tokens, completion_tokens}` emitido en `llm done`.
   - Acumuladores `sessionTokens` (prompt + completion); complementar o reemplazar el tope `SESSION_MAX_LLM_CHARS` con `SESSION_MAX_TOKENS` (los tokens son la métrica real; dejar chars como fallback si `usage` viene vacío).
   - Unificar el reporte en `turn_end`: `turn_stats {gen, hop_ms, stt_ms, llm_ttft_ms, llm_total_ms, tts_synth_ms, tokens}` — los datos ya existen como logs (`llm first_token_ms`, `tts synth_ms`); solo hace falta agregarlos y emitirlos juntos.

3. **`memory.ts`**: log persistente de cada turno de LLM con usage (patrón existente `memory.log`); `llmChars` se suma desde el usage real si viene.

4. **`server.ts` `/health`**: exponer tokens acumulados de la sesión activa (o por sesión vía el event `usage`).

## Edge cases

- **Respuesta cancelada por barge-in a mitad de stream** → OpenRouter igual reporta el usage del completion parcial; hay que contarlo (es exactamente lo que hace s2s: preserva usage a través de cancelaciones). En `runOrchestrator` el abort por gen no debe perder el usage ya acumulado.
- **Fallback de modelo** → el usage pertenece al modelo que realmente respondió (`model_used`, ya se tiene `provider`).
- `usage` vacío (algún provider no lo manda) → contabilizar 0 y loggear aviso, no romper el turno.

## Verificación

- Hacer 3 turnos con barge-in en uno; revisar en `node/logs/*.jsonl` que cada turno tiene su `llm_turn` con tokens y que el total cuadra con la factura de OpenRouter.
- El event `turn_stats` aparece en la UI por cada turno terminado.

## Doc updates

- `FLOW.md`: event `usage` y `turn_stats` en el contrato Node → cliente.
- `AGENTS.md`: mención del tope por tokens (`SESSION_MAX_TOKENS`).
