import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadEnv(file: string) {
  let raw: string;
  try {
    raw = fs.readFileSync(file, "utf8");
  } catch {
    return;
  }
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 1) continue;
    const key = trimmed.slice(0, eq).trim();
    let val = trimmed.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    if (process.env[key] === undefined) process.env[key] = val;
  }
}

loadEnv(path.join(__dirname, "../.env"));

export const PUBLIC = path.resolve(path.join(__dirname, "../../frontend/dist"));
export const LOG_DIR = path.resolve(path.join(__dirname, "../logs"));
export const PORT = Number(process.env.PORT || 8787);
export const SPEECH_URL = process.env.SPEECH_URL || "ws://127.0.0.1:8765/ws/speech-in";
export const TTS_URL = process.env.TTS_URL || "ws://127.0.0.1:8765/ws/tts";
export const TTS_VOICE = process.env.TTS_VOICE || "ef_dora";
export const TTS_LANG = process.env.TTS_LANG || "es";
export const LLM_URL = process.env.LLM_URL || "http://192.168.0.29:8080/v1";
export const ORCHESTRATOR_MODEL = process.env.ORCHESTRATOR_MODEL || "ggml-org/gemma-4-E4B-it-GGUF";
export const AGENTIC_MODEL = process.env.AGENTIC_MODEL || ORCHESTRATOR_MODEL;

export const QUESTION_TIMEOUT_MS = Number(process.env.QUESTION_TIMEOUT_MS || 60 * 60 * 1000);
export const ASK_TIMEOUT_MS = Number(process.env.ASK_TIMEOUT_MS || 10 * 60 * 1000);
export const WINDOW_CHARS = Number(process.env.WINDOW_CHARS || 32000);
export const TTS_MAX_CHARS = Number(process.env.TTS_MAX_CHARS || 600);
export const ANSWER_MAX_CHARS = Number(process.env.ANSWER_MAX_CHARS || 800);
export const SESSION_MAX_LLM_CHARS = Number(process.env.SESSION_MAX_LLM_CHARS || 500000);
export const MERGE_MS = Number(process.env.MERGE_MS || 1500);
export const ECHO_TAIL_MS = Number(process.env.ECHO_TAIL_MS || 300);
export const ECHO_MIN_MS = Number(process.env.ECHO_MIN_MS || 400);
export const AGENTIC_MAX_ROUNDS = Number(process.env.AGENTIC_MAX_ROUNDS || 12);
export const CHAIN_MAX_DEPTH = Number(process.env.CHAIN_MAX_DEPTH || 4);
