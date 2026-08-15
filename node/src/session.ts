import { randomUUID } from "node:crypto";
import { WebSocket } from "ws";
import type { Llm } from "./llm.js";
import { Brain, ORCH_SYSTEM } from "./agents.js";
import {
  ECHO_MIN_MS,
  ECHO_TAIL_MS,
  LOG_DIR,
  MERGE_MS,
  TTS_LANG,
  TTS_VOICE,
} from "./config.js";
import { Memory } from "./memory.js";
import { QuestionQueue } from "./questions.js";
import { ms, now, sanitizeTts } from "./text.js";
import { sendJson, sendRaw } from "./wsutil.js";

export type VoiceState =
  | "idle"
  | "listening"
  | "user_speaking"
  | "transcribing"
  | "thinking"
  | "speaking";

export type LogChannel = "stt" | "llm_conv" | "llm_agentic" | "minions" | "tts";

export function wantMerge(opts: {
  lastSpeechEnd: number;
  now: number;
  assistantSpoke: boolean;
  state: VoiceState;
  mergeMs?: number;
}): boolean {
  if (opts.assistantSpoke) return false;
  if (opts.state !== "transcribing" && opts.state !== "thinking" && opts.state !== "listening") {
    return false;
  }
  if (!opts.lastSpeechEnd) return false;
  return opts.now - opts.lastSpeechEnd < (opts.mergeMs ?? MERGE_MS);
}

export function isEchoTail(durationMs: number, startedDuringTts: boolean, minMs = ECHO_MIN_MS) {
  return durationMs < minMs && startedDuringTts;
}

{
  if (wantMerge({ lastSpeechEnd: 0, now: 100, assistantSpoke: false, state: "transcribing" })) {
    throw new Error("merge needs lastSpeechEnd");
  }
  if (
    !wantMerge({
      lastSpeechEnd: 1000,
      now: 2000,
      assistantSpoke: false,
      state: "transcribing",
      mergeMs: 1500,
    })
  ) {
    throw new Error("merge window");
  }
  if (
    wantMerge({
      lastSpeechEnd: 1000,
      now: 2000,
      assistantSpoke: true,
      state: "transcribing",
      mergeMs: 1500,
    })
  ) {
    throw new Error("merge after assistant");
  }
  if (wantMerge({ lastSpeechEnd: 1, now: 2, assistantSpoke: false, state: "speaking" })) {
    throw new Error("merge not from speaking");
  }
  if (!isEchoTail(200, true) || isEchoTail(500, true) || isEchoTail(200, false)) {
    throw new Error("echo tail");
  }
}

type TtsJob = { id: string; text: string; gen: number; tSpeak: number; gotPcm: boolean };

type PendingStt = {
  gen: number;
  durationMs: number;
  tSpeechEnd: number;
  startedDuringTts: boolean;
};

export class Session {
  gen = 0;
  state: VoiceState = "idle";
  echo = false;
  micOn = false;
  toolRunning = false;
  llmBusy = false;
  llmAbort: AbortController | null = null;
  closed = false;
  assistantSpoke = false;
  speech?: WebSocket;
  tts?: WebSocket;
  onPythonDead: (() => void) | null = null;

  readonly questions = new QuestionQueue();
  readonly memory: Memory;
  readonly brain: Brain;
  readonly id: string;

  private ttsQueue: TtsJob[] = [];
  private ttsInFlight: TtsJob | null = null;
  private dropTtsBinary = false;
  private lastSpeechEnd = 0;
  private lastTtsEnd = 0;
  private ttsPlaying = false;
  private currentStartDuringTts = false;
  private pendingStt: PendingStt[] = [];
  private mergeHold = "";
  private lastUserText = "";
  private turnId = "----";
  private pythonBound = false;

  constructor(
    readonly client: WebSocket,
    readonly llm: Llm | null,
  ) {
    this.id = randomUUID().slice(0, 8);
    this.memory = new Memory(LOG_DIR, this.id, ORCH_SYSTEM);
    this.brain = new Brain(this);
    this.questions.onEnqueue = () => {
      this.emitState();
      this.maybeSpeakQuestion();
    };
    this.questions.onResolve = (info) => {
      this.logTurn(
        `question_resolved ${info.id} ${info.status}${info.text ? ` text=${info.text}` : ""}`,
        "llm_agentic",
      );
      sendJson(this.client, {
        type: "question_resolved",
        id: info.id,
        status: info.status,
        text: info.text,
      });
      this.emitState();
    };
  }

  emitState() {
    sendJson(this.client, {
      type: "state",
      state: this.state,
      gen: this.gen,
      pending_question: this.questions.head()?.text ?? null,
      tool_running: this.toolRunning,
      echo: this.echo,
    });
  }

  setState(state: VoiceState) {
    const prev = this.state;
    this.state = state;
    this.emitState();
    if (state === "listening" && prev !== "listening") void this.onEnterListening();
  }

  setMic(on: boolean) {
    this.micOn = on;
    if (on && this.state === "idle") this.setState("listening");
    if (!on && this.state === "listening") this.setState("idle");
  }

  logTurn(line: string, channel?: LogChannel) {
    console.log(`[${this.turnId}|g${this.gen}] ${line}`);
    this.memory.log({ type: "turn_log", turn: this.turnId, gen: this.gen, line });
    if (channel) sendJson(this.client, { type: "log", ts: now(), channel, text: line });
  }

  abortVoicePath() {
    const had = this.llmBusy || this.ttsInFlight || this.ttsQueue.length || this.llmAbort;
    if (had) this.logTurn("abort voice path", "tts");
    this.llmAbort?.abort();
    this.llmAbort = null;
    this.llmBusy = false;
    this.ttsQueue = [];
    if (this.ttsPlaying || this.ttsInFlight) this.lastTtsEnd = now();
    this.ttsInFlight = null;
    this.dropTtsBinary = true;
    this.ttsPlaying = false;
    sendJson(this.tts, { type: "cancel", id: "" });
  }

  bindPython(speech: WebSocket, tts: WebSocket) {
    this.unbindPython();
    this.speech = speech;
    this.tts = tts;
    this.pythonBound = true;
    speech.on("message", (data, isBinary) => this.onSpeechMsg(data, isBinary));
    tts.on("message", (data, isBinary) => this.onTtsMsg(data, isBinary));
    const dead = () => {
      if (this.closed || !this.pythonBound) return;
      this.pythonBound = false;
      this.abortVoicePath();
      this.onPythonDead?.();
    };
    speech.on("close", dead);
    tts.on("close", dead);
    speech.on("error", (err) => {
      this.logTurn(`speech ws error ${String(err)}`, "stt");
      sendJson(this.client, { type: "error", message: String(err) });
    });
    tts.on("error", (err) => {
      this.logTurn(`tts ws error ${String(err)}`, "tts");
      sendJson(this.client, { type: "error", message: String(err) });
    });
  }

  unbindPython() {
    this.pythonBound = false;
    try {
      this.speech?.close();
    } catch {
      /* ignore */
    }
    try {
      this.tts?.close();
    } catch {
      /* ignore */
    }
    this.speech = undefined;
    this.tts = undefined;
  }

  destroy() {
    this.closed = true;
    this.abortVoicePath();
    this.questions.clear();
    this.brain.killAll();
    this.unbindPython();
  }

  onClientPcm(data: WebSocket.RawData) {
    sendRaw(this.speech, data, true);
  }

  onSpeechStart() {
    const t = now();
    const startedDuringTts =
      this.ttsPlaying || (this.lastTtsEnd > 0 && t - this.lastTtsEnd < ECHO_TAIL_MS);
    const merge = wantMerge({
      lastSpeechEnd: this.lastSpeechEnd,
      now: t,
      assistantSpoke: this.assistantSpoke,
      state: this.state,
    });
    if (merge) {
      this.logTurn("speech_start merge", "stt");
      this.llmAbort?.abort();
      this.llmAbort = null;
      this.llmBusy = false;
      if (this.lastUserText && !this.mergeHold) this.mergeHold = this.lastUserText;
      this.setState("user_speaking");
      sendJson(this.client, { type: "speech_start" });
      return;
    }
    this.gen += 1;
    this.turnId = randomUUID().slice(0, 4);
    this.abortVoicePath();
    this.assistantSpoke = false;
    this.mergeHold = "";
    this.lastUserText = "";
    this.currentStartDuringTts = startedDuringTts;
    this.logTurn("speech_start", "stt");
    this.setState("user_speaking");
    sendJson(this.client, { type: "speech_start" });
  }

  onSpeechEnd(durationMs: number) {
    this.lastSpeechEnd = now();
    this.pendingStt.push({
      gen: this.gen,
      durationMs,
      tSpeechEnd: this.lastSpeechEnd,
      startedDuringTts: this.currentStartDuringTts,
    });
    this.logTurn(`speech_end duration_ms=${durationMs}`, "stt");
    this.setState("transcribing");
    sendJson(this.client, { type: "speech_end", duration_ms: durationMs });
  }

  onTranscript(text: string, sttMs?: number) {
    const pending = this.pendingStt.shift();
    const t = String(text || "").trim();
    if (!pending) {
      this.logTurn("transcript unmatched drop", "stt");
      return;
    }
    const hop = pending.tSpeechEnd ? ms(pending.tSpeechEnd) : 0;
    if (pending.gen !== this.gen) {
      this.logTurn(`transcript drop gen=${pending.gen} current=${this.gen}`, "stt");
      return;
    }
    if (isEchoTail(pending.durationMs, pending.startedDuringTts)) {
      this.logTurn(`transcript echo-tail drop duration_ms=${pending.durationMs}`, "stt");
      this.goIdleOrListening();
      return;
    }
    sendJson(this.client, { type: "transcript", text: t, stt_ms: sttMs });
    if (!t) {
      this.logTurn(`transcript empty stt_ms=${sttMs ?? ""} hop_ms=${hop}`, "stt");
      this.goIdleOrListening();
      return;
    }
    this.logTurn(`transcript stt_ms=${sttMs ?? ""} hop_ms=${hop} text=${t}`, "stt");
    if (this.state === "user_speaking") {
      this.mergeHold = [this.mergeHold, t].filter(Boolean).join(" ");
      return;
    }
    const full = [this.mergeHold, t].filter(Boolean).join(" ");
    this.mergeHold = "";
    this.lastUserText = full;
    this.beginUserTurn(full);
  }

  beginUserTurn(text: string) {
    if (this.echo) {
      this.speak(text, this.gen);
      return;
    }
    this.llmBusy = true;
    this.setState("thinking");
    void this.brain.runOrchestrator(text, this.gen);
  }

  speakAssistant(text: string, gen: number) {
    if (gen !== this.gen || this.closed) return;
    const t = sanitizeTts(text);
    if (!t) return;
    this.assistantSpoke = true;
    this.logTurn(`assistant ${t}`, "llm_conv");
    sendJson(this.client, { type: "assistant", text: t });
    this.speak(t, gen);
  }

  speak(text: string, gen: number) {
    if (gen !== this.gen || this.closed) return;
    const t = sanitizeTts(text);
    if (!t) return;
    this.ttsQueue.push({ id: randomUUID(), text: t, gen, tSpeak: now(), gotPcm: false });
    this.pumpTts();
  }

  speakError(message: string, gen: number) {
    this.logTurn(`error ${message}`, "llm_conv");
    sendJson(this.client, { type: "error", message });
    this.speakAssistant(message, gen);
  }

  maybeFinishTurn() {
    if (this.llmBusy || this.ttsInFlight || this.ttsQueue.length) return;
    sendJson(this.client, { type: "turn_end" });
    this.goIdleOrListening();
  }

  goIdleOrListening() {
    if (this.llmBusy || this.ttsInFlight || this.ttsQueue.length) return;
    this.setState(this.micOn ? "listening" : "idle");
  }

  private async onEnterListening() {
    this.maybeSpeakQuestion();
    void this.brain.compactIfNeeded();
  }

  maybeSpeakQuestion() {
    const h = this.questions.head();
    if (!h || h.spoken) return;
    if (this.ttsInFlight || this.ttsQueue.length) return;
    if (this.state === "user_speaking" || this.state === "transcribing") return;
    this.questions.markSpoken(h.id);
    sendJson(this.client, { type: "question_asked", id: h.id, text: h.text });
    this.logTurn(`question_asked ${h.id} ${h.text}`, "llm_agentic");
    this.speak(h.text, this.gen);
  }

  private pumpTts() {
    if (this.ttsInFlight) return;
    while (this.ttsQueue.length) {
      const job = this.ttsQueue.shift();
      if (!job) break;
      if (job.gen !== this.gen) continue;
      this.ttsInFlight = job;
      this.logTurn(`tts speak id=${job.id} chars=${job.text.length}`, "tts");
      sendJson(this.tts, {
        type: "speak",
        id: job.id,
        text: job.text,
        voice: TTS_VOICE,
        lang: TTS_LANG,
      });
      return;
    }
    this.maybeFinishTurn();
    this.maybeSpeakQuestion();
  }

  private onSpeechMsg(data: WebSocket.RawData, isBinary: boolean) {
    if (isBinary) {
      sendRaw(this.client, data, true);
      return;
    }
    let msg: { type?: string; text?: string; stt_ms?: number; duration_ms?: number; message?: string };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      sendRaw(this.client, data, false);
      return;
    }
    if (msg.type === "speech_start") this.onSpeechStart();
    else if (msg.type === "speech_end") this.onSpeechEnd(Number(msg.duration_ms || 0));
    else if (msg.type === "transcript") this.onTranscript(String(msg.text || ""), msg.stt_ms);
    else if (msg.type === "error") {
      this.logTurn(`speech error ${msg.message ?? ""}`, "stt");
      sendJson(this.client, { type: "error", message: msg.message });
    }
    else sendRaw(this.client, data, false);
  }

  private onTtsMsg(data: WebSocket.RawData, isBinary: boolean) {
    if (isBinary) {
      if (this.dropTtsBinary || !this.ttsInFlight) return;
      if (!this.ttsInFlight.gotPcm) {
        this.ttsInFlight.gotPcm = true;
        this.logTurn(`tts first_pcm_ms=${ms(this.ttsInFlight.tSpeak)} id=${this.ttsInFlight.id}`, "tts");
      }
      sendRaw(this.client, data, true);
      return;
    }
    let msg: { type?: string; id?: string; synth_ms?: number; sample_rate?: number; message?: string };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "audio_start") {
      const id = String(msg.id || "");
      const job = this.ttsInFlight;
      if (job && job.id === id && job.gen === this.gen) {
        this.dropTtsBinary = false;
        this.ttsPlaying = true;
        this.logTurn(`tts audio_start_ms=${ms(job.tSpeak)} id=${id}`, "tts");
        this.setState("speaking");
        sendJson(this.client, msg);
      } else {
        this.dropTtsBinary = true;
      }
      return;
    }
    if (msg.type === "audio_end" || msg.type === "error") {
      const id = String(msg.id || this.ttsInFlight?.id || "");
      const job = this.ttsInFlight;
      if (!job || job.id !== id) return;
      const extra = msg.type === "audio_end" ? ` synth_ms=${msg.synth_ms ?? ""}` : "";
      this.logTurn(`tts ${msg.type} total_ms=${ms(job.tSpeak)}${extra} id=${id}`, "tts");
      this.ttsInFlight = null;
      this.ttsPlaying = false;
      this.lastTtsEnd = now();
      if (msg.type === "audio_end") sendJson(this.client, msg);
      else sendJson(this.client, { type: "error", message: msg.message || "tts error" });
      if (this.llmBusy && !this.ttsQueue.length) this.setState("thinking");
      this.pumpTts();
    }
  }
}
