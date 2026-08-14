import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { WebSocketServer, WebSocket } from "ws";
import { OpenRouter } from "@openrouter/sdk";

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

const PUBLIC = path.resolve(path.join(__dirname, "../public"));
const PORT = Number(process.env.PORT || 8787);
const SPEECH_URL = process.env.SPEECH_URL || "ws://127.0.0.1:8765/ws/speech-in";
const TTS_URL = process.env.TTS_URL || "ws://127.0.0.1:8765/ws/tts";
const TTS_VOICE = process.env.TTS_VOICE || "ef_dora";
const TTS_LANG = process.env.TTS_LANG || "es";
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || "deepseek/deepseek-chat";
const SYSTEM_PROMPT =
  "Respondé en español, como si hablaras en voz alta. Dos a cuatro oraciones. Sin markdown, listas ni código.";

const openrouter = OPENROUTER_API_KEY
  ? new OpenRouter({ apiKey: OPENROUTER_API_KEY })
  : null;

type ChatMsg = { role: "system" | "user" | "assistant"; content: string };

type Turn = {
  id: string;
  tSpeechEnd: number;
  tTranscript: number;
  tLlm: number;
  gotFirstToken: boolean;
  gotFirstSentence: boolean;
};

type TtsWait = { turnId: string; tSpeak: number; gotPcm: boolean };

function now() {
  return performance.now();
}

function ms(t0: number) {
  return Math.round(now() - t0);
}

function newTurnId() {
  return randomUUID().slice(0, 4);
}

function logTurn(id: string, line: string) {
  console.log(`[${id}] ${line}`);
}

function errMsg(err: unknown) {
  return err instanceof Error ? err.message : String(err);
}

function takeSentences(buf: string, flushAll: boolean) {
  const sentences: string[] = [];
  const re = /[.!?](?:[ \t\n]+|$)|[\n]+/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(buf))) {
    const piece = buf.slice(last, m.index + m[0].length).trim();
    if (piece) sentences.push(piece);
    last = m.index + m[0].length;
  }
  if (flushAll) {
    const rest = buf.slice(last).trim();
    if (rest) sentences.push(rest);
    return { sentences, rest: "" };
  }
  return { sentences, rest: buf.slice(last) };
}

{
  const a = takeSentences("Hi. How are you? ", false);
  if (a.sentences.join("|") !== "Hi.|How are you?" || a.rest !== "") {
    throw new Error("takeSentences punct");
  }
  const b = takeSentences("Hi. more", false);
  if (b.sentences.join("|") !== "Hi." || b.rest !== "more") {
    throw new Error("takeSentences rest");
  }
  const c = takeSentences("no period", true);
  if (c.sentences.join("|") !== "no period") throw new Error("takeSentences flush");
}

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function sendFile(res: http.ServerResponse, urlPath: string) {
  const rel = urlPath === "/" ? "index.html" : urlPath.replace(/^\/+/, "");
  const filePath = path.resolve(path.join(PUBLIC, rel));
  if (filePath !== PUBLIC && !filePath.startsWith(PUBLIC + path.sep)) {
    res.writeHead(403);
    res.end("forbidden");
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(err.code === "ENOENT" ? 404 : 500);
      res.end(err.code === "ENOENT" ? "not found" : "error");
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host}`);
  if (url.pathname === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        ok: true,
        speech: SPEECH_URL,
        tts: TTS_URL,
        llm: Boolean(openrouter),
        model: openrouter ? OPENROUTER_MODEL : null,
      }),
    );
    return;
  }
  sendFile(res, url.pathname);
});

const wss = new WebSocketServer({ server, path: "/ws" });

function sleep(msWait: number) {
  return new Promise((r) => setTimeout(r, msWait));
}

async function openSocket(url: string, retries = 40): Promise<WebSocket> {
  let last: unknown;
  for (let i = 0; i < retries; i++) {
    try {
      const ws = await new Promise<WebSocket>((resolve, reject) => {
        const sock = new WebSocket(url);
        const onErr = (err: Error) => {
          sock.close();
          reject(err);
        };
        sock.once("open", () => {
          sock.off("error", onErr);
          resolve(sock);
        });
        sock.once("error", onErr);
      });
      return ws;
    } catch (err) {
      last = err;
      await sleep(500);
    }
  }
  throw last instanceof Error ? last : new Error(`cannot connect ${url}`);
}

function sendJson(ws: WebSocket | undefined, obj: unknown) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function sendRaw(ws: WebSocket | undefined, data: WebSocket.RawData, binary: boolean) {
  if (ws?.readyState !== WebSocket.OPEN) return;
  if (binary) ws.send(data, { binary: true });
  else ws.send(typeof data === "string" ? data : data.toString());
}

function deltaText(chunk: unknown): string {
  if (!chunk || typeof chunk !== "object") return "";
  const choices = (chunk as { choices?: Array<{ delta?: { content?: unknown } }> }).choices;
  const c = choices?.[0]?.delta?.content;
  return typeof c === "string" ? c : "";
}

function chunkProvider(chunk: unknown): string {
  if (!chunk || typeof chunk !== "object") return "";
  const o = chunk as Record<string, unknown>;
  if (typeof o.provider === "string") return o.provider;
  const meta = o.openrouterMetadata ?? o.openrouter_metadata;
  if (meta && typeof meta === "object") {
    const attempts = (meta as { attempts?: Array<{ provider?: unknown }> }).attempts;
    const p = attempts?.[0]?.provider;
    if (typeof p === "string") return p;
  }
  return "";
}

function aborted(err: unknown) {
  const name =
    err && typeof err === "object" && "name" in err ? String((err as { name: unknown }).name) : "";
  return name === "AbortError" || name === "RequestAbortedError";
}

wss.on("connection", async (client) => {
  let echo = false;
  let llmBusy = false;
  let pendingSpeak = 0;
  let llmAbort: AbortController | null = null;
  let speech: WebSocket | undefined;
  let tts: WebSocket | undefined;
  const messages: ChatMsg[] = [{ role: "system", content: SYSTEM_PROMPT }];
  let turn: Turn | null = null;
  const ttsWaits = new Map<string, TtsWait>();
  let ttsActiveId = "";

  const closeUp = () => {
    llmAbort?.abort();
    llmAbort = null;
    try {
      speech?.close();
    } catch {
      /* ignore */
    }
    try {
      tts?.close();
    } catch {
      /* ignore */
    }
  };

  function maybeTurnEnd() {
    if (llmBusy || pendingSpeak > 0) return;
    sendJson(client, { type: "turn_end" });
  }

  function abortTurn() {
    const hadWork = llmBusy || pendingSpeak > 0 || llmAbort;
    if (hadWork && turn) logTurn(turn.id, "abort");
    llmAbort?.abort();
    llmAbort = null;
    llmBusy = false;
    pendingSpeak = 0;
    ttsWaits.clear();
    ttsActiveId = "";
    sendJson(tts, { type: "cancel", id: "" });
  }

  function speakTts(text: string) {
    const t = String(text || "").trim();
    if (!t) return;
    const id = randomUUID();
    pendingSpeak++;
    const turnId = turn?.id ?? newTurnId();
    ttsWaits.set(id, { turnId, tSpeak: now(), gotPcm: false });
    logTurn(turnId, `tts speak id=${id} chars=${t.length}`);
    sendJson(tts, {
      type: "speak",
      id,
      text: t,
      voice: TTS_VOICE,
      lang: TTS_LANG,
    });
  }

  function speakAssistant(text: string) {
    const t = String(text || "").trim();
    if (!t) return;
    if (turn && !turn.gotFirstSentence) {
      turn.gotFirstSentence = true;
      logTurn(turn.id, `llm first_sentence_ms=${ms(turn.tLlm)} chars=${t.length}`);
    }
    sendJson(client, { type: "assistant", text: t });
    speakTts(t);
  }

  async function runLlm(userText: string) {
    abortTurn();
    if (!openrouter) {
      sendJson(client, { type: "error", message: "OPENROUTER_API_KEY missing in node/.env" });
      return;
    }
    if (!turn) turn = { id: newTurnId(), tSpeechEnd: 0, tTranscript: now(), tLlm: 0, gotFirstToken: false, gotFirstSentence: false };
    const ac = new AbortController();
    llmAbort = ac;
    llmBusy = true;
    const userMsg: ChatMsg = { role: "user", content: userText };
    messages.push(userMsg);
    let buf = "";
    let full = "";
    turn.tLlm = now();
    turn.gotFirstToken = false;
    turn.gotFirstSentence = false;
    logTurn(turn.id, "llm start sort=latency");
    try {
      const stream = (await openrouter.chat.send(
        {
          xOpenRouterMetadata: "enabled",
          chatRequest: {
            model: OPENROUTER_MODEL,
            messages,
            stream: true,
            provider: { sort: "latency" },
          },
        },
        { signal: ac.signal },
      )) as AsyncIterable<unknown>;
      let provider = "";
      for await (const chunk of stream) {
        if (ac.signal.aborted) return;
        if (!provider) provider = chunkProvider(chunk);
        const piece = deltaText(chunk);
        if (!piece) continue;
        if (turn && !turn.gotFirstToken) {
          turn.gotFirstToken = true;
          logTurn(
            turn.id,
            `llm first_token_ms=${ms(turn.tLlm)}${provider ? ` provider=${provider}` : ""}`,
          );
        }
        buf += piece;
        full += piece;
        const { sentences, rest } = takeSentences(buf, false);
        buf = rest;
        for (const s of sentences) speakAssistant(s);
      }
      const { sentences } = takeSentences(buf, true);
      for (const s of sentences) speakAssistant(s);
    } catch (err) {
      if (aborted(err) || ac.signal.aborted) return;
      sendJson(client, { type: "error", message: errMsg(err) });
    } finally {
      if (turn && turn.tLlm) logTurn(turn.id, `llm done total_ms=${ms(turn.tLlm)} chars=${full.length}`);
      if (full.trim()) {
        const idx = messages.indexOf(userMsg);
        if (idx >= 0) messages.splice(idx + 1, 0, { role: "assistant", content: full.trim() });
      }
      if (llmAbort !== ac) return;
      llmAbort = null;
      llmBusy = false;
      maybeTurnEnd();
    }
  }

  client.on("close", closeUp);
  client.on("error", closeUp);

  try {
    speech = await openSocket(SPEECH_URL);
    tts = await openSocket(TTS_URL);
  } catch (err) {
    sendJson(client, {
      type: "error",
      message: `python upstream: ${errMsg(err)}. Is uvicorn on 8765?`,
    });
    client.close();
    return;
  }

  speech.on("message", (data, isBinary) => {
    if (isBinary) {
      sendRaw(client, data, true);
      return;
    }
    let msg: { type?: string; text?: string; stt_ms?: number; duration_ms?: number };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      sendRaw(client, data, false);
      return;
    }
    if (msg.type === "speech_start") {
      abortTurn();
      logTurn(turn?.id ?? "----", "speech_start");
    }
    if (msg.type === "speech_end") {
      turn = {
        id: newTurnId(),
        tSpeechEnd: now(),
        tTranscript: 0,
        tLlm: 0,
        gotFirstToken: false,
        gotFirstSentence: false,
      };
      logTurn(turn.id, `speech_end duration_ms=${msg.duration_ms ?? ""}`);
    }
    sendRaw(client, data, false);
    if (msg.type !== "transcript") return;
    const text = String(msg.text || "").trim();
    if (!turn) {
      turn = {
        id: newTurnId(),
        tSpeechEnd: 0,
        tTranscript: now(),
        tLlm: 0,
        gotFirstToken: false,
        gotFirstSentence: false,
      };
    } else {
      turn.tTranscript = now();
    }
    const hop = turn.tSpeechEnd ? ms(turn.tSpeechEnd) : 0;
    if (!text) {
      logTurn(turn.id, `transcript empty stt_ms=${msg.stt_ms ?? ""} hop_ms=${hop}`);
      return;
    }
    logTurn(turn.id, `transcript stt_ms=${msg.stt_ms ?? ""} hop_ms=${hop} text=${text}`);
    if (echo) {
      speakTts(text);
      return;
    }
    runLlm(text);
  });

  tts.on("message", (data, isBinary) => {
    sendRaw(client, data, isBinary);
    if (isBinary) {
      const wait = ttsWaits.get(ttsActiveId);
      if (wait && !wait.gotPcm) {
        wait.gotPcm = true;
        logTurn(wait.turnId, `tts first_pcm_ms=${ms(wait.tSpeak)} id=${ttsActiveId}`);
      }
      return;
    }
    let msg: { type?: string; id?: string; synth_ms?: number };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "audio_start") {
      ttsActiveId = String(msg.id || "");
      const wait = ttsWaits.get(ttsActiveId);
      if (wait) logTurn(wait.turnId, `tts audio_start_ms=${ms(wait.tSpeak)} id=${ttsActiveId}`);
    }
    if (msg.type === "audio_end" || msg.type === "error") {
      const id = String(msg.id || ttsActiveId);
      const wait = ttsWaits.get(id);
      if (!wait) return;
      const extra = msg.type === "audio_end" ? ` synth_ms=${msg.synth_ms ?? ""}` : "";
      logTurn(wait.turnId, `tts ${msg.type} total_ms=${ms(wait.tSpeak)}${extra} id=${id}`);
      ttsWaits.delete(id);
      if (pendingSpeak > 0) pendingSpeak--;
      maybeTurnEnd();
    }
  });

  speech.on("close", () => sendJson(client, { type: "error", message: "speech-in closed" }));
  tts.on("close", () => sendJson(client, { type: "error", message: "tts closed" }));
  speech.on("error", (err) => sendJson(client, { type: "error", message: String(err) }));
  tts.on("error", (err) => sendJson(client, { type: "error", message: String(err) }));

  client.on("message", (data, isBinary) => {
    if (isBinary) {
      if (pendingSpeak === 0) sendRaw(speech, data, true);
      return;
    }
    let msg: { type?: string; enabled?: boolean; text?: string };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "echo") {
      echo = Boolean(msg.enabled);
      return;
    }
    if (msg.type === "speak") {
      const text = String(msg.text || "").trim();
      if (!text) return;
      turn = {
        id: newTurnId(),
        tSpeechEnd: 0,
        tTranscript: now(),
        tLlm: 0,
        gotFirstToken: false,
        gotFirstSentence: false,
      };
      logTurn(turn.id, `client speak chars=${text.length}`);
      speakTts(text);
      return;
    }
    if (msg.type === "cancel") abortTurn();
  });

  sendJson(client, { type: "ready", echo, llm: Boolean(openrouter), model: OPENROUTER_MODEL });
});

server.listen(PORT, "0.0.0.0", () => {
  const llm = openrouter ? OPENROUTER_MODEL : "off";
  console.log(`voya bff http://127.0.0.1:${PORT}  speech=${SPEECH_URL}  tts=${TTS_URL}  llm=${llm}`);
  console.log("use headphones — TTS playback is muted from speech-in, but speakers still leak");
});
