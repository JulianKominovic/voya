import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { WebSocketServer, WebSocket } from "ws";
import { OpenRouter } from "@openrouter/sdk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadEnv(file) {
  let raw;
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

function takeSentences(buf, flushAll) {
  const sentences = [];
  const re = /[.!?](?:[ \t\n]+|$)|[\n]+/g;
  let last = 0;
  let m;
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

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function sendFile(res, urlPath) {
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

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function openSocket(url, retries = 40) {
  let last;
  for (let i = 0; i < retries; i++) {
    try {
      const ws = await new Promise((resolve, reject) => {
        const sock = new WebSocket(url);
        const onErr = (err) => {
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
  throw last || new Error(`cannot connect ${url}`);
}

function sendJson(ws, obj) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function sendRaw(ws, data, binary) {
  if (ws?.readyState !== WebSocket.OPEN) return;
  if (binary) ws.send(data, { binary: true });
  else ws.send(typeof data === "string" ? data : data.toString());
}

function deltaText(chunk) {
  const c = chunk?.choices?.[0]?.delta?.content;
  if (typeof c === "string") return c;
  return "";
}

function aborted(err) {
  const name = err?.name || "";
  return name === "AbortError" || name === "RequestAbortedError";
}

wss.on("connection", async (client) => {
  let echo = false;
  let llmBusy = false;
  let pendingSpeak = 0;
  let llmAbort = null;
  let speech;
  let tts;
  const messages = [{ role: "system", content: SYSTEM_PROMPT }];

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
    llmAbort?.abort();
    llmAbort = null;
    llmBusy = false;
    pendingSpeak = 0;
    sendJson(tts, { type: "cancel", id: "" });
  }

  function speakTts(text) {
    const t = String(text || "").trim();
    if (!t) return;
    pendingSpeak++;
    sendJson(tts, {
      type: "speak",
      id: randomUUID(),
      text: t,
      voice: TTS_VOICE,
      lang: TTS_LANG,
    });
  }

  function speakAssistant(text) {
    const t = String(text || "").trim();
    if (!t) return;
    sendJson(client, { type: "assistant", text: t });
    speakTts(t);
  }

  async function runLlm(userText) {
    abortTurn();
    if (!openrouter) {
      sendJson(client, { type: "error", message: "OPENROUTER_API_KEY missing in node/.env" });
      return;
    }
    const ac = new AbortController();
    llmAbort = ac;
    llmBusy = true;
    const userMsg = { role: "user", content: userText };
    messages.push(userMsg);
    let buf = "";
    let full = "";
    try {
      const stream = await openrouter.chat.send(
        {
          chatRequest: {
            model: OPENROUTER_MODEL,
            messages,
            stream: true,
          },
        },
        { signal: ac.signal },
      );
      for await (const chunk of stream) {
        if (ac.signal.aborted) return;
        const piece = deltaText(chunk);
        if (!piece) continue;
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
      sendJson(client, { type: "error", message: String(err.message || err) });
    } finally {
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
      message: `python upstream: ${err.message}. Is uvicorn on 8765?`,
    });
    client.close();
    return;
  }

  speech.on("message", (data, isBinary) => {
    if (isBinary) {
      sendRaw(client, data, true);
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      sendRaw(client, data, false);
      return;
    }
    if (msg.type === "speech_start") abortTurn();
    sendRaw(client, data, false);
    if (msg.type !== "transcript") return;
    const text = String(msg.text || "").trim();
    if (!text) return;
    if (echo) {
      speakTts(text);
      return;
    }
    runLlm(text);
  });

  tts.on("message", (data, isBinary) => {
    sendRaw(client, data, isBinary);
    if (isBinary) return;
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "audio_end" || msg.type === "error") {
      if (pendingSpeak === 0 && !llmBusy) return;
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
    let msg;
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
