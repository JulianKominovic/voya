import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { WebSocketServer, WebSocket } from "ws";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.resolve(path.join(__dirname, "../public"));
const PORT = Number(process.env.PORT || 8787);
const SPEECH_URL = process.env.SPEECH_URL || "ws://127.0.0.1:8765/ws/speech-in";
const TTS_URL = process.env.TTS_URL || "ws://127.0.0.1:8765/ws/tts";
const TTS_VOICE = process.env.TTS_VOICE || "ef_dora";
const TTS_LANG = process.env.TTS_LANG || "es";

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
    res.end(JSON.stringify({ ok: true, speech: SPEECH_URL, tts: TTS_URL }));
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

wss.on("connection", async (client) => {
  let echo = true;
  let speaking = false;
  let speech;
  let tts;

  const closeUp = () => {
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
    sendRaw(client, data, isBinary);
    if (isBinary) return;
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "transcript" && echo && String(msg.text || "").trim()) {
      speaking = true;
      sendJson(tts, {
        type: "speak",
        id: randomUUID(),
        text: String(msg.text).trim(),
        voice: TTS_VOICE,
        lang: TTS_LANG,
      });
    }
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
    if (msg.type === "audio_start") speaking = true;
    if (msg.type === "audio_end" || msg.type === "error") speaking = false;
  });

  speech.on("close", () => sendJson(client, { type: "error", message: "speech-in closed" }));
  tts.on("close", () => sendJson(client, { type: "error", message: "tts closed" }));
  speech.on("error", (err) => sendJson(client, { type: "error", message: String(err) }));
  tts.on("error", (err) => sendJson(client, { type: "error", message: String(err) }));

  client.on("message", (data, isBinary) => {
    if (isBinary) {
      if (!speaking) sendRaw(speech, data, true);
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
      speaking = true;
      sendJson(tts, {
        type: "speak",
        id: msg.id || randomUUID(),
        text,
        voice: msg.voice || TTS_VOICE,
        lang: msg.lang || TTS_LANG,
      });
      return;
    }
    if (msg.type === "cancel") {
      speaking = false;
      sendJson(tts, { type: "cancel", id: msg.id || "" });
    }
  });

  sendJson(client, { type: "ready", echo });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`voya bff http://127.0.0.1:${PORT}  speech=${SPEECH_URL}  tts=${TTS_URL}`);
  console.log("use headphones — TTS playback is muted from speech-in, but speakers still leak");
});
