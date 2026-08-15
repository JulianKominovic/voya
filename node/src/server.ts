import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { WebSocketServer } from "ws";
import { OpenRouter } from "@openrouter/sdk";
import {
  AGENTIC_MODEL,
  OPENROUTER_API_KEY,
  ORCHESTRATOR_MODEL,
  PORT,
  PUBLIC,
  SPEECH_URL,
  TTS_URL,
} from "./config.js";
import { Session } from "./session.js";
import { errMsg, sleep } from "./text.js";
import { openSocket, sendJson } from "./wsutil.js";

const openrouter = OPENROUTER_API_KEY
  ? new OpenRouter({ apiKey: OPENROUTER_API_KEY })
  : null;

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
        orchestrator: openrouter ? ORCHESTRATOR_MODEL : null,
        agentic: openrouter ? AGENTIC_MODEL : null,
      }),
    );
    return;
  }
  sendFile(res, url.pathname);
});

const wss = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", async (client) => {
  const session = new Session(client, openrouter);
  let reconnecting = false;

  const closeUp = () => session.destroy();
  client.on("close", closeUp);
  client.on("error", closeUp);

  async function attachPython() {
    if (session.closed) return;
    try {
      const speech = await openSocket(SPEECH_URL);
      const tts = await openSocket(TTS_URL);
      if (session.closed) {
        try {
          speech.close();
        } catch {
          /* ignore */
        }
        try {
          tts.close();
        } catch {
          /* ignore */
        }
        return;
      }
      session.bindPython(speech, tts);
      session.emitState();
    } catch (err) {
      session.logTurn(`python upstream: ${errMsg(err)}. Is uvicorn on 8765?`, "stt");
      sendJson(client, {
        type: "error",
        message: `python upstream: ${errMsg(err)}. Is uvicorn on 8765?`,
      });
      if (session.closed) return;
      await sleep(500);
      return attachPython();
    }
  }

  session.onPythonDead = () => {
    if (session.closed || reconnecting) return;
    reconnecting = true;
    session.logTurn("python ws closed, reconnecting", "stt");
    sendJson(client, { type: "error", message: "python ws closed, reconnecting" });
    void attachPython().finally(() => {
      reconnecting = false;
    });
  };

  client.on("message", (data, isBinary) => {
    if (isBinary) {
      session.onClientPcm(data);
      return;
    }
    let msg: { type?: string; enabled?: boolean; on?: boolean; text?: string };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === "echo") {
      session.echo = Boolean(msg.enabled);
      session.emitState();
      return;
    }
    if (msg.type === "mic") {
      session.setMic(Boolean(msg.on));
      return;
    }
    if (msg.type === "speak") {
      const text = String(msg.text || "").trim();
      if (!text) return;
      session.logTurn(`client speak chars=${text.length}`, "tts");
      session.speak(text, session.gen);
      return;
    }
    if (msg.type === "cancel") {
      session.abortVoicePath();
      session.goIdleOrListening();
    }
  });

  await attachPython();
  if (session.closed) return;
  sendJson(client, {
    type: "ready",
    echo: session.echo,
    llm: Boolean(openrouter),
    orchestrator: ORCHESTRATOR_MODEL,
    agentic: AGENTIC_MODEL,
  });
  session.emitState();
});

server.listen(PORT, "0.0.0.0", () => {
  const llm = openrouter ? `orch=${ORCHESTRATOR_MODEL} agentic=${AGENTIC_MODEL}` : "off";
  console.log(`voya bff http://0.0.0.0:${PORT}  speech=${SPEECH_URL}  tts=${TTS_URL}  llm=${llm}`);
  console.log("use headphones — mic is always forwarded (barge-in); speakers still leak");
});
