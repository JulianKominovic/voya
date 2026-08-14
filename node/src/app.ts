function $<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
}

const talkBtn = $<HTMLButtonElement>("talk");
const stopBtn = $<HTMLButtonElement>("stop");
const synthBtn = $<HTMLButtonElement>("synth");
const cancelBtn = $<HTMLButtonElement>("cancel");
const echoBox = $<HTMLInputElement>("echo");
const stateEl = $("state");
const logEl = $("log");
const transcriptEl = $("transcript");
const assistantEl = $("assistant");
const latSttEl = $("lat-stt");
const latLlmEl = $("lat-llm");
const latTtsEl = $("lat-tts");
const synthText = $<HTMLTextAreaElement>("synth-text");

let ws: WebSocket | undefined;
let captureCtx: AudioContext | undefined;
let playCtx: AudioContext | undefined;
let mediaStream: MediaStream | undefined;
let processor: ScriptProcessorNode | undefined;
let capturing = false;
let player: PcmPlayer | undefined;
let tSpeechEnd = 0;
let tSpeak = 0;
let tTranscript = 0;
let awaitingFirstAudio = false;
let uiState = "idle";
let assistantBuf = "";

function log(line: string) {
  const t = new Date().toISOString().slice(11, 23);
  logEl.textContent += `${t} ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setState(name: string) {
  uiState = name;
  stateEl.textContent = name;
  stateEl.className = "pill";
  if (name === "listening" || name === "speaking") stateEl.classList.add("on");
  if (name === "speech" || name === "transcribing") stateEl.classList.add("speech");
  if (name === "thinking") stateEl.classList.add("think");
  if (name === "error") stateEl.classList.add("err");
}

function downsample(input: Float32Array, inRate: number, outRate: number) {
  if (inRate === outRate) return input;
  const ratio = inRate / outRate;
  const outLen = Math.round(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const frac = idx - i0;
    const a = input[i0] || 0;
    const b = input[i0 + 1] || a;
    out[i] = a + frac * (b - a);
  }
  return out;
}

function floatToS16(float32: Float32Array) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function s16ToFloat(buf: ArrayBuffer) {
  const i16 = new Int16Array(buf);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}

class PcmPlayer {
  ctx: AudioContext;
  next = 0;
  sources: AudioBufferSourceNode[] = [];

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
  }

  push(float32: Float32Array) {
    if (!float32.length) return;
    const buf = this.ctx.createBuffer(1, float32.length, this.ctx.sampleRate);
    buf.getChannelData(0).set(float32);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const t = this.ctx.currentTime;
    if (this.next < t + 0.1) this.next = t + 0.1;
    src.start(this.next);
    this.next += buf.duration;
    this.sources.push(src);
    src.onended = () => {
      this.sources = this.sources.filter((s) => s !== src);
    };
  }

  stop() {
    for (const src of this.sources) {
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources = [];
    this.next = 0;
  }
}

function connectWs() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const sock = new WebSocket(`${proto}://${location.host}/ws`);
  ws = sock;
  sock.binaryType = "arraybuffer";
  sock.onopen = () => {
    log("ws open");
    sock.send(JSON.stringify({ type: "echo", enabled: echoBox.checked }));
  };
  sock.onclose = () => log("ws close");
  sock.onerror = () => log("ws error");
  sock.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      if (awaitingFirstAudio) {
        awaitingFirstAudio = false;
        if (tSpeak) {
          const elapsed = Math.round(performance.now() - tSpeak);
          latTtsEl.textContent = `${elapsed} ms`;
          log(`tts_first_chunk_ms=${elapsed}`);
        }
        if (tTranscript) {
          const elapsed = Math.round(performance.now() - tTranscript);
          latLlmEl.textContent = `${elapsed} ms`;
          log(`turn_first_chunk_ms=${elapsed}`);
        }
      }
      player?.push(s16ToFloat(ev.data as ArrayBuffer));
      return;
    }
    let msg: {
      type?: string;
      echo?: boolean;
      llm?: boolean;
      model?: string;
      message?: string;
      duration_ms?: number;
      stt_ms?: number;
      text?: string;
      id?: string;
      sample_rate?: number;
      synth_ms?: number;
    };
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    onJson(msg);
  };
}

function onJson(msg: {
  type?: string;
  echo?: boolean;
  llm?: boolean;
  model?: string;
  message?: string;
  duration_ms?: number;
  stt_ms?: number;
  text?: string;
  id?: string;
  sample_rate?: number;
  synth_ms?: number;
}) {
  if (msg.type === "ready") {
    log(`ready echo=${msg.echo} llm=${msg.llm} model=${msg.model || ""}`);
    return;
  }
  if (msg.type === "error") {
    log(`error ${msg.message}`);
    setState("error");
    return;
  }
  if (msg.type === "speech_start") {
    player?.stop();
    awaitingFirstAudio = false;
    setState("speech");
    log("speech_start");
    return;
  }
  if (msg.type === "speech_end") {
    tSpeechEnd = performance.now();
    setState("transcribing");
    log(`speech_end duration_ms=${msg.duration_ms}`);
    return;
  }
  if (msg.type === "transcript") {
    const elapsed = tSpeechEnd ? Math.round(performance.now() - tSpeechEnd) : msg.stt_ms;
    latSttEl.textContent = `${elapsed} ms`;
    transcriptEl.textContent = msg.text || "(empty)";
    log(`transcript stt_ms=${msg.stt_ms} ${msg.text}`);
    if (!msg.text?.trim()) {
      if (capturing) setState("listening");
      else setState("idle");
      return;
    }
    tTranscript = performance.now();
    awaitingFirstAudio = true;
    assistantBuf = "";
    assistantEl.textContent = "—";
    if (echoBox.checked) {
      tSpeak = tTranscript;
      setState("speaking");
    } else {
      tSpeak = 0;
      setState("thinking");
    }
    return;
  }
  if (msg.type === "assistant") {
    assistantBuf = assistantBuf ? `${assistantBuf} ${msg.text}` : msg.text || "";
    assistantEl.textContent = assistantBuf;
    log(`assistant ${msg.text}`);
    setState("speaking");
    return;
  }
  if (msg.type === "audio_start") {
    setState("speaking");
    log(`audio_start id=${msg.id} ${msg.sample_rate} Hz`);
    return;
  }
  if (msg.type === "audio_end") {
    log(`audio_end id=${msg.id} synth_ms=${msg.synth_ms}`);
    return;
  }
  if (msg.type === "turn_end") {
    awaitingFirstAudio = false;
    if (uiState === "error") return;
    if (capturing) setState("listening");
    else setState("idle");
  }
}

async function startMic() {
  connectWs();
  captureCtx = new AudioContext();
  playCtx = new AudioContext({ sampleRate: 24000 });
  await captureCtx.resume();
  await playCtx.resume();
  player = new PcmPlayer(playCtx);
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      channelCount: 1,
    },
  });
  const src = captureCtx.createMediaStreamSource(mediaStream);
  processor = captureCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    if (!capturing || !ws || ws.readyState !== WebSocket.OPEN || !captureCtx) return;
    const input = e.inputBuffer.getChannelData(0);
    const down = downsample(input, captureCtx.sampleRate, 16000);
    ws.send(floatToS16(down).buffer as ArrayBuffer);
  };
  const mute = captureCtx.createGain();
  mute.gain.value = 0;
  src.connect(processor);
  processor.connect(mute);
  mute.connect(captureCtx.destination);
  capturing = true;
  talkBtn.disabled = true;
  stopBtn.disabled = false;
  setState("listening");
  log(`mic ${captureCtx.sampleRate} Hz → 16000, play ${playCtx.sampleRate} Hz`);
}

function stopMic() {
  capturing = false;
  processor?.disconnect();
  mediaStream?.getTracks().forEach((t) => t.stop());
  captureCtx?.close();
  processor = undefined;
  mediaStream = undefined;
  talkBtn.disabled = false;
  stopBtn.disabled = true;
  if (uiState !== "speaking" && uiState !== "thinking") setState("idle");
  log("mic stopped");
}

talkBtn.addEventListener("click", () => {
  startMic().catch((err: unknown) => {
    log(String(err));
    setState("error");
  });
});
stopBtn.addEventListener("click", stopMic);

echoBox.addEventListener("change", () => {
  connectWs();
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "echo", enabled: echoBox.checked }));
  }
});

synthBtn.addEventListener("click", async () => {
  connectWs();
  if (!playCtx) {
    playCtx = new AudioContext({ sampleRate: 24000 });
    await playCtx.resume();
    player = new PcmPlayer(playCtx);
  }
  const text = synthText.value.trim();
  if (!text) return;
  tSpeak = performance.now();
  tTranscript = 0;
  awaitingFirstAudio = true;
  player?.stop();
  setState("speaking");
  const sock = ws;
  if (!sock) return;
  const send = () => sock.send(JSON.stringify({ type: "speak", id: crypto.randomUUID(), text }));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
});

cancelBtn.addEventListener("click", () => {
  player?.stop();
  awaitingFirstAudio = false;
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "cancel", id: "" }));
  }
  if (capturing) setState("listening");
  else setState("idle");
});

connectWs();
