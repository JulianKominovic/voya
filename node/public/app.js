const $ = (id) => document.getElementById(id);

const talkBtn = $("talk");
const stopBtn = $("stop");
const synthBtn = $("synth");
const cancelBtn = $("cancel");
const echoBox = $("echo");
const stateEl = $("state");
const logEl = $("log");
const transcriptEl = $("transcript");
const latSttEl = $("lat-stt");
const latTtsEl = $("lat-tts");
const synthText = $("synth-text");

let ws;
let captureCtx;
let playCtx;
let mediaStream;
let processor;
let capturing = false;
let player;
let tSpeechEnd = 0;
let tSpeak = 0;
let awaitingFirstAudio = false;
let uiState = "idle";

function log(line) {
  const t = new Date().toISOString().slice(11, 23);
  logEl.textContent += `${t} ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setState(name) {
  uiState = name;
  stateEl.textContent = name;
  stateEl.className = "pill";
  if (name === "listening" || name === "speaking") stateEl.classList.add("on");
  if (name === "speech" || name === "transcribing") stateEl.classList.add("speech");
  if (name === "error") stateEl.classList.add("err");
}

function downsample(input, inRate, outRate) {
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

function floatToS16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function s16ToFloat(buf) {
  const i16 = new Int16Array(buf);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}

class PcmPlayer {
  constructor(ctx) {
    this.ctx = ctx;
    this.next = 0;
    this.sources = [];
  }

  push(float32) {
    if (!float32.length) return;
    const buf = this.ctx.createBuffer(1, float32.length, this.ctx.sampleRate);
    buf.getChannelData(0).set(float32);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    if (this.next < now + 0.1) this.next = now + 0.1;
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
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    log("ws open");
    ws.send(JSON.stringify({ type: "echo", enabled: echoBox.checked }));
  };
  ws.onclose = () => log("ws close");
  ws.onerror = () => log("ws error");
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      if (awaitingFirstAudio) {
        awaitingFirstAudio = false;
        const ms = Math.round(performance.now() - tSpeak);
        latTtsEl.textContent = `${ms} ms`;
        log(`tts_first_chunk_ms=${ms}`);
      }
      player?.push(s16ToFloat(ev.data));
      return;
    }
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    onJson(msg);
  };
}

function onJson(msg) {
  if (msg.type === "ready") {
    log(`ready echo=${msg.echo}`);
    return;
  }
  if (msg.type === "error") {
    log(`error ${msg.message}`);
    setState("error");
    return;
  }
  if (msg.type === "speech_start") {
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
    const ms = tSpeechEnd ? Math.round(performance.now() - tSpeechEnd) : msg.stt_ms;
    latSttEl.textContent = `${ms} ms`;
    transcriptEl.textContent = msg.text || "(vacío)";
    log(`transcript stt_ms=${msg.stt_ms} ${msg.text}`);
    if (echoBox.checked && msg.text?.trim()) {
      tSpeak = performance.now();
      awaitingFirstAudio = true;
      setState("speaking");
    } else if (capturing) {
      setState("listening");
    } else {
      setState("idle");
    }
    return;
  }
  if (msg.type === "audio_start") {
    player?.stop();
    setState("speaking");
    log(`audio_start id=${msg.id} ${msg.sample_rate} Hz`);
    return;
  }
  if (msg.type === "audio_end") {
    log(`audio_end id=${msg.id} synth_ms=${msg.synth_ms}`);
    awaitingFirstAudio = false;
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
    if (!capturing || !ws || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const down = downsample(input, captureCtx.sampleRate, 16000);
    ws.send(floatToS16(down).buffer);
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
  processor = null;
  mediaStream = null;
  talkBtn.disabled = false;
  stopBtn.disabled = true;
  if (uiState !== "speaking") setState("idle");
  log("mic stopped");
}

talkBtn.addEventListener("click", () => {
  startMic().catch((err) => {
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
  awaitingFirstAudio = true;
  setState("speaking");
  const send = () => ws.send(JSON.stringify({ type: "speak", id: crypto.randomUUID(), text }));
  if (ws.readyState === WebSocket.OPEN) send();
  else ws.addEventListener("open", send, { once: true });
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
