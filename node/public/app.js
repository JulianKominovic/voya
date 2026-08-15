"use strict";
function $(id) {
    const el = document.getElementById(id);
    if (!el)
        throw new Error(`#${id} missing`);
    return el;
}
const talkBtn = $("talk");
const stopBtn = $("stop");
const synthBtn = $("synth");
const cancelBtn = $("cancel");
const echoBox = $("echo");
const stateEl = $("state");
const logEl = $("log");
const logBody = $("log-body");
const transcriptEl = $("transcript");
const assistantEl = $("assistant");
const questionEl = $("question");
const latSttEl = $("lat-stt");
const latLlmEl = $("lat-llm");
const latTtsEl = $("lat-tts");
const synthText = $("synth-text");
let ws;
let captureCtx;
let playCtx;
let mediaStream;
let processor;
let capturing = false;
let player;
let aec = !new URLSearchParams(location.search).has("noaec");
let pcMic;
let pcTts;
let ttsDest;
let renderEl;
let tSpeechEnd = 0;
let tSpeak = 0;
let tTranscript = 0;
let awaitingFirstAudio = false;
let uiState = "idle";
let assistantBuf = "";
const COLUMNS = ["stt", "llm_conv", "llm_agentic", "minions", "tts"];
const MAX_ROWS = 300;
function addRow(channel, text, ts) {
    const tr = document.createElement("tr");
    const t = new Date(ts ?? Date.now()).toISOString().slice(11, 23);
    const tdTime = document.createElement("td");
    tdTime.className = "time";
    tdTime.textContent = t;
    tr.appendChild(tdTime);
    for (const col of COLUMNS) {
        const td = document.createElement("td");
        if (col === channel) {
            td.className = `evt ${col}`;
            td.textContent = text;
            td.title = text;
        }
        tr.appendChild(td);
    }
    logBody.appendChild(tr);
    while (logBody.rows.length > MAX_ROWS)
        logBody.deleteRow(0);
    logEl.scrollTop = logEl.scrollHeight;
}
function setState(name) {
    uiState = name;
    stateEl.textContent = name;
    stateEl.className = "pill";
    if (name === "listening" || name === "speaking")
        stateEl.classList.add("on");
    if (name === "user_speaking" || name === "speech" || name === "transcribing")
        stateEl.classList.add("speech");
    if (name === "thinking")
        stateEl.classList.add("think");
    if (name === "error")
        stateEl.classList.add("err");
}
function downsample(input, inRate, outRate) {
    if (inRate === outRate)
        return input;
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
    for (let i = 0; i < i16.length; i++)
        f32[i] = i16[i] / 32768;
    return f32;
}
class PcmPlayer {
    ctx;
    out;
    next = 0;
    sources = [];
    constructor(ctx, out) {
        this.ctx = ctx;
        this.out = out ?? ctx.destination;
    }
    push(float32) {
        if (!float32.length)
            return;
        const buf = this.ctx.createBuffer(1, float32.length, this.ctx.sampleRate);
        buf.getChannelData(0).set(float32);
        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.out);
        const t = this.ctx.currentTime;
        if (this.next < t + 0.1)
            this.next = t + 0.1;
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
            }
            catch {
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
        addRow("stt", "ws open");
        sock.send(JSON.stringify({ type: "echo", enabled: echoBox.checked }));
    };
    sock.onclose = () => addRow("stt", "ws close");
    sock.onerror = () => addRow("stt", "ws error");
    sock.onmessage = (ev) => {
        if (typeof ev.data !== "string") {
            if (awaitingFirstAudio) {
                awaitingFirstAudio = false;
                if (tSpeak) {
                    const elapsed = Math.round(performance.now() - tSpeak);
                    latTtsEl.textContent = `${elapsed} ms`;
                    addRow("tts", `tts_first_chunk_ms=${elapsed}`);
                }
                if (tTranscript) {
                    const elapsed = Math.round(performance.now() - tTranscript);
                    latLlmEl.textContent = `${elapsed} ms`;
                    addRow("llm_conv", `turn_first_chunk_ms=${elapsed}`);
                }
            }
            player?.push(s16ToFloat(ev.data));
            return;
        }
        let msg;
        try {
            msg = JSON.parse(ev.data);
        }
        catch {
            return;
        }
        onJson(msg);
    };
}
function onJson(msg) {
    if (msg.type === "log") {
        if (msg.channel === "stt" || msg.channel === "llm_conv" || msg.channel === "llm_agentic" || msg.channel === "minions" || msg.channel === "tts") {
            addRow(msg.channel, msg.text || "", msg.ts);
        }
        return;
    }
    if (msg.type === "ready") {
        addRow("llm_conv", `ready orch=${msg.orchestrator || ""} agentic=${msg.agentic || ""}`);
        return;
    }
    if (msg.type === "state") {
        if (msg.state)
            setState(msg.state);
        questionEl.textContent = msg.pending_question || "—";
        return;
    }
    if (msg.type === "question_asked") {
        questionEl.textContent = msg.text || "—";
        return;
    }
    if (msg.type === "question_resolved") {
        return;
    }
    if (msg.type === "error") {
        setState("error");
        return;
    }
    if (msg.type === "speech_start") {
        player?.stop();
        awaitingFirstAudio = false;
        return;
    }
    if (msg.type === "speech_end") {
        tSpeechEnd = performance.now();
        return;
    }
    if (msg.type === "transcript") {
        const elapsed = tSpeechEnd ? Math.round(performance.now() - tSpeechEnd) : msg.stt_ms;
        latSttEl.textContent = `${elapsed} ms`;
        transcriptEl.textContent = msg.text || "(empty)";
        if (msg.text?.trim()) {
            tTranscript = performance.now();
            awaitingFirstAudio = true;
            assistantBuf = "";
            assistantEl.textContent = "—";
            if (echoBox.checked)
                tSpeak = tTranscript;
            else
                tSpeak = 0;
        }
        return;
    }
    if (msg.type === "assistant") {
        assistantBuf = assistantBuf ? `${assistantBuf} ${msg.text}` : msg.text || "";
        assistantEl.textContent = assistantBuf;
        return;
    }
    if (msg.type === "turn_end") {
        awaitingFirstAudio = false;
    }
}
async function setupLoopbackAec(micStream) {
    if (!playCtx)
        throw new Error("no playCtx");
    const a = new RTCPeerConnection();
    const b = new RTCPeerConnection();
    pcMic = a;
    pcTts = b;
    a.onicecandidate = (e) => {
        if (e.candidate)
            b.addIceCandidate(e.candidate).catch(() => { });
    };
    b.onicecandidate = (e) => {
        if (e.candidate)
            a.addIceCandidate(e.candidate).catch(() => { });
    };
    a.onconnectionstatechange = () => addRow("stt", `aec pcMic ${a.connectionState}`);
    a.addTrack(micStream.getAudioTracks()[0], micStream);
    ttsDest = playCtx.createMediaStreamDestination();
    b.addTrack(ttsDest.stream.getAudioTracks()[0], ttsDest.stream);
    player = new PcmPlayer(playCtx, ttsDest);
    const el = document.createElement("audio");
    el.autoplay = true;
    renderEl = el;
    a.ontrack = (ev) => {
        el.srcObject = ev.streams[0];
        el.play().catch(() => addRow("stt", "render autoplay blocked"));
    };
    const processed = new Promise((resolve, reject) => {
        b.ontrack = (ev) => resolve(ev.streams[0]);
        setTimeout(() => reject(new Error("no processed mic track")), 5000);
    });
    const offer = await a.createOffer();
    await a.setLocalDescription(offer);
    await b.setRemoteDescription(offer);
    const answer = await b.createAnswer();
    await b.setLocalDescription(answer);
    await a.setRemoteDescription(answer);
    return processed;
}
async function startMic() {
    connectWs();
    captureCtx = new AudioContext();
    playCtx = new AudioContext({ sampleRate: 24000 });
    await captureCtx.resume();
    await playCtx.resume();
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            channelCount: 1,
        },
    });
    let processed;
    if (aec) {
        try {
            processed = await setupLoopbackAec(mediaStream);
            addRow("stt", "aec loopback ok");
        }
        catch (err) {
            addRow("stt", `aec fallback: ${err}`);
            aec = false;
            player = new PcmPlayer(playCtx);
            processed = mediaStream;
        }
    }
    else {
        player = new PcmPlayer(playCtx);
        processed = mediaStream;
    }
    const src = captureCtx.createMediaStreamSource(processed);
    processor = captureCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
        if (!capturing || !ws || ws.readyState !== WebSocket.OPEN || !captureCtx)
            return;
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
    if (ws?.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: "mic", on: true }));
    else {
        const sock = ws;
        sock?.addEventListener("open", () => sock.send(JSON.stringify({ type: "mic", on: true })), { once: true });
    }
    addRow("stt", `mic ${captureCtx.sampleRate} Hz → 16000, play ${playCtx.sampleRate} Hz`);
}
function stopMic() {
    capturing = false;
    processor?.disconnect();
    pcMic?.close();
    pcTts?.close();
    pcMic = undefined;
    pcTts = undefined;
    ttsDest = undefined;
    if (renderEl) {
        renderEl.srcObject = null;
        renderEl = undefined;
    }
    mediaStream?.getTracks().forEach((t) => t.stop());
    captureCtx?.close();
    processor = undefined;
    mediaStream = undefined;
    talkBtn.disabled = false;
    stopBtn.disabled = true;
    if (ws?.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: "mic", on: false }));
    if (uiState !== "speaking" && uiState !== "thinking")
        setState("idle");
    addRow("stt", "mic stopped");
}
talkBtn.addEventListener("click", () => {
    startMic().catch((err) => {
        addRow("stt", String(err));
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
    if (!text)
        return;
    tSpeak = performance.now();
    tTranscript = 0;
    awaitingFirstAudio = true;
    player?.stop();
    setState("speaking");
    const sock = ws;
    if (!sock)
        return;
    const send = () => sock.send(JSON.stringify({ type: "speak", id: crypto.randomUUID(), text }));
    if (sock.readyState === WebSocket.OPEN)
        send();
    else
        sock.addEventListener("open", send, { once: true });
});
cancelBtn.addEventListener("click", () => {
    player?.stop();
    awaitingFirstAudio = false;
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "cancel", id: "" }));
    }
    if (capturing)
        setState("listening");
    else
        setState("idle");
});
connectWs();
