import { useCallback, useEffect, useRef, useState } from "react";
import { PcmPlayer, SentAudioRecorder, loadWorklets, s16ToFloat, startLevelLoop } from "./audio";

type Channel = "stt" | "llm_conv" | "llm_agentic" | "minions" | "tts";

const COLUMNS: Channel[] = ["stt", "llm_conv", "llm_agentic", "minions", "tts"];
const MAX_ROWS = 300;

const GATE_OFF_DB = -66;
const GATE_MAX_DB = -3;

function loadGateThreshold(): number {
  const raw = localStorage.getItem("voya.gate");
  if (raw === null || raw === "") return GATE_OFF_DB;
  const v = Number(raw);
  if (!Number.isFinite(v)) return GATE_OFF_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(v)));
}

type LogRow = { id: number; time: string; channel: Channel; text: string };

const PILL_STYLES: Record<string, string> = {
  listening: "text-accent border-[#245c48]",
  speaking: "text-accent border-[#245c48]",
  user_speaking: "text-warn border-[#5c4a20]",
  speech: "text-warn border-[#5c4a20]",
  transcribing: "text-warn border-[#5c4a20]",
  thinking: "text-blue-300 border-[#1e3a5f]",
  error: "text-err border-[#5c2020]",
};

const CHANNEL_COLORS: Record<Channel, string> = {
  stt: "text-warn",
  llm_conv: "text-blue-300",
  llm_agentic: "text-violet-300",
  minions: "text-accent",
  tts: "text-pink-300",
};

const BTN = "rounded-lg px-3.5 py-2 font-semibold cursor-pointer disabled:opacity-45";
const BTN_ACCENT = `${BTN} bg-accent text-[#042014]`;
const BTN_GHOST = `${BTN} bg-transparent text-ink border border-line`;
const PANEL = "bg-panel border border-line rounded-lg";
const OUT = `${PANEL} min-h-12 p-3 mb-3`;

export default function App() {
  const [rows, setRows] = useState<LogRow[]>([]);
  const [state, setState] = useState("idle");
  const [question, setQuestion] = useState("—");
  const [transcript, setTranscript] = useState("—");
  const [assistant, setAssistant] = useState("—");
  const [latStt, setLatStt] = useState("—");
  const [latLlm, setLatLlm] = useState("—");
  const [latTts, setLatTts] = useState("—");
  const [echo, setEcho] = useState(() => localStorage.getItem("voya.echo") !== "0");
  const [synthText, setSynthText] = useState(
    () => localStorage.getItem("voya.synthText") ?? "Hello. This is a Kokoro test.",
  );
  const [micOn, setMicOn] = useState(false);
  const [muted, setMuted] = useState(false);
  const [noiseGateDb, setNoiseGateDb] = useState<number>(loadGateThreshold);
  const [replayUrl, setReplayUrl] = useState("");

  const wsRef = useRef<WebSocket | undefined>(undefined);
  const connectRef = useRef<() => WebSocket | undefined>(() => undefined);
  const captureCtxRef = useRef<AudioContext | undefined>(undefined);
  const playCtxRef = useRef<AudioContext | undefined>(undefined);
  const mediaStreamRef = useRef<MediaStream | undefined>(undefined);
  const captureNodeRef = useRef<AudioWorkletNode | undefined>(undefined);
  const micAnalyserRef = useRef<AnalyserNode | undefined>(undefined);
  const recorderRef = useRef<SentAudioRecorder | undefined>(undefined);
  const meterStopRef = useRef<() => void>(() => {});
  const meterElRef = useRef<HTMLSpanElement | null>(null);
  const playerRef = useRef<PcmPlayer | undefined>(undefined);
  const aecRef = useRef(true);
  const pcMicRef = useRef<RTCPeerConnection | undefined>(undefined);
  const pcTtsRef = useRef<RTCPeerConnection | undefined>(undefined);
  const ttsDestRef = useRef<MediaStreamAudioDestinationNode | undefined>(undefined);
  const renderElRef = useRef<HTMLAudioElement | undefined>(undefined);
  const capturingRef = useRef(false);
  const awaitingFirstAudioRef = useRef(false);
  const tSpeechEndRef = useRef(0);
  const tSpeakRef = useRef(0);
  const tTranscriptRef = useRef(0);
  const assistantBufRef = useRef("");
  const echoRef = useRef(echo);
  const stateRef = useRef(state);
  const idRef = useRef(0);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    echoRef.current = echo;
  }, [echo]);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const addRow = useCallback((channel: Channel, text: string, ts?: number) => {
    const time = new Date(ts ?? Date.now()).toISOString().slice(11, 23);
    setRows((prev) => {
      const next = [...prev, { id: idRef.current++, time, channel, text }];
      return next.length > MAX_ROWS ? next.slice(next.length - MAX_ROWS) : next;
    });
  }, []);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [rows]);

  const onJson = useCallback(
    (msg: {
      type?: string;
      text?: string;
      ts?: number;
      stt_ms?: number;
      state?: string;
      pending_question?: string | null;
      orchestrator?: string;
      agentic?: string;
      channel?: string;
    }) => {
      if (msg.type === "log") {
        if (
          msg.channel === "stt" ||
          msg.channel === "llm_conv" ||
          msg.channel === "llm_agentic" ||
          msg.channel === "minions" ||
          msg.channel === "tts"
        ) {
          addRow(msg.channel as Channel, msg.text || "", msg.ts);
        }
        return;
      }
      if (msg.type === "ready") {
        addRow("llm_conv", `ready orch=${msg.orchestrator || ""} agentic=${msg.agentic || ""}`);
        return;
      }
      if (msg.type === "state") {
        if (msg.state) setState(msg.state);
        setQuestion(msg.pending_question || "—");
        return;
      }
      if (msg.type === "question_asked") {
        setQuestion(msg.text || "—");
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
        playerRef.current?.stop();
        awaitingFirstAudioRef.current = false;
        recorderRef.current?.start();
        return;
      }
      if (msg.type === "speech_end") {
        tSpeechEndRef.current = performance.now();
        const rec = recorderRef.current?.stop();
        if (rec) {
          setReplayUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return URL.createObjectURL(rec.audio);
          });
        }
        return;
      }
      if (msg.type === "transcript") {
        const elapsed = tSpeechEndRef.current ? Math.round(performance.now() - tSpeechEndRef.current) : msg.stt_ms;
        setLatStt(`${elapsed} ms`);
        setTranscript(msg.text || "(empty)");
        if (msg.text?.trim()) {
          tTranscriptRef.current = performance.now();
          awaitingFirstAudioRef.current = true;
          assistantBufRef.current = "";
          setAssistant("—");
          tSpeakRef.current = echoRef.current ? tTranscriptRef.current : 0;
        }
        return;
      }
      if (msg.type === "assistant") {
        assistantBufRef.current = assistantBufRef.current
          ? `${assistantBufRef.current} ${msg.text}`
          : msg.text || "";
        setAssistant(assistantBufRef.current);
        return;
      }
      if (msg.type === "turn_end") {
        awaitingFirstAudioRef.current = false;
      }
    },
    [addRow],
  );

  useEffect(() => {
    const connect = () => {
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return ws;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/ws`);
      sock.binaryType = "arraybuffer";
      wsRef.current = sock;
      sock.onopen = () => {
        addRow("stt", "ws open");
        sock.send(JSON.stringify({ type: "echo", enabled: echoRef.current }));
      };
      sock.onclose = () => addRow("stt", "ws close");
      sock.onerror = () => addRow("stt", "ws error");
      sock.onmessage = (ev) => {
        if (typeof ev.data !== "string") {
          if (awaitingFirstAudioRef.current) {
            awaitingFirstAudioRef.current = false;
            if (tSpeakRef.current) {
              const elapsed = Math.round(performance.now() - tSpeakRef.current);
              setLatTts(`${elapsed} ms`);
              addRow("tts", `tts_first_chunk_ms=${elapsed}`);
            }
            if (tTranscriptRef.current) {
              const elapsed = Math.round(performance.now() - tTranscriptRef.current);
              setLatLlm(`${elapsed} ms`);
              addRow("llm_conv", `turn_first_chunk_ms=${elapsed}`);
            }
          }
          playerRef.current?.push(s16ToFloat(ev.data as ArrayBuffer));
          return;
        }
        let msg: {
          type?: string;
          text?: string;
          ts?: number;
          stt_ms?: number;
          state?: string;
          pending_question?: string | null;
          orchestrator?: string;
          agentic?: string;
          channel?: string;
        };
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        onJson(msg);
      };
      return sock;
    };
    connectRef.current = connect;
    connect();
  }, [addRow, onJson]);

  async function setupLoopbackAec(micStream: MediaStream): Promise<MediaStream> {
    const playCtx = playCtxRef.current;
    if (!playCtx) throw new Error("no playCtx");
    const a = new RTCPeerConnection();
    const b = new RTCPeerConnection();
    pcMicRef.current = a;
    pcTtsRef.current = b;
    a.onicecandidate = (e) => {
      if (e.candidate) b.addIceCandidate(e.candidate).catch(() => {});
    };
    b.onicecandidate = (e) => {
      if (e.candidate) a.addIceCandidate(e.candidate).catch(() => {});
    };
    a.onconnectionstatechange = () => addRow("stt", `aec pcMic ${a.connectionState}`);
    a.addTrack(micStream.getAudioTracks()[0], micStream);
    ttsDestRef.current = playCtx.createMediaStreamDestination();
    b.addTrack(ttsDestRef.current.stream.getAudioTracks()[0], ttsDestRef.current.stream);
    await loadWorklets(playCtx);
    playerRef.current = new PcmPlayer(playCtx, ttsDestRef.current);
    const el = document.createElement("audio");
    el.autoplay = true;
    renderElRef.current = el;
    a.ontrack = (ev) => {
      el.srcObject = ev.streams[0];
      el.play().catch(() => addRow("stt", "render autoplay blocked"));
    };
    const processed = new Promise<MediaStream>((resolve, reject) => {
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
    connectRef.current();
    captureCtxRef.current = new AudioContext();
    playCtxRef.current = new AudioContext({ sampleRate: 24000 });
    await captureCtxRef.current.resume();
    await playCtxRef.current.resume();
    mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    let processed: MediaStream;
    if (aecRef.current) {
      try {
        processed = await setupLoopbackAec(mediaStreamRef.current);
        addRow("stt", "aec loopback ok");
      } catch (err) {
        addRow("stt", `aec fallback: ${err}`);
        aecRef.current = false;
        await loadWorklets(playCtxRef.current);
        playerRef.current = new PcmPlayer(playCtxRef.current);
        processed = mediaStreamRef.current;
      }
    } else {
      await loadWorklets(playCtxRef.current);
      playerRef.current = new PcmPlayer(playCtxRef.current);
      processed = mediaStreamRef.current;
    }
    const captureCtx = captureCtxRef.current;
    await loadWorklets(captureCtx);
    const src = captureCtx.createMediaStreamSource(processed);
    micAnalyserRef.current = captureCtx.createAnalyser();
    micAnalyserRef.current.fftSize = 256;
    const node = new AudioWorkletNode(captureCtx, "mic-capture");
    const mute = captureCtx.createGain();
    mute.gain.value = 0;
    node.port.onmessage = (e) => {
      const d = e.data;
      if (!d) return;
      if (d.kind === "pcm") {
        recorderRef.current?.append(d.buffer);
        if (capturingRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(d.buffer);
        }
      }
    };
    captureNodeRef.current = node;
    src.connect(micAnalyserRef.current);
    micAnalyserRef.current.connect(node);
    node.connect(mute);
    mute.connect(captureCtx.destination);
    node.port.postMessage({ kind: "gate", enabled: noiseGateDb > GATE_OFF_DB, thresholdDb: noiseGateDb });
    node.port.postMessage({ kind: "enable", value: true });
    recorderRef.current = new SentAudioRecorder(16000);
    capturingRef.current = true;
    setMicOn(true);
    setState("listening");
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "mic", on: true }));
    } else {
      wsRef.current?.addEventListener(
        "open",
        () => wsRef.current?.send(JSON.stringify({ type: "mic", on: true })),
        { once: true },
      );
    }
    addRow("stt", `mic ${captureCtx.sampleRate} Hz → 16000 worklet, play ${playCtxRef.current.sampleRate} Hz`);
    meterStopRef.current = startLevelLoop(
      () => (stateRef.current === "speaking" ? playerRef.current?.analyser : micAnalyserRef.current),
      meterElRef.current,
    );
  }

  function stopMic() {
    capturingRef.current = false;
    meterStopRef.current();
    captureNodeRef.current?.port.postMessage({ kind: "enable", value: false });
    captureNodeRef.current?.disconnect();
    captureNodeRef.current = undefined;
    micAnalyserRef.current?.disconnect();
    micAnalyserRef.current = undefined;
    pcMicRef.current?.close();
    pcTtsRef.current?.close();
    pcMicRef.current = undefined;
    pcTtsRef.current = undefined;
    ttsDestRef.current = undefined;
    if (renderElRef.current) {
      renderElRef.current.srcObject = null;
      renderElRef.current = undefined;
    }
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    captureCtxRef.current?.close();
    mediaStreamRef.current = undefined;
    setMicOn(false);
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: "mic", on: false }));
    if (state !== "speaking" && state !== "thinking") setState("idle");
    addRow("stt", "mic stopped");
  }

  async function synth() {
    connectRef.current();
    if (!playCtxRef.current) {
      playCtxRef.current = new AudioContext({ sampleRate: 24000 });
      await playCtxRef.current.resume();
      await loadWorklets(playCtxRef.current);
      playerRef.current = new PcmPlayer(playCtxRef.current);
    }
    const text = synthText.trim();
    if (!text) return;
    tSpeakRef.current = performance.now();
    tTranscriptRef.current = 0;
    awaitingFirstAudioRef.current = true;
    playerRef.current?.stop();
    setState("speaking");
    const sock = wsRef.current;
    if (!sock) return;
    const send = () => sock.send(JSON.stringify({ type: "speak", id: crypto.randomUUID(), text }));
    if (sock.readyState === WebSocket.OPEN) send();
    else sock.addEventListener("open", send, { once: true });
  }

  function cancel() {
    playerRef.current?.stop();
    awaitingFirstAudioRef.current = false;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "cancel", id: "" }));
    }
    if (capturingRef.current) setState("listening");
    else setState("idle");
  }

  function onEchoChange(checked: boolean) {
    setEcho(checked);
    localStorage.setItem("voya.echo", checked ? "1" : "0");
    connectRef.current();
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "echo", enabled: checked }));
    }
  }

  function toggleMute() {
    const next = !muted;
    setMuted(next);
    for (const t of mediaStreamRef.current?.getTracks() ?? []) t.enabled = !next;
  }

  function onGateChange(value: number) {
    setNoiseGateDb(value);
    localStorage.setItem("voya.gate", String(value));
    captureNodeRef.current?.port.postMessage({
      kind: "gate",
      enabled: value > GATE_OFF_DB,
      thresholdDb: value,
    });
  }

  const indicatorActive = ["listening", "user_speaking", "speech", "transcribing", "speaking", "thinking"].includes(state);

  return (
    <main className="max-w-[1160px] mx-auto px-4 pt-6 pb-12">
      <h1 className="text-lg font-semibold mb-1">Voya — voice + LLM</h1>
      <p className="text-muted mb-5">
        Chrome + headphones. Mic always on (barge-in). Whisper → orchestrator → Kokoro.
      </p>
      <div className="flex gap-2.5 flex-wrap items-center mb-3">
        <button
          type="button"
          disabled={micOn}
          onClick={() => void startMic().catch((err: unknown) => {
            addRow("stt", String(err));
            setState("error");
          })}
          className={BTN_ACCENT}
        >
          Speak
        </button>
        <button type="button" disabled={!micOn} onClick={stopMic} className={BTN_GHOST}>
          Stop
        </button>
        <button type="button" disabled={!micOn} onClick={toggleMute} className={BTN_GHOST}>
          {muted ? "Unmute" : "Mute"}
        </button>
        <span
          className={`inline-flex items-center gap-1.5 border rounded-full px-2.5 py-1 text-[11px] uppercase tracking-wider ${
            PILL_STYLES[state] ?? "text-muted border-line"
          }`}
        >
          {state}
        </span>
        <span
          ref={meterElRef}
          className={`indicator ${indicatorActive ? "" : "hidden"} ${
            state === "thinking" ? "text-blue-300" : "text-accent"
          }`}
        >
          {state === "thinking" ? (
            <span className="ind-dots">
              <span />
              <span />
              <span />
            </span>
          ) : (
            <span className="ind-bars">
              <span />
              <span />
              <span />
              <span />
              <span />
            </span>
          )}
        </span>
        <label className="text-muted flex gap-1.5 items-center">
          <span>gate</span>
          <input
            type="range"
            min={GATE_OFF_DB}
            max={GATE_MAX_DB}
            step={1}
            value={noiseGateDb}
            onChange={(e) => onGateChange(Number(e.target.value))}
            className="w-24"
          />
          <span className="tabular-nums">{noiseGateDb <= GATE_OFF_DB ? "off" : `${noiseGateDb} dB`}</span>
        </label>
        <label className="text-muted flex gap-1.5 items-center">
          <input type="checkbox" checked={echo} onChange={(e) => onEchoChange(e.target.checked)} />
          echo TTS
        </label>
      </div>
      <div className="mb-3">
        <textarea
          rows={2}
          value={synthText}
          onChange={(e) => {
            setSynthText(e.target.value);
            localStorage.setItem("voya.synthText", e.target.value);
          }}
          placeholder="Synthesize this…"
          className="w-full bg-panel text-ink border border-line rounded-lg p-2.5 font-inherit resize-y"
        />
      </div>
      <div className="flex gap-2.5 mb-3">
        <button type="button" onClick={() => void synth()} className={BTN_GHOST}>
          Synthesize
        </button>
        <button type="button" onClick={cancel} className={BTN_GHOST}>
          Cancelar
        </button>
      </div>
      <div className="grid grid-cols-3 gap-2 my-4">
        <div className={PANEL}>
          <span className="block text-muted text-[11px] px-3 pt-2.5">speech_end → transcript</span>
          <strong className="block px-3 pb-2.5 text-xl tabular-nums">{latStt}</strong>
        </div>
        <div className={PANEL}>
          <span className="block text-muted text-[11px] px-3 pt-2.5">transcript → first chunk</span>
          <strong className="block px-3 pb-2.5 text-xl tabular-nums">{latLlm}</strong>
        </div>
        <div className={PANEL}>
          <span className="block text-muted text-[11px] px-3 pt-2.5">speak → primer chunk</span>
          <strong className="block px-3 pb-2.5 text-xl tabular-nums">{latTts}</strong>
        </div>
      </div>
      <div className={OUT}>{transcript}</div>
      {replayUrl && (
        <div className={`${OUT} flex flex-col gap-1.5`}>
          <span className="text-muted text-[11px]">Audio sent to the model</span>
          <audio controls preload="metadata" src={replayUrl} className="w-full h-8" />
        </div>
      )}
      <div className={`${OUT} text-warn`} title="pending question">
        {question}
      </div>
      <div className={OUT}>{assistant}</div>
      <div ref={logRef} className="h-90 overflow-auto bg-panel border border-line rounded-lg font-mono text-xs text-muted">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="sticky top-0 z-10 text-left px-2 py-1 border-b border-line bg-panel text-ink text-[11px] font-semibold uppercase tracking-wider">
                datetime
              </th>
              {["STT", "LLM Conversacional", "LLM Agentico", "Minions", "TTS"].map((h) => (
                <th
                  key={h}
                  className="sticky top-0 z-10 text-left px-2 py-1 border-b border-line bg-panel text-ink text-[11px] font-semibold uppercase tracking-wider"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-line align-top">
                <td className="px-2 py-0.5 whitespace-nowrap tabular-nums">{r.time}</td>
                {COLUMNS.map((col) => (
                  <td
                    key={col}
                    title={col === r.channel ? r.text : undefined}
                    className={`px-2 py-0.5 ${col === r.channel ? `max-w-[280px] wrap-anywhere ${CHANNEL_COLORS[col]}` : ""}`}
                  >
                    {col === r.channel ? r.text : ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
