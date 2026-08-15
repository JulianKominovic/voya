import micCaptureWorklet from "./worklets/mic-capture.worklet";
import playbackWorklet from "./worklets/audio-playback.worklet";

const workletPromises = new WeakMap<AudioContext, Promise<void>>();

export function loadWorklets(ctx: AudioContext): Promise<void> {
  let p = workletPromises.get(ctx);
  if (!p) {
    p = (async () => {
      await loadWorklet(ctx, "mic-capture", micCaptureWorklet);
      await loadWorklet(ctx, "audio-playback", playbackWorklet);
    })();
    workletPromises.set(ctx, p);
  }
  return p;
}

async function loadWorklet(ctx: AudioContext, name: string, source: string) {
  const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function s16ToFloat(buf: ArrayBuffer) {
  const i16 = new Int16Array(buf);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}

export class PcmPlayer {
  node: AudioWorkletNode;
  analyser: AnalyserNode;

  constructor(ctx: AudioContext, out?: AudioNode) {
    this.node = new AudioWorkletNode(ctx, "audio-playback");
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.node.connect(this.analyser);
    this.analyser.connect(out ?? ctx.destination);
    this.node.port.postMessage({ kind: "config", inputRate: 24000 });
  }

  push(float32: Float32Array) {
    if (float32.length) {
      this.node.port.postMessage({ kind: "audio", samples: float32 }, [float32.buffer]);
    }
  }

  stop() {
    this.node.port.postMessage({ kind: "clear" });
  }
}

export function pcm16ToWavBlob(pcm: Uint8Array, sampleRate: number): Blob {
  const dataLength = pcm.byteLength - (pcm.byteLength % 2);
  const wav = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wav);
  const ascii = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, dataLength, true);
  new Uint8Array(wav, 44).set(pcm.subarray(0, dataLength));
  return new Blob([wav], { type: "audio/wav" });
}

type Chunk = { start: number; end: number; bytes: Uint8Array };

export class SentAudioRecorder {
  private chunks: Chunk[] = [];
  private sentSamples = 0;
  private activeStart = -1;
  private maxSamples: number;

  constructor(
    private sampleRate: number,
    maxMs = 120_000,
  ) {
    this.maxSamples = (sampleRate * maxMs) / 1000;
  }

  append(buffer: ArrayBuffer) {
    const even = buffer.byteLength - (buffer.byteLength % 2);
    if (even <= 0) return;
    const bytes = new Uint8Array(buffer.slice(0, even));
    const start = this.sentSamples;
    this.chunks.push({ start, end: start + even / 2, bytes });
    this.sentSamples = start + even / 2;
    const floor = Math.max(0, this.sentSamples - this.maxSamples);
    while (this.chunks.length && this.chunks[0].end <= floor) this.chunks.shift();
  }

  start() {
    this.activeStart = this.sentSamples;
  }

  stop(): { audio: Blob; durationMs: number } | null {
    if (this.activeStart < 0) return null;
    const startSample = Math.max(this.activeStart, this.chunks[0]?.start ?? 0);
    const endSample = this.sentSamples;
    this.activeStart = -1;
    if (endSample <= startSample) return null;
    let length = 0;
    const parts: Uint8Array[] = [];
    for (const c of this.chunks) {
      const lo = Math.max(startSample, c.start);
      const hi = Math.min(endSample, c.end);
      if (hi <= lo) continue;
      const from = (lo - c.start) * 2;
      const to = (hi - c.start) * 2;
      const part = c.bytes.slice(from, to);
      parts.push(part);
      length += part.byteLength;
    }
    if (!length) return null;
    const pcm = new Uint8Array(length);
    let offset = 0;
    for (const p of parts) {
      pcm.set(p, offset);
      offset += p.byteLength;
    }
    return {
      audio: pcm16ToWavBlob(pcm, this.sampleRate),
      durationMs: (length / 2 / this.sampleRate) * 1000,
    };
  }

  reset() {
    this.chunks = [];
    this.sentSamples = 0;
    this.activeStart = -1;
  }
}

const LEVEL_EDGES = [2, 5, 9, 16, 28, 52];

export function startLevelLoop(
  getAnalyser: () => AnalyserNode | undefined,
  el: HTMLElement | null,
): () => void {
  if (!el) return () => {};
  const bands = new Float32Array(5);
  const buf = new Uint8Array(128);
  let raf = 0;
  const tick = () => {
    raf = requestAnimationFrame(tick);
    const analyser = getAnalyser();
    if (!analyser) return;
    analyser.getByteFrequencyData(buf);
    for (let b = 0; b < 5; b++) {
      const lo = LEVEL_EDGES[b];
      const hi = LEVEL_EDGES[b + 1];
      let sum = 0;
      let n = 0;
      for (let i = lo; i < hi && i < buf.length; i++) {
        sum += buf[i];
        n += 1;
      }
      const target = n > 0 ? sum / (n * 255) : 0;
      const prev = bands[b];
      const k = target > prev ? 0.6 : 0.18;
      const next = prev + (target - prev) * k;
      bands[b] = next;
      el.style.setProperty(`--bar${b}`, next.toFixed(3));
    }
  };
  tick();
  return () => {
    cancelAnimationFrame(raf);
    for (let i = 0; i < 5; i++) el.style.removeProperty(`--bar${i}`);
  };
}
