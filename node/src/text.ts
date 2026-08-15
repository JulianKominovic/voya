import { TTS_MAX_CHARS } from "./config.js";

export function takeSentences(buf: string, flushAll: boolean) {
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

export function forTts(text: string, system: string) {
  const raw = String(text || "").replace(/<think>[\s\S]*?<\/think>/gi, " ");
  const sys = system.toLowerCase();
  const { sentences } = takeSentences(raw, true);
  return sentences.filter((s) => {
    if (/^instructions\s*:/i.test(s)) return false;
    if (/^(debo |el usuario |cuando el usuario |parece que )/i.test(s)) return false;
    const n = s.toLowerCase().replace(/[.!?…]+$/g, "").trim();
    if (!n) return false;
    if (sys.includes(n)) return false;
    return true;
  });
}

export function sanitizeTts(text: string, cap = TTS_MAX_CHARS) {
  let t = String(text || "").trim();
  if (!t) return "";
  t = t.replace(/<think>[\s\S]*?<\/think>/gi, " ");
  t = t.replace(/```[\s\S]*?```/g, " ");
  t = t.replace(/`([^`]+)`/g, "$1");
  t = t.replace(/!\[[^\]]*]\([^)]*\)/g, " ");
  t = t.replace(/\[([^\]]+)]\([^)]*\)/g, "$1");
  t = t.replace(/^#{1,6}\s+/gm, "");
  t = t.replace(/^\s*[-*+]\s+/gm, "");
  t = t.replace(/^\s*\d+\.\s+/gm, "");
  t = t.replace(/\*\*([^*]+)\*\*/g, "$1");
  t = t.replace(/\*([^*]+)\*/g, "$1");
  t = t.replace(/\s+/g, " ").trim();
  if (t.length > cap) t = t.slice(0, cap).trim();
  return t;
}

export function truncate(text: string, cap: number) {
  const t = String(text || "").trim();
  if (t.length <= cap) return t;
  return t.slice(0, cap).trim();
}

export function errMsg(err: unknown) {
  if (!(err instanceof Error)) return String(err);
  const o = err as Error & { statusCode?: unknown; body?: unknown };
  const status = typeof o.statusCode === "number" ? o.statusCode : 0;
  const body = typeof o.body === "string" ? o.body : "";
  if (!status && !body) return err.message;
  const bits: string[] = [];
  if (status) bits.push(String(status));
  if (body) {
    try {
      const e = JSON.parse(body)?.error as
        | { message?: unknown; metadata?: { raw?: unknown; provider_name?: unknown } }
        | undefined;
      const meta = e?.metadata ?? {};
      if (typeof meta.provider_name === "string" && meta.provider_name) bits.push(meta.provider_name);
      const raw = String(meta.raw || e?.message || "").trim();
      if (raw && raw !== err.message) bits.push(raw);
    } catch {
      bits.push(body);
    }
  }
  const extra = bits.join(" ").replace(/\s+/g, " ").trim().slice(0, 240);
  if (!extra || extra === err.message) return err.message;
  return `${err.message}: ${extra}`;
}

export function aborted(err: unknown) {
  const name =
    err && typeof err === "object" && "name" in err ? String((err as { name: unknown }).name) : "";
  return name === "AbortError" || name === "RequestAbortedError";
}

export function now() {
  return performance.now();
}

export function ms(t0: number) {
  return Math.round(now() - t0);
}

export function sleep(msWait: number) {
  return new Promise((r) => setTimeout(r, msWait));
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
  const s = sanitizeTts("**Hola** `mundo`\n- item\n\nresto", 20);
  if (s.includes("*") || s.includes("`") || s.length > 20) throw new Error("sanitizeTts");
  if (truncate("abcdef", 3) !== "abc") throw new Error("truncate");
  const sys = "Sos el lado hablado de Voya. No inventes el estado de la cola.";
  const leaked = forTts(
    "No inventes el estado de la cola. Debo reconocer el error. Tenés razón, me equivoqué.",
    sys,
  );
  if (leaked.join("|") !== "Tenés razón, me equivoqué.") throw new Error("forTts leak");
  if (forTts('Instructions: action es "cancel". ¿Todo bien?', sys).join("|") !== "¿Todo bien?") {
    throw new Error("forTts instructions");
  }
  const orErr = Object.assign(new Error("Provider returned error"), {
    statusCode: 429,
    body: JSON.stringify({
      error: {
        message: "Provider returned error",
        metadata: { provider_name: "Novita", raw: "rate-limited upstream" },
      },
    }),
  });
  if (errMsg(orErr) !== "Provider returned error: 429 Novita rate-limited upstream") {
    throw new Error("errMsg");
  }
}
