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

export function sanitizeTts(text: string, cap = TTS_MAX_CHARS) {
  let t = String(text || "").trim();
  if (!t) return "";
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
  return err instanceof Error ? err.message : String(err);
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
}
