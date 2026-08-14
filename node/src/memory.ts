import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { WINDOW_CHARS } from "./config.js";

export type Role = "system" | "user" | "assistant" | "tool";

export type ToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

export type Msg = {
  role: Role;
  content: string | null;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
};

function charsOf(messages: Msg[]) {
  let n = 0;
  for (const m of messages) n += (m.content || "").length + JSON.stringify(m.tool_calls || "").length;
  return n;
}

export class Memory {
  messages: Msg[];
  pendingInjects: string[] = [];
  llmChars = 0;
  readonly logPath: string;

  constructor(logDir: string, sessionId: string, system: string) {
    fs.mkdirSync(logDir, { recursive: true });
    this.logPath = path.join(logDir, `${sessionId}.jsonl`);
    this.messages = [{ role: "system", content: system }];
  }

  log(event: Record<string, unknown>) {
    try {
      fs.appendFileSync(this.logPath, JSON.stringify({ t: Date.now(), ...event }) + "\n");
    } catch {
      /* disk full / perms — don't kill the session */
    }
  }

  addChars(n: number) {
    this.llmChars += n;
  }

  inject(text: string) {
    const t = String(text || "").trim();
    if (t) this.pendingInjects.push(t);
  }

  flushInjects() {
    for (const t of this.pendingInjects) {
      this.messages.push({ role: "system", content: t });
      this.log({ type: "inject", text: t });
    }
    this.pendingInjects = [];
  }

  push(msg: Msg) {
    this.messages.push(msg);
    this.log({ type: "msg", role: msg.role, content: msg.content, tool_calls: msg.tool_calls });
  }

  window(pendingQuestion?: string): Msg[] {
    this.flushInjects();
    const extra: Msg[] = [];
    if (pendingQuestion) {
      extra.push({
        role: "system",
        content: `Pregunta pendiente (única activa, FIFO). El usuario puede estar respondiéndola ahora:\n${pendingQuestion}`,
      });
    }
    return extra.length ? [...this.messages, ...extra] : this.messages.slice();
  }

  overBudget(limit = WINDOW_CHARS) {
    return charsOf(this.messages) > limit;
  }

  applySummary(summary: string, keep = 4) {
    const sys = this.messages[0];
    const recent = this.messages.slice(-keep);
    this.messages = [
      sys,
      { role: "system", content: `Resumen del contexto anterior:\n${summary}` },
      ...recent.filter((m) => m !== sys),
    ];
    this.log({ type: "compact", summary, kept: recent.length });
  }
}

{
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "voya-mem-"));
  try {
    const m = new Memory(dir, "t", "sys");
    m.push({ role: "user", content: "hola" });
    m.inject("previo");
    const w = m.window("¿ok?");
    if (!w.some((x) => x.content?.includes("previo"))) throw new Error("memory inject");
    if (!w.some((x) => x.content?.includes("¿ok?"))) throw new Error("memory pending");
    if (!fs.existsSync(m.logPath)) throw new Error("memory log");
    m.applySummary("sum", 1);
    if (!m.messages.some((x) => x.content?.includes("sum"))) throw new Error("memory compact");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}
