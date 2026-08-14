import type { OpenRouter } from "@openrouter/sdk";
import { randomUUID } from "node:crypto";
import {
  AGENTIC_MAX_ROUNDS,
  AGENTIC_MODEL,
  ANSWER_MAX_CHARS,
  ASK_TIMEOUT_MS,
  CHAIN_MAX_DEPTH,
  ORCHESTRATOR_MODEL,
  SESSION_MAX_LLM_CHARS,
  WINDOW_CHARS,
} from "./config.js";
import type { Msg, ToolCall } from "./memory.js";
import type { Session } from "./session.js";
import { aborted, errMsg, now, takeSentences, truncate } from "./text.js";

export const ORCH_SYSTEM = `Sos el lado hablado de Voya. Español oral, dos a cuatro oraciones. Sin markdown, listas ni código.
Tools:
- ask: trabajo de fondo (pensar, descomponer, minions). Si va a tardar, decí algo corto antes.
- pending_question: si el contexto muestra una pregunta pendiente, registrá la respuesta del usuario o cancelala si pide olvidarla o cambió de tema de forma definitiva. Si cancelás por cambio de tema, primero decilo en voz alta.
No inventes el estado de la cola. El usuario puede no responder: si hay timeout, no insistas.`;

const AGENTIC_SYSTEM = `Sos el agente agentic de Voya. No hablás con el usuario: devolvé el resultado con el tool answer (corto, sin markdown).
Tools: create_minion, list_minions, kill_minion, talk_to, ask_user, answer.
ask_user encola una pregunta FIFO; el usuario puede tardar o nunca responder. Si timeout o cancel, seguí con tu mejor juicio o abortá con answer.
Los minions no tienen tools de archivos: les das una tarea en texto y ellos piensan o preguntan con ask.
answer es obligatorio para devolverle al orchestrator.`;

const MINION_SYSTEM = `Trabajás una tarea en segundo plano. Un solo tool: ask, para pedirle input al agente. Toda pregunta lleva contexto completo: qué estás haciendo, por qué preguntás, dónde trabás.
Cuando termines, escribí el resultado final en texto, sin tool.`;

const ORCH_TOOLS = [
  fnTool("ask", "Consulta al agente agentic (trabajo de fondo).", {
    question: { type: "string" },
  }, ["question"]),
  fnTool("pending_question", "Resolver la pregunta pendiente: answer o cancel.", {
    action: { type: "string", enum: ["answer", "cancel"] },
    text: { type: "string" },
  }, ["action"]),
];

const AGENTIC_TOOLS = [
  fnTool("create_minion", "Crea un minion de fondo con una tarea en texto.", {
    task: { type: "string" },
  }, ["task"]),
  fnTool("list_minions", "Lista minions activos.", {}, []),
  fnTool("kill_minion", "Mata un minion.", { id: { type: "string" } }, ["id"]),
  fnTool("talk_to", "Manda un mensaje a un minion.", {
    id: { type: "string" },
    message: { type: "string" },
  }, ["id", "message"]),
  fnTool("ask_user", "Encola una pregunta para el usuario (FIFO, bloquea hasta answer/timeout/cancel).", {
    question: { type: "string" },
  }, ["question"]),
  fnTool("answer", "Respuesta corta para el orchestrator.", { text: { type: "string" } }, ["text"]),
];

const MINION_TOOLS = [
  fnTool("ask", "Pregunta al agente agentic. Incluí contexto completo.", {
    question: { type: "string" },
  }, ["question"]),
];

function fnTool(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: string[],
) {
  return {
    type: "function" as const,
    function: {
      name,
      description,
      parameters: { type: "object", properties, required },
    },
  };
}

function parseArgs(raw: string): Record<string, unknown> {
  try {
    const v = JSON.parse(raw || "{}");
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  } catch {
    return { _raw: raw };
  }
}

function str(v: unknown) {
  return typeof v === "string" ? v : v == null ? "" : String(v);
}

function deltaText(chunk: unknown): string {
  if (!chunk || typeof chunk !== "object") return "";
  const choices = (chunk as { choices?: Array<{ delta?: { content?: unknown } }> }).choices;
  const c = choices?.[0]?.delta?.content;
  return typeof c === "string" ? c : "";
}

function chunkMeta(chunk: unknown): { provider: string; strategy: string } {
  if (!chunk || typeof chunk !== "object") return { provider: "", strategy: "" };
  const o = chunk as Record<string, unknown>;
  let provider = typeof o.provider === "string" ? o.provider : "";
  let strategy = "";
  const meta = (o.openrouterMetadata ?? o.openrouter_metadata) as
    | {
        strategy?: unknown;
        attempts?: Array<{ provider?: unknown }>;
        endpoints?: { available?: Array<{ provider?: unknown; selected?: unknown }> };
      }
    | undefined;
  if (meta && typeof meta === "object") {
    if (typeof meta.strategy === "string") strategy = meta.strategy;
    const sel = meta.endpoints?.available?.find((e) => e.selected === true)?.provider;
    if (typeof sel === "string") provider = sel;
    else {
      const p = meta.attempts?.[0]?.provider;
      if (typeof p === "string") provider = p;
    }
  }
  return { provider, strategy };
}

function deltaToolDeltas(chunk: unknown) {
  if (!chunk || typeof chunk !== "object") return [];
  const choices = (
    chunk as {
      choices?: Array<{ delta?: { toolCalls?: unknown; tool_calls?: unknown } }>;
    }
  ).choices;
  const delta = choices?.[0]?.delta;
  const tcs = delta?.toolCalls ?? delta?.tool_calls;
  return Array.isArray(tcs) ? tcs : [];
}

function toOrMessages(messages: Msg[]) {
  return messages.map((m) => {
    if (m.role === "tool") {
      return { role: "tool", content: m.content ?? "", toolCallId: m.tool_call_id };
    }
    if (m.tool_calls?.length) {
      return {
        role: "assistant" as const,
        content: m.content,
        toolCalls: m.tool_calls.map((t) => ({
          id: t.id,
          type: "function" as const,
          function: t.function,
        })),
      };
    }
    return { role: m.role, content: m.content };
  });
}

class Serial {
  private tail: Promise<void> = Promise.resolve();
  enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.tail.then(fn, fn);
    this.tail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }
}

function withTimeout<T>(p: Promise<T>, msWait: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(label)), msWait);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

type AccTool = { id: string; name: string; arguments: string };

async function streamChat(opts: {
  openrouter: OpenRouter;
  model: string;
  messages: Msg[];
  tools?: unknown[];
  signal?: AbortSignal;
  onSentence?: (s: string) => void;
  onFirstToken?: (provider: string) => void;
}): Promise<{ text: string; toolCalls: AccTool[]; provider: string; strategy: string }> {
  const stream = (await opts.openrouter.chat.send(
    {
      xOpenRouterMetadata: "enabled",
      chatRequest: {
        model: opts.model,
        messages: toOrMessages(opts.messages) as never,
        stream: true,
        provider: {
          sort: "latency",
          ...(opts.tools?.length ? { requireParameters: true } : {}),
        },
        ...(opts.tools?.length ? { tools: opts.tools as never } : {}),
      },
    },
    opts.signal ? { signal: opts.signal } : {},
  )) as AsyncIterable<unknown>;

  let buf = "";
  let full = "";
  let provider = "";
  let strategy = "";
  let first = false;
  const acc: AccTool[] = [];

  for await (const chunk of stream) {
    if (opts.signal?.aborted) break;
    const meta = chunkMeta(chunk);
    if (meta.provider) provider = meta.provider;
    if (meta.strategy) strategy = meta.strategy;
    const piece = deltaText(chunk);
    if (piece) {
      if (!first) {
        first = true;
        opts.onFirstToken?.(provider);
      }
      buf += piece;
      full += piece;
      if (opts.onSentence) {
        const { sentences, rest } = takeSentences(buf, false);
        buf = rest;
        for (const s of sentences) opts.onSentence(s);
      }
    }
    for (const raw of deltaToolDeltas(chunk)) {
      if (!raw || typeof raw !== "object") continue;
      const d = raw as {
        index?: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      };
      const i = d.index ?? acc.length;
      if (!acc[i]) acc[i] = { id: "", name: "", arguments: "" };
      if (d.id) acc[i].id = d.id;
      if (d.function?.name) acc[i].name += d.function.name;
      if (d.function?.arguments) acc[i].arguments += d.function.arguments;
    }
  }
  if (opts.onSentence) {
    const { sentences } = takeSentences(buf, true);
    for (const s of sentences) opts.onSentence(s);
  }
  const toolCalls = acc.filter((t) => t.name);
  for (const t of toolCalls) if (!t.id) t.id = randomUUID();
  return { text: full, toolCalls, provider, strategy };
}

function asToolCalls(calls: AccTool[]): ToolCall[] {
  return calls.map((c) => ({
    id: c.id,
    type: "function",
    function: { name: c.name, arguments: c.arguments },
  }));
}

type Minion = {
  id: string;
  task: string;
  killed: boolean;
  messages: Msg[];
  serial: Serial;
};

type AgenticJob = {
  kind: "ask" | "minion_ask" | "minion_done";
  text: string;
  minionId?: string;
  chain: string[];
};

export class Brain {
  private agentic = new Serial();
  private agenticMsgs: Msg[] = [{ role: "system", content: AGENTIC_SYSTEM }];
  private minions = new Map<string, Minion>();

  constructor(private s: Session) {}

  killAll() {
    for (const m of this.minions.values()) m.killed = true;
    this.minions.clear();
  }

  async compactIfNeeded() {
    const mem = this.s.memory;
    if (!mem.overBudget() || !this.s.openrouter) return;
    const keep = 4;
    const old = mem.messages.slice(1, -keep);
    if (!old.length) return;
    const blob = old
      .map((m) => `${m.role}: ${m.content || JSON.stringify(m.tool_calls || "")}`)
      .join("\n")
      .slice(0, WINDOW_CHARS);
    try {
      const { text } = await streamChat({
        openrouter: this.s.openrouter,
        model: ORCHESTRATOR_MODEL,
        messages: [
          {
            role: "system",
            content: "Resumí en español, 8-12 oraciones, hechos y decisiones. Sin markdown.",
          },
          { role: "user", content: blob },
        ],
      });
      if (this.s.state !== "listening" && this.s.state !== "idle") return;
      mem.applySummary(text.trim() || "(sin resumen)", keep);
    } catch (err) {
      if (this.s.state !== "listening" && this.s.state !== "idle") return;
      mem.applySummary(`(compact falló: ${errMsg(err)})`, keep);
    }
  }

  async runOrchestrator(userText: string, gen: number) {
    const s = this.s;
    if (!s.openrouter) {
      s.speakError("OPENROUTER_API_KEY missing in node/.env", gen);
      s.llmBusy = false;
      s.maybeFinishTurn();
      return;
    }
    if (s.memory.llmChars > SESSION_MAX_LLM_CHARS) {
      s.llmBusy = false;
      s.speakError("tope de uso de LLM en esta sesión", gen);
      s.maybeFinishTurn();
      return;
    }
    const ac = new AbortController();
    s.llmAbort = ac;
    s.llmBusy = true;
    s.memory.push({ role: "user", content: userText });
    s.memory.addChars(userText.length);
    const pending = s.questions.head()?.text;
    const messages = s.memory.window(pending);
    const t0 = now();
    s.logTurn(`llm start model=${ORCHESTRATOR_MODEL} sort=latency`);
    let full = "";
    let provider = "";
    let strategy = "";
    try {
      while (true) {
        if (ac.signal.aborted || s.gen !== gen) return;
        const { text, toolCalls, provider: p, strategy: st } = await streamChat({
          openrouter: s.openrouter,
          model: ORCHESTRATOR_MODEL,
          messages,
          tools: ORCH_TOOLS,
          signal: ac.signal,
          onSentence: (sent) => s.speakAssistant(sent, gen),
          onFirstToken: (prov) =>
            s.logTurn(`llm first_token_ms=${Math.round(now() - t0)}${prov ? ` provider=${prov}` : ""}`),
        });
        provider = p;
        strategy = st;
        full += text;
        s.memory.addChars(text.length);
        if (ac.signal.aborted || s.gen !== gen) {
          s.memory.log({ type: "orch_abort", text, gen });
          return;
        }
        if (!toolCalls.length) {
          if (text.trim()) s.memory.push({ role: "assistant", content: text.trim() });
          return;
        }
        messages.push({
          role: "assistant",
          content: text || null,
          tool_calls: asToolCalls(toolCalls),
        });
        s.memory.push({
          role: "assistant",
          content: text || null,
          tool_calls: asToolCalls(toolCalls),
        });
        for (const tc of toolCalls) {
          const out = await this.execOrchTool(tc, gen, ac);
          messages.push({ role: "tool", tool_call_id: tc.id, content: out });
          s.memory.push({ role: "tool", tool_call_id: tc.id, content: out });
          if (s.gen !== gen || ac.signal.aborted) return;
        }
      }
    } catch (err) {
      if (aborted(err) || ac.signal.aborted || s.gen !== gen) return;
      s.speakError(errMsg(err), gen);
    } finally {
      s.logTurn(
        `llm done total_ms=${Math.round(now() - t0)} chars=${full.length}${provider ? ` provider=${provider}${strategy ? `/${strategy}` : ""}` : ""}`,
      );
      if (s.llmAbort === ac) {
        s.llmAbort = null;
        s.llmBusy = false;
        s.maybeFinishTurn();
      }
    }
  }

  private async execOrchTool(tc: AccTool, gen: number, ac: AbortController) {
    const args = parseArgs(tc.arguments);
    if (tc.name === "pending_question") {
      const action = str(args.action);
      if (action === "answer") {
        const ok = this.s.questions.answer(str(args.text));
        return ok ? "ok" : "no hay pregunta pendiente";
      }
      if (action === "cancel") {
        const ok = this.s.questions.cancel();
        return ok ? "cancelled" : "no hay pregunta pendiente";
      }
      return "action inválida";
    }
    if (tc.name !== "ask") return `tool desconocido: ${tc.name}`;
    const question = str(args.question).trim();
    if (!question) return "question vacío";
    this.s.toolRunning = true;
    this.s.emitState();
    try {
      const result = await withTimeout(
        this.agenticAsk({ kind: "ask", text: question, chain: [] }),
        ASK_TIMEOUT_MS,
        "ask timeout",
      );
      const clipped = truncate(result, ANSWER_MAX_CHARS);
      if (this.s.gen !== gen || ac.signal.aborted) {
        this.s.memory.inject(`Resultado de una consulta anterior: ${clipped}`);
        return clipped;
      }
      return clipped;
    } catch (err) {
      const msg = `error: ${errMsg(err)}`;
      if (this.s.gen !== gen || ac.signal.aborted) {
        this.s.memory.inject(`Consulta anterior: ${msg}`);
        return msg;
      }
      return msg;
    } finally {
      this.s.toolRunning = false;
      this.s.emitState();
    }
  }

  private agenticAsk(job: AgenticJob): Promise<string> {
    return this.agentic.enqueue(() => this.runAgentic(job));
  }

  private async runAgentic(job: AgenticJob): Promise<string> {
    const s = this.s;
    s.memory.log({ type: "agentic_job", ...job });
    if (job.chain.length > CHAIN_MAX_DEPTH) return "error: profundidad máxima de cadena";
    if (!s.openrouter) return "error: no hay LLM";
    const user =
      job.kind === "minion_ask"
        ? `Pregunta del minion ${job.minionId} (cadena ${job.chain.join(">") || "-"}):\n${job.text}`
        : job.kind === "minion_done"
          ? job.text
          : `Pedido del orchestrator:\n${job.text}`;
    this.agenticMsgs.push({ role: "user", content: user });
    s.memory.addChars(user.length);
    for (let round = 0; round < AGENTIC_MAX_ROUNDS; round++) {
      const { text, toolCalls, provider } = await streamChat({
        openrouter: s.openrouter,
        model: AGENTIC_MODEL,
        messages: this.agenticMsgs,
        tools: AGENTIC_TOOLS,
      });
      s.memory.addChars(text.length);
      s.memory.log({ type: "agentic_round", round, text, tools: toolCalls.map((t) => t.name), provider });
      if (!toolCalls.length) return truncate(text.trim() || "(sin respuesta)", ANSWER_MAX_CHARS);
      this.agenticMsgs.push({
        role: "assistant",
        content: text || null,
        tool_calls: asToolCalls(toolCalls),
      });
      let answered: string | null = null;
      for (const tc of toolCalls) {
        const out = await this.execAgenticTool(tc, job.chain);
        this.agenticMsgs.push({ role: "tool", tool_call_id: tc.id, content: out });
        if (tc.name === "answer") answered = out;
      }
      if (answered !== null) return truncate(answered, ANSWER_MAX_CHARS);
    }
    return "error: demasiadas herramientas";
  }

  private async execAgenticTool(tc: AccTool, chain: string[]): Promise<string> {
    const args = parseArgs(tc.arguments);
    if (tc.name === "answer") return str(args.text).trim() || "(vacío)";
    if (tc.name === "list_minions") {
      const list = [...this.minions.values()]
        .filter((m) => !m.killed)
        .map((m) => `${m.id}: ${m.task.slice(0, 80)}`);
      return list.length ? list.join("\n") : "(ninguno)";
    }
    if (tc.name === "create_minion") {
      const task = str(args.task).trim();
      if (!task) return "task vacío";
      const id = randomUUID().slice(0, 6);
      const m: Minion = {
        id,
        task,
        killed: false,
        messages: [
          { role: "system", content: MINION_SYSTEM },
          { role: "user", content: task },
        ],
        serial: new Serial(),
      };
      this.minions.set(id, m);
      void m.serial.enqueue(() => this.runMinion(m, chain));
      return `created ${id}`;
    }
    if (tc.name === "kill_minion") {
      const m = this.minions.get(str(args.id));
      if (!m) return "no existe";
      m.killed = true;
      this.minions.delete(m.id);
      return "killed";
    }
    if (tc.name === "talk_to") {
      const id = str(args.id);
      const message = str(args.message).trim();
      const m = this.minions.get(id);
      if (!m || m.killed) return "no existe";
      if (chain.includes(id)) return "error: loop — ese minion ya está en la cadena";
      m.messages.push({ role: "user", content: message });
      void m.serial.enqueue(() => this.runMinion(m, chain));
      return "ok";
    }
    if (tc.name === "ask_user") {
      const question = str(args.question).trim();
      if (!question) return "question vacío";
      const result = await this.s.questions.enqueue(question);
      return JSON.stringify(result);
    }
    return `tool desconocido: ${tc.name}`;
  }

  private async runMinion(m: Minion, parentChain: string[]) {
    const s = this.s;
    if (m.killed || !s.openrouter) return;
    const chain = [...parentChain, m.id];
    s.memory.log({ type: "minion_run", id: m.id, chain });
    for (let round = 0; round < AGENTIC_MAX_ROUNDS && !m.killed; round++) {
      const { text, toolCalls, provider } = await streamChat({
        openrouter: s.openrouter,
        model: AGENTIC_MODEL,
        messages: m.messages,
        tools: MINION_TOOLS,
      });
      s.memory.addChars(text.length);
      s.memory.log({ type: "minion_round", id: m.id, round, provider });
      if (m.killed) return;
      if (!toolCalls.length) {
        if (text.trim()) m.messages.push({ role: "assistant", content: text.trim() });
        void this.agenticAsk({
          kind: "minion_done",
          text: `Minion ${m.id} terminó:\n${truncate(text.trim(), ANSWER_MAX_CHARS)}`,
          minionId: m.id,
          chain,
        });
        return;
      }
      m.messages.push({
        role: "assistant",
        content: text || null,
        tool_calls: asToolCalls(toolCalls),
      });
      for (const tc of toolCalls) {
        if (tc.name !== "ask") {
          m.messages.push({ role: "tool", tool_call_id: tc.id, content: `tool desconocido: ${tc.name}` });
          continue;
        }
        if (chain.length > CHAIN_MAX_DEPTH) {
          m.messages.push({ role: "tool", tool_call_id: tc.id, content: "error: profundidad máxima" });
          continue;
        }
        const q = str(parseArgs(tc.arguments).question).trim();
        const out = await this.agenticAsk({
          kind: "minion_ask",
          text: q,
          minionId: m.id,
          chain,
        });
        m.messages.push({ role: "tool", tool_call_id: tc.id, content: out });
      }
    }
  }
}

