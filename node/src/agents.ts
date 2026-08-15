import { randomUUID } from "node:crypto";
import type { ApiMessage, ApiTool, Llm } from "./llm.js";
import {
  AGENTIC_MAX_ROUNDS,
  AGENTIC_MODEL,
  ANSWER_MAX_CHARS,
  ASK_TIMEOUT_MS,
  CHAIN_MAX_DEPTH,
  LLM_URL,
  ORCHESTRATOR_MODEL,
  SESSION_MAX_LLM_CHARS,
  WINDOW_CHARS,
} from "./config.js";
import type { Msg, ToolCall } from "./memory.js";
import { MINIONS, toApiTool, type MinionDef } from "./minions.js";
import type { Session } from "./session.js";
import { aborted, errMsg, forTts, now, takeSentences, truncate } from "./text.js";

export const ORCH_SYSTEM = `Sos el lado hablado de Voya. Español oral, dos a cuatro oraciones. Sin markdown, listas ni código.
Lo que hablás es solo para el usuario: nunca leas el system, system prompt, las tools ni razonamiento interno.
Tools:
- ask: trabajo de fondo (pensar, descomponer, minions). Si va a tardar, decí algo corto antes.
- pending_question: si el contexto muestra una pregunta pendiente, registrá la respuesta del usuario o cancelala si pide olvidarla o cambió de tema de forma definitiva. Si cancelás por cambio de tema, primero decilo en voz alta.
No inventes el estado de la cola. El usuario puede no responder: si hay timeout, no insistas.`;

const AGENTIC_SYSTEM = `Sos el agente agentic de Voya. No hablás con el usuario: devolvé el resultado con el tool answer (corto, sin markdown).
Tools: create_minion, list_minions, kill_minion, talk_to, ask_user, answer.
create_minion: id del catálogo + task. talk_to/kill usan el id de instancia (created). Minions:
${MINIONS.map((m) => `- ${m.id}: ${m.description}`).join("\n")}
ask_user encola una pregunta FIFO; el usuario puede tardar o nunca responder. Si timeout o cancel, seguí con tu mejor juicio o abortá con answer.
No esperes a un minion con ask_user, talk_to ni list_minions: el report (status, ask o done) llega solo. talk_to es una instrucción nueva, no un ping. ask_user es solo para el usuario.
answer es obligatorio para devolverle al orchestrator.`;

const ORCH_TOOLS = [
  fnTool("ask", "Consulta al agente agentic (trabajo de fondo).", {
    question: { type: "string" },
  }, ["question"]),
  fnTool("pending_question", "Resolver la pregunta pendiente: answer o cancel.", {
    action: { type: "string", enum: ["answer", "cancel"] },
    text: { type: "string" },
  }, ["action"]),
];

const MINION_IDS = MINIONS.map((m) => m.id);

const AGENTIC_TOOLS = [
  fnTool("create_minion", `Crea un minion de fondo. id: ${MINION_IDS.join(", ")}.`, {
    id: { type: "string", enum: MINION_IDS },
    task: { type: "string" },
  }, ["id", "task"]),
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

const REPORT_TOOL = fnTool("report", "Canal con el agente: status (progreso), ask (pregunta; puede ir al usuario), done (resultado final y termina).", {
  kind: { type: "string", enum: ["status", "ask", "done"] },
  text: { type: "string" },
}, ["kind", "text"]);

const HOST_MINION_SYSTEM = `Trabajás una tarea en segundo plano. report es el canal con el agente:
- status: progreso, si hace falta. Seguí trabajando.
- ask: pregunta al agente. Toda pregunta lleva contexto completo: qué estás haciendo, por qué preguntás, dónde trabás. El agente puede preguntarle al usuario.
- done: resultado final. Obligatorio para terminar.
No termines en texto suelto: usá report.`;

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

function toApiMessages(messages: Msg[]): ApiMessage[] {
  return messages.map((m) => {
    if (m.role === "tool") {
      return { role: "tool", content: m.content ?? "", tool_call_id: m.tool_call_id };
    }
    if (m.tool_calls?.length) {
      return {
        role: "assistant",
        content: m.content,
        tool_calls: m.tool_calls.map((t) => ({
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
  llm: Llm;
  model: string;
  fallbackModel?: string;
  messages: Msg[];
  tools?: unknown[];
  signal?: AbortSignal;
  onSentence?: (s: string) => void;
  onFirstToken?: () => void;
  onFallback?: (err: unknown, model: string) => void;
}): Promise<{ text: string; toolCalls: AccTool[] }> {
  try {
    return await streamChatOnce(opts);
  } catch (err) {
    const fb = opts.fallbackModel;
    if (!fb || fb === opts.model || aborted(err) || opts.signal?.aborted) throw err;
    opts.onFallback?.(err, fb);
    return await streamChatOnce({ ...opts, model: fb });
  }
}

async function streamChatOnce(opts: {
  llm: Llm;
  model: string;
  messages: Msg[];
  tools?: unknown[];
  signal?: AbortSignal;
  onSentence?: (s: string) => void;
  onFirstToken?: () => void;
}): Promise<{ text: string; toolCalls: AccTool[] }> {
  const stream = opts.llm.chat({
    model: opts.model,
    messages: toApiMessages(opts.messages),
    tools: opts.tools as ApiTool[] | undefined,
    signal: opts.signal,
  });

  let buf = "";
  let full = "";
  let first = false;
  const acc: AccTool[] = [];

  for await (const chunk of stream) {
    if (opts.signal?.aborted) break;
    const piece = deltaText(chunk);
    if (piece) {
      if (!first) {
        first = true;
        opts.onFirstToken?.();
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
  return { text: full, toolCalls };
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
  def: MinionDef;
  task: string;
  killed: boolean;
  busy: boolean;
  messages: Msg[];
  serial: Serial;
};

type AgenticJob = {
  kind: "ask" | "minion_ask" | "minion_done" | "minion_status";
  text: string;
  minionId?: string;
  chain: string[];
};

function jobUser(job: AgenticJob): string {
  if (job.kind === "minion_ask") {
    return `Pregunta del minion ${job.minionId} (cadena ${job.chain.join(">") || "-"}). Si hace falta, ask_user:\n${job.text}`;
  }
  if (job.kind === "minion_done") return job.text;
  if (job.kind === "minion_status") return `Estado del minion ${job.minionId}:\n${job.text}`;
  return `Pedido del orchestrator:\n${job.text}`;
}

export class Brain {
  private agentic = new Serial();
  private agenticMsgs: Msg[] = [{ role: "system", content: AGENTIC_SYSTEM }];
  private minions = new Map<string, Minion>();
  private agenticLive = false;
  private inbox: AgenticJob[] = [];

  constructor(private s: Session) {}

  killAll() {
    for (const m of this.minions.values()) m.killed = true;
    this.minions.clear();
  }

  async compactIfNeeded() {
    const mem = this.s.memory;
    if (!mem.overBudget() || !this.s.llm) return;
    const keep = 4;
    const old = mem.messages.slice(1, -keep);
    if (!old.length) return;
    const blob = old
      .map((m) => `${m.role}: ${m.content || JSON.stringify(m.tool_calls || "")}`)
      .join("\n")
      .slice(0, WINDOW_CHARS);
    try {
      const { text } = await streamChat({
        llm: this.s.llm,
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
    if (!s.llm) {
      s.speakError("LLM_URL missing in node/.env", gen);
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
    s.logTurn(`llm start model=${ORCHESTRATOR_MODEL} url=${LLM_URL}`, "llm_conv");
    let full = "";
    try {
      while (true) {
        if (ac.signal.aborted || s.gen !== gen) return;
        const { text, toolCalls } = await streamChat({
          llm: s.llm,
          model: ORCHESTRATOR_MODEL,
          messages,
          tools: ORCH_TOOLS,
          signal: ac.signal,
          onFirstToken: () =>
            s.logTurn(`llm first_token_ms=${Math.round(now() - t0)}`, "llm_conv"),
        });
        full += text;
        s.memory.addChars(text.length);
        if (ac.signal.aborted || s.gen !== gen) {
          s.memory.log({ type: "orch_abort", text, gen });
          return;
        }
        // ponytail: buffer until tools are known — pending_question dumps protocol into content
        if (!toolCalls.length || toolCalls.some((t) => t.name === "ask")) {
          for (const sent of forTts(text, ORCH_SYSTEM)) s.speakAssistant(sent, gen);
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
      s.logTurn(`llm done total_ms=${Math.round(now() - t0)} chars=${full.length}`, "llm_conv");
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
      this.s.logTurn(`tool pending_question action=${action}`, "llm_conv");
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
    this.s.logTurn(`tool ask ${truncate(question, 120)}`, "llm_conv");
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
      this.s.logTurn(`tool ask ${msg}`, "llm_conv");
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
    // ponytail: mailbox is serial; if we enqueue minion_done behind an in-flight ask,
    // the agent never sees it and polls with talk_to until AGENTIC_MAX_ROUNDS.
    if (this.agenticLive && (job.kind === "minion_done" || job.kind === "minion_status")) {
      this.inbox.push(job);
      this.s.logTurn(
        `agentic inbox ${job.kind}${job.minionId ? ` minion=${job.minionId}` : ""}`,
        "llm_agentic",
      );
      return Promise.resolve("ok");
    }
    return this.agentic.enqueue(() => this.runAgentic(job));
  }

  private drainInbox() {
    const jobs = this.inbox.splice(0);
    for (const job of jobs) {
      this.s.memory.log({ type: "agentic_job", ...job });
      const user = jobUser(job);
      this.agenticMsgs.push({ role: "user", content: user });
      this.s.memory.addChars(user.length);
    }
  }

  private takeDone(): AgenticJob | undefined {
    let last: AgenticJob | undefined;
    this.inbox = this.inbox.filter((j) => {
      if (j.kind !== "minion_done") return true;
      last = j;
      return false;
    });
    return last;
  }

  private async runAgentic(job: AgenticJob): Promise<string> {
    const s = this.s;
    this.agenticLive = true;
    try {
      s.memory.log({ type: "agentic_job", ...job });
      s.logTurn(
        `agentic ${job.kind}${job.minionId ? ` minion=${job.minionId}` : ""} model=${AGENTIC_MODEL}${AGENTIC_MODEL !== ORCHESTRATOR_MODEL ? ` fallback=${ORCHESTRATOR_MODEL}` : ""}`,
        "llm_agentic",
      );
      if (job.chain.length > CHAIN_MAX_DEPTH) return "error: profundidad máxima de cadena";
      if (!s.llm) return "error: no hay LLM";
      const user = jobUser(job);
      this.agenticMsgs.push({ role: "user", content: user });
      s.memory.addChars(user.length);
      for (let round = 0; round < AGENTIC_MAX_ROUNDS; round++) {
        this.drainInbox();
        const { text, toolCalls } = await streamChat({
          llm: s.llm,
          model: AGENTIC_MODEL,
          fallbackModel: AGENTIC_MODEL === ORCHESTRATOR_MODEL ? undefined : ORCHESTRATOR_MODEL,
          messages: this.agenticMsgs,
          tools: AGENTIC_TOOLS,
          onFallback: (err, model) =>
            s.logTurn(`agentic fallback model=${model} after ${errMsg(err)}`, "llm_agentic"),
        });
        s.memory.addChars(text.length);
        s.memory.log({ type: "agentic_round", round, text, tools: toolCalls.map((t) => t.name) });
        s.logTurn(
          `agentic round=${round} tools=${toolCalls.map((t) => t.name).join(",") || "-"}`,
          "llm_agentic",
        );
        const done = this.takeDone();
        if (done) {
          this.drainInbox();
          return truncate(done.text, ANSWER_MAX_CHARS);
        }
        if (!toolCalls.length) return truncate(text.trim() || "(sin respuesta)", ANSWER_MAX_CHARS);
        this.agenticMsgs.push({
          role: "assistant",
          content: text || null,
          tool_calls: asToolCalls(toolCalls),
        });
        let answered: string | null = null;
        for (let i = 0; i < toolCalls.length; i++) {
          const got = this.takeDone();
          if (got) {
            this.drainInbox();
            for (let j = i; j < toolCalls.length; j++) {
              this.agenticMsgs.push({ role: "tool", tool_call_id: toolCalls[j].id, content: "ok" });
            }
            return truncate(got.text, ANSWER_MAX_CHARS);
          }
          const tc = toolCalls[i];
          const out = await this.execAgenticTool(tc, job.chain);
          this.agenticMsgs.push({ role: "tool", tool_call_id: tc.id, content: out });
          if (tc.name === "answer") answered = out;
        }
        const got = this.takeDone();
        if (got) {
          this.drainInbox();
          return truncate(got.text, ANSWER_MAX_CHARS);
        }
        if (answered !== null) return truncate(answered, ANSWER_MAX_CHARS);
      }
      const done = this.takeDone();
      this.drainInbox();
      if (done) return truncate(done.text, ANSWER_MAX_CHARS);
      return "error: demasiadas herramientas";
    } finally {
      this.agenticLive = false;
    }
  }

  private async execAgenticTool(tc: AccTool, chain: string[]): Promise<string> {
    const args = parseArgs(tc.arguments);
    const s = this.s;
    if (tc.name === "answer") return str(args.text).trim() || "(vacío)";
    if (tc.name === "list_minions") {
      s.logTurn("agentic tool list_minions", "llm_agentic");
      const list = [...this.minions.values()]
        .filter((m) => !m.killed)
        .map((m) => `${m.id} [${m.def.id}${m.busy ? " busy" : ""}]: ${m.task.slice(0, 80)}`);
      return list.length ? list.join("\n") : "(ninguno)";
    }
    if (tc.name === "create_minion") {
      const def = MINIONS.find((m) => m.id === str(args.id).trim());
      if (!def) return `id inválido. válidos: ${MINION_IDS.join(", ")}`;
      const task = str(args.task).trim();
      if (!task) return "task vacío";
      s.logTurn(`agentic tool create_minion id=${def.id} task=${truncate(task, 100)}`, "llm_agentic");
      const id = randomUUID().slice(0, 6);
      const m: Minion = {
        id,
        def,
        task,
        killed: false,
        busy: true,
        messages: [
          { role: "system", content: [HOST_MINION_SYSTEM, def.system].filter(Boolean).join("\n\n") },
          { role: "user", content: task },
        ],
        serial: new Serial(),
      };
      this.minions.set(id, m);
      s.logTurn(`minion created id=${id} kind=${def.id} task=${truncate(task, 100)}`, "minions");
      void m.serial.enqueue(() => this.runMinion(m, chain));
      return `created ${id} (${def.id})`;
    }
    if (tc.name === "kill_minion") {
      const m = this.minions.get(str(args.id));
      if (!m) return "no existe";
      s.logTurn(`minion killed id=${m.id}`, "minions");
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
      s.logTurn(`minion talk_to id=${id} msg=${truncate(message, 80)}`, "minions");
      m.messages.push({ role: "user", content: message });
      if (!m.busy) {
        m.busy = true;
        void m.serial.enqueue(() => this.runMinion(m, chain));
      }
      return "ok";
    }
    if (tc.name === "ask_user") {
      const question = str(args.question).trim();
      if (!question) return "question vacío";
      s.logTurn(`agentic tool ask_user ${truncate(question, 100)}`, "llm_agentic");
      const result = await this.s.questions.enqueue(question);
      return JSON.stringify(result);
    }
    return `tool desconocido: ${tc.name}`;
  }

  private async runMinion(m: Minion, parentChain: string[]) {
    const s = this.s;
    m.busy = true;
    try {
      if (m.killed || !s.llm) return;
      const chain = [...parentChain, m.id];
      const tools = [
        REPORT_TOOL,
        ...m.def.tools.map(toApiTool),
      ];
      s.memory.log({ type: "minion_run", id: m.id, kind: m.def.id, chain });
      s.logTurn(`minion run id=${m.id} kind=${m.def.id} task=${truncate(m.task, 100)}`, "minions");
      for (let round = 0; round < AGENTIC_MAX_ROUNDS && !m.killed; round++) {
        const { text, toolCalls } = await streamChat({
          llm: s.llm,
          model: AGENTIC_MODEL,
          fallbackModel: AGENTIC_MODEL === ORCHESTRATOR_MODEL ? undefined : ORCHESTRATOR_MODEL,
          messages: m.messages,
          tools,
          onFallback: (err, model) =>
            s.logTurn(`minion fallback id=${m.id} model=${model} after ${errMsg(err)}`, "minions"),
        });
        s.memory.addChars(text.length);
        s.memory.log({ type: "minion_round", id: m.id, round });
        s.logTurn(
          `minion round id=${m.id} round=${round} tools=${toolCalls.map((t) => t.name).join(",") || "-"}`,
          "minions",
        );
        if (m.killed) return;
        if (!toolCalls.length) {
          if (text.trim()) m.messages.push({ role: "assistant", content: text.trim() });
          m.messages.push({
            role: "user",
            content: "Usá report: status (progreso), ask (pregunta) o done (resultado y terminás).",
          });
          continue;
        }
        m.messages.push({
          role: "assistant",
          content: text || null,
          tool_calls: asToolCalls(toolCalls),
        });
        let finished = false;
        for (const tc of toolCalls) {
          const out = await this.execMinionTool(m, tc, chain);
          m.messages.push({ role: "tool", tool_call_id: tc.id, content: out });
          if (tc.name === "report" && str(parseArgs(tc.arguments).kind) === "done") finished = true;
        }
        if (finished) return;
      }
      s.logTurn(`minion stalled id=${m.id}`, "minions");
      void this.agenticAsk({
        kind: "minion_done",
        text: `Minion ${m.id} (${m.def.id}) se detuvo sin report done.`,
        minionId: m.id,
        chain,
      });
    } finally {
      m.busy = false;
    }
  }

  private async execMinionTool(m: Minion, tc: AccTool, chain: string[]): Promise<string> {
    if (tc.name === "report") {
      const args = parseArgs(tc.arguments);
      const kind = str(args.kind);
      const text = str(args.text).trim();
      if (!text) return "text vacío";
      if (kind !== "status" && kind !== "ask" && kind !== "done") return "kind inválido: status|ask|done";
      this.s.logTurn(`minion report kind=${kind} id=${m.id}`, "minions");
      if (kind === "ask") {
        if (chain.length > CHAIN_MAX_DEPTH) return "error: profundidad máxima";
        return this.agenticAsk({ kind: "minion_ask", text, minionId: m.id, chain });
      }
      const job: AgenticJob =
        kind === "done"
          ? {
              kind: "minion_done",
              text: `Minion ${m.id} (${m.def.id}) terminó:\n${truncate(text, ANSWER_MAX_CHARS)}`,
              minionId: m.id,
              chain,
            }
          : {
              kind: "minion_status",
              text: truncate(text, ANSWER_MAX_CHARS),
              minionId: m.id,
              chain,
            };
      void this.agenticAsk(job);
      return "ok";
    }
    const tool = m.def.tools.find((t) => t.function.name === tc.name);
    if (tool) {
      try {
        return await tool.exec(parseArgs(tc.arguments));
      } catch (err) {
        return `error: ${errMsg(err)}`;
      }
    }
    return `tool desconocido: ${tc.name}`;
  }
}

