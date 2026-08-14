import { randomUUID } from "node:crypto";
import { QUESTION_TIMEOUT_MS } from "./config.js";

export type QuestionStatus = "answer" | "timeout" | "cancelled";

export type QuestionResult = { status: QuestionStatus; text?: string };

export type Question = {
  id: string;
  text: string;
  spoken: boolean;
};

type Item = Question & {
  resolve: (r: QuestionResult) => void;
  timer: ReturnType<typeof setTimeout>;
};

export class QuestionQueue {
  private items: Item[] = [];
  onEnqueue: (() => void) | null = null;
  onResolve: ((info: QuestionResult & { id: string }) => void) | null = null;

  head(): Question | undefined {
    const h = this.items[0];
    if (!h) return undefined;
    return { id: h.id, text: h.text, spoken: h.spoken };
  }

  markSpoken(id: string) {
    const h = this.items[0];
    if (h && h.id === id) h.spoken = true;
  }

  enqueue(text: string, timeoutMs = QUESTION_TIMEOUT_MS): Promise<QuestionResult> {
    const t = String(text || "").trim();
    return new Promise((resolve) => {
      const id = randomUUID().slice(0, 8);
      const item: Item = {
        id,
        text: t,
        spoken: false,
        resolve,
        timer: setTimeout(() => this.finish(id, { status: "timeout" }), timeoutMs),
      };
      this.items.push(item);
      this.onEnqueue?.();
    });
  }

  answer(text: string): boolean {
    const h = this.items[0];
    if (!h) return false;
    this.finish(h.id, { status: "answer", text: String(text || "").trim() });
    return true;
  }

  cancel(id?: string): boolean {
    const h = this.items[0];
    if (!h) return false;
    if (id && h.id !== id) return false;
    this.finish(h.id, { status: "cancelled" });
    return true;
  }

  clear() {
    for (const it of this.items) {
      clearTimeout(it.timer);
      it.resolve({ status: "cancelled" });
    }
    this.items = [];
  }

  private finish(id: string, result: QuestionResult) {
    const i = this.items.findIndex((x) => x.id === id);
    if (i < 0) return;
    const item = this.items[i];
    clearTimeout(item.timer);
    this.items.splice(i, 1);
    item.resolve(result);
    this.onResolve?.({ id, ...result });
  }
}

{
  const q = new QuestionQueue();
  void q.enqueue("hola?");
  if (q.head()?.text !== "hola?" || q.head()?.spoken) throw new Error("queue head");
  q.markSpoken(q.head()!.id);
  if (!q.head()?.spoken) throw new Error("queue spoken");
  void q.enqueue("segunda");
  if (!q.answer("sí")) throw new Error("queue answer");
  if (q.head()?.text !== "segunda") throw new Error("queue fifo");
  if (!q.cancel()) throw new Error("queue cancel");
  if (q.head()) throw new Error("queue empty");
}
