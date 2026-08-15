export type ApiTool = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
};

export type ApiMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
};

// Cliente mínimo OpenAI-compatible para llama.cpp (local, baja latencia).
export class Llm {
  constructor(private baseUrl: string) {}

  async *chat(opts: {
    model: string;
    messages: ApiMessage[];
    tools?: ApiTool[];
    signal?: AbortSignal;
  }): AsyncIterable<unknown> {
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: opts.model,
        messages: opts.messages,
        stream: true,
        ...(opts.tools?.length ? { tools: opts.tools } : {}),
      }),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "");
      const err = new Error(`llama.cpp ${res.status}: ${body.slice(0, 240)}`) as Error & {
        statusCode?: number;
        body?: string;
      };
      err.statusCode = res.status;
      err.body = body;
      throw err;
    }
    const dec = new TextDecoder();
    let buf = "";
    for await (const raw of res.body) {
      buf += dec.decode(raw as BufferSource, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        try {
          yield JSON.parse(payload);
        } catch {
          /* chunk malformado: skip */
        }
      }
    }
  }
}
