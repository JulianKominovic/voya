import { WebSocket } from "ws";
import { errMsg, sleep } from "./text.js";

export function sendJson(ws: WebSocket | undefined, obj: unknown) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

export function sendRaw(ws: WebSocket | undefined, data: WebSocket.RawData, binary: boolean) {
  if (ws?.readyState !== WebSocket.OPEN) return;
  if (binary) ws.send(data, { binary: true });
  else ws.send(typeof data === "string" ? data : data.toString());
}

export async function openSocket(url: string, retries = 40): Promise<WebSocket> {
  let last: unknown;
  for (let i = 0; i < retries; i++) {
    try {
      const ws = await new Promise<WebSocket>((resolve, reject) => {
        const sock = new WebSocket(url);
        const onErr = (err: Error) => {
          sock.close();
          reject(err);
        };
        sock.once("open", () => {
          sock.off("error", onErr);
          resolve(sock);
        });
        sock.once("error", onErr);
      });
      return ws;
    } catch (err) {
      last = err;
      await sleep(500);
    }
  }
  throw last instanceof Error ? last : new Error(`cannot connect ${url}`);
}

export { errMsg };
