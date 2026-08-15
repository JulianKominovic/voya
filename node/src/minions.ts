import type { ChatFunctionToolFunction } from "@openrouter/sdk/models";
import { exec } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

export type MinionTool = ChatFunctionToolFunction & {
  exec: (args: Record<string, unknown>) => Promise<string>;
};

export type MinionDef = {
  id: string;
  name: string;
  description: string;
  system: string; // instrucciones de este minion (el host inyecta el protocolo aparte)
  tools: MinionTool[];
};

export function toOrTool({ exec: _exec, ...t }: MinionTool): ChatFunctionToolFunction {
  return t;
}

const execAsync = promisify(exec);
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT_CAP = 16_000;

function clip(s: string) {
  const t = s.trim();
  return t.length <= OUT_CAP ? t : `${t.slice(0, OUT_CAP)}\n…(truncado)`;
}

async function runShell(args: Record<string, unknown>): Promise<string> {
  const command = typeof args.command === "string" ? args.command.trim() : "";
  if (!command) return "command vacío";
  try {
    const { stdout, stderr } = await execAsync(command, {
      cwd: REPO,
      timeout: 60_000,
      maxBuffer: 256 * 1024,
    });
    return clip([stdout, stderr].filter(Boolean).join("\n")) || "(sin output)";
  } catch (err) {
    const e = err as {
      stdout?: string;
      stderr?: string;
      code?: unknown;
      killed?: boolean;
      message?: string;
    };
    return clip(
      [e.killed ? "timeout" : `exit ${e.code ?? "?"}`, e.stdout, e.stderr].filter(Boolean).join("\n") ||
        e.message ||
        "error",
    );
  }
}

const shell: MinionTool = {
  type: "function",
  function: {
    name: "shell",
    description: `Corre un comando en la shell. cwd=${REPO}. stdout+stderr; timeout 60s.`,
    parameters: {
      type: "object",
      properties: { command: { type: "string", description: "Comando a ejecutar (sh)." } },
      required: ["command"],
    },
  },
  exec: runShell,
};

const dev: MinionDef = {
  id: "developer",
  name: "Developer",
  description: "Desarrollador de software con acceso a la terminal y editor de código.",
  system: "Sos un desarrollador de software. Completá la tarea de código que te asignen. Usá shell para comandos.",
  tools: [shell],
};

export const MINIONS: MinionDef[] = [dev];
