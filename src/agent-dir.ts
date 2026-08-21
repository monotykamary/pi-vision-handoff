// pi's agent-dir resolution in-process, so no module on the extension load
// path imports the pi-coding-agent runtime (its graph costs ~1s per extension
// in pi's loader).

import { homedir } from "node:os";
import { join } from "node:path";

const expandTilde = (value: string): string => {
  if (value === "~") return homedir();
  return value.startsWith("~/") ? join(homedir(), value.slice(2)) : value;
};

export const resolveAgentDir = (): string => {
  const override = process.env.PI_CODING_AGENT_DIR;
  return override ? expandTilde(override) : join(homedir(), ".pi", "agent");
};
