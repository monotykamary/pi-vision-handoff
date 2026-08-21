import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// The extension resolves its config directory in-process (src/agent-dir.ts)
// instead of importing pi's getAgentDir(), which drags the pi-coding-agent
// module graph into startup. The resolver must still honor the
// PI_CODING_AGENT_DIR override and fall back to ~/.pi/agent.

describe("config dir resolves in-process", () => {
  it("src/index.ts uses the local agent-dir resolver", () => {
    const srcPath = join(process.cwd(), "src", "index.ts");
    const source = readFileSync(srcPath, "utf8");

    expect(source).toContain('import { resolveAgentDir } from "./agent-dir.js"');
    expect(source).toContain("resolveAgentDir()");

    // Must not pull the pi-coding-agent runtime into the load path.
    expect(source).not.toContain("@earendil-works/pi-coding-agent");
  });

  it("resolver honors the env override and the ~/.pi/agent fallback", () => {
    const resolverPath = join(process.cwd(), "src", "agent-dir.ts");
    const source = readFileSync(resolverPath, "utf8");

    expect(source).toContain("PI_CODING_AGENT_DIR");
    expect(source).toContain('".pi"');
    expect(source).toContain("homedir()");
  });

  it("configures the extensions/ subdir with the pi-vision-handoff.json name", () => {
    const srcPath = join(process.cwd(), "src", "index.ts");
    const source = readFileSync(srcPath, "utf8");

    expect(source).toContain('CONFIG_SUBDIR = "extensions"');
    expect(source).toContain('CONFIG_FILENAME = "pi-vision-handoff.json"');
    expect(source).toContain("join(resolveAgentDir(), CONFIG_SUBDIR, CONFIG_FILENAME)");
  });
});
