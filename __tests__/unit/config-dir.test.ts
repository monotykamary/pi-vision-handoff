import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// Mirror of pi-hide-providers' config-dir test: ensures the extension resolves
// its config directory through pi's getAgentDir() rather than hardcoding paths.

describe("config dir uses getAgentDir", () => {
  it("src/index.ts imports getAgentDir from @earendil-works/pi-coding-agent", () => {
    const srcPath = join(process.cwd(), "src", "index.ts");
    const source = readFileSync(srcPath, "utf8");

    expect(source).toMatch(
      /import\s*\{[^}]*getAgentDir[^}]*\}\s*from\s*["']@earendil-works\/pi-coding-agent["']/,
    );
    expect(source).toMatch(/getAgentDir\(\)/);

    // Must NOT hardcode ".pi"/"agent" via homedir() + join().
    expect(source).not.toMatch(
      /import\s*\{[^}]*homedir[^}]*\}\s*from\s*["']node:os["']/,
    );
    expect(source).not.toMatch(/join\s*\([^)]*["']\.pi["']/);
  });

  it("configures the extensions/ subdir with the pi-vision-handoff.json name", () => {
    const srcPath = join(process.cwd(), "src", "index.ts");
    const source = readFileSync(srcPath, "utf8");

    expect(source).toMatch(/CONFIG_SUBDIR\s*=\s*["']extensions["']/);
    expect(source).toMatch(/CONFIG_FILENAME\s*=\s*["']pi-vision-handoff\.json["']/);
    expect(source).toMatch(/join\(getAgentDir\(\),\s*CONFIG_SUBDIR,\s*CONFIG_FILENAME\)/);
  });
});
