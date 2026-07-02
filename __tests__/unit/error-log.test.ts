import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  appendVisionError,
  formatErrorLogLine,
  getErrorLogDir,
  getErrorLogPath,
  rotateLogIfNeeded,
  MAX_LOG_BYTES,
  type VisionErrorLogEntry,
} from "../../src/error-log.js";

// All path resolution goes through getAgentDir(), which honors
// $PI_CODING_AGENT_DIR — so redirect it to a fresh temp dir per test and the
// real path-resolution code path gets exercised end-to-end (no fs mocking).
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pi-vh-log-"));
  process.env.PI_CODING_AGENT_DIR = dir;
});

afterEach(() => {
  delete process.env.PI_CODING_AGENT_DIR;
  rmSync(dir, { recursive: true, force: true });
});

function fullEntry(over: Partial<VisionErrorLogEntry> = {}): VisionErrorLogEntry {
  return {
    timestamp: "2026-07-02T00:00:00.000Z",
    phase: "batch",
    reason: "boom",
    visionModel: "litellm/kimi-k2.7",
    imageHashes: ["abc123"],
    imageCount: 1,
    ...over,
  };
}

describe("path resolution", () => {
  it("getErrorLogDir / getErrorLogPath resolve under $PI_CODING_AGENT_DIR", () => {
    expect(getErrorLogDir()).toBe(join(dir, "logs", "pi-vision-handoff"));
    expect(getErrorLogPath()).toBe(join(dir, "logs", "pi-vision-handoff", "errors.log"));
  });
});

describe("formatErrorLogLine", () => {
  it("emits one JSON object terminated by a newline", () => {
    const line = formatErrorLogLine(fullEntry());
    expect(line.endsWith("\n")).toBe(true);
    expect(JSON.parse(line)).toEqual(fullEntry());
  });

  it("round-trips optional fields (stopReason, timedOut, errorStack, config)", () => {
    const entry = fullEntry({
      stopReason: "error",
      timedOut: true,
      timeoutMs: 120000,
      errorMessage: "upstream 500",
      errorStack: "Error: boom\n    at foo",
      config: { maxTokens: 4096, thinking: true, thinkingLevel: "high" },
    });
    expect(JSON.parse(formatErrorLogLine(entry))).toEqual(entry);
  });
});

describe("appendVisionError", () => {
  it("creates the log dir lazily and writes a parseable JSONL line with a filled ISO timestamp", () => {
    expect(existsSync(getErrorLogDir())).toBe(false);
    appendVisionError({
      phase: "single",
      reason: "vision model returned an empty description",
      visionModel: "litellm/kimi-k2.7",
      imageHashes: ["deadbeef"],
      imageCount: 1,
    });
    expect(existsSync(getErrorLogPath())).toBe(true);
    const raw = readFileSync(getErrorLogPath(), "utf8");
    const parsed = JSON.parse(raw) as VisionErrorLogEntry;
    expect(parsed.phase).toBe("single");
    expect(parsed.reason).toBe("vision model returned an empty description");
    expect(parsed.visionModel).toBe("litellm/kimi-k2.7");
    expect(parsed.imageHashes).toEqual(["deadbeef"]);
    expect(parsed.imageCount).toBe(1);
    expect(parsed.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/);
  });

  it("appends one line per call (does not overwrite)", () => {
    appendVisionError({ phase: "batch", reason: "first", visionModel: null, imageHashes: [], imageCount: 0 });
    appendVisionError({ phase: "warn", reason: "unknown error", visionModel: null, imageHashes: ["h"], imageCount: 1 });
    const lines = readFileSync(getErrorLogPath(), "utf8").trimEnd().split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]).reason).toBe("first");
    expect(JSON.parse(lines[1]).reason).toBe("unknown error");
  });

  it("never throws when the log directory cannot be created (best-effort)", () => {
    // Make the configured agent dir a FILE, so creating logs/ under it fails
    // (mkdirSync recursive hits ENOTDIR). Logging must swallow this rather than
    // break a describer turn.
    rmSync(dir, { recursive: true, force: true });
    writeFileSync(dir, "i am a file, not a dir");
    expect(() =>
      appendVisionError({ phase: "batch", reason: "x", visionModel: null, imageHashes: [], imageCount: 0 }),
    ).not.toThrow();
  });
});

describe("rotateLogIfNeeded", () => {
  beforeEach(() => {
    // These tests pre-populate the log file directly (bypassing appendVisionError,
    // which would create the dir itself), so ensure the dir exists first.
    mkdirSync(getErrorLogDir(), { recursive: true });
  });

  it("is a no-op when the file is under the cap", () => {
    const path = getErrorLogPath();
    appendFileSync(path, "small\n");
    rotateLogIfNeeded(path, MAX_LOG_BYTES);
    expect(existsSync(path)).toBe(true);
    expect(existsSync(path + ".1")).toBe(false);
  });

  it("is a no-op when the file does not exist", () => {
    const path = getErrorLogPath();
    expect(() => rotateLogIfNeeded(path, MAX_LOG_BYTES)).not.toThrow();
    expect(existsSync(path)).toBe(false);
  });

  it("rotates the active log to .1 once it reaches the cap (overwriting a stale backup)", () => {
    const path = getErrorLogPath();
    const backup = path + ".1";
    // Pre-existing backup must be overwritten, not preserved alongside.
    appendFileSync(backup, "stale backup\n");
    const cap = 64;
    appendFileSync(path, "x".repeat(cap));
    expect(statSync(path).size).toBeGreaterThanOrEqual(cap);
    rotateLogIfNeeded(path, cap);
    // Old active content is now the backup; active file is gone (cleared for a
    // fresh log — the next appendVisionError recreates it).
    expect(existsSync(path)).toBe(false);
    expect(existsSync(backup)).toBe(true);
    expect(readFileSync(backup, "utf8")).toBe("x".repeat(cap));
  });

  it("appendVisionError rotates before writing once the active log exceeds the cap", () => {
    const path = getErrorLogPath();
    const backup = path + ".1";
    // Pre-fill the active log PAST the real cap so appendVisionError's internal
    // rotateLogIfNeeded() moves it to .1 before writing the new entry into a
    // fresh active log.
    appendFileSync(path, "x".repeat(MAX_LOG_BYTES + 1));
    appendVisionError({
      phase: "batch",
      reason: "after-rotate",
      visionModel: null,
      imageHashes: [],
      imageCount: 0,
    });
    expect(existsSync(backup)).toBe(true);
    expect(readFileSync(backup, "utf8")).toBe("x".repeat(MAX_LOG_BYTES + 1));
    const active = readFileSync(path, "utf8");
    expect(JSON.parse(active).reason).toBe("after-rotate");
  });
});
