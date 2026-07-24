import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Mock completeSimple() so the describer's stopReason handling is observable
// without a provider. The describer routes through completeSimple (not
// complete): complete()/stream() silently drops the `reasoning` ThinkingLevel,
// so only completeSimple()/streamSimple() translate it to provider-specific
// reasoningEffort. `complete` is mocked to throw so any accidental revert
// fails loudly instead of silently dropping thinking. The mocks are reconfigured
// per-test via completeSimple.mockResolvedValueOnce.
const completeSimple = vi.fn();
const complete = vi.fn(() => {
  throw new Error("describer must call completeSimple, not complete");
});
vi.mock("@earendil-works/pi-ai/compat", () => ({
  completeSimple: (...args: unknown[]) => completeSimple(...(args as Parameters<typeof completeSimple>)),
  complete: (...args: unknown[]) => complete(...(args as Parameters<typeof complete>)),
}));

import { runBatch, describeSingle, resolveMaxTokens, type DescriberDeps } from "../../src/describer.js";
import { DESCRIPTION_TRUNCATED_MARKER, type ExtractedImage, type VisionHandoffConfig } from "../../src/index.js";
import { imageHash } from "../../src/image.js";
import { getErrorLogPath, type VisionErrorLogEntry } from "../../src/error-log.js";

// The describer now appends a structured error-log entry on every failure
// (src/error-log.ts → ~/.pi/agent/logs/pi-vision-handoff/errors.log). Redirect
// $PI_CODING_AGENT_DIR to a temp dir for every test so the suite never writes
// to the user's real log, and so the wiring can be asserted by reading it back.
let logDir: string;
let savedEnv: string | undefined;
beforeEach(() => {
  logDir = mkdtempSync(join(tmpdir(), "pi-vh-desc-"));
  savedEnv = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = logDir;
});
afterEach(() => {
  if (savedEnv === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = savedEnv;
  rmSync(logDir, { recursive: true, force: true });
});

/** Read+parse the error log written this test (one entry per line). */
function readLog(): VisionErrorLogEntry[] {
  if (!existsSync(getErrorLogPath())) return [];
  return readFileSync(getErrorLogPath(), "utf8")
    .trimEnd()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as VisionErrorLogEntry);
}

const img = (data: string, mimeType = "image/png"): ExtractedImage => ({ data, mimeType });

function fakeResponse(opts: Partial<AssistantMessage> & { text?: string }): AssistantMessage {
  const { text = "a description", stopReason = "stop", ...rest } = opts;
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    usage: { input: 0, output: 0, totalTokens: 0 },
    model: "p/id",
    provider: "p",
    stopReason,
    ...rest,
  } as unknown as AssistantMessage;
}

const cfg: VisionHandoffConfig = {
  enabled: true,
  visionModel: "p/id",
  autoHandoff: true,
  handoffModels: [],
  prewarmPastedImages: false,
  asyncClipboardHandoff: false,
  maxTokens: undefined,
  cacheMax: 50,
  maxDescriptionLines: 0,
  thinking: false,
  thinkingLevel: "medium",
};

const modelRegistry = {
  getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: {} }),
} as any;

const visionModel = { provider: "p", id: "id", maxTokens: 4096, contextWindow: 131072 } as any;

function makeDeps(): DescriberDeps {
  return {
    reportUsage: vi.fn(),
    setLastError: vi.fn(),
  };
}

describe("describeSingle stopReason handling", () => {
  beforeEach(() => completeSimple.mockReset());

  it("returns the raw text on a clean stop", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "full description", stopReason: "stop" }));
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    expect(out).toBe("full description");
  });

  it("appends the truncation marker on stopReason length (not silent)", async () => {
    completeSimple.mockResolvedValueOnce(
      fakeResponse({ text: "partial description cut off mid-sente", stopReason: "length" }),
    );
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    expect(out).toBe(`partial description cut off mid-sente${DESCRIPTION_TRUNCATED_MARKER}`);
  });

  it("returns null (no marker) on an aborted/error stop, surfacing the error", async () => {
    const deps = makeDeps();
    completeSimple.mockResolvedValueOnce(fakeResponse({ stopReason: "error", errorMessage: "boom" }));
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, deps);
    expect(out).toBeNull();
    expect(deps.setLastError).toHaveBeenCalledWith(expect.stringContaining("boom"));
  });

  it("returns null for an empty text response and surfaces an empty-description error", async () => {
    const deps = makeDeps();
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "", stopReason: "stop" }));
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, deps);
    expect(out).toBeNull();
    expect(deps.setLastError).toHaveBeenCalledWith("vision model returned an empty description");
  });

  it("passes the vision model's declared maxTokens when cfg.maxTokens is unset", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    const opts = completeSimple.mock.calls[0][2];
    expect(opts.maxTokens).toBe(4096); // visionModel.maxTokens, not undefined
  });

  it("passes a configured cfg.maxTokens over the model's declared max", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    const capped = { ...cfg, maxTokens: 512 };
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, capped, makeDeps());
    const opts = completeSimple.mock.calls[0][2];
    expect(opts.maxTokens).toBe(512);
  });
});

describe("thinking (reasoning) passthrough", () => {
  const reasoningModel = { ...visionModel, reasoning: true } as any;
  const nonReasoningModel = { ...visionModel, reasoning: false } as any;

  beforeEach(() => completeSimple.mockReset());

  it("omits reasoning when thinking is disabled in config", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(img("AAA"), "", reasoningModel, modelRegistry, { ...cfg, thinking: false }, makeDeps());
    expect(completeSimple.mock.calls[0][2].reasoning).toBeUndefined();
  });

  it("passes the configured thinkingLevel when thinking is on and the model reasons", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(
      img("AAA"),
      "",
      reasoningModel,
      modelRegistry,
      { ...cfg, thinking: true, thinkingLevel: "high" },
      makeDeps(),
    );
    expect(completeSimple.mock.calls[0][2].reasoning).toBe("high");
  });

  it("omits reasoning when thinking is on but the model has no reasoning support", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(
      img("AAA"),
      "",
      nonReasoningModel,
      modelRegistry,
      { ...cfg, thinking: true, thinkingLevel: "high" },
      makeDeps(),
    );
    expect(completeSimple.mock.calls[0][2].reasoning).toBeUndefined();
  });
});

describe("provider routing", () => {
  beforeEach(() => {
    completeSimple.mockReset();
    complete.mockReset();
  });

  it("uses a matching extension provider stream instead of the global compatibility registry", async () => {
    const response = fakeResponse({ text: "custom provider description", stopReason: "stop" });
    const result = vi.fn().mockResolvedValue(response);
    const streamSimple = vi.fn((_model: unknown, _context: unknown, _options: unknown) => ({ result }));
    const customRegistry = {
      ...modelRegistry,
      getRegisteredProviderConfig: vi.fn(() => ({ api: "custom-api", streamSimple })),
    } as any;
    const customModel = { ...visionModel, api: "custom-api" } as any;

    const out = await describeSingle(img("AAA"), "", customModel, customRegistry, cfg, makeDeps());

    expect(out).toBe("custom provider description");
    expect(streamSimple).toHaveBeenCalledTimes(1);
    expect(streamSimple.mock.calls[0][0]).toBe(customModel);
    expect(streamSimple.mock.calls[0][2]).toMatchObject({ apiKey: "k", maxTokens: 4096 });
    expect(result).toHaveBeenCalledTimes(1);
    expect(completeSimple).not.toHaveBeenCalled();
  });

  it("falls back to completeSimple when no matching extension stream is registered", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "built-in description", stopReason: "stop" }));
    const registry = {
      ...modelRegistry,
      getRegisteredProviderConfig: vi.fn(() => undefined),
    } as any;

    const out = await describeSingle(img("AAA"), "", visionModel, registry, cfg, makeDeps());

    expect(out).toBe("built-in description");
    expect(completeSimple).toHaveBeenCalledTimes(1);
  });
});

describe("routes through completeSimple (not complete)", () => {
  // Regression guard: the describer used to call complete(), which silently
  // drops the `reasoning` ThinkingLevel — only completeSimple()/streamSimple()
  // translate it into provider-specific reasoningEffort. If the describer ever
  // calls complete(), the throwing mock fails the test with a clear message.
  beforeEach(() => {
    completeSimple.mockReset();
    complete.mockReset();
    complete.mockImplementation(() => {
      throw new Error("describer must call completeSimple, not complete");
    });
  });

  it("describeSingle calls completeSimple and never complete", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    expect(completeSimple).toHaveBeenCalledTimes(1);
    expect(complete).not.toHaveBeenCalled();
  });

  it("runBatch calls completeSimple and never complete", async () => {
    const a = img("AAA");
    const ha = imageHash(a.mimeType, a.data);
    completeSimple.mockResolvedValueOnce(
      fakeResponse({ text: ["<<<IMAGE 1>>>", "one", "<<<END>>>"].join("\n"), stopReason: "stop" }),
    );
    await runBatch([{ img: a, hash: ha }], "", visionModel, modelRegistry, cfg, makeDeps());
    expect(completeSimple).toHaveBeenCalledTimes(1);
    expect(complete).not.toHaveBeenCalled();
  });
});

describe("resolveMaxTokens", () => {
  const baseCfg: VisionHandoffConfig = {
    enabled: true,
    visionModel: "p/id",
    autoHandoff: true,
    handoffModels: [],
    prewarmPastedImages: false,
    asyncClipboardHandoff: false,
    maxTokens: undefined,
    cacheMax: 50,
    maxDescriptionLines: 0,
    thinking: false,
    thinkingLevel: "medium",
  };
  const model = (maxTokens: number, contextWindow: number) =>
    ({ provider: "p", id: "id", maxTokens, contextWindow }) as any;
  const cfg = (maxTokens?: number): VisionHandoffConfig => ({ ...baseCfg, maxTokens });

  it("uses the model's declared max when cfg.maxTokens is unset and it fits the window", () => {
    expect(resolveMaxTokens(cfg(undefined), model(4096, 131072))).toBe(4096);
  });

  it("clamps a model whose maxTokens equals its full contextWindow (the lilac/kimi 400 bug)", () => {
    // model declares maxTokens == contextWindow == 262144; providers reject
    // input + maxTokens > contextWindow. We clamp to contextWindow - reserve.
    const out = resolveMaxTokens(cfg(undefined), model(262144, 262144));
    expect(out).toBe(262144 - 8192);
    expect(out).toBeLessThan(262144);
  });

  it("a configured cfg.maxTokens wins and is also clamped to the window", () => {
    expect(resolveMaxTokens(cfg(512), model(4096, 131072))).toBe(512);
    // configured cap exceeds window-reserve → clamped down
    expect(resolveMaxTokens(cfg(200000), model(262144, 131072))).toBe(131072 - 8192);
  });

  it("returns undefined when neither cfg nor the model declares a usable max", () => {
    expect(resolveMaxTokens(cfg(undefined), model(0, 8192))).toBeUndefined();
    expect(resolveMaxTokens(cfg(undefined), model(-1, 8192))).toBeUndefined();
  });

  it("never returns a value below 1 (even if the reserve exceeds the window)", () => {
    // tiny contextWindow smaller than the reserve: clamp floors at 1 so we
    // still request SOMETHING rather than 0 / negative.
    expect(resolveMaxTokens(cfg(4096), model(4096, 1024))).toBe(1);
  });
});

describe("runBatch stopReason handling", () => {
  beforeEach(() => completeSimple.mockReset());

  it("marks only the LAST parsed image on stopReason length (the one cut off mid-stream)", async () => {
    const a = img("AAA");
    const b = img("BBB");
    const c = img("CCC");
    const ha = imageHash(a.mimeType, a.data);
    const hb = imageHash(b.mimeType, b.data);
    const hc = imageHash(c.mimeType, c.data);
    completeSimple.mockResolvedValueOnce(
      fakeResponse({
        text: [
          "<<<IMAGE 1>>>",
          "desc one",
          "<<<END>>>",
          "<<<IMAGE 2>>>",
          "desc two",
          "<<<END>>>",
          "<<<IMAGE 3>>>",
          "desc three partial cut",
        ].join("\n"),
        stopReason: "length",
      }),
    );
    const out = await runBatch(
      [
        { img: a, hash: ha },
        { img: b, hash: hb },
        { img: c, hash: hc },
      ],
      "",
      visionModel,
      modelRegistry,
      cfg,
      makeDeps(),
    );
    expect(out.get(ha)).toBe("desc one");
    expect(out.get(hb)).toBe("desc two");
    // last parsed (highest index) is the truncated one
    expect(out.get(hc)).toBe(`desc three partial cut${DESCRIPTION_TRUNCATED_MARKER}`);
  });

  it("does not append a marker on a clean batch stop", async () => {
    const a = img("AAA");
    const b = img("BBB");
    const ha = imageHash(a.mimeType, a.data);
    const hb = imageHash(b.mimeType, b.data);
    completeSimple.mockResolvedValueOnce(
      fakeResponse({
        text: ["<<<IMAGE 1>>>", "one", "<<<END>>>", "<<<IMAGE 2>>>", "two", "<<<END>>>"].join("\n"),
        stopReason: "stop",
      }),
    );
    const out = await runBatch([{ img: a, hash: ha }, { img: b, hash: hb }], "", visionModel, modelRegistry, cfg, makeDeps());
    expect(out.get(ha)).toBe("one");
    expect(out.get(hb)).toBe("two");
    expect(out.get(hb)).not.toContain(DESCRIPTION_TRUNCATED_MARKER);
  });

  it("returns an empty map and surfaces the error on an aborted batch", async () => {
    const deps = makeDeps();
    const a = img("AAA");
    const ha = imageHash(a.mimeType, a.data);
    completeSimple.mockResolvedValueOnce(fakeResponse({ stopReason: "aborted" }));
    const out = await runBatch([{ img: a, hash: ha }], "", visionModel, modelRegistry, cfg, deps);
    expect(out.size).toBe(0);
    expect(deps.setLastError).toHaveBeenCalled();
  });
});

describe("error logging (src/error-log.ts)", () => {
  beforeEach(() => completeSimple.mockReset());

  it("logs a batch entry on a stopReason error", async () => {
    const a = img("AAA");
    const ha = imageHash(a.mimeType, a.data);
    completeSimple.mockResolvedValueOnce(fakeResponse({ stopReason: "error", errorMessage: "boom" }));
    await runBatch([{ img: a, hash: ha }], "", visionModel, modelRegistry, cfg, makeDeps());
    const entries = readLog();
    expect(entries).toHaveLength(1);
    expect(entries[0].phase).toBe("batch");
    expect(entries[0].stopReason).toBe("error");
    expect(entries[0].reason).toContain("error");
    expect(entries[0].reason).toContain("boom");
    expect(entries[0].visionModel).toBe("p/id");
    expect(entries[0].imageHashes).toEqual([ha]);
    expect(entries[0].imageCount).toBe(1);
    expect(entries[0].config?.thinking).toBe(false);
  });

  it("logs a single entry on an empty description", async () => {
    completeSimple.mockResolvedValueOnce(fakeResponse({ text: "", stopReason: "stop" }));
    const a = img("AAA");
    await describeSingle(a, "", visionModel, modelRegistry, cfg, makeDeps());
    const entries = readLog();
    expect(entries).toHaveLength(1);
    expect(entries[0].phase).toBe("single");
    expect(entries[0].reason).toBe("vision model returned an empty description");
    expect(entries[0].imageHashes).toEqual([imageHash(a.mimeType, a.data)]);
  });

  it("logs a single entry with a stack on a thrown error", async () => {
    completeSimple.mockRejectedValueOnce(new Error("network down"));
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    const entries = readLog();
    expect(entries).toHaveLength(1);
    expect(entries[0].phase).toBe("single");
    expect(entries[0].reason).toBe("network down");
    expect(entries[0].errorStack).toContain("network down");
    expect(entries[0].timedOut).toBeFalsy();
  });

  it("logs an auth failure (no API key) with the provider error", async () => {
    const failingRegistry = {
      getApiKeyAndHeaders: async () => ({ ok: false, error: "missing API key" }),
    } as any;
    const a = img("AAA");
    const ha = imageHash(a.mimeType, a.data);
    await runBatch([{ img: a, hash: ha }], "", visionModel, failingRegistry, cfg, makeDeps());
    const entries = readLog();
    expect(entries).toHaveLength(1);
    expect(entries[0].phase).toBe("batch");
    expect(entries[0].reason).toBe("missing API key");
    expect(entries[0].imageHashes).toEqual([ha]);
  });

  it("does not log a deliberate user abort (a cancel isn't a troubleshooting error)", async () => {
    completeSimple.mockRejectedValueOnce(new Error("aborted"));
    const ac = new AbortController();
    ac.abort();
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps(), ac.signal);
    expect(readLog()).toHaveLength(0);
  });
});
