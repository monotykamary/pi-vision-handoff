import { describe, it, expect, beforeEach, vi } from "vitest";
import type { AssistantMessage } from "@earendil-works/pi-ai";

// Mock complete() so the describer's stopReason handling is observable without
// a provider. The mock is reconfigured per-test via mockComplete.mockResolvedValueOnce.
const complete = vi.fn();
vi.mock("@earendil-works/pi-ai/compat", () => ({
  complete: (...args: unknown[]) => complete(...(args as Parameters<typeof complete>)),
}));

import { runBatch, describeSingle, resolveMaxTokens, type DescriberDeps } from "../../src/describer.js";
import { DESCRIPTION_TRUNCATED_MARKER, type ExtractedImage, type VisionHandoffConfig } from "../../src/index.js";
import { imageHash } from "../../src/image.js";

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
  beforeEach(() => complete.mockReset());

  it("returns the raw text on a clean stop", async () => {
    complete.mockResolvedValueOnce(fakeResponse({ text: "full description", stopReason: "stop" }));
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    expect(out).toBe("full description");
  });

  it("appends the truncation marker on stopReason length (not silent)", async () => {
    complete.mockResolvedValueOnce(
      fakeResponse({ text: "partial description cut off mid-sente", stopReason: "length" }),
    );
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    expect(out).toBe(`partial description cut off mid-sente${DESCRIPTION_TRUNCATED_MARKER}`);
  });

  it("returns null (no marker) on an aborted/error stop, surfacing the error", async () => {
    const deps = makeDeps();
    complete.mockResolvedValueOnce(fakeResponse({ stopReason: "error", errorMessage: "boom" }));
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, deps);
    expect(out).toBeNull();
    expect(deps.setLastError).toHaveBeenCalledWith(expect.stringContaining("boom"));
  });

  it("returns null for an empty text response and surfaces an empty-description error", async () => {
    const deps = makeDeps();
    complete.mockResolvedValueOnce(fakeResponse({ text: "", stopReason: "stop" }));
    const out = await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, deps);
    expect(out).toBeNull();
    expect(deps.setLastError).toHaveBeenCalledWith("vision model returned an empty description");
  });

  it("passes the vision model's declared maxTokens when cfg.maxTokens is unset", async () => {
    complete.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, cfg, makeDeps());
    const opts = complete.mock.calls[0][2];
    expect(opts.maxTokens).toBe(4096); // visionModel.maxTokens, not undefined
  });

  it("passes a configured cfg.maxTokens over the model's declared max", async () => {
    complete.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    const capped = { ...cfg, maxTokens: 512 };
    await describeSingle(img("AAA"), "", visionModel, modelRegistry, capped, makeDeps());
    const opts = complete.mock.calls[0][2];
    expect(opts.maxTokens).toBe(512);
  });
});

describe("thinking (reasoning) passthrough", () => {
  const reasoningModel = { ...visionModel, reasoning: true } as any;
  const nonReasoningModel = { ...visionModel, reasoning: false } as any;

  beforeEach(() => complete.mockReset());

  it("omits reasoning when thinking is disabled in config", async () => {
    complete.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(img("AAA"), "", reasoningModel, modelRegistry, { ...cfg, thinking: false }, makeDeps());
    expect(complete.mock.calls[0][2].reasoning).toBeUndefined();
  });

  it("passes the configured thinkingLevel when thinking is on and the model reasons", async () => {
    complete.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(
      img("AAA"),
      "",
      reasoningModel,
      modelRegistry,
      { ...cfg, thinking: true, thinkingLevel: "high" },
      makeDeps(),
    );
    expect(complete.mock.calls[0][2].reasoning).toBe("high");
  });

  it("omits reasoning when thinking is on but the model has no reasoning support", async () => {
    complete.mockResolvedValueOnce(fakeResponse({ text: "ok", stopReason: "stop" }));
    await describeSingle(
      img("AAA"),
      "",
      nonReasoningModel,
      modelRegistry,
      { ...cfg, thinking: true, thinkingLevel: "high" },
      makeDeps(),
    );
    expect(complete.mock.calls[0][2].reasoning).toBeUndefined();
  });
});

describe("resolveMaxTokens", () => {
  const baseCfg: VisionHandoffConfig = {
    enabled: true,
    visionModel: "p/id",
    autoHandoff: true,
    handoffModels: [],
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
  beforeEach(() => complete.mockReset());

  it("marks only the LAST parsed image on stopReason length (the one cut off mid-stream)", async () => {
    const a = img("AAA");
    const b = img("BBB");
    const c = img("CCC");
    const ha = imageHash(a.mimeType, a.data);
    const hb = imageHash(b.mimeType, b.data);
    const hc = imageHash(c.mimeType, c.data);
    complete.mockResolvedValueOnce(
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
    complete.mockResolvedValueOnce(
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
    complete.mockResolvedValueOnce(fakeResponse({ stopReason: "aborted" }));
    const out = await runBatch([{ img: a, hash: ha }], "", visionModel, modelRegistry, cfg, deps);
    expect(out.size).toBe(0);
    expect(deps.setLastError).toHaveBeenCalled();
  });
});
