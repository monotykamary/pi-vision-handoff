import { describe, it, expect, beforeEach, vi } from "vitest";

// Mock the describer so the loader's batching is observable without a provider.
const runBatch = vi.fn(async (misses: { hash: string }[]) => {
  const out = new Map<string, string>();
  for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
  return out;
});
const describeSingle = vi.fn(async (img: { data: string }) => `single-${img.data}`);

vi.mock("../../src/describer.js", () => ({
  runBatch: (...args: unknown[]) => runBatch(...(args as Parameters<typeof runBatch>)),
  describeSingle: (...args: unknown[]) => describeSingle(...(args as Parameters<typeof describeSingle>)),
}));

import { DescriptionLoader, UNAVAILABLE, type LoaderDeps } from "../../src/dataloader.js";
import { imageHash } from "../../src/image.js";
import type { ExtractedImage, VisionHandoffConfig } from "../../src/index.js";

const img = (data: string, mimeType = "image/png"): ExtractedImage => ({ data, mimeType });
const hashOf = (e: ExtractedImage) => imageHash(e.mimeType, e.data);

function makeLoader(overrides: Partial<LoaderDeps> = {}): DescriptionLoader {
  const setLastError = vi.fn();
  const reportUsage = vi.fn();
  const cfg: VisionHandoffConfig = {
    enabled: true,
    visionModel: "p/id",
    autoHandoff: true,
    handoffModels: [],
    maxTokens: undefined,
    cacheMax: 50,
    maxDescriptionLines: 0,
  };
  const deps: LoaderDeps = {
    getConfig: () => cfg,
    resolveVisionModel: () => ({}) as any,
    reportUsage,
    setLastError,
    ...overrides,
  };
  return new DescriptionLoader(deps);
}

describe("DescriptionLoader batching", () => {
  beforeEach(() => {
    runBatch.mockClear();
    describeSingle.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) => {
      const out = new Map<string, string>();
      for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
      return out;
    });
  });

  it("coalesces two loads in the same frame into ONE runBatch call", async () => {
    const loader = makeLoader();
    loader.bindTurnContext({ modelRegistry: {} as any });
    const a = img("AAA");
    const b = img("BBB");

    const pa = loader.loadDescription(a);
    const pb = loader.loadDescription(b);
    const [da, db] = await Promise.all([pa, pb]);

    expect(runBatch).toHaveBeenCalledTimes(1);
    expect(runBatch.mock.calls[0][0]).toHaveLength(2);
    expect(da).toBe(`[Image: desc-for-${hashOf(a)}]`);
    expect(db).toBe(`[Image: desc-for-${hashOf(b)}]`);
  });

  it("memoizes: a repeat load in a later frame is a cache hit (no new call)", async () => {
    const loader = makeLoader();
    loader.bindTurnContext({ modelRegistry: {} as any });
    const a = img("AAA");

    await loader.loadDescription(a);
    const callsBefore = runBatch.mock.calls.length;
    await loader.loadDescription(a);
    expect(runBatch.mock.calls.length).toBe(callsBefore);
  });

  it("resolves UNAVAILABLE and evicts the cache on a failed batch, so the next turn re-attempts", async () => {
    const loader = makeLoader();
    loader.bindTurnContext({ modelRegistry: {} as any });
    const a = img("AAA");

    runBatch.mockResolvedValueOnce(new Map<string, string>());
    const first = await loader.loadDescription(a);
    expect(first).toBe(UNAVAILABLE);
    // failed → not cached: a fresh load triggers a new batch
    const second = await loader.loadDescription(a);
    expect(second).not.toBe(UNAVAILABLE);
    expect(runBatch.mock.calls.length).toBe(2);
  });

  it("clears the last error before each fresh attempt", async () => {
    const setLastError = vi.fn();
    const loader = makeLoader({ setLastError });
    loader.bindTurnContext({ modelRegistry: {} as any });
    await loader.loadDescription(img("AAA"));
    expect(setLastError).toHaveBeenCalledWith(null);
  });

  it("resolves UNAVAILABLE without a provider call when no vision model is resolved", async () => {
    const loader = makeLoader({ resolveVisionModel: () => null });
    loader.bindTurnContext({ modelRegistry: {} as any });
    const out = await loader.loadDescription(img("AAA"));
    expect(out).toBe(UNAVAILABLE);
    expect(runBatch).not.toHaveBeenCalled();
  });

  it("implements Disposable: [Symbol.dispose] resets in-flight batch + turn context", async () => {
    const loader = makeLoader();
    loader.bindTurnContext({ modelRegistry: {} as any });
    loader.setPendingTurnPrompt("hello");
    loader.loadDescription(img("AAA")); // schedule a dispatch, don't await

    using scoped = loader;
    scoped[Symbol.dispose]();
    // After dispose, a new load must still work (batch/turn were reset, but the
    // loader is reusable — rebind and load again).
    loader.bindTurnContext({ modelRegistry: {} as any });
    const out = await loader.loadDescription(img("BBB"));
    expect(out).not.toBe(UNAVAILABLE);
  });
});
