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

function makeLoader(
  overrides: Partial<LoaderDeps> = {},
  cfgOverrides: Partial<VisionHandoffConfig> = {},
): DescriptionLoader {
  const setLastError = vi.fn();
  const reportUsage = vi.fn();
  const cfg: VisionHandoffConfig = {
    enabled: true,
    visionModel: "p/id",
    autoHandoff: true,
    handoffModels: [],
    prewarmPastedImages: false,
    maxTokens: undefined,
    cacheMax: 50,
    maxDescriptionLines: 0,
    thinking: false,
    thinkingLevel: "medium",
    ...cfgOverrides,
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

  it("coalesces loads issued from separate event-loop callbacks in the same phase (cross-IO batching)", async () => {
    // Two setImmediate callbacks run in the same check phase. The loader's
    // dispatch is itself a setImmediate scheduled DURING that phase, so it
    // runs in the NEXT check phase — after both loads have landed in the
    // batch. Under process.nextTick, dispatch would drain between the two
    // callbacks and split them into two single-image calls.
    const loader = makeLoader();
    loader.bindTurnContext({ modelRegistry: {} as any });
    const a = img("AAA");
    const b = img("BBB");

    const [da, db] = await new Promise<[string, string]>((resolve) => {
      let pa!: Promise<string>;
      let pb!: Promise<string>;
      setImmediate(() => {
        pa = loader.loadDescription(a);
      });
      setImmediate(() => {
        pb = loader.loadDescription(b);
      });
      setImmediate(() => {
        Promise.all([pa, pb]).then(resolve);
      });
    });

    expect(runBatch).toHaveBeenCalledTimes(1);
    expect(da).toBe(`[Image: desc-for-${hashOf(a)}]`);
    expect(db).toBe(`[Image: desc-for-${hashOf(b)}]`);
  });

  it("resolves EVERY callback when a duplicate load pushes a second callback (no hang)", async () => {
    // cacheMax=1 forces a mid-frame eviction: load A, load B (evicts A), then
    // load A again. A's first cache entry is gone, so the third load can't
    // short-circuit on the cache and pushes a SECOND callback for A into the
    // same batch (callbacks.length=3, keys.length=2). Dispatch must resolve
    // all three by hash — a key-indexed loop would skip the duplicate and hang.
    const loader = makeLoader({}, { cacheMax: 1 });
    loader.bindTurnContext({ modelRegistry: {} as any });
    const a = img("AAA");
    const b = img("BBB");

    const pa = loader.loadDescription(a);
    const pb = loader.loadDescription(b);
    const pa2 = loader.loadDescription(a); // duplicate after A was evicted
    const [da, db, da2] = await Promise.all([pa, pb, pa2]);

    expect(runBatch).toHaveBeenCalledTimes(1);
    expect(runBatch.mock.calls[0][0]).toHaveLength(2); // A and B, deduped
    expect(da).toBe(`[Image: desc-for-${hashOf(a)}]`);
    expect(db).toBe(`[Image: desc-for-${hashOf(b)}]`);
    expect(da2).toBe(`[Image: desc-for-${hashOf(a)}]`); // duplicate resolved, not hung
  });
});
