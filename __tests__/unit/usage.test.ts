import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import {
  EMPTY_ENERGY_CAPTURE,
  buildUsageRecord,
  describeAls,
  fetchInterceptorRefcount,
  installFetchInterceptor,
  readEnergyFromTee,
  uninstallFetchInterceptor,
  type DescribeContext,
  type VisionHandoffEnergyCapture,
  type VisionHandoffUsageRecord,
} from "../../src/index.js";
import type { Api, AssistantMessage, Model, Usage } from "@earendil-works/pi-ai";

// ── helpers ──────────────────────────────────────────────────────────────────

/** Build a ReadableStream from SSE-shaped text lines (joined with "\n"). */
function sseBody(lines: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  const data = lines.join("\n") + "\n";
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(enc.encode(data));
      controller.close();
    },
  });
}

/** A ReadableStream whose bytes do NOT end with a newline (exercises the
 *  final-partial-line path in readEnergyFromTee). */
function sseBodyRaw(text: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(enc.encode(text));
      controller.close();
    },
  });
}

function sseResponse(lines: string[], status = 200): Response {
  return new Response(sseBody(lines), {
    status,
    headers: { "content-type": "text/event-stream" },
  });
}

function makeUsage(over: Partial<Usage> = {}): Usage {
  return {
    input: 100,
    output: 50,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 150,
    cost: { input: 0.001, output: 0.002, cacheRead: 0, cacheWrite: 0, total: 0.003 },
    ...over,
  };
}

function makeMessage(over: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text: "a description" }],
    api: "openai-completions",
    provider: "neuralwatt",
    model: "response-model",
    usage: makeUsage(),
    stopReason: "stop",
    timestamp: Date.now(),
    ...over,
  } as AssistantMessage;
}

function makeModel(over: Partial<Model<Api>> = {}): Model<Api> {
  return { id: "vision-model-id", provider: "vision-provider", ...over } as Model<Api>;
}

// ── readEnergyFromTee ─────────────────────────────────────────────────────────

describe("readEnergyFromTee", () => {
  it("parses energy, cost, and mcr-session comment lines", async () => {
    const capture = await readEnergyFromTee(
      sseBody([
        ': energy {"energy_joules": 12.5, "mcr": {"apc_hit_rate": 0.85}}',
        ': mcr-session {"session_fp": "fp123", "safe_drop_before": 7}',
        ': cost {"request_cost_usd": 0.0042}',
      ]),
    );
    expect(capture.energyJoules).toBe(12.5);
    expect(capture.energyRaw).toEqual({ energy_joules: 12.5, mcr: { apc_hit_rate: 0.85 } });
    expect(capture.mcrSessionRaw).toEqual({ session_fp: "fp123", safe_drop_before: 7 });
    expect(capture.costUsd).toBe(0.0042);
    expect(capture.costRaw).toEqual({ request_cost_usd: 0.0042 });
  });

  it("accumulates multiple energy comments (sums joules)", async () => {
    const capture = await readEnergyFromTee(
      sseBody([': energy {"energy_joules": 3}', ': energy {"energy_joules": 4.5}']),
    );
    expect(capture.energyJoules).toBe(7.5);
    // latest-wins for the raw payload (mirrors pi-neuralwatt-provider semantics)
    expect(capture.energyRaw).toEqual({ energy_joules: 4.5 });
  });

  it("returns an empty capture for a stream with no comment lines", async () => {
    const capture = await readEnergyFromTee(
      sseBody(['data: {"choices": []}', "data: [DONE]"]),
    );
    expect(capture).toEqual(EMPTY_ENERGY_CAPTURE);
    expect(capture.energyJoules).toBe(0);
    expect(capture.energyRaw).toBeNull();
  });

  it("returns an empty capture for an empty stream", async () => {
    const capture = await readEnergyFromTee(sseBody([]));
    expect(capture).toEqual(EMPTY_ENERGY_CAPTURE);
  });

  it("skips malformed comment lines without throwing", async () => {
    const capture = await readEnergyFromTee(
      sseBody([': energy {not json', ': cost {"request_cost_usd": 0.5}', ': mcr-session oops}']),
    );
    // malformed energy + mcr skipped; valid cost still captured
    expect(capture.energyJoules).toBe(0);
    expect(capture.energyRaw).toBeNull();
    expect(capture.mcrSessionRaw).toBeNull();
    expect(capture.costUsd).toBe(0.5);
  });

  it("processes a final partial line with no trailing newline", async () => {
    const capture = await readEnergyFromTee(sseBodyRaw(': energy {"energy_joules": 9}'));
    expect(capture.energyJoules).toBe(9);
    expect(capture.energyRaw).toEqual({ energy_joules: 9 });
  });

  it("returns a fresh object each call (does not share state)", async () => {
    const a = await readEnergyFromTee(sseBody([]));
    const b = await readEnergyFromTee(sseBody([]));
    expect(a).not.toBe(b);
    expect(a).toEqual(EMPTY_ENERGY_CAPTURE);
  });

  it("does not mutate EMPTY_ENERGY_CAPTURE", async () => {
    await readEnergyFromTee(sseBody([': energy {"energy_joules": 5}', ': cost {"request_cost_usd": 0.1}']));
    expect(EMPTY_ENERGY_CAPTURE.energyJoules).toBe(0);
    expect(EMPTY_ENERGY_CAPTURE.energyRaw).toBeNull();
    expect(EMPTY_ENERGY_CAPTURE.costUsd).toBe(0);
    expect(EMPTY_ENERGY_CAPTURE.costRaw).toBeNull();
  });
});

// ── buildUsageRecord ─────────────────────────────────────────────────────────

describe("buildUsageRecord", () => {
  it("carries model + tokens from the response, energy fields omitted when no energy", () => {
    const rec = buildUsageRecord(makeMessage(), EMPTY_ENERGY_CAPTURE, makeModel(), "hash-1");
    expect(rec).not.toBeNull();
    expect(rec!.imageHash).toBe("hash-1");
    expect(rec!.model).toBe("response-model");
    expect(rec!.provider).toBe("neuralwatt");
    expect(rec!.usage.totalTokens).toBe(150);
    expect(rec!.usage.input).toBe(100);
    expect(rec!.usage.output).toBe(50);
    // energy fields OMITTED (not zeroed) so consumers can filter "no energy"
    expect("energyJoules" in rec!).toBe(false);
    expect("costUsd" in rec!).toBe(false);
    expect("energyRaw" in rec!).toBe(false);
    expect("mcrSessionRaw" in rec!).toBe(false);
    expect("costRaw" in rec!).toBe(false);
  });

  it("reports energy when tokens are zero but SSE energy was captured", () => {
    const msg = makeMessage({
      usage: makeUsage({ input: 0, output: 0, totalTokens: 0 }),
    });
    const capture: VisionHandoffEnergyCapture = {
      ...EMPTY_ENERGY_CAPTURE,
      energyJoules: 12.5,
      energyRaw: { energy_joules: 12.5 },
    };
    const rec = buildUsageRecord(msg, capture, makeModel(), "h");
    expect(rec).not.toBeNull();
    expect(rec!.energyJoules).toBe(12.5);
    expect(rec!.energyRaw).toEqual({ energy_joules: 12.5 });
    // usage still present (the zero-token one)
    expect(rec!.usage.totalTokens).toBe(0);
  });

  it("reports tokens AND energy together when both are present", () => {
    const capture: VisionHandoffEnergyCapture = {
      ...EMPTY_ENERGY_CAPTURE,
      energyJoules: 9,
      costUsd: 0.004,
      energyRaw: { energy_joules: 9 },
      costRaw: { request_cost_usd: 0.004 },
      mcrSessionRaw: { session_fp: "fp", safe_drop_before: 3 },
    };
    const rec = buildUsageRecord(makeMessage(), capture, makeModel(), "h");
    expect(rec!.usage.totalTokens).toBe(150);
    expect(rec!.energyJoules).toBe(9);
    expect(rec!.costUsd).toBe(0.004);
    expect(rec!.mcrSessionRaw).toEqual({ session_fp: "fp", safe_drop_before: 3 });
    expect(rec!.costRaw).toEqual({ request_cost_usd: 0.004 });
  });

  it("returns null when there are no tokens AND no energy", () => {
    const msg = makeMessage({
      usage: makeUsage({
        input: 0,
        output: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      }),
    });
    expect(buildUsageRecord(msg, EMPTY_ENERGY_CAPTURE, makeModel(), "h")).toBeNull();
  });

  it("falls back to the vision model id/provider when the response omits them", () => {
    const rec = buildUsageRecord(
      makeMessage({ model: "", provider: "" }),
      EMPTY_ENERGY_CAPTURE,
      makeModel(),
      "h",
    );
    expect(rec!.model).toBe("vision-model-id");
    expect(rec!.provider).toBe("vision-provider");
  });

  it("passes responseModel and responseId through", () => {
    const rec = buildUsageRecord(
      makeMessage({ responseModel: "k2.6-actual", responseId: "rid-123" }),
      EMPTY_ENERGY_CAPTURE,
      makeModel(),
      "h",
    );
    expect(rec!.responseModel).toBe("k2.6-actual");
    expect(rec!.responseId).toBe("rid-123");
  });

  it("produces a record assignable to VisionHandoffUsageRecord", () => {
    const rec: VisionHandoffUsageRecord | null = buildUsageRecord(
      makeMessage(),
      EMPTY_ENERGY_CAPTURE,
      makeModel(),
      "h",
    );
    expect(rec).not.toBeNull();
  });

  it("omits imageHashes for a single-image call (unchanged shape)", () => {
    const rec = buildUsageRecord(makeMessage(), EMPTY_ENERGY_CAPTURE, makeModel(), "h");
    expect(rec!.imageHash).toBe("h");
    expect("imageHashes" in rec!).toBe(false);
  });

  it("lists imageHashes only for a genuine batch (length > 1), with the first as representative", () => {
    const rec = buildUsageRecord(
      makeMessage(),
      EMPTY_ENERGY_CAPTURE,
      makeModel(),
      "h1",
      ["h1", "h2", "h3"],
    );
    expect(rec!.imageHash).toBe("h1");
    expect(rec!.imageHashes).toEqual(["h1", "h2", "h3"]);
  });

  it("does not set imageHashes when a single-element array is passed", () => {
    const rec = buildUsageRecord(makeMessage(), EMPTY_ENERGY_CAPTURE, makeModel(), "h1", ["h1"]);
    expect(rec!.imageHash).toBe("h1");
    expect("imageHashes" in rec!).toBe(false);
  });
});

// ── fetch interceptor + ALS routing ──────────────────────────────────────────

describe("fetch interceptor + AsyncLocalStorage routing", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    // restore refcount to 0 (which restores globalThis.fetch) then belt-and-braces
    while (fetchInterceptorRefcount() > 0) uninstallFetchInterceptor();
    globalThis.fetch = originalFetch;
  });

  it("install patches globalThis.fetch and uninstall restores it (refcounted)", () => {
    const myFetch = (async () => new Response()) as typeof globalThis.fetch;
    globalThis.fetch = myFetch;

    expect(fetchInterceptorRefcount()).toBe(0);
    installFetchInterceptor();
    expect(fetchInterceptorRefcount()).toBe(1);
    expect(globalThis.fetch).not.toBe(myFetch);

    // nested install — still the same interceptor, refcount bumped
    installFetchInterceptor();
    expect(fetchInterceptorRefcount()).toBe(2);
    expect(globalThis.fetch).not.toBe(myFetch);

    // first uninstall does NOT restore (refcount still > 0)
    uninstallFetchInterceptor();
    expect(fetchInterceptorRefcount()).toBe(1);
    expect(globalThis.fetch).not.toBe(myFetch);

    // second uninstall restores the original
    uninstallFetchInterceptor();
    expect(fetchInterceptorRefcount()).toBe(0);
    expect(globalThis.fetch).toBe(myFetch);
  });

  it("tees the response body and stashes the reader on the active ALS store", async () => {
    const mockFetch = vi.fn(async () =>
      sseResponse([': energy {"energy_joules": 7.25}', ': cost {"request_cost_usd": 0.002}']),
    );
    globalThis.fetch = mockFetch as typeof globalThis.fetch;
    installFetchInterceptor();

    const store: DescribeContext = { energyReader: undefined };
    let sdkBody: ReadableStream<Uint8Array> | undefined;
    await describeAls.run(store, async () => {
      const res = await globalThis.fetch("https://describe.test/v1");
      sdkBody = res.body ?? undefined;
    });

    // energy reader got the parsed comment lines
    expect(store.energyReader).toBeDefined();
    const capture = await store.energyReader!;
    expect(capture.energyJoules).toBe(7.25);
    expect(capture.costUsd).toBe(0.002);

    // the SDK copy of the body is independently consumable (tee gives two branches)
    expect(sdkBody).toBeDefined();
    const text = await new Response(sdkBody!).text();
    expect(text).toContain(': energy {"energy_joules": 7.25}');
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("passes the original response through untouched when called outside a describer ALS store", async () => {
    const original = sseResponse([': energy {"energy_joules": 99}']);
    const mockFetch = vi.fn(async () => original);
    globalThis.fetch = mockFetch as typeof globalThis.fetch;
    installFetchInterceptor();

    // no describeAls.run — getStore() returns undefined, so no tee / no new Response
    const res = await globalThis.fetch("https://x");
    expect(res).toBe(original);
  });

  it("skips the tee for bodiless responses even inside an ALS store", async () => {
    const bodiless = new Response(null, { status: 204 });
    const mockFetch = vi.fn(async () => bodiless);
    globalThis.fetch = mockFetch as typeof globalThis.fetch;
    installFetchInterceptor();

    const store: DescribeContext = { energyReader: undefined };
    await describeAls.run(store, async () => {
      const res = await globalThis.fetch("https://x");
      expect(res).toBe(bodiless); // untouched, no new Response
    });
    expect(store.energyReader).toBeUndefined();
  });

  it("routes each concurrent fetch's teed body to its own ALS store (no cross-contamination)", async () => {
    const bodies: Record<string, string[]> = {
      "https://a": [': energy {"energy_joules": 1.5}', ': cost {"request_cost_usd": 0.01}'],
      "https://b": [': energy {"energy_joules": 88}'],
    };
    const mockFetch = vi.fn(async (input: RequestInfo | URL) =>
      sseResponse(bodies[String(input)] ?? ["nope"]),
    );
    globalThis.fetch = mockFetch as typeof globalThis.fetch;
    installFetchInterceptor();

    const storeA: DescribeContext = { energyReader: undefined };
    const storeB: DescribeContext = { energyReader: undefined };

    // fire both concurrently, each inside its own ALS context
    const pA = describeAls.run(storeA, async () => {
      await globalThis.fetch("https://a");
      return storeA.energyReader!;
    });
    const pB = describeAls.run(storeB, async () => {
      await globalThis.fetch("https://b");
      return storeB.energyReader!;
    });
    const [a, b] = await Promise.all([pA, pB]);

    expect(a.energyJoules).toBe(1.5);
    expect(a.costUsd).toBe(0.01);
    expect(b.energyJoules).toBe(88);
    expect(b.costUsd).toBe(0);
    expect(b.costRaw).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});
