/**
 * Describer usage + energy capture for pi-vision-handoff.
 *
 * One record is produced per REAL describer provider call (cache hits emit
 * nothing):
 *   - model + tokens: from completeSimple()'s AssistantMessage.usage
 *   - energy + cost + raw MCR/energy/cost payloads: from Neuralwatt SSE comment
 *     lines parsed out of the teed response body (readEnergyFromTee). Present
 *     ONLY when the vision model is a Neuralwatt model — non-Neuralwatt models
 *     produce no comment lines, so the energy fields are OMITTED (not zeroed)
 *     for easy downstream filtering ("no energy" vs "zero energy").
 *
 * The caller (vision-handoff.ts) persists the record via pi.appendEntry (replays
 * on session resume/branch) AND emits it on pi.events so a live consumer can
 * filter on the one channel for tokens AND energy.
 *
 * Split out of vision-handoff.ts so the pure pieces (readEnergyFromTee,
 * buildUsageRecord) and the concurrency-safe fetch interceptor are unit-testable
 * through the normal src/ import path — vision-handoff.ts runs readConfig() at
 * module load and pulls in the TUI selector, so it is not unit-test-friendly.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import type { Api, AssistantMessage, Model, Usage } from "@earendil-works/pi-ai";

/** Custom session-entry type persisted via pi.appendEntry. */
export const USAGE_ENTRY_TYPE = "vision-handoff-usage";

/** Event-bus channel emitted via pi.events. */
export const USAGE_EVENT_CHANNEL = "vision-handoff:usage";

/** Parsed Neuralwatt SSE-comment energy/cost/MCR data for one describer call. */
export interface VisionHandoffEnergyCapture {
  energyJoules: number;
  costUsd: number;
  energyRaw: Record<string, unknown> | null;
  mcrSessionRaw: Record<string, unknown> | null;
  costRaw: Record<string, unknown> | null;
}

/** Shared empty capture. Safe to share: readEnergyFromTee returns a fresh object
 *  and never mutates this; buildUsageRecord only reads from its argument. */
export const EMPTY_ENERGY_CAPTURE: VisionHandoffEnergyCapture = {
  energyJoules: 0,
  costUsd: 0,
  energyRaw: null,
  mcrSessionRaw: null,
  costRaw: null,
};

/** A single describer-call usage record. Energy fields are present only when
 *  Neuralwatt SSE energy comments were captured (omitted, not zeroed, otherwise).
 *
 *  One record is emitted per REAL describer provider call (cache hits emit
 *  nothing). A batched call that describes several images at once still emits a
 *  single record: `imageHash` is the representative (first) member and
 *  `imageHashes` lists every image the call covered, so consumers can attribute
 *  the call's tokens/energy to each member image without double-counting. */
export interface VisionHandoffUsageRecord {
  /** Representative image hash (first member of the batch). */
  imageHash: string;
  /** All image hashes covered by this describer call. Present only for batched
   *  calls (length > 1); omitted for single-image calls. */
  imageHashes?: string[];
  model: string;
  provider: string;
  responseModel?: string;
  responseId?: string;
  usage: Usage;
  /** Neuralwatt energy — present only when SSE energy comments were captured. */
  energyJoules?: number;
  costUsd?: number;
  energyRaw?: Record<string, unknown> | null;
  mcrSessionRaw?: Record<string, unknown> | null;
  costRaw?: Record<string, unknown> | null;
}

/**
 * Parse Neuralwatt SSE comment lines (`: energy`, `: cost`, `: mcr-session`)
 * from a teed response body into a fresh capture object. Mirrors
 * pi-neuralwatt-provider's readEnergyFromTee but returns a result instead of
 * mutating module state — vision-handoff describes images concurrently (the
 * before_agent_start warm-up fires several in parallel), so each call needs its
 * own capture routed via {@link describeAls}. For non-Neuralwatt vision models
 * no comment lines are present and the returned capture stays empty.
 */
export async function readEnergyFromTee(
  body: ReadableStream<Uint8Array>,
): Promise<VisionHandoffEnergyCapture> {
  const result: VisionHandoffEnergyCapture = { ...EMPTY_ENERGY_CAPTURE };
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function processLine(line: string): void {
    const trimmed = line.trim();
    if (trimmed.startsWith(": energy ")) {
      try {
        const energy = JSON.parse(trimmed.slice(9));
        result.energyJoules += energy.energy_joules || 0;
        result.energyRaw = energy;
      } catch {
        // malformed energy comment — ignore
      }
    } else if (trimmed.startsWith(": mcr-session ")) {
      try {
        result.mcrSessionRaw = JSON.parse(trimmed.slice(14));
      } catch {
        // malformed mcr-session comment — ignore
      }
    } else if (trimmed.startsWith(": cost ")) {
      try {
        const cost = JSON.parse(trimmed.slice(7));
        result.costUsd += cost.request_cost_usd || 0;
        result.costRaw = cost;
      } catch {
        // malformed cost comment — ignore
      }
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) processLine(line);
    }
  } catch {
    // tee stream may error if the main stream is aborted — that's fine
  }

  const final = decoder.decode(new Uint8Array(0), { stream: false });
  const remaining = (buffer + final).trim();
  if (remaining) processLine(remaining);

  try {
    reader.releaseLock();
  } catch {
    // ignore
  }
  return result;
}

/**
 * Build a usage record from a describer response + energy capture. Returns
 * null when there is nothing meaningful to report (e.g. a provider-level
 * failure with zero tokens and no energy) so the caller can skip emitting.
 *
 * Energy fields are OMITTED entirely (not zeroed) when no Neuralwatt SSE energy
 * comments were captured, so consumers can distinguish "no energy" from
 * "zero energy".
 */
export function buildUsageRecord(
  response: AssistantMessage,
  capture: VisionHandoffEnergyCapture,
  visionModel: Model<Api>,
  imageHash: string,
  imageHashes?: string[],
): VisionHandoffUsageRecord | null {
  const hasEnergy = !!(
    capture.energyRaw ||
    capture.costRaw ||
    capture.mcrSessionRaw ||
    capture.energyJoules > 0 ||
    capture.costUsd > 0
  );
  const hasUsage =
    !!response.usage &&
    (response.usage.totalTokens > 0 || response.usage.input > 0 || response.usage.output > 0);
  if (!hasUsage && !hasEnergy) return null;

  const record: VisionHandoffUsageRecord = {
    imageHash,
    model: response.model || visionModel.id,
    provider: response.provider || visionModel.provider,
    responseModel: response.responseModel,
    responseId: response.responseId,
    usage: response.usage,
  };
  // Present only for genuine batched calls (more than one image). Keeps the
  // single-image record shape unchanged for existing consumers.
  if (imageHashes && imageHashes.length > 1) {
    record.imageHashes = imageHashes;
  }
  if (hasEnergy) {
    record.energyJoules = capture.energyJoules;
    record.costUsd = capture.costUsd;
    record.energyRaw = capture.energyRaw;
    record.mcrSessionRaw = capture.mcrSessionRaw;
    record.costRaw = capture.costRaw;
  }
  return record;
}

// ── Concurrency-safe fetch interceptor ─────────────────────────────────────
//
// The only body-interception point for completeSimple() is globalThis.fetch (pi-ai's
// StreamOptions.onResponse exposes headers only, not the body where the
// `: energy` SSE comments live). before_agent_start fires several describeImage()
// calls fire-and-forget, so a naïve save/patch/restore of globalThis.fetch would
// clobber under concurrency. The fix is a refcounted shared interceptor
// (installed only while ≥1 describe is in flight, so non-describe fetches pass
// through unmodified) + AsyncLocalStorage to route each teed response body to
// the describe call that issued it. Nested patches from other extensions (e.g.
// pi-neuralwatt-provider's streamNeuralwatt, which also tees for its own energy
// display) chain on top and restore back to this interceptor, so both tees read
// the same comment lines independently.

/** Per-describer-call ALS slot carrying the energy tee reader. */
export interface DescribeContext {
  energyReader: Promise<VisionHandoffEnergyCapture> | undefined;
}

/** Routes each teed response body to the describe call that issued it. */
export const describeAls = new AsyncLocalStorage<DescribeContext>();

let fetchInterceptRefCount = 0;
let savedRealFetch: typeof globalThis.fetch | null = null;

/** Install the globalThis.fetch interceptor. Refcounted: the first caller
 *  patches fetch; later callers just bump the count. Idempotent per install. */
export function installFetchInterceptor(): void {
  if (fetchInterceptRefCount === 0) {
    savedRealFetch = globalThis.fetch;
    globalThis.fetch = interceptedFetch;
  }
  fetchInterceptRefCount++;
}

/** Remove the interceptor. Refcounted: only the last caller restores fetch. */
export function uninstallFetchInterceptor(): void {
  if (fetchInterceptRefCount > 0) fetchInterceptRefCount--;
  if (fetchInterceptRefCount === 0 && savedRealFetch) {
    globalThis.fetch = savedRealFetch;
    savedRealFetch = null;
  }
}

/** Current refcount — 0 means the interceptor is not installed. Test hook. */
export function fetchInterceptorRefcount(): number {
  return fetchInterceptRefCount;
}

async function interceptedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const real = savedRealFetch ?? globalThis.fetch;
  const response = await real(input, init);
  const store = describeAls.getStore();
  // Outside a describer call (no ALS store) or for bodiless responses: pass
  // through untouched.
  if (!store || !response.body) return response;
  const [bodyForSdk, bodyForEnergy] = response.body.tee();
  store.energyReader = readEnergyFromTee(bodyForEnergy);
  return new Response(bodyForSdk, {
    headers: response.headers,
    status: response.status,
    statusText: response.statusText,
  });
}
