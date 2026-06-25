/**
 * Facebook DataLoader pattern for image descriptions.
 *
 * `loadDescription(img)` returns a memoized Promise for the description and
 * pushes the image's key (hash) into the CURRENT batch. All `load()` calls in
 * the same execution frame (+ its microtask cascade) coalesce into ONE batch
 * object. Dispatch is scheduled via `setImmediate` (see `enqueuePostPromiseJob`),
 * so every load in the frame — and every load from a separate I/O callback in
 * the same poll iteration — lands in the single batch before the ONE vision
 * call fires. Each load()'s promise then resolves with its description.
 *
 * Mapping: the `read` tools are the `load()` callers. Their `tool_result`
 * handler awaits the shared batch — so N parallel reads coalesce into the SAME
 * single vision call and all resolve together, the descriptions landing in the
 * tool results BEFORE the agent's next turn. pi fires each read's `tool_result`
 * as that read's I/O completes (poll phase); the loader's `setImmediate`
 * dispatch defers to the check phase, AFTER the whole poll iteration, so reads
 * completing together (the common case for cached local files) land in ONE
 * batch — and reads completing in separate iterations get separate calls, but
 * always in parallel, never sequential. The agent's tool-result wait is free
 * time, so this adds zero latency to the critical path. `context` then sees
 * text-described tool results (no image blocks to swap); any remaining images
 * (user-attached, custom-injected) are cache hits.
 *
 * All mutable state (batch, cache, turn context) lives on the instance — no
 * module-level globals — and the class implements `Disposable` so a `using`
 * binding (or an explicit `reset()`) cleanly abandons an in-flight batch and
 * clears turn context, e.g. on session reset.
 */

import type { Api, Model } from "@earendil-works/pi-ai";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import {
  IMAGE_PLACEHOLDER_PREFIX,
  IMAGE_PLACEHOLDER_SUFFIX,
  wrapDescription,
  type ExtractedImage,
  type VisionHandoffConfig,
} from "./index.js";
import { imageHash } from "./image.js";
import { runBatch, describeSingle, type DescriberDeps } from "./describer.js";

/** Resolved when a description couldn't be obtained (graceful degradation).
 *  Failures are NOT cached, so the next turn re-attempts. */
export const UNAVAILABLE = `${IMAGE_PLACEHOLDER_PREFIX}description unavailable${IMAGE_PLACEHOLDER_SUFFIX}`;

interface DescriptionBatch {
  keys: string[];
  imgs: ExtractedImage[];
  /** One per `loadDescription()` call. A duplicate load (same hash, but its
   *  first cache entry was evicted mid-frame so it couldn't short-circuit on
   *  the cache) pushes a second callback for an existing key — so
   *  `callbacks.length` can exceed `keys.length`. `dispatchBatch` resolves by
   *  hash so every callback is reached. */
  callbacks: { hash: string; resolve: (v: string) => void; reject: (e: Error) => void }[];
}

/** Engine-provided resolver for the configured vision model. */
export interface VisionModelResolver {
  (registry: ModelRegistry, ref: string): Model<Api> | null;
}

/** Dependencies the loader can't own itself (held by the engine). */
export interface LoaderDeps extends DescriberDeps {
  /** Read the current config (reloaded on session_start / config writes). */
  getConfig(): VisionHandoffConfig;
  /** Resolve the configured vision model against a registry. */
  resolveVisionModel: VisionModelResolver;
}

/** Defer `fn` to the next check phase (`setImmediate`), so every
 *  `loadDescription()` — whether called from sync code, a microtask cascade,
 *  or a separate I/O callback in the same poll iteration — coalesces into one
 *  batch before the single vision call fires.
 *
 *  Why `setImmediate` and not DataLoader's classic `process.nextTick`:
 *  nextTick drains between I/O callbacks in the poll phase, so dispatch would
 *  fire after the first parallel `read`'s `tool_result` but before the
 *  second's — splitting N reads into N single-image calls. `setImmediate` runs
 *  in the check phase, AFTER the whole poll phase, so all `tool_result`
 *  handlers that fire in one poll iteration land in ONE batch. The check phase
 *  also runs after the microtask queue drains, so loads issued from a `.then`
 *  cascade (e.g. the clipboard pre-warm) still coalesce. */
function enqueuePostPromiseJob(fn: () => void): void {
  setImmediate(fn);
}

export class DescriptionLoader implements Disposable {
  private readonly cache = new Map<string, Promise<string>>();
  private batch: DescriptionBatch | null = null;
  private dispatchScheduled = false;
  private turnModelRegistry: ModelRegistry | null = null;
  private turnVisionModel: Model<Api> | null = null;
  private turnSignal: AbortSignal | undefined;
  private pendingTurnPrompt = "";

  constructor(private readonly deps: LoaderDeps) {}

  /** Bind the turn context (model registry, resolved vision model, abort signal)
   *  for the loader's next dispatch. Called from every handler that may trigger
   *  `loadDescription()`. */
  bindTurnContext(ctx: { modelRegistry: ModelRegistry; signal?: AbortSignal }): void {
    this.turnModelRegistry = ctx.modelRegistry;
    this.turnSignal = ctx.signal;
    const resolved = this.deps.resolveVisionModel(ctx.modelRegistry, this.deps.getConfig().visionModel!);
    if (resolved) this.turnVisionModel = resolved;
  }

  /** Capture this turn's user prompt so every image in the turn is described in
   *  the same request context. */
  setPendingTurnPrompt(prompt: string): void {
    this.pendingTurnPrompt = prompt;
  }

  /** Load an image's description. Returns a memoized Promise: cache hits return
   *  the existing (in-flight or resolved) promise; misses push the image's key
   *  into the current batch and schedule dispatch. Callers in the same frame
   *  share ONE batch → ONE vision call. Failures resolve to {@link UNAVAILABLE}
   *  and are NOT cached, so the next turn re-attempts. */
  loadDescription(img: ExtractedImage): Promise<string> {
    const hash = imageHash(img.mimeType, img.data);
    const cached = this.cache.get(hash);
    if (cached) return cached;

    if (!this.batch) {
      this.batch = { keys: [], imgs: [], callbacks: [] };
      this.scheduleDispatch();
    }
    const batch = this.batch;
    let idx = batch.keys.indexOf(hash);
    if (idx === -1) {
      batch.keys.push(hash);
      batch.imgs.push(img);
      idx = batch.keys.length - 1;
    } else {
      // Same image loaded twice in the frame (the first load's cache entry was
      // evicted mid-frame, else the second load would have short-circuited on
      // the cache). Share the one key/image slot but give this caller its own
      // promise — dispatch resolves it by hash alongside the first caller's.
      batch.imgs[idx] = img;
    }
    const promise = new Promise<string>((resolve, reject) => {
      batch.callbacks.push({ hash, resolve, reject });
    });
    const cfg = this.deps.getConfig();
    if (this.cache.size >= cfg.cacheMax) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(hash, promise);
    return promise;
  }

  /** Abandon any in-flight batch and clear turn context. Used on session reset
   *  (the batch's turn is gone) — also exposed via `[Symbol.dispose]` so a
   *  `using` binding can scope a loader lifetime. The description cache is
   *  preserved across turns (descriptions are stable per hash). */
  reset(): void {
    this.batch = null;
    this.dispatchScheduled = false;
    this.turnModelRegistry = null;
    this.turnVisionModel = null;
    this.turnSignal = undefined;
    this.pendingTurnPrompt = "";
  }

  [Symbol.dispose](): void {
    this.reset();
  }

  private scheduleDispatch(): void {
    if (this.dispatchScheduled) return;
    this.dispatchScheduled = true;
    enqueuePostPromiseJob(() => this.dispatchBatch());
  }

  /** Dispatch the current batch: ONE batched `runBatch` vision call for every
   *  key collected this frame, then resolve each load()'s promise with its
   *  description (or {@link UNAVAILABLE} on failure). Failures are evicted from
   *  the cache so the next turn re-attempts.
   *
   *  Results are resolved BY HASH and fanned out to EVERY callback: a batch can
   *  hold more callbacks than keys when the same image was loaded twice in one
   *  frame (its first cache entry was evicted mid-frame, so the second load
   *  pushed a second callback for the same hash). Indexing callbacks by key
   *  would skip those duplicates and hang their promises; iterating callbacks
   *  and looking up each one's hash fans the one result to all of them. */
  private async dispatchBatch(): Promise<void> {
    this.dispatchScheduled = false;
    const batch = this.batch;
    this.batch = null;
    if (!batch || batch.keys.length === 0) return;
    if (!this.turnVisionModel || !this.turnModelRegistry) {
      for (const cb of batch.callbacks) cb.resolve(UNAVAILABLE);
      return;
    }
    if (this.turnSignal?.aborted) {
      for (const key of batch.keys) this.cache.delete(key);
      for (const cb of batch.callbacks) cb.resolve(UNAVAILABLE);
      return;
    }
    this.deps.setLastError(null); // clear before a fresh attempt
    const misses = batch.keys.map((k, i) => ({ hash: k, img: batch.imgs[i] }));
    const cfg = this.deps.getConfig();
    const parsed = await runBatch(
      misses,
      this.pendingTurnPrompt,
      this.turnVisionModel,
      this.turnModelRegistry,
      cfg,
      this.deps,
      this.turnSignal,
    );
    // Build per-hash results, then fan them out to every callback. This reaches
    // duplicate-hash callbacks that a key-indexed loop would have skipped (and
    // left hanging).
    const results = new Map<string, string>();
    for (let i = 0; i < batch.keys.length; i++) {
      const key = batch.keys[i];
      const raw = parsed.get(key);
      if (raw) {
        const final = wrapDescription(raw, cfg);
        results.set(key, final);
        // Cache the resolved value so later loads (this frame or next) hit.
        this.cache.set(key, Promise.resolve(final));
      } else {
        // Genuine failure — do NOT cache; next turn re-attempts (and surfaces
        // the real error). Resolve (not reject) with UNAVAILABLE to match the
        // graceful-degradation contract and avoid unhandled rejections.
        this.cache.delete(key);
      }
    }
    for (const cb of batch.callbacks) {
      cb.resolve(results.get(cb.hash) ?? UNAVAILABLE);
    }
  }
}
