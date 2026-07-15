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
  /** Backoff (ms) before retrying a totally-failed describer batch. Defaults
   *  to {@link DESCRIBE_RETRY_BACKOFF_MS} when unset; 0 skips the wait (tests). */
  retryBackoffMs?: number;
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

/** Default backoff (ms) before retrying a totally-failed describer batch. A
 *  short backoff covers a transient provider blip (network hiccup, momentary
 *  429) without adding meaningful latency to the tool-result phase (free
 *  time). Generous enough to let a rate-limited provider recover, short enough
 *  that a genuinely broken vision model doesn't stall a turn. */
const DESCRIBE_RETRY_BACKOFF_MS = 500;

/** Resolve after `ms`, or immediately if `signal` is already aborted or aborts
 *  during the wait. Used by the retry backoff so a user cancel (ESC) doesn't
 *  wait the full backoff before the retry is skipped. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (ms <= 0 || signal.aborted) return resolve();
    const onAbort = () => {
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export class DescriptionLoader implements Disposable {
  private readonly cache = new Map<string, Promise<string>>();
  private batch: DescriptionBatch | null = null;
  private dispatchScheduled = false;
  private turnModelRegistry: ModelRegistry | null = null;
  private turnVisionModel: Model<Api> | null = null;
  /** Turn-level abort controller. In-flight describer batches (`runBatch`)
   *  are wired to `turnAbortController.signal` so a user cancel (ESC) aborts
   *  them EVEN WHEN the batch was dispatched before the run's live abort
   *  signal existed — the `before_agent_start` / paste-time prewarm case,
   *  where `ctx.signal` is `undefined` (the agent run hasn't started yet).
   *  The run's live signal, once it arrives via `bindTurnContext`, is forwarded
   *  into `turnAbortController.abort()`. Because `runBatch` holds a reference
   *  to this controller's signal (not a snapshot of `ctx.signal`), a batch
   *  that started during prewarm becomes abortable the instant a later
   *  `bindTurnContext` (from `tool_result`/`context`) brings the live signal.
   *  Reset per turn via {@link resetTurnAbort} so a previous turn's cancel
   *  can't poison the next turn's prewarm. */
  private turnAbortController = new AbortController();
  /** The run signal currently forwarded into {@link turnAbortController}, so
   *  the abort listener is attached at most once per signal (the same signal
   *  is bound by every `tool_result`/`context` event in a turn). */
  private wiredSignal: AbortSignal | undefined;
  private pendingTurnPrompt = "";

  constructor(private readonly deps: LoaderDeps) {}

  /** Bind the turn context (model registry, resolved vision model, abort
   *  signal) for the loader's next dispatch. Called from every handler that
   *  may trigger `loadDescription()`.
   *
   *  The abort signal is forwarded into the loader's {@link turnAbortController}
   *  rather than stored directly. Storing `ctx.signal` directly would leave a
   *  prewarm-dispatched batch (started in `before_agent_start`, where
   *  `ctx.signal` is `undefined` because the run hasn't started) with no abort
   *  wire — ESC couldn't cancel it, so the `tool_result` handler would only
   *  discard the result AFTER the vision call ran to completion. Forwarding
   *  the live signal into a stable, loader-owned controller lets an in-flight
   *  prewarm batch be aborted the moment the live signal arrives. */
  bindTurnContext(ctx: { modelRegistry: ModelRegistry; signal?: AbortSignal }): void {
    this.turnModelRegistry = ctx.modelRegistry;
    const resolved = this.deps.resolveVisionModel(ctx.modelRegistry, this.deps.getConfig().visionModel!);
    if (resolved) this.turnVisionModel = resolved;
    const signal = ctx.signal;
    if (!signal) return;
    if (signal.aborted) {
      this.turnAbortController.abort();
      this.wiredSignal = signal;
      return;
    }
    if (signal === this.wiredSignal) return; // already forwarded this signal
    this.wiredSignal = signal;
    signal.addEventListener("abort", () => this.turnAbortController.abort(), { once: true });
  }

  /** Reset the turn-level abort controller for a fresh turn. Call at turn
   *  boundaries (`before_agent_start`, paste-time prewarm) so a previous turn's
   *  cancel doesn't leave {@link turnAbortController} aborted — which would
   *  make every subsequent dispatch short-circuit to UNAVAILABLE. A
   *  non-aborted controller is reused (avoids orphaning an in-flight paste
   *  prewarm holding its signal); only an aborted one is replaced. `wiredSignal`
   *  is always cleared so the next live signal re-wires. Safe at a real turn
   *  boundary, where the prior turn's batch has settled. */
  resetTurnAbort(): void {
    if (this.turnAbortController.signal.aborted) {
      this.turnAbortController = new AbortController();
    }
    this.wiredSignal = undefined;
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
    // Fresh controller on session reset (no in-flight batch to orphan).
    this.turnAbortController = new AbortController();
    this.wiredSignal = undefined;
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
    if (this.turnAbortController.signal.aborted) {
      for (const key of batch.keys) this.cache.delete(key);
      for (const cb of batch.callbacks) cb.resolve(UNAVAILABLE);
      return;
    }
    this.deps.setLastError(null); // clear before a fresh attempt
    const misses = batch.keys.map((k, i) => ({ hash: k, img: batch.imgs[i] }));
    const cfg = this.deps.getConfig();
    let parsed = await runBatch(
      misses,
      this.pendingTurnPrompt,
      this.turnVisionModel,
      this.turnModelRegistry,
      cfg,
      this.deps,
      this.turnAbortController.signal,
    );
    // Retry once on a TOTAL batch failure (the batched call itself failed:
    // auth, network, timeout, empty, or stopReason "error"). A transient blip
    // would otherwise cost the agent a full turn — UNAVAILABLE this turn, not
    // cached, re-attempted next turn. Retrying the whole batch recovers it
    // within the same tool-result phase (free time). Skip when the turn was
    // cancelled (ESC); an auth failure re-fails cheaply (runBatch re-checks the
    // API key before the vision call), so no vision call is wasted on a
    // permanent auth error. Partial failures (some images described) are left
    // as-is — only a totally-empty result retries.
    if (parsed.size === 0 && misses.length > 0 && !this.turnAbortController.signal.aborted) {
      await sleep(this.deps.retryBackoffMs ?? DESCRIBE_RETRY_BACKOFF_MS, this.turnAbortController.signal);
      if (!this.turnAbortController.signal.aborted) {
        this.deps.setLastError(null);
        parsed = await runBatch(
          misses,
          this.pendingTurnPrompt,
          this.turnVisionModel,
          this.turnModelRegistry,
          cfg,
          this.deps,
          this.turnAbortController.signal,
        );
      }
    }
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
