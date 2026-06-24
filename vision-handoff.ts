/**
 * pi-vision-handoff — give text-only models vision by proxying image input
 * through a vision-capable model of your choice.
 *
 * Extracted from the GLM 5.1 vision-handoff pipeline in pi-umans-provider and
 * generalized: instead of a hardcoded describer, the user picks any
 * vision-capable model from the registry via an interactive picker, and the
 * choice is persisted to ~/.pi/agent/extensions/pi-vision-handoff.json.
 *
 * Pipeline (provider-agnostic via @earendil-works/pi-ai's complete()):
 *   before_agent_start → warm the description cache for attached images
 *   tool_result (read) → describe read-tool images and INSERT the description
 *     as a text block before each image, keeping the image so the TUI still
 *     renders it (kitty). pi-ai later strips the image for non-vision models,
 *     leaving the description text for the model.
 *   before_provider_request → swap remaining image blocks in the payload for
 *     text (catches user-attached images for vision-capable handoff targets)
 *
 * Image blocks are detected by shape across the four formats pi uses:
 *   openai-completions: { type: "image_url",  image_url: { url: "data:..." } }
 *   openai-responses:   { type: "input_image", image_url: "data:..." }
 *   anthropic-messages: { type: "image", source: { type: "base64", media_type, data } }
 *   pi-ai internal:     { type: "image", data, mimeType }   ← read tool / ToolResultEvent
 *
 * Descriptions are cached per image hash (LRU, size = config.cacheMax) so the
 * swap is instant by the time before_provider_request fires.
 */

import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, sep } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { complete } from "@earendil-works/pi-ai/compat";
import type { Api, ImageContent, Message, Model, TextContent } from "@earendil-works/pi-ai";
import {
  DEFAULT_USER_PROMPT_PREFIX,
  DEFAULT_VISION_PROMPT,
  HANDOFF_COMMAND_DESCRIPTION,
  IMAGE_PLACEHOLDER_PREFIX,
  IMAGE_PLACEHOLDER_SUFFIX,
  NON_VISION_IMAGE_NOTE,
  USAGE_ENTRY_TYPE,
  USAGE_EVENT_CHANNEL,
  EMPTY_ENERGY_CAPTURE,
  batchUserPrompt,
  buildUsageRecord,
  describeAls,
  describeTimeoutMs,
  extractImageFromBlock,
  formatModelRef,
  installFetchInterceptor,
  isVisionModel,
  parseModelRef,
  parseBatchedDescriptions,
  readConfig,
  stripNonVisionImageNote,
  uninstallFetchInterceptor,
  wrapDescription,
  writeConfig,
  type DescribeContext,
  type ExtractedImage,
  type VisionHandoffConfig,
  type VisionHandoffEnergyCapture,
  type VisionHandoffUsageRecord,
} from "./src/index.js";
import { VisionModelSelectorComponent, type VisionModelSelectorResult } from "./src/vision-model-selector.js";

const UNAVAILABLE = `${IMAGE_PLACEHOLDER_PREFIX}description unavailable${IMAGE_PLACEHOLDER_SUFFIX}`;

// Usage reporter; wired to pi.appendEntry + pi.events.emit in the default
// export. No-op until then so the describer is safe to call before wiring.
let reportUsage: (record: VisionHandoffUsageRecord) => void = () => {};

let config: VisionHandoffConfig = readConfig();

/** User prompt for the current agent turn, captured from before_agent_start. */
let pendingTurnPrompt: string | null = null;

const visionCache = new Map<string, Promise<string>>();
let visionModelCache: { ref: string; model: Model<Api> } | null = null;
let visionModelUnresolvedRef: string | null = null;

/** Most recent describer failure message (auth error, network error, abort,
 *  empty response, etc.). Set by the describer's catch blocks; surfaced to the
 *  user by the `context` handler via ctx.ui.notify so a broken vision model
 *  stops looking like a silent "extension doesn't work" — you see the actual
 *  provider error. Cleared at the start of each describer attempt. */
let lastDescriberError: string | null = null;

/** Image hashes we've already warned the user about this session. Prevents the
 *  `context` hook (which fires before every LLM turn) from re-warning on the
 *  same failing images every turn — describer failures aren't cached, so
 *  without this guard a broken vision model would spam a warning per turn.
 *  Cleared on `session_start`. */
const warnedHashes = new Set<string>();

function isConfigured(cfg: VisionHandoffConfig): boolean {
  return cfg.enabled && !!cfg.visionModel;
}

function isHandoffTarget(
  model: { provider?: string; id?: string; input?: ("text" | "image")[] } | undefined | null,
  cfg: VisionHandoffConfig,
): boolean {
  if (!model || !model.provider || !model.id) return false;
  const ref = formatModelRef(model.provider, model.id);
  if (cfg.handoffModels.includes(ref)) return true;
  if (cfg.autoHandoff && !isVisionModel(model)) return true;
  return false;
}

function resolveVisionModel(modelRegistry: ModelRegistry, ref: string): Model<Api> | null {
  if (visionModelCache && visionModelCache.ref === ref) return visionModelCache.model;
  const parsed = parseModelRef(ref);
  if (!parsed) return null;
  const model = modelRegistry.find(parsed.provider, parsed.id);
  if (!model) return null;
  visionModelCache = { ref, model };
  return model;
}

function imageHash(mimeType: string, data: string): string {
  return crypto.createHash("sha256").update(`${mimeType}\x00${data}`).digest("hex").slice(0, 32);
}

/** Sniff an image MIME type from magic bytes (mirrors pi's
 *  detectSupportedImageMimeType). Returns null for unsupported/animated. */
function sniffImageMime(buf: Buffer): string | null {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    return buf[3] === 0xf7 ? null : "image/jpeg";
  }
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (buf.length >= 8 && png.every((b, i) => buf[i] === b)) return "image/png";
  if (buf.length >= 3 && buf.subarray(0, 3).toString("ascii") === "GIF") return "image/gif";
  if (buf.length >= 12 && buf.subarray(0, 4).toString("ascii") === "RIFF" && buf.subarray(8, 12).toString("ascii") === "WEBP") {
    return "image/webp";
  }
  return null;
}

/** Read an image file from disk into an ExtractedImage (base64 + sniffed MIME).
 *  SYNCHRONOUS by design: the dataloader requires all `loadDescription()`
 *  calls in a frame to land in the SAME batch (dispatched after the microtask
 *  cascade settles). An async read would suspend the handler on I/O, letting
 *  the dispatch fire mid-read and split one batch into two vision calls.
 *  Clipboard images are small (a few MB), so a sync read is sub-millisecond —
 *  the same cost pi pays when the agent reads the file, just front-loaded.
 *  Returns null if the file can't be read or isn't a supported image. */
function readImageFile(filePath: string): ExtractedImage | null {
  try {
    const buf = readFileSync(filePath);
    const mimeType = sniffImageMime(buf);
    if (!mimeType) return null;
    return { data: buf.toString("base64"), mimeType };
  } catch {
    return null;
  }
}

/** Regex matching pi's pasted-clipboard temp image file paths anywhere in the
 *  prompt text. pi writes pasted clipboard images to `<tmpdir>/pi-clipboard-<uuid>.<ext>`
 *  and inserts the path as text at the cursor, so on a non-vision model these
 *  arrive as path tokens in the user prompt — NOT as `event.images`. Matching
 *  them lets us pre-warm the describer at paste-enter (concurrent with the
 *  agent's first response) instead of waiting for the agent to `read` them. */
const CLIPBOARD_IMAGE_PATH_RE = /(\S*pi-clipboard-[^\s]+\.(?:png|jpe?g|gif|webp))/gi;

/** Extract pasted clipboard image file paths from a prompt. Confined to the
 *  OS temp directory so an attacker-crafted prompt can't trick the extension
 *  into reading arbitrary files — only pi's own clipboard temp files qualify. */
function findClipboardImagePaths(prompt: string): string[] {
  const tmp = tmpdir();
  const paths = new Set<string>();
  for (const m of prompt.matchAll(CLIPBOARD_IMAGE_PATH_RE)) {
    const p = m[1];
    if (!p) continue;
    const abs = isAbsolute(p) ? p : join(tmp, p);
    // Ensure the resolved candidate stays inside the temp directory.
    if (abs.startsWith(tmp + sep) || abs === tmp) paths.add(abs);
  }
  return [...paths];
}

/** Wire the turn's abort signal into a describer call's AbortController so a
 *  user cancel propagates immediately. Returns a handle to detach the
 *  listener (call in `finally`) and to check whether the abort came from the
 *  user (to suppress the spurious "description failed" warning on a deliberate
 *  cancel). Aborting the controller causes the in-flight `complete()` to
 *  reject/abort promptly instead of blocking the turn's abort until timeout. */
function wireTurnAbort(
  turnSignal: AbortSignal | undefined,
  controller: AbortController,
): { detach: () => void; userAborted: () => boolean } | null {
  if (!turnSignal) return null;
  const onAbort = () => controller.abort();
  if (turnSignal.aborted) controller.abort();
  else turnSignal.addEventListener("abort", onAbort, { once: true });
  return {
    detach: () => turnSignal.removeEventListener("abort", onAbort),
    userAborted: () => turnSignal.aborted,
  };
}

// Facebook DataLoader pattern for image descriptions. `loadDescription(img)`
// returns a memoized Promise for the description and pushes the image's key
// (hash) into the CURRENT batch. All `load()` calls in the same execution
// frame (+ its microtask cascade) coalesce into ONE batch object. Dispatch is
// scheduled after the microtask queue settles (via enqueuePostPromiseJob), so
// every load in the frame lands in the single batch before the ONE vision call
// fires. Each load()'s promise then resolves with its description.
//
// Mapping: the `read` tools are the `load()` callers. Their `tool_result`
// handler awaits the shared batch — so N parallel reads (pi runs `read` via
// Promise.all) block on the SAME single vision call and all resolve together,
// the descriptions landing in the tool results BEFORE the agent's next turn.
// The agent's tool-result wait is free time, so this adds zero latency to the
// critical path (the describer no longer blocks the `context` transform).
// `context` then sees text-described tool results (no image blocks to swap);
// any remaining images (user-attached, custom-injected) are cache hits.
interface DescriptionBatch {
  keys: string[];
  imgs: ExtractedImage[];
  callbacks: { resolve: (v: string) => void; reject: (e: Error) => void }[];
}
let currentBatch: DescriptionBatch | null = null;
let dispatchScheduled = false;

// Turn context, bound from before_agent_start / tool_result / context so the
// loader's dispatch can resolve the vision model + signal without the caller
// passing them through load(). Re-bound at the start of each handler.
let turnModelRegistry: ModelRegistry | null = null;
let turnVisionModel: Model<Api> | null = null;
let turnSignal: AbortSignal | undefined;

const resolvedMicrotask = Promise.resolve();

/** Defer `fn` until after the current microtask cascade settles, so every
 *  `loadDescription()` in the same frame coalesces into one batch before the
 *  single vision call fires. Mirrors DataLoader's enqueuePostPromiseJob: a
 *  Promise Job enqueues a global Job (process.nextTick), guaranteeing dispatch
 *  runs after "PromiseJobs" ends — after all `load()` callers in the frame
 *  have registered their keys. */
function enqueuePostPromiseJob(fn: () => void): void {
  resolvedMicrotask.then(() => process.nextTick(fn));
}

/** Bind the turn context (model registry, resolved vision model, abort signal)
 *  for the loader's next dispatch. Called from every handler that may trigger
 *  `loadDescription()`. */
function bindTurnContext(ctx: { modelRegistry: ModelRegistry; signal?: AbortSignal }): void {
  turnModelRegistry = ctx.modelRegistry;
  turnSignal = ctx.signal;
  const resolved = resolveVisionModel(ctx.modelRegistry, config.visionModel!);
  if (resolved) turnVisionModel = resolved;
}

/** Load an image's description. Returns a memoized Promise: cache hits return
 *  the existing (in-flight or resolved) promise; misses push the image's key
 *  into the current batch and schedule dispatch. Callers in the same frame
 *  share ONE batch → ONE vision call. Failures resolve to UNAVAILABLE and are
 *  NOT cached, so the next turn re-attempts. */
function loadDescription(img: ExtractedImage): Promise<string> {
  const hash = imageHash(img.mimeType, img.data);
  const cached = visionCache.get(hash);
  if (cached) return cached;

  if (!currentBatch) {
    currentBatch = { keys: [], imgs: [], callbacks: [] };
    scheduleDispatch();
  }
  const batch = currentBatch;
  let idx = batch.keys.indexOf(hash);
  if (idx === -1) {
    batch.keys.push(hash);
    batch.imgs.push(img);
    idx = batch.keys.length - 1;
  } else {
    // Same image loaded twice in the frame — share one cache slot, but give
    // this caller its own promise.
    batch.imgs[idx] = img;
  }
  const promise = new Promise<string>((resolve, reject) => {
    batch.callbacks.push({ resolve, reject });
  });
  if (visionCache.size >= config.cacheMax) {
    const firstKey = visionCache.keys().next().value;
    if (firstKey !== undefined) visionCache.delete(firstKey);
  }
  visionCache.set(hash, promise);
  return promise;
}

function scheduleDispatch(): void {
  if (dispatchScheduled) return;
  dispatchScheduled = true;
  enqueuePostPromiseJob(dispatchBatch);
}

/** Dispatch the current batch: ONE batched `runBatchRaw` vision call for every
 *  key collected this frame, then resolve each load()'s promise with its
 *  description (or UNAVAILABLE on failure). Failures are evicted from the cache
 *  so the next turn re-attempts. */
async function dispatchBatch(): Promise<void> {
  dispatchScheduled = false;
  const batch = currentBatch;
  currentBatch = null;
  if (!batch || batch.keys.length === 0) return;
  if (!turnVisionModel || !turnModelRegistry) {
    for (const cb of batch.callbacks) cb.resolve(UNAVAILABLE);
    return;
  }
  if (turnSignal?.aborted) {
    for (let i = 0; i < batch.keys.length; i++) {
      visionCache.delete(batch.keys[i]);
      batch.callbacks[i].resolve(UNAVAILABLE);
    }
    return;
  }
  lastDescriberError = null;
  const misses = batch.keys.map((k, i) => ({ hash: k, img: batch.imgs[i] }));
  const parsed = await runBatchRaw(
    misses,
    pendingTurnPrompt ?? "",
    turnVisionModel,
    turnModelRegistry,
    config,
    turnSignal,
  );
  for (let i = 0; i < batch.keys.length; i++) {
    const raw = parsed.get(batch.keys[i]);
    if (raw) {
      const final = wrapDescription(raw, config);
      visionCache.set(batch.keys[i], Promise.resolve(final));
      batch.callbacks[i].resolve(final);
    } else {
      // Genuine failure — do NOT cache; next turn re-attempts (and surfaces
      // the real error). Resolve (not reject) with UNAVAILABLE to match the
      // graceful-degradation contract and avoid unhandled rejections.
      visionCache.delete(batch.keys[i]);
      batch.callbacks[i].resolve(UNAVAILABLE);
    }
  }
}

/** Describe a single image with one `complete()` call and return the RAW
 *  description (no `[Image: …]` envelope, no truncation). Returns null on any
 *  genuine failure (auth, abort/error, empty) so the caller can cache the
 *  `UNAVAILABLE` placeholder only when a real describer attempt failed —
 *  never merely because a batched delimiter-parse missed.
 *
 *  This is the robust single-image path (no delimiters to cooperate with) and
 *  the per-image fallback used when a batched call's response couldn't be split
 *  back into per-image sections. It also powers the common single-image case
 *  directly via `runBatchRaw`'s fallback for count === 1. */
async function describeSingleRaw(
  img: ExtractedImage,
  userPrompt: string,
  visionModel: Model<Api>,
  modelRegistry: ModelRegistry,
  cfg: VisionHandoffConfig,
  turnSignal?: AbortSignal,
): Promise<string | null> {
  const auth = await modelRegistry.getApiKeyAndHeaders(visionModel);
  if (!auth.ok || !auth.apiKey) {
    lastDescriberError =
      (!auth.ok && "error" in auth && typeof auth.error === "string" && auth.error) ||
      `No API key for vision model "${visionModel.provider}/${visionModel.id}"`;
    return null;
  }
  const prefix = cfg.userPromptPrefix ?? DEFAULT_USER_PROMPT_PREFIX;
  const systemPrompt = cfg.prompt ?? DEFAULT_VISION_PROMPT;
  const content: (TextContent | ImageContent)[] = [
    { type: "text", text: batchUserPrompt(1, userPrompt, prefix) },
    { type: "image", data: img.data, mimeType: img.mimeType } satisfies ImageContent,
  ];
  const userMessage: Message = { role: "user", content, timestamp: Date.now() };
  const timeoutMs = describeTimeoutMs(1);
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  // Propagate the turn's abort signal so a user cancel kills the in-flight
  // describer call immediately instead of waiting for the timeout. Without
  // this, aborting a turn blocks until the describer resolves (up to the
  // timeout) because pi awaits the context transform before processing abort.
  const turnAbort = wireTurnAbort(turnSignal, controller);
  const describeCtx: DescribeContext = { energyReader: undefined };
  installFetchInterceptor();
  try {
    const response = await describeAls.run(describeCtx, async () =>
      complete(
        visionModel,
        { systemPrompt, messages: [userMessage] },
        { apiKey: auth.apiKey, headers: auth.headers, signal: controller.signal, maxTokens: cfg.maxTokens },
      ),
    );
    let capture: VisionHandoffEnergyCapture = EMPTY_ENERGY_CAPTURE;
    if (describeCtx.energyReader) {
      try {
        capture = await describeCtx.energyReader;
      } catch {
        // tee aborted with the main stream — keep the empty capture
      }
    }
    const hash = imageHash(img.mimeType, img.data);
    const record = buildUsageRecord(response, capture, visionModel, hash);
    if (record) reportUsage(record);
    if (response.stopReason === "aborted" || response.stopReason === "error") {
      if (turnAbort?.userAborted()) {
        // user cancelled the turn — don't surface a warning for that
      } else if (timedOut) {
        lastDescriberError = `describer timed out after ${timeoutMs / 1000}s`;
      } else {
        lastDescriberError = `vision model returned stopReason "${response.stopReason}"${response.errorMessage ? ": " + response.errorMessage : ""}`;
      }
      return null;
    }
    const text = response.content
      .filter((c): c is TextContent => c.type === "text")
      .map((c) => c.text)
      .join("\n")
      .trim();
    if (!text) lastDescriberError = "vision model returned an empty description";
    return text || null;
  } catch (err) {
    lastDescriberError = timedOut
      ? `describer timed out after ${timeoutMs / 1000}s`
      : err instanceof Error ? err.message : String(err);
    return null;
  } finally {
    if (describeCtx.energyReader) describeCtx.energyReader.catch(() => {});
    uninstallFetchInterceptor();
    clearTimeout(timer);
    turnAbort?.detach();
  }
}

async function runBatchRaw(
  misses: { img: ExtractedImage; hash: string }[],
  userPrompt: string,
  visionModel: Model<Api>,
  modelRegistry: ModelRegistry,
  cfg: VisionHandoffConfig,
  turnSignal?: AbortSignal,
): Promise<Map<string, string>> {
  const out = new Map<string, string>();
  const auth = await modelRegistry.getApiKeyAndHeaders(visionModel);
  if (!auth.ok || !auth.apiKey) {
    lastDescriberError =
      (!auth.ok && "error" in auth && typeof auth.error === "string" && auth.error) ||
      `No API key for vision model "${visionModel.provider}/${visionModel.id}"`;
    return out;
  }

  const prefix = cfg.userPromptPrefix ?? DEFAULT_USER_PROMPT_PREFIX;
  const systemPrompt = cfg.prompt ?? DEFAULT_VISION_PROMPT;

  const content: (TextContent | ImageContent)[] = [
    { type: "text", text: batchUserPrompt(misses.length, userPrompt, prefix) },
    ...misses.map((m) => ({ type: "image", data: m.img.data, mimeType: m.img.mimeType } satisfies ImageContent)),
  ];
  const userMessage: Message = { role: "user", content, timestamp: Date.now() };

  const timeoutMs = describeTimeoutMs(misses.length);
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  // Propagate the turn's abort signal so a user cancel kills the in-flight
  // batched describer call immediately (see describeSingleRaw for rationale).
  const turnAbort = wireTurnAbort(turnSignal, controller);
  // Energy/token capture for this describer call. The fetch interceptor is
  // refcount-installed around the complete() window and routes the teed
  // response body to this describe's AsyncLocalStorage slot. For non-Neuralwatt
  // vision models no SSE energy comments are present and the capture stays
  // empty (energy fields omitted from the record). One record is emitted per
  // real call (this whole batch), with imageHashes listing every member so
  // consumers can attribute tokens/energy per image without double-counting.
  const describeCtx: DescribeContext = { energyReader: undefined };
  installFetchInterceptor();
  try {
    const response = await describeAls.run(describeCtx, async () =>
      complete(
        visionModel,
        { systemPrompt, messages: [userMessage] },
        { apiKey: auth.apiKey, headers: auth.headers, signal: controller.signal, maxTokens: cfg.maxTokens },
      ),
    );
    let capture: VisionHandoffEnergyCapture = EMPTY_ENERGY_CAPTURE;
    if (describeCtx.energyReader) {
      try {
        capture = await describeCtx.energyReader;
      } catch {
        // tee aborted with the main stream — keep the empty capture
      }
    }
    const hashes = misses.map((m) => m.hash);
    const record = buildUsageRecord(response, capture, visionModel, hashes[0], hashes.length > 1 ? hashes : undefined);
    if (record) reportUsage(record);
    if (response.stopReason === "aborted" || response.stopReason === "error") {
      if (turnAbort?.userAborted()) {
        // user cancelled the turn — don't surface a warning for that
      } else if (timedOut) {
        lastDescriberError = `describer timed out after ${timeoutMs / 1000}s`;
      } else {
        lastDescriberError = `vision model returned stopReason "${response.stopReason}"${response.errorMessage ? ": " + response.errorMessage : ""}`;
      }
      return out;
    }
    const text = response.content
      .filter((c): c is TextContent => c.type === "text")
      .map((c) => c.text)
      .join("\n")
      .trim();
    if (!text) {
      lastDescriberError = "vision model returned an empty description";
      return out;
    }
    const parsed = parseBatchedDescriptions(text, misses.length);
    for (let i = 0; i < misses.length; i++) {
      const d = parsed[i];
      if (d) out.set(misses[i].hash, d);
    }
    // Fallback: a failed delimiter-parse is NOT a failed description — it's a
    // failed batching. Describe each unparsed image with its own single-image
    // call (no delimiters to cooperate with, so this always works for any
    // vision model that can describe one image). The calls run in parallel so
    // their results still arrive together ("same instant"), not sequentially.
    // This only runs when the batched response couldn't be split — a
    // cooperative model never triggers it, so the common case stays one call.
    const unparsed = misses.filter((m) => !out.has(m.hash));
    if (unparsed.length > 0) {
      const fallbacks = await Promise.all(
        unparsed.map((m) => describeSingleRaw(m.img, userPrompt, visionModel, modelRegistry, cfg, turnSignal)),
      );
      for (let i = 0; i < unparsed.length; i++) {
        if (fallbacks[i]) out.set(unparsed[i].hash, fallbacks[i]!);
      }
    }
    return out;
  } catch (err) {
    lastDescriberError = timedOut
      ? `describer timed out after ${timeoutMs / 1000}s`
      : err instanceof Error ? err.message : String(err);
    return out;
  } finally {
    if (describeCtx.energyReader) describeCtx.energyReader.catch(() => {});
    uninstallFetchInterceptor();
    clearTimeout(timer);
    turnAbort?.detach();
  }
}

function notifyUnresolvedVisionModel(ctx: ExtensionContext, ref: string): void {
  if (visionModelUnresolvedRef === ref) return;
  visionModelUnresolvedRef = ref;
  if (ctx.hasUI) {
    ctx.ui.notify(
      `pi-vision-handoff: configured vision model "${ref}" was not found in the registry — run /vision-handoff to pick a model.`,
      "warning",
    );
  }
}

export default function (pi: ExtensionAPI) {
  config = readConfig();

  // Wire the usage reporter to pi's persistence + event bus. appendEntry
  // persists the record so it replays on session resume/branch, and the event
  // lets live consumers filter on one channel for tokens AND energy. Each call
  // is independently guarded so a persistence/emit failure never breaks a
  // describer turn. Re-assigned every factory invocation (pi re-runs the
  // factory on /new, /resume, fork, /reload) so the closure always references
  // the live pi.
  reportUsage = (record: VisionHandoffUsageRecord) => {
    try {
      pi.appendEntry(USAGE_ENTRY_TYPE, record);
    } catch {
      // never break the describer on persistence failure
    }
    try {
      pi.events?.emit(USAGE_EVENT_CHANNEL, record);
    } catch {
      // never break the describer on emit failure
    }
  };

  pi.on("session_start", async () => {
    // Reload in case the user edited the config on disk from another session.
    config = readConfig();
    visionModelCache = null;
    pendingTurnPrompt = null;
    warnedHashes.clear();
    // Reset the DataLoader: drop any in-flight batch (its turn is gone) and
    // clear the dispatch flag so the next load() schedules a fresh dispatch.
    currentBatch = null;
    dispatchScheduled = false;
    turnModelRegistry = null;
    turnVisionModel = null;
    turnSignal = undefined;
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (!isConfigured(config)) return;
    if (!isHandoffTarget(ctx.model, config)) return;

    // Capture this turn's user prompt so every image in the turn — attached or
    // read via the read tool — is described in the same request context.
    pendingTurnPrompt = event.prompt || "";
    bindTurnContext(ctx);

    if (!resolveVisionModel(ctx.modelRegistry, config.visionModel!)) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    // PRE-WARM at paste-enter via the DataLoader. Two image sources land here:
    //
    // 1. Attached image blocks (event.images) — vision-capable targets where
    //    the user message itself carries image blocks (e.g. `pi --image`).
    //
    // 2. Pasted clipboard image FILE PATHS in the prompt text — the common
    //    non-vision flow. pi's `handleClipboardImagePaste` writes each pasted
    //    image to `<tmpdir>/pi-clipboard-<uuid>.<ext>` and inserts the PATH as
    //    text at the cursor; on a non-vision model these arrive as path tokens
    //    in `event.prompt`, NOT as `event.images`. We scan the prompt for those
    //    temp paths, read the files, and `loadDescription()` them so the ONE
    //    batched vision call starts the instant you press enter — CONCURRENT
    //    with the agent's first response generation — instead of waiting for
    //    the agent to `read` the files. By the time the agent's `read` tool
    //    results fire, `tool_result`'s `loadDescription()` is a cache hit.
    //
    // Both sources flow through the same loader: `load()` is synchronous and
    // memoized, so all images in this frame (attached + clipboard-path)
    // coalesce into ONE batch dispatched after the microtask cascade.
    for (const image of event.images ?? []) {
      if (!image || image.type !== "image" || !image.data) continue;
      loadDescription({ data: image.data, mimeType: image.mimeType || "image/png" }).catch(() => {});
    }

    // Pasted clipboard image paths in the prompt text — read the files and
    // warm the loader. Read synchronously (see readImageFile) so all
    // loadDescription() calls — attached + clipboard-path — land in ONE batch
    // frame → ONE vision call. Fire-and-forget: a file that can't be read or
    // isn't an image is skipped (the agent's `read` will still describe it via
    // `tool_result` if the agent reads it).
    for (const p of findClipboardImagePaths(event.prompt || "")) {
      const img = readImageFile(p);
      if (img) loadDescription(img).catch(() => {});
    }
  });

  // The PRIMARY injection point: the `read` tool's `tool_result` handler.
  // When the agent reads image files, this fires for each read result. It
  // calls the loader's `loadDescription(img)` for every image block and
  // AWAITS the shared batch — so N parallel reads (pi runs `read` via
  // Promise.all) coalesce into ONE batched vision call (DataLoader: all
  // load() calls in the same microtask frame share one batch, dispatched
  // after the cascade settles) and all resolve together. The descriptions
  // replace the image blocks in the returned `content`, so by the time the
  // agent's next turn starts the tool results already carry text — the
  // agent never sees raw image blocks it can't process.
  //
  // Why block here and not in `context`: the tool-result phase is free time
  // (the agent is just waiting for tool results), so running the describer
  // here adds zero latency to the critical path. `context` then becomes a
  // cache-hit no-op for read images. (If this handler didn't await, the
  // describer would instead block the `context` transform — the agent's
  // critical path right before the LLM call — which is the bug this fixes.)
  pi.on("tool_result", async (event, ctx) => {
    if (!isConfigured(config)) return;
    if (event.toolName !== "read") return;
    const content = event.content;
    if (!Array.isArray(content)) return;

    // Collect image blocks in this read result. (The read tool emits image
    // blocks even for non-vision models — they reach here untouched.)
    const imgs: ExtractedImage[] = [];
    for (let i = 0; i < content.length; i++) {
      const img = extractImageFromBlock(content[i]);
      if (img) imgs.push(img);
    }
    if (imgs.length === 0) return;
    if (!isHandoffTarget(ctx.model, config)) return;
    bindTurnContext(ctx);
    if (!resolveVisionModel(ctx.modelRegistry, config.visionModel!)) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    // load() each image — synchronous calls that push into the current batch
    // and return memoized promises — then await them all. Parallel reads'
    // load() calls land in the SAME batch (same microtask frame), so this is
    // ONE vision call for the whole read set, not N. Awaiting here runs the
    // describer during the tool-result phase (free time — the agent is just
    // waiting for tool results), so the batch is COMPLETE before `context`
    // fires, making `context` a non-blocking cache hit instead of a cold miss
    // on the critical path.
    //
    // We do NOT mutate the result content here: returning undefined keeps the
    // image block in storage so kitty renders it inline and `/resume` retains
    // it. The actual image→text swap happens in the `context` hook (on the
    // cloned LLM-bound payload only), by which point these are cache hits.
    const descs = await Promise.all(imgs.map((img) => loadDescription(img)));

    // On user abort, leave the result untouched — pi is tearing the turn
    // down and the LLM-bound content won't be sent.
    if (ctx.signal?.aborted) return;

    // Surface a failure once per image per session (the loader resolves
    // failures to UNAVAILABLE rather than rejecting, so this is where we
    // detect them for the read path). The `context` hook re-checks but the
    // per-image warnedHashes dedup prevents double-warning.
    if (ctx.hasUI) {
      const newlyFailed: string[] = [];
      for (let i = 0; i < imgs.length; i++) {
        if (descs[i] === UNAVAILABLE) {
          const h = imageHash(imgs[i].mimeType, imgs[i].data);
          if (!warnedHashes.has(h)) newlyFailed.push(h);
        }
      }
      if (newlyFailed.length > 0) {
        for (const h of newlyFailed) warnedHashes.add(h);
        const reason = lastDescriberError || "unknown error";
        ctx.ui.notify(
          `pi-vision-handoff: image description failed — ${reason}. Vision model: ${config.visionModel}`,
          "warning",
        );
      }
    }

    // Strip pi's `[Current model does not support images…]` note from text
    // blocks — since the handoff replaces the image with a description in the
    // `context` hook, that note is misleading (the agent WILL receive the
    // image's content, as text). Keep the image block itself so kitty still
    // renders it inline and `/resume` retains it; the `context` hook swaps the
    // image for its description in the LLM-bound clone before the next turn.
    let stripped = false;
    const next = content.slice();
    for (let i = 0; i < next.length; i++) {
      const block = next[i];
      if (block && typeof block === "object" && (block as { type: string }).type === "text") {
        const text = (block as { text: string }).text;
        if (typeof text === "string" && text.includes(NON_VISION_IMAGE_NOTE)) {
          const cleaned = stripNonVisionImageNote(text);
          if (cleaned !== text) {
            next[i] = { type: "text", text: cleaned };
            stripped = true;
          }
        }
      }
    }
    if (stripped) return { content: next as (TextContent | ImageContent)[] };
  });

  // The FALLBACK injection point: the `context` event fires as the agent's
  // `transformContext`, BEFORE pi-ai's `downgradeUnsupportedImages` strips
  // image blocks and BEFORE `convertToLlm`. It catches any image blocks that
  // didn't go through the `read` tool's `tool_result` handler — user-attached
  // images (for vision-capable handoff targets), custom extension-injected
  // messages, or reads that somehow bypassed the handler. `emitContext` does a
  // `structuredClone`, so swapping here touches only the LLM-bound payload.
  //
  // Read images are already text by this point (the `tool_result` handler
  // replaced them), so this is usually a no-op for the common paste-and-read
  // flow. For the images it does find, `loadDescription()` is a cache hit
  // (warmed by `before_agent_start`) or queues into the loader's current batch.
  pi.on("context", async (event, ctx) => {
    if (!isConfigured(config)) return;

    const messages = event.messages as unknown as Array<Record<string, unknown>>;
    if (!Array.isArray(messages)) return;

    const byHash = new Map<string, ExtractedImage>();
    let anyImage = false;
    for (const msg of messages) {
      const content = msg.content;
      if (!Array.isArray(content)) continue;
      for (let i = 0; i < content.length; i++) {
        const img = extractImageFromBlock(content[i]);
        if (!img) continue;
        anyImage = true;
        byHash.set(imageHash(img.mimeType, img.data), img);
      }
    }
    if (!anyImage) return;
    if (!isHandoffTarget(ctx.model, config)) return;
    bindTurnContext(ctx);
    if (!resolveVisionModel(ctx.modelRegistry, config.visionModel!)) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    // Cache hits (warmed by before_agent_start / tool_result) resolve
    // instantly; any remaining misses queue into the loader's current batch.
    const imgs = [...byHash.values()];
    const descArr = await Promise.all(imgs.map((img) => loadDescription(img)));
    const descs = new Map<string, string>();
    for (let i = 0; i < imgs.length; i++) {
      descs.set(imageHash(imgs[i].mimeType, imgs[i].data), descArr[i]);
    }

    const userAborted = !!ctx.signal?.aborted;
    if (userAborted) return;

    // Surface failures for any image not already warned about this session
    // (read images were warned in the tool_result handler; this catches
    // user-attached / custom-injected images). Per-image dedup via warnedHashes.
    if (ctx.hasUI) {
      const newlyFailed: string[] = [];
      for (let i = 0; i < imgs.length; i++) {
        if (descArr[i] === UNAVAILABLE) {
          const h = imageHash(imgs[i].mimeType, imgs[i].data);
          if (!warnedHashes.has(h)) newlyFailed.push(h);
        }
      }
      if (newlyFailed.length > 0) {
        for (const h of newlyFailed) warnedHashes.add(h);
        const reason = lastDescriberError || "unknown error";
        ctx.ui.notify(
          `pi-vision-handoff: image description failed — ${reason}. Vision model: ${config.visionModel}`,
          "warning",
        );
      }
    }

    let changed = false;
    for (const msg of messages) {
      const content = msg.content;
      if (!Array.isArray(content)) continue;
      let touched = false;
      const next: unknown[] = [];
      for (let i = 0; i < content.length; i++) {
        const block = content[i];
        const img = extractImageFromBlock(block);
        if (img) {
          next.push({ type: "text", text: descs.get(imageHash(img.mimeType, img.data)) ?? UNAVAILABLE });
          touched = true;
        } else {
          next.push(block);
        }
      }
      if (touched) {
        msg.content = next;
        changed = true;
      }
    }
    if (changed) return { messages: event.messages };
  });

  pi.on("model_select", (event, ctx) => {
    if (!ctx.hasUI) return;
    if (!isConfigured(config)) return;
    const model = event.model;
    if (!model) return;
    if (isHandoffTarget(model, config) && !isVisionModel(model)) {
      ctx.ui.notify(
        `pi-vision-handoff: active — images will be described by ${config.visionModel}`,
        "info",
      );
    }
  });

  pi.registerCommand("vision-handoff", {
    description: HANDOFF_COMMAND_DESCRIPTION,
    getArgumentCompletions(prefix: string) {
      const subcommands = ["select", "model", "status", "enable", "disable", "auto", "add", "remove", "clear", "help"];
      const matches = subcommands.filter((s) => s.startsWith(prefix));
      return matches.length > 0 ? matches.map((s) => ({ value: s, label: s })) : null;
    },
    handler: async (args, ctx) => {
      await handleHandoffCommand(ctx, args.trim());
    },
  });
}

async function handleHandoffCommand(ctx: ExtensionCommandContext, args: string): Promise<void> {
  const parts = args.split(/\s+/);
  const subcommand = parts[0]?.toLowerCase() ?? "";
  const rest = parts.slice(1).join(" ");

  // /vision-handoff (no args) or /vision-handoff select — interactive picker
  if (!subcommand || subcommand === "select") {
    await showSelector(ctx);
    return;
  }

  if (subcommand === "help") {
    ctx.ui.notify(
      [
        "pi-vision-handoff commands:",
        "  /vision-handoff                 Open interactive picker to choose the vision model",
        "  /vision-handoff select         Same as /vision-handoff",
        "  /vision-handoff model <p/id>   Set the vision model directly",
        "  /vision-handoff status         Show current config and active state",
        "  /vision-handoff enable         Enable vision handoff",
        "  /vision-handoff disable        Disable vision handoff (keeps configured model)",
        "  /vision-handoff auto <on|off>  Toggle automatic handoff for all non-vision models",
        "  /vision-handoff add <p/id>     Force handoff for an extra model",
        "  /vision-handoff remove <p/id>  Stop forcing handoff for a model",
        "  /vision-handoff clear          Clear the configured vision model",
        "  /vision-handoff help           This message",
        "",
        "Config: ~/.pi/agent/extensions/pi-vision-handoff.json",
        "Mechanism: before_agent_start warms a description cache; before_provider_request",
        "  swaps image blocks in the payload for the cached text description.",
      ].join("\n"),
      "info",
    );
    return;
  }

  if (subcommand === "status") {
    showStatus(ctx);
    return;
  }

  if (subcommand === "enable") {
    updateConfig(ctx, (c) => ({ ...c, enabled: true }), "Vision handoff enabled.");
    return;
  }

  if (subcommand === "disable") {
    updateConfig(ctx, (c) => ({ ...c, enabled: false }), "Vision handoff disabled.");
    return;
  }

  if (subcommand === "auto") {
    const value = rest.toLowerCase();
    if (value !== "on" && value !== "off") {
      ctx.ui.notify("Usage: /vision-handoff auto <on|off>", "warning");
      return;
    }
    const on = value === "on";
    updateConfig(
      ctx,
      (c) => ({ ...c, autoHandoff: on }),
      `Automatic handoff for non-vision models ${on ? "on" : "off"}.`,
    );
    return;
  }

  if (subcommand === "clear") {
    updateConfig(
      ctx,
      (c) => ({ ...c, visionModel: null }),
      "Vision model cleared — handoff inactive until you pick a model.",
    );
    return;
  }

  if (subcommand === "model") {
    if (!rest) {
      ctx.ui.notify("Usage: /vision-handoff model <provider/id>", "warning");
      return;
    }
    const parsed = parseModelRef(rest);
    if (!parsed) {
      ctx.ui.notify(`Invalid model reference: "${rest}". Use "provider/id".`, "error");
      return;
    }
    const model = ctx.modelRegistry.find(parsed.provider, parsed.id);
    if (!model) {
      ctx.ui.notify(`Model not found: ${rest}. Use /vision-handoff to pick from the list.`, "error");
      return;
    }
    const ref = formatModelRef(parsed.provider, parsed.id);
    updateConfig(ctx, (c) => ({ ...c, visionModel: ref }), `Vision model set to ${ref}.`);
    if (!isVisionModel(model)) {
      ctx.ui.notify(
        `Note: ${ref} does not declare image input — it may not describe images well.`,
        "warning",
      );
    }
    return;
  }

  if (subcommand === "add") {
    if (!rest) {
      ctx.ui.notify("Usage: /vision-handoff add <provider/id>", "warning");
      return;
    }
    const parsed = parseModelRef(rest);
    if (!parsed) {
      ctx.ui.notify(`Invalid model reference: "${rest}". Use "provider/id".`, "error");
      return;
    }
    const ref = formatModelRef(parsed.provider, parsed.id);
    updateConfig(
      ctx,
      (c) => ({ ...c, handoffModels: Array.from(new Set([...c.handoffModels, ref])) }),
      `Added ${ref} to handoff targets.`,
    );
    return;
  }

  if (subcommand === "remove") {
    if (!rest) {
      ctx.ui.notify("Usage: /vision-handoff remove <provider/id>", "warning");
      return;
    }
    const parsed = parseModelRef(rest);
    if (!parsed) {
      ctx.ui.notify(`Invalid model reference: "${rest}". Use "provider/id".`, "error");
      return;
    }
    const ref = formatModelRef(parsed.provider, parsed.id);
    const before = config.handoffModels.length;
    updateConfig(
      ctx,
      (c) => ({ ...c, handoffModels: c.handoffModels.filter((m) => m !== ref) }),
      `Removed ${ref} from handoff targets.`,
    );
    if (config.handoffModels.length === before) {
      ctx.ui.notify(`Note: ${ref} was not in the handoff list.`, "info");
    }
    return;
  }

  ctx.ui.notify(`Unknown subcommand: "${subcommand}". Use /vision-handoff help for usage.`, "warning");
}

function updateConfig(
  ctx: ExtensionCommandContext,
  transform: (c: VisionHandoffConfig) => VisionHandoffConfig,
  message: string,
): void {
  const next = transform(config);
  const path = writeConfig(next);
  config = next;
  visionModelCache = null;
  visionModelUnresolvedRef = null;
  ctx.ui.notify(`${message} (config: ${path})`, "info");
}

async function showSelector(ctx: ExtensionCommandContext): Promise<void> {
  if (!ctx.hasUI) {
    ctx.ui.notify("/vision-handoff requires interactive mode.", "error");
    return;
  }

  const allModels = ctx.modelRegistry
    .getAll()
    .map((m) => ({ provider: m.provider, id: m.id, name: m.name, input: m.input }));

  const result = await ctx.ui.custom<VisionModelSelectorResult>((tui, theme, _kb, done) => {
    const selector = new VisionModelSelectorComponent(theme, allModels, config.visionModel, (r) => done(r));
    return {
      render(width: number) {
        return selector.render(width);
      },
      invalidate() {
        selector.invalidate();
      },
      handleInput(data: string) {
        selector.handleInput(data);
        tui.requestRender();
      },
    };
  });

  if (!result || result.cancelled) {
    ctx.ui.notify("Vision handoff picker cancelled.", "info");
    return;
  }

  const ref = result.ref;
  updateConfig(ctx, (c) => ({ ...c, visionModel: ref }), ref ? `Vision model set to ${ref}` : "Vision model cleared");
  if (!ref) {
    ctx.ui.notify("Handoff is inactive until you pick a vision model.", "warning");
  }
}

function showStatus(ctx: ExtensionCommandContext): void {
  const lines: string[] = [];
  lines.push(`Vision handoff: ${config.enabled ? "enabled" : "disabled"}`);
  lines.push(`Vision model: ${config.visionModel ?? "(none — pick one with /vision-handoff)"}`);
  lines.push(`Auto handoff (non-vision models): ${config.autoHandoff ? "on" : "off"}`);
  lines.push(`Handoff targets (explicit): ${config.handoffModels.length ? config.handoffModels.join(", ") : "(none)"}`);
  lines.push(`maxTokens: ${config.maxTokens} · cacheMax: ${config.cacheMax} · maxDescriptionLines: ${config.maxDescriptionLines === 0 ? "unbounded" : config.maxDescriptionLines}`);

  const model = ctx.model;
  let active = false;
  if (isConfigured(config) && model) {
    active = isHandoffTarget(model, config);
  }
  lines.push(
    `Active for current model (${model ? formatModelRef(model.provider, model.id) : "none"}): ${active ? "yes" : "no"}`,
  );

  ctx.ui.notify(lines.join("\n"), "info");
}
