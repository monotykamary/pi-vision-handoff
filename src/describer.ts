/**
 * The vision describer: calls a vision-capable model via pi-ai's `complete()`
 * to produce text descriptions of images.
 *
 * Two entry points:
 *   - {@link runBatch}: ONE batched call describing N images at once (the
 *     dataloader's dispatch path). The model is asked to emit one delimited
 *     `<<<IMAGE k>>> … <<<END>>>` section per image so the response can be
 *     split back into per-image descriptions.
 *   - {@link describeSingle}: one call for one image (no delimiters). The
 *     robust per-image fallback used when a batched response couldn't be split.
 *
 * Resource lifetimes (fetch interceptor, timeout timer, turn-abort wire) are
 * managed with the `using` keyword via the {@link Disposable} guards in
 * `dispose.ts`, replacing the manual `try`/`finally` cleanup the old code
 * carried. Disposing is lexical and exception-safe: a thrown `complete()`
 * still tears down the timer, uninstalls the interceptor, and detaches the
 * abort listener.
 */

import type { Api, ImageContent, Message, Model, TextContent } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai/compat";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import {
  batchUserPrompt,
  DEFAULT_USER_PROMPT_PREFIX,
  DEFAULT_VISION_PROMPT,
  describeTimeoutMs,
  parseBatchedDescriptions,
  type ExtractedImage,
  type VisionHandoffConfig,
} from "./index.js";
import {
  buildUsageRecord,
  describeAls,
  EMPTY_ENERGY_CAPTURE,
  type DescribeContext,
  type VisionHandoffEnergyCapture,
  type VisionHandoffUsageRecord,
} from "./usage.js";
import { imageHash } from "./image.js";
import { abortWireGuard, fetchInterceptorGuard, timeoutGuard, type AbortWire } from "./dispose.js";

/** Dependencies the describer can't own itself (held by the engine). */
export interface DescriberDeps {
  /** Report a usage+energy record for one real describer call (cache hits emit none). */
  reportUsage(record: VisionHandoffUsageRecord): void;
  /** Set the most-recent describer failure message (surfaced to the user by the engine).
   *  Pass `null` to clear before a fresh attempt. */
  setLastError(msg: string | null): void;
}

/** The result of a batched describer call: per-image raw descriptions keyed by hash. */
export type BatchResult = Map<string, string>;

/** Describe N images with ONE batched `complete()` call and split the response
 *  back into per-image descriptions. Returns a map keyed by image hash; an
 *  image whose section failed to parse is omitted (the caller treats omission
 *  as "description unavailable"). On a genuine call failure (auth, abort,
 *  empty) the map is empty and `deps.setLastError` records the reason.
 *
 *  A failed delimiter-parse is a failed BATCH, not a failed description: the
 *  unparsed images fall back to parallel single-image calls (no delimiters to
 *  cooperate with), so the common cooperative case stays one call while an
 *  uncooperative model still gets every image described. */
export async function runBatch(
  misses: { img: ExtractedImage; hash: string }[],
  userPrompt: string,
  visionModel: Model<Api>,
  modelRegistry: ModelRegistry,
  cfg: VisionHandoffConfig,
  deps: DescriberDeps,
  turnSignal?: AbortSignal,
): Promise<BatchResult> {
  const out: BatchResult = new Map();
  const auth = await modelRegistry.getApiKeyAndHeaders(visionModel);
  if (!auth.ok || !auth.apiKey) {
    deps.setLastError(
      !auth.ok ? auth.error : `No API key for vision model "${visionModel.provider}/${visionModel.id}"`,
    );
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
  using fetchGuard = fetchInterceptorGuard();
  using timer = timeoutGuard(timeoutMs, () => {
    timedOut = true;
    controller.abort();
  });
  using abortWire = abortWireGuard(turnSignal, controller);

  const describeCtx: DescribeContext = { energyReader: undefined };
  try {
    const response = await describeAls.run(describeCtx, async () =>
      complete(
        visionModel,
        { systemPrompt, messages: [userMessage] },
        { apiKey: auth.apiKey, headers: auth.headers, signal: controller.signal, maxTokens: cfg.maxTokens },
      ),
    );
    const capture = await readCapture(describeCtx);
    const hashes = misses.map((m) => m.hash);
    const record = buildUsageRecord(response, capture, visionModel, hashes[0], hashes.length > 1 ? hashes : undefined);
    if (record) deps.reportUsage(record);
    if (response.stopReason === "aborted" || response.stopReason === "error") {
      setStopReasonError(deps, response.stopReason, response.errorMessage, abortWire, timedOut, timeoutMs);
      return out;
    }
    const text = response.content
      .filter((c): c is TextContent => c.type === "text")
      .map((c) => c.text)
      .join("\n")
      .trim();
    if (!text) {
      deps.setLastError("vision model returned an empty description");
      return out;
    }
    const parsed = parseBatchedDescriptions(text, misses.length);
    for (let i = 0; i < misses.length; i++) {
      const d = parsed[i];
      if (d) out.set(misses[i].hash, d);
    }
    // Fallback: a failed delimiter-parse is NOT a failed description — it's a
    // failed batching. Describe each unparsed image with its own single-image
    // call (no delimiters to cooperate with). The calls run in parallel so
    // their results still arrive together, not sequentially.
    const unparsed = misses.filter((m) => !out.has(m.hash));
    if (unparsed.length > 0) {
      const fallbacks = await Promise.all(
        unparsed.map((m) => describeSingle(m.img, userPrompt, visionModel, modelRegistry, cfg, deps, turnSignal)),
      );
      for (let i = 0; i < unparsed.length; i++) {
        if (fallbacks[i]) out.set(unparsed[i].hash, fallbacks[i]!);
      }
    }
    return out;
  } catch (err) {
    deps.setLastError(
      timedOut ? `describer timed out after ${timeoutMs / 1000}s` : err instanceof Error ? err.message : String(err),
    );
    return out;
  } finally {
    // The `using` guards above already released the fetch interceptor, timer,
    // and abort wire. Only the energy tee needs an explicit unhandled-rejection
    // swallow: if the main stream aborted, the tee rejects too.
    describeCtx.energyReader?.catch(() => {});
  }
}

/** Describe a single image with one `complete()` call and return the RAW
 *  description (no envelope, no truncation). Returns null on any genuine
 *  failure (auth, abort/error, empty) so the caller only caches `UNAVAILABLE`
 *  when a real describer attempt failed. */
export async function describeSingle(
  img: ExtractedImage,
  userPrompt: string,
  visionModel: Model<Api>,
  modelRegistry: ModelRegistry,
  cfg: VisionHandoffConfig,
  deps: DescriberDeps,
  turnSignal?: AbortSignal,
): Promise<string | null> {
  const auth = await modelRegistry.getApiKeyAndHeaders(visionModel);
  if (!auth.ok || !auth.apiKey) {
    deps.setLastError(
      !auth.ok ? auth.error : `No API key for vision model "${visionModel.provider}/${visionModel.id}"`,
    );
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
  using fetchGuard = fetchInterceptorGuard();
  using timer = timeoutGuard(timeoutMs, () => {
    timedOut = true;
    controller.abort();
  });
  using abortWire = abortWireGuard(turnSignal, controller);

  const describeCtx: DescribeContext = { energyReader: undefined };
  try {
    const response = await describeAls.run(describeCtx, async () =>
      complete(
        visionModel,
        { systemPrompt, messages: [userMessage] },
        { apiKey: auth.apiKey, headers: auth.headers, signal: controller.signal, maxTokens: cfg.maxTokens },
      ),
    );
    const capture = await readCapture(describeCtx);
    const hash = imageHash(img.mimeType, img.data);
    const record = buildUsageRecord(response, capture, visionModel, hash);
    if (record) deps.reportUsage(record);
    if (response.stopReason === "aborted" || response.stopReason === "error") {
      setStopReasonError(deps, response.stopReason, response.errorMessage, abortWire, timedOut, timeoutMs);
      return null;
    }
    const text = response.content
      .filter((c): c is TextContent => c.type === "text")
      .map((c) => c.text)
      .join("\n")
      .trim();
    if (!text) deps.setLastError("vision model returned an empty description");
    return text || null;
  } catch (err) {
    deps.setLastError(
      timedOut ? `describer timed out after ${timeoutMs / 1000}s` : err instanceof Error ? err.message : String(err),
    );
    return null;
  } finally {
    describeCtx.energyReader?.catch(() => {});
  }
}

/** Read the energy tee for a describer call, if one was captured. Returns the
 *  empty capture when there is no reader (non-Neuralwatt models) or the tee
 *  aborted with the main stream. */
async function readCapture(describeCtx: DescribeContext): Promise<VisionHandoffEnergyCapture> {
  if (!describeCtx.energyReader) return EMPTY_ENERGY_CAPTURE;
  try {
    return await describeCtx.energyReader;
  } catch {
    return EMPTY_ENERGY_CAPTURE;
  }
}

/** Translate a non-OK stopReason into a user-facing failure message, unless the
 *  abort came from the user cancelling the turn (then stay silent — no warning
 *  for a deliberate cancel). */
function setStopReasonError(
  deps: DescriberDeps,
  stopReason: string,
  errorMessage: string | undefined,
  abortWire: AbortWire,
  timedOut: boolean,
  timeoutMs: number,
): void {
  if (abortWire.userAborted()) return; // user cancelled the turn — no warning
  if (timedOut) {
    deps.setLastError(`describer timed out after ${timeoutMs / 1000}s`);
    return;
  }
  deps.setLastError(`vision model returned stopReason "${stopReason}"${errorMessage ? ": " + errorMessage : ""}`);
}
