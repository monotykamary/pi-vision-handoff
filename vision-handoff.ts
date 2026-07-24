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
 *   before_agent_start → warm the description cache for attached images AND
 *     pasted clipboard image file paths found in the prompt text (pre-warm at
 *     paste-enter, concurrent with the agent's first response).
 *   optional async clipboard fallback → race matching read tool calls against a
 *     non-blocking steering-message injection; matching reads cancel delivery.
 *   tool_result (read) → loadDescription() each read-tool image and AWAIT the
 *     shared batch, so N parallel reads coalesce into ONE vision call. The
 *     descriptions land in the tool results before the agent's next turn.
 *   context → swap any remaining image blocks for their (now cached) text
 *     description on the cloned LLM-bound payload.
 *
 * This file is the wiring layer: pi event handlers + the /vision-handoff
 * command. The dataloader (`src/dataloader.ts`), describer (`src/describer.ts`),
 * image IO (`src/image.ts`), resource guards (`src/dispose.ts`), config/types
 * (`src/index.ts`), and usage/energy (`src/usage.ts`) live in `src/`.
 *
 * Image blocks are detected by shape across the four formats pi uses — see
 * `extractImageFromBlock` in `src/index.ts`. Descriptions are cached per image
 * hash (LRU, size = config.cacheMax) so the swap is instant by the time
 * `context` fires.
 */

import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext, ModelRegistry } from "@earendil-works/pi-coding-agent";
import type { Api, ImageContent, Model, TextContent } from "@earendil-works/pi-ai";
import { isAbsolute, resolve } from "node:path";
import {
  extractImageFromBlock,
  formatModelRef,
  isThinkingLevel,
  isVisionModel,
  NON_VISION_IMAGE_NOTE,
  parseModelRef,
  readConfig,
  stripNonVisionImageNote,
  writeConfig,
  HANDOFF_COMMAND_DESCRIPTION,
  type ExtractedImage,
  type VisionHandoffConfig,
} from "./src/index.js";
import {
  USAGE_ENTRY_TYPE,
  USAGE_EVENT_CHANNEL,
  type VisionHandoffUsageRecord,
} from "./src/usage.js";
import { DescriptionLoader, UNAVAILABLE, type LoaderDeps } from "./src/dataloader.js";
import { imageHash, findPastedImagePaths, readImageBuffer, readImageBufferBounded, resolvePrewarmImage, isOmittedImageNote } from "./src/image.js";
import { appendVisionError } from "./src/error-log.js";
import { resizeImage } from "@earendil-works/pi-coding-agent";
import { VisionModelSelectorComponent, type VisionModelSelectorResult } from "./src/vision-model-selector.js";
import { PrewarmEditor } from "./src/prewarm-editor.js";
import { Text } from "@earendil-works/pi-tui";

let config: VisionHandoffConfig = readConfig();

/** Most recent describer failure message (auth error, network error, abort,
 *  empty response, etc.). Set by the describer via the loader deps; surfaced
 *  to the user by the `context`/`tool_result` handler via ctx.ui.notify so a
 *  broken vision model stops looking like a silent "extension doesn't work".
 *  Cleared at the start of each describer attempt. */
let lastDescriberError: string | null = null;

/** Image hashes we've already warned the user about this session. Prevents the
 *  `context` hook (which fires before every LLM turn) from re-warning on the
 *  same failing images every turn — describer failures aren't cached, so
 *  without this guard a broken vision model would spam a warning per turn.
 *  Cleared on `session_start`. */
const warnedHashes = new Set<string>();

/** Current model, tracked so the paste-time prewarm gate can skip prewarming
 *  when the active model is vision-capable (handoff won't run → a prewarm
 *  would be a wasted vision call). Updated in session_start and model_select. */
let currentModel: Model<Api> | undefined | null;

/** Whether the paste-time prewarm editor is installed this session. False in
 *  non-TUI modes or when another extension already replaced the editor. */
let editorInstalled = false;

let visionModelCache: { ref: string; model: Model<Api> } | null = null;
let visionModelUnresolvedRef: string | null = null;

interface PendingAsyncClipboardHandoff {
  token: symbol;
  paths: Set<string>;
  cancelled: boolean;
}

interface PreparedClipboardImage {
  path: string;
  image: ExtractedImage;
  description: Promise<string>;
}

let pendingAsyncClipboardHandoff: PendingAsyncClipboardHandoff | null = null;

function cancelAsyncClipboardHandoff(): void {
  if (!pendingAsyncClipboardHandoff) return;
  pendingAsyncClipboardHandoff.cancelled = true;
  pendingAsyncClipboardHandoff = null;
}

function isPendingClipboardRead(input: unknown, cwd: string): boolean {
  const pending = pendingAsyncClipboardHandoff;
  if (!pending || !input || typeof input !== "object") return false;
  const raw = (input as { path?: unknown }).path;
  if (typeof raw !== "string") return false;
  const path = raw.startsWith("@") ? raw.slice(1) : raw;
  const absolute = isAbsolute(path) ? path : resolve(cwd, path);
  return pending.paths.has(absolute);
}

function scheduleAsyncClipboardInjection(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  paths: string[],
  prepared: Promise<PreparedClipboardImage | null>[],
): void {
  cancelAsyncClipboardHandoff();
  const pending: PendingAsyncClipboardHandoff = {
    token: Symbol("async-clipboard-handoff"),
    paths: new Set(paths),
    cancelled: false,
  };
  pendingAsyncClipboardHandoff = pending;

  void (async () => {
    const entries = (await Promise.all(prepared)).filter(
      (entry): entry is PreparedClipboardImage => entry !== null,
    );
    if (entries.length === 0) {
      if (pendingAsyncClipboardHandoff?.token === pending.token) {
        pendingAsyncClipboardHandoff = null;
      }
      return;
    }
    const descriptions = await Promise.all(entries.map((entry) => entry.description));

    // Always yield out of before_agent_start, even when paste-time prewarm made
    // every description an immediate cache hit. This lets Pi enter the active
    // agent run before sendMessage queues the fallback as a steering message.
    await new Promise<void>((resolveImmediate) => setImmediate(resolveImmediate));
    if (
      pending.cancelled ||
      pendingAsyncClipboardHandoff?.token !== pending.token ||
      !config.asyncClipboardHandoff ||
      !isConfigured(config)
    ) {
      return;
    }

    warnFailedImages(
      ctx,
      entries.map((entry) => entry.image),
      descriptions,
      lastDescriberError ?? "unknown error",
    );
    const content = [
      "Asynchronous vision handoff for pasted image path(s):",
      ...entries.map(
        (entry, index) => `${entry.path}\n${descriptions[index] ?? UNAVAILABLE}`,
      ),
    ].join("\n\n");

    pendingAsyncClipboardHandoff = null;
    try {
      pi.sendMessage(
        {
          customType: "vision-handoff-async",
          content,
          display: true,
          details: { imageCount: entries.length },
        },
        { deliverAs: "steer", triggerTurn: true },
      );
    } catch {
      // The session may have been replaced/reloaded while the description was
      // in flight. The stale extension instance must not affect the new one.
    }
  })().catch(() => {
    if (pendingAsyncClipboardHandoff?.token === pending.token) {
      pendingAsyncClipboardHandoff = null;
    }
  });
}

// Usage reporter; wired to pi.appendEntry + pi.events.emit in the default
// export. No-op until then so the describer is safe to call before wiring.
let reportUsage: (record: VisionHandoffUsageRecord) => void = () => {};

function resolveVisionModel(modelRegistry: ModelRegistry, ref: string): Model<Api> | null {
  if (visionModelCache && visionModelCache.ref === ref) return visionModelCache.model;
  const parsed = parseModelRef(ref);
  if (!parsed) return null;
  const model = modelRegistry.find(parsed.provider, parsed.id);
  if (!model) return null;
  visionModelCache = { ref, model };
  return model;
}

const loaderDeps: LoaderDeps = {
  getConfig: () => config,
  resolveVisionModel,
  reportUsage: (record) => reportUsage(record),
  setLastError: (msg) => {
    lastDescriberError = msg;
  },
};
const loader = new DescriptionLoader(loaderDeps);

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

/** Whether paste-time prewarm should fire for a text change right now: the
 *  opt-in flag is on, handoff is configured, and the active model is a handoff
 *  target (so the prewarmed description will actually be consumed — a
 *  vision-capable model needs no handoff, so prewarming would waste a call). */
function shouldPrewarmPaste(): boolean {
  return config.prewarmPastedImages && isConfigured(config) && isHandoffTarget(currentModel, config);
}

/** Prewarm one pasted clipboard image path through the dataloader at paste
 *  time. Mirrors the before_agent_start clipboard prewarm, but binds only the
 *  model registry (no turn signal — the agent is idle at paste time) and
 *  resets the turn prompt to "" so a stale previous-turn prompt can't leak
 *  into the description. The user's question isn't typed yet at paste time, so
 *  the description is generated without question context (the documented
 *  tradeoff of the opt-in). If the user submits before this dispatch fires,
 *  before_agent_start overwrites the prompt — a benign bonus, not a bug. */
function prewarmClipboardPath(path: string, modelRegistry: ModelRegistry): void {
  const read = readImageBuffer(path);
  if (!read) return;
  resolvePrewarmImage(read.buf, read.mimeType, resizeImage)
    .then((img) => {
      if (!img) return;
      // Paste happens while the agent is idle (no run → no live signal).
      // Reset the turn-abort controller so a previous turn's ESC doesn't leave
      // it aborted and short-circuit this prewarm; the next submit's
      // `before_agent_start` resets again. The prewarm batch uses the loader's
      // controller and becomes abortable once a run starts and binds a live
      // signal.
      loader.resetTurnAbort();
      loader.setPendingTurnPrompt("");
      loader.bindTurnContext({ modelRegistry });
      loader.loadDescription(img).catch(() => {});
    })
    .catch(() => {});
}

/** Install the paste-time prewarm editor wrapper for this session. TUI only,
 *  and only when no other extension has replaced the editor — installing over
 *  a custom editor would clobber its input handling and break clipboard paste
 *  (pi wires paste-image to the outermost editor only). When a custom editor
 *  is present, paste-time prewarm is unavailable; submit-time prewarm
 *  (before_agent_start) still covers clipboard paths. */
function installPrewarmEditor(ctx: ExtensionContext): void {
  if (ctx.mode !== "tui") {
    editorInstalled = false;
    return;
  }
  if (ctx.ui.getEditorComponent()) {
    editorInstalled = false;
    return;
  }
  editorInstalled = true;
  ctx.ui.setEditorComponent((_tui, theme, keybindings) =>
    new PrewarmEditor(_tui, theme, keybindings, {
      modelRegistry: ctx.modelRegistry,
      shouldPrewarm: shouldPrewarmPaste,
      prewarmPath: prewarmClipboardPath,
    }),
  );
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

/** Notify once per failing image per session (dedup via warnedHashes), and log
 *  EVERY failure (even in headless/SDK mode with no UI) to the error log so the
 *  user can troubleshoot. The log entry's `phase: "warn"` carries the failing
 *  image hashes and the surfaced reason — when the reason is "unknown error",
 *  the real cause lives in the matching `batch`/`single` entry for those hashes. */
function warnFailedImages(
  ctx: ExtensionContext,
  imgs: ExtractedImage[],
  descs: string[],
  reason: string,
): void {
  const newlyFailed: string[] = [];
  for (let i = 0; i < imgs.length; i++) {
    if (descs[i] === UNAVAILABLE) {
      const h = imageHash(imgs[i].mimeType, imgs[i].data);
      if (!warnedHashes.has(h)) newlyFailed.push(h);
    }
  }
  if (newlyFailed.length === 0) return;
  for (const h of newlyFailed) warnedHashes.add(h);
  // Always log: troubleshooting must work in headless/SDK mode too, where the
  // notify below never fires.
  appendVisionError({
    phase: "warn",
    reason,
    visionModel: config.visionModel,
    imageHashes: newlyFailed,
    imageCount: newlyFailed.length,
    activeModel: ctx.model ? formatModelRef(ctx.model.provider, ctx.model.id) : undefined,
  });
  if (!ctx.hasUI) return;
  ctx.ui.notify(
    `pi-vision-handoff: image description failed — ${reason}. Vision model: ${config.visionModel}`,
    "warning",
  );
}

export default function (pi: ExtensionAPI) {
  config = readConfig();

  pi.registerMessageRenderer("vision-handoff-async", (message, { expanded }, theme) => {
    const details = message.details as { imageCount?: number } | undefined;
    const count = details?.imageCount ?? 1;
    const label = `Vision handoff · ${count} pasted image${count === 1 ? "" : "s"}`;
    const hint = expanded ? "Ctrl+O to collapse" : "Ctrl+O to expand";
    const summary = theme.fg("dim", `👁 ${label} · ${hint}`);
    if (!expanded) return new Text(summary, 0, 0);
    // Apply the dim (grey) style per line: the TUI appends a full SGR reset at
    // the end of each rendered line, so a single style wrapper would only tint
    // the first line. Text wraps with wrapTextWithAnsi (ANSI-preserving), and
    // each line here carries its own dim code so wrapped continuations stay grey.
    const contentStr = typeof message.content === "string" ? message.content : "";
    const body = contentStr.length > 0
      ? contentStr.split("\n").map((line) => theme.fg("dim", line)).join("\n")
      : "";
    return new Text(body ? `${summary}\n${body}` : summary, 0, 0);
  });

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

  pi.on("session_start", async (_event, ctx) => {
    cancelAsyncClipboardHandoff();
    // Reload in case the user edited the config on disk from another session.
    config = readConfig();
    visionModelCache = null;
    visionModelUnresolvedRef = null;
    warnedHashes.clear();
    loader.reset();
    currentModel = ctx.model;
    installPrewarmEditor(ctx);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    cancelAsyncClipboardHandoff();
    if (!isConfigured(config)) return;
    if (!isHandoffTarget(ctx.model, config)) return;

    // Fresh turn → fresh turn-abort controller. `before_agent_start` fires
    // BEFORE the agent run starts, so `ctx.signal` is undefined here (the run's
    // abort signal doesn't exist yet — it's created in `agent.prompt()` →
    // `runWithLifecycle`, which runs AFTER this event). The prewarm below
    // dispatches describer batches now, and they must be abortable once the
    // run's live signal arrives — so reset the loader's turn-abort controller
    // here, and let the later `tool_result`/`context` binds forward the live
    // signal into it. Without this reset, a previous turn's ESC would leave
    // the controller aborted and every dispatch would short-circuit to
    // UNAVAILABLE.
    loader.resetTurnAbort();

    // Capture this turn's user prompt so every image in the turn — attached or
    // read via the read tool — is described in the same request context.
    loader.setPendingTurnPrompt(event.prompt || "");
    loader.bindTurnContext(ctx);

    if (!resolveVisionModel(ctx.modelRegistry, config.visionModel!)) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    // PRE-WARM at paste-enter via the DataLoader. Two image sources land here:
    //
    // 1. Attached image blocks (event.images) — vision-capable targets where
    //    the user message itself carries image blocks (e.g. `pi --image`).
    //
    // 2. Pasted image FILE PATHS in the prompt text — the common non-vision
    //    flow. pi's `handleClipboardImagePaste` (and other paste mechanisms like
    //    localterm-paste) write each pasted image to a temp file and insert the
    //    PATH as
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
    // coalesce into ONE batch dispatched via `setImmediate` after the
    // microtask cascade settles.
    for (const image of event.images ?? []) {
      if (!image || image.type !== "image" || !image.data) continue;
      loader.loadDescription({ data: image.data, mimeType: image.mimeType || "image/png" }).catch(() => {});
    }

    // Pasted clipboard image paths in the prompt text — resolve each to the
    // SAME ExtractedImage pi's `read` tool will emit, then warm the loader so
    // the `tool_result`'s `loadDescription()` is a cache hit (no wasted vision
    // call). pi's read tool resizes images by default: for a small image it
    // returns the raw bytes unchanged (our raw read matches), but for an
    // oversized image it RE-ENCODES via Photon — so we run the same
    // `resizeImage` pipeline to produce a matching key. The resize branch is
    // async (worker thread) and fire-and-forget; its `loadDescription()` lands
    // in a later batch than the no-resize images, so a mixed small+oversized
    // paste may split into two vision calls (still no WASTED call — each
    // describes an image the agent will see). A file that can't be read, isn't
    // a supported image, or fails to resize is skipped (the agent's `read`
    // will still describe it via `tool_result` if it emits an image block).
    const clipboardPaths = findPastedImagePaths(event.prompt || "");
    const preparedClipboardImages = clipboardPaths.map(
      async (path): Promise<PreparedClipboardImage | null> => {
        try {
          const read = readImageBuffer(path);
          if (!read) return null;
          const image = await resolvePrewarmImage(read.buf, read.mimeType, resizeImage);
          if (!image) return null;
          const description = loader.loadDescription(image);
          description.catch(() => {});
          return { path, image, description };
        } catch {
          return null;
        }
      },
    );

    if (config.asyncClipboardHandoff && clipboardPaths.length > 0) {
      scheduleAsyncClipboardInjection(pi, ctx, clipboardPaths, preparedClipboardImages);
    } else {
      // Start every preparation task even when no consumer awaits it. Each task
      // calls loadDescription as soon as resize completes, preserving the
      // original fire-and-forget submit-time prewarm behavior.
      for (const prepared of preparedClipboardImages) prepared.catch(() => {});
    }
  });

  // A direct read and a nested pi.read both emit a read tool_call. If it targets
  // one of this turn's pasted clipboard paths before the async fallback is
  // injected, the normal tool_result/context path wins and the queued custom
  // message is cancelled. The in-flight description is deliberately retained:
  // the read result reuses it as a cache/in-flight hit.
  pi.on("tool_call", (event, ctx) => {
    if (event.toolName !== "read" && event.toolName !== "pi.read") return;
    if (isPendingClipboardRead(event.input, ctx.cwd)) {
      cancelAsyncClipboardHandoff();
    }
  });

  pi.on("session_shutdown", () => {
    cancelAsyncClipboardHandoff();
  });

  // The PRIMARY injection point: the `read` tool's `tool_result` handler.
  // When the agent reads image files, this fires for each read result. It
  // calls the loader's `loadDescription(img)` for every image block and
  // AWAITS the shared batch — so N parallel reads (pi runs `read` via
  // Promise.all) coalesce into ONE batched vision call: pi fires each read's
  // `tool_result` as its I/O completes (poll phase), and the loader's
  // `setImmediate` dispatch defers to the check phase AFTER the whole poll
  // iteration, so reads completing together land in ONE batch and all resolve
  // together.
  //
  // Both a direct agent `read` and a NESTED `pi.read` inside fabric_exec reach
  // this handler and take the SAME path: WARM the cache (during the free
  // tool-result phase), strip pi's misleading non-vision note, and KEEP the
  // image block so kitty renders it inline and /resume retains it. The
  // `context` hook swaps image→description on the LLM-bound clone before the
  // next turn, so the text-only agent receives the description as text. This
  // holds for fabric too: pi-fabric re-attaches the nested read's image to the
  // fabric_exec tool-result content, which IS the agent's message context — so
  // the `context` hook sees and swaps it, exactly like a direct read.
  pi.on("tool_result", async (event, ctx) => {
    if (!isConfigured(config)) return;
    if (event.toolName !== "read") return;
    const content = event.content;
    if (!Array.isArray(content)) return;

    // Collect image blocks in this read result. (The read tool emits image
    // blocks even for non-vision models — they reach here untouched.)
    if (!isHandoffTarget(ctx.model, config)) return;
    const imgs: ExtractedImage[] = [];
    let hasImageBlock = false;
    for (const block of content) {
      const img = extractImageFromBlock(block);
      if (img) {
        imgs.push(img);
        hasImageBlock = true;
      }
    }

    // Fallback: `read` detected an image but `processImage` failed (Photon
    // unavailable / decode fail / convert fail / couldn't resize below the
    // inline limit), so it emitted a "[Image omitted: …]" text note with NO
    // image block. The image-block path above never sees it, so the image
    // would go undescribed and the model would be told the image was "omitted".
    // Re-read the raw file and describe its bytes directly — the vision model
    // decodes them itself (no Photon needed). This recovers the Photon-
    // unavailable and under-vision-model-limit cases; APNG/unsupported are
    // rejected by the sniff (the vision model can't decode them either).
    let omittedNoteIndices: number[] = [];
    if (!hasImageBlock) {
      for (let i = 0; i < content.length; i++) {
        const block = content[i];
        if (
          block &&
          typeof block === "object" &&
          (block as { type: string }).type === "text" &&
          typeof (block as { text: string }).text === "string" &&
          isOmittedImageNote((block as { text: string }).text)
        ) {
          omittedNoteIndices.push(i);
        }
      }
      if (omittedNoteIndices.length > 0) {
        const inputPath = event.input.path;
        if (typeof inputPath === "string") {
          const resolved = isAbsolute(inputPath) ? inputPath : resolve(ctx.cwd, inputPath);
          const read = readImageBufferBounded(resolved);
          if (read) imgs.push({ data: read.buf.toString("base64"), mimeType: read.mimeType });
        }
      }
    }

    if (imgs.length === 0) return;
    loader.bindTurnContext(ctx);
    if (!resolveVisionModel(ctx.modelRegistry, config.visionModel!)) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    // load() each image — synchronous calls that push into the current batch
    // and return memoized promises — then await them all. pi fires each read's
    // `tool_result` event as that read's I/O completes (poll phase); the
    // loader's `setImmediate` dispatch defers to the check phase, AFTER the
    // whole poll iteration, so reads completing together (the common case for
    // cached local files) land in ONE batch — ONE vision call for the whole
    // read set, not N. Awaiting here runs the describer during the tool-result
    // phase (free time), so the batch is COMPLETE before `context` fires,
    // making `context` a non-blocking cache hit instead of a cold miss.
    const descs = await Promise.all(imgs.map((img) => loader.loadDescription(img)));

    // On user abort, leave the result untouched — pi is tearing the turn
    // down and the LLM-bound content won't be sent.
    if (ctx.signal?.aborted) return;

    warnFailedImages(ctx, imgs, descs, lastDescriberError ?? "unknown error");

    // Recovery path: no image block was emitted, so there's nothing for the
    // `context` hook to swap. Replace each "[Image omitted: …]" note directly
    // with its description (the recovered raw-bytes image is the only img here,
    // so descs aligns 1:1 with omittedNoteIndices).
    if (!hasImageBlock) {
      const recovered = content.slice();
      for (let i = 0; i < omittedNoteIndices.length && i < descs.length; i++) {
        recovered[omittedNoteIndices[i]] = { type: "text", text: descs[i] };
      }
      return { content: recovered as (TextContent | ImageContent)[] };
    }

    // Warm the cache (done above) and strip pi's
    // `[Current model does not support images…]` note from text blocks —
    // since the handoff replaces the image with a description in the
    // `context` hook, that note is misleading (the agent WILL receive the
    // image's content, as text). Keep the image block itself so kitty still
    // renders it inline and `/resume` retains it; the `context` hook swaps
    // the image for its description in the LLM-bound clone before the next turn.
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
      for (const block of content) {
        const img = extractImageFromBlock(block);
        if (!img) continue;
        anyImage = true;
        byHash.set(imageHash(img.mimeType, img.data), img);
      }
    }
    if (!anyImage) return;
    if (!isHandoffTarget(ctx.model, config)) return;
    loader.bindTurnContext(ctx);
    if (!resolveVisionModel(ctx.modelRegistry, config.visionModel!)) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    // Cache hits (warmed by before_agent_start / tool_result) resolve
    // instantly; any remaining misses queue into the loader's current batch.
    const imgs = [...byHash.values()];
    const descArr = await Promise.all(imgs.map((img) => loader.loadDescription(img)));
    const descs = new Map<string, string>();
    for (let i = 0; i < imgs.length; i++) {
      descs.set(imageHash(imgs[i].mimeType, imgs[i].data), descArr[i]);
    }

    if (ctx.signal?.aborted) return;

    warnFailedImages(ctx, imgs, descArr, lastDescriberError ?? "unknown error");

    let changed = false;
    for (const msg of messages) {
      const content = msg.content;
      if (!Array.isArray(content)) continue;
      let touched = false;
      const next: unknown[] = [];
      for (const block of content) {
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
    currentModel = event.model;
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
      const subcommands = ["select", "model", "status", "enable", "disable", "auto", "thinking", "prewarm", "fallback", "add", "remove", "clear", "help"];
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
        "  /vision-handoff thinking <off|minimal|low|medium|high|xhigh|max>",
        "                               Set the vision describer's thinking effort (off = disabled)",
        "  /vision-handoff prewarm <on|off>",
        "                               Toggle describing pasted images at paste-time (opt-in, off by default)",
        "  /vision-handoff fallback <on|off>",
        "                               Inject pasted-image descriptions asynchronously when no matching read wins",
        "  /vision-handoff add <p/id>     Force handoff for an extra model",
        "  /vision-handoff remove <p/id>  Stop forcing handoff for a model",
        "  /vision-handoff clear          Clear the configured vision model",
        "  /vision-handoff help           This message",
        "",
        "Config: ~/.pi/agent/extensions/pi-vision-handoff.json",
        "Mechanism: before_agent_start warms a description cache; tool_result loads",
        "  read images through a dataloader (one batched vision call); context swaps",
        "  image blocks in the payload for the cached text description.",
        "  prewarm on wraps the editor to describe pasted images at paste-time.",
        "  fallback on asynchronously injects a collapsed description unless a matching read wins.",
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

  if (subcommand === "thinking") {
    handleThinkingSubcommand(ctx, rest);
    return;
  }

  if (subcommand === "prewarm") {
    handlePrewarmSubcommand(ctx, rest);
    return;
  }

  if (subcommand === "fallback") {
    handleFallbackSubcommand(ctx, rest);
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
  if (!next.asyncClipboardHandoff) cancelAsyncClipboardHandoff();
  const path = writeConfig(next);
  config = next;
  visionModelCache = null;
  visionModelUnresolvedRef = null;
  ctx.ui.notify(`${message} (config: ${path})`, "info");
}

/** Resolve a `/vision-handoff thinking <level>` argument into a
 *  `(thinking, thinkingLevel)` pair. `off` disables thinking; any of the
 *  {@link THINKING_LEVELS} enables it at that effort. */
function handleThinkingSubcommand(ctx: ExtensionCommandContext, rest: string): void {
  const arg = rest.trim().toLowerCase();
  if (!arg) {
    ctx.ui.notify(
      `Thinking: ${config.thinking ? `on (${config.thinkingLevel})` : "off"}.\n` +
        `Usage: /vision-handoff thinking <off|minimal|low|medium|high|xhigh|max>`,
      "info",
    );
    return;
  }
  if (arg === "off") {
    updateConfig(ctx, (c) => ({ ...c, thinking: false }), "Vision describer thinking off.");
    return;
  }
  if (!isThinkingLevel(arg)) {
    ctx.ui.notify(
      `Unknown thinking level: "${arg}". Use off, minimal, low, medium, high, xhigh, or max.`,
      "error",
    );
    return;
  }
  const level = arg;
  updateConfig(
    ctx,
    (c) => ({ ...c, thinking: true, thinkingLevel: level }),
    `Vision describer thinking on (${level}).`,
  );
}

/** Handle `/vision-handoff prewarm <on|off>` — toggle paste-time prewarm. */
function handlePrewarmSubcommand(ctx: ExtensionCommandContext, rest: string): void {
  const value = rest.trim().toLowerCase();
  if (!value) {
    ctx.ui.notify(
      `Paste-time prewarm: ${config.prewarmPastedImages ? "on" : "off"}.\n` +
        `Usage: /vision-handoff prewarm <on|off>`,
      "info",
    );
    return;
  }
  if (value !== "on" && value !== "off") {
    ctx.ui.notify("Usage: /vision-handoff prewarm <on|off>", "warning");
    return;
  }
  const on = value === "on";
  const note = on
    ? editorInstalled
      ? "Paste-time prewarm on — pasted images are described the instant their path lands in the prompt (before submit)."
      : "Paste-time prewarm on — but another custom editor extension is active, so it's unavailable. Submit-time prewarm still works; disable the other editor extension (or /vision-handoff prewarm off) to silence this."
    : "Paste-time prewarm off — images are described at submit time (default).";
  updateConfig(ctx, (c) => ({ ...c, prewarmPastedImages: on }), note);
}

/** Handle /vision-handoff fallback <on|off>. */
function handleFallbackSubcommand(ctx: ExtensionCommandContext, rest: string): void {
  const value = rest.trim().toLowerCase();
  if (!value) {
    ctx.ui.notify(
      `Async pasted-path fallback: ${config.asyncClipboardHandoff ? "on" : "off"}.\n` +
        "Usage: /vision-handoff fallback <on|off>",
      "info",
    );
    return;
  }
  if (value !== "on" && value !== "off") {
    ctx.ui.notify("Usage: /vision-handoff fallback <on|off>", "warning");
    return;
  }
  const on = value === "on";
  updateConfig(
    ctx,
    (c) => ({ ...c, asyncClipboardHandoff: on }),
    on
      ? "Async pasted-path fallback on — a matching read wins the race; otherwise the description is injected as a collapsed message."
      : "Async pasted-path fallback off.",
  );
}

async function showSelector(ctx: ExtensionCommandContext): Promise<void> {
  if (!ctx.hasUI) {
    ctx.ui.notify("/vision-handoff requires interactive mode.", "error");
    return;
  }

  const allModels = ctx.modelRegistry
    .getAll()
    .map((m) => ({ provider: m.provider, id: m.id, name: m.name, input: m.input, reasoning: m.reasoning }));

  if (ctx.mode !== "tui") {
    const modelItems = ["None", ...allModels.map((m) => `${m.provider}/${m.id}`)];
    const modelPick = await ctx.ui.select("Vision model", modelItems);
    if (modelPick === undefined) return;
    const ref = modelPick === "None" ? null : modelPick;
    const thinkingPick = await ctx.ui.select("Thinking", ["on", "off"]);
    if (thinkingPick === undefined) return;
    const thinking = thinkingPick === "on";
    const thinkingLevel = config.thinkingLevel;
    const fallbackPick = await ctx.ui.select("Async pasted-path fallback", ["on", "off"]);
    if (fallbackPick === undefined) return;
    const asyncClipboardHandoff = fallbackPick === "on";
    const thinkingNote = thinking
      ? `thinking on (${thinkingLevel})${ref ? " \u2014 applies only if the vision model supports reasoning" : ""}`
      : "thinking off";
    updateConfig(
      ctx,
      (c) => ({ ...c, visionModel: ref, thinking, thinkingLevel, asyncClipboardHandoff }),
      ref ? `Vision model set to ${ref} \u00b7 ${thinkingNote}` : `Vision model cleared \u00b7 ${thinkingNote}`,
    );
    if (!ref) {
      ctx.ui.notify("Handoff is inactive until you pick a vision model.", "warning");
    }
    return;
  }

  const result = await ctx.ui.custom<VisionModelSelectorResult>((tui, theme, _kb, done) => {
    const selector = new VisionModelSelectorComponent(
      theme,
      allModels,
      config.visionModel,
      config.thinking,
      config.thinkingLevel,
      config.asyncClipboardHandoff,
      (r) => done(r),
    );
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
  const thinking = result.thinking;
  const thinkingLevel = result.thinkingLevel;
  const asyncClipboardHandoff = result.asyncClipboardHandoff;
  // Fold the thinking state into the single updateConfig notify so the
  // "thinking off" message can't overwrite the model-change message.
  const thinkingNote = thinking
    ? `thinking on (${thinkingLevel})${ref ? " — applies only if the vision model supports reasoning" : ""}`
    : "thinking off";
  updateConfig(
    ctx,
    (c) => ({ ...c, visionModel: ref, thinking, thinkingLevel, asyncClipboardHandoff }),
    ref ? `Vision model set to ${ref} · ${thinkingNote}` : `Vision model cleared · ${thinkingNote}`,
  );
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
  lines.push(`Thinking: ${config.thinking ? `on (${config.thinkingLevel})` : "off"}`);
  lines.push(`Paste-time prewarm: ${config.prewarmPastedImages ? `on${editorInstalled ? "" : " (inactive — another custom editor is active)"}` : "off"}`);
  lines.push(`Async pasted-path fallback: ${config.asyncClipboardHandoff ? "on" : "off"}`);
  lines.push(`maxTokens: ${config.maxTokens ?? "unbounded"} · cacheMax: ${config.cacheMax} · maxDescriptionLines: ${config.maxDescriptionLines === 0 ? "unbounded" : config.maxDescriptionLines}`);

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
