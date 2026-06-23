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
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { complete } from "@earendil-works/pi-ai";
import type { Api, ImageContent, Message, Model, TextContent } from "@earendil-works/pi-ai";
import {
  DEFAULT_USER_PROMPT_PREFIX,
  DEFAULT_VISION_PROMPT,
  DESCRIBE_TIMEOUT_MS,
  HANDOFF_COMMAND_DESCRIPTION,
  IMAGE_PLACEHOLDER_PREFIX,
  IMAGE_PLACEHOLDER_SUFFIX,
  USAGE_ENTRY_TYPE,
  USAGE_EVENT_CHANNEL,
  EMPTY_ENERGY_CAPTURE,
  buildUsageRecord,
  describeAls,
  extractImageFromBlock,
  formatModelRef,
  insertImageDescriptions,
  installFetchInterceptor,
  isVisionModel,
  makeReplacementText,
  parseModelRef,
  readConfig,
  truncateDescription,
  uninstallFetchInterceptor,
  writeConfig,
  type DescribeContext,
  type VisionHandoffConfig,
  type VisionHandoffEnergyCapture,
  type VisionHandoffUsageRecord,
} from "./src/index.js";
import { VisionModelSelectorComponent, type VisionModelSelectorResult } from "./src/vision-model-selector.js";

const UNAVAILABLE = `${IMAGE_PLACEHOLDER_PREFIX}description unavailable${IMAGE_PLACEHOLDER_SUFFIX}`;

// Usage reporter; wired to pi.appendEntry + pi.events.emit in the default
// export. No-op until then so describeImage is safe to call before wiring.
let reportUsage: (record: VisionHandoffUsageRecord) => void = () => {};

let config: VisionHandoffConfig = readConfig();

/** User prompt for the current agent turn, captured from before_agent_start. */
let pendingTurnPrompt: string | null = null;

const visionCache = new Map<string, Promise<string>>();
let visionModelCache: { ref: string; model: Model<Api> } | null = null;
let visionModelUnresolvedRef: string | null = null;

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

async function describeImage(
  data: string,
  mimeType: string,
  userPrompt: string,
  visionModel: Model<Api>,
  modelRegistry: ModelRegistry,
  cfg: VisionHandoffConfig,
): Promise<string> {
  const key = imageHash(mimeType, data);
  const cached = visionCache.get(key);
  if (cached) return cached;

  const promise = (async (): Promise<string> => {
    const auth = await modelRegistry.getApiKeyAndHeaders(visionModel);
    if (!auth.ok || !auth.apiKey) return UNAVAILABLE;

    const prefix = cfg.userPromptPrefix ?? DEFAULT_USER_PROMPT_PREFIX;
    const systemPrompt = cfg.prompt ?? DEFAULT_VISION_PROMPT;

    const content: (TextContent | ImageContent)[] = [];
    if (userPrompt && userPrompt.trim()) {
      content.push({ type: "text", text: prefix + userPrompt });
    } else {
      content.push({ type: "text", text: "Describe this image." } satisfies TextContent);
    }
    content.push({ type: "image", data, mimeType } satisfies ImageContent);

    const userMessage: Message = {
      role: "user",
      content,
      timestamp: Date.now(),
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), DESCRIBE_TIMEOUT_MS);
    // Energy/token capture for this describer call. The fetch interceptor is
    // refcount-installed around the complete() window and routes the teed
    // response body to this describe's AsyncLocalStorage slot — so concurrent
    // describes (before_agent_start warm-up fires several in parallel) each
    // get their own capture without clobbering globalThis.fetch. For non-
    // Neuralwatt vision models no SSE energy comments are present and the
    // capture stays empty (energy fields omitted from the record).
    const describeCtx: DescribeContext = { energyReader: undefined };
    installFetchInterceptor();
    try {
      const response = await describeAls.run(describeCtx, async () =>
        complete(
          visionModel,
          { systemPrompt, messages: [userMessage] },
          {
            apiKey: auth.apiKey,
            headers: auth.headers,
            signal: controller.signal,
            maxTokens: cfg.maxTokens,
          },
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
      const record = buildUsageRecord(response, capture, visionModel, key);
      if (record) reportUsage(record);
      if (response.stopReason === "aborted" || response.stopReason === "error") {
        return UNAVAILABLE;
      }
      const description = response.content
        .filter((c): c is TextContent => c.type === "text")
        .map((c) => c.text)
        .join("\n")
        .trim();
      if (!description) return UNAVAILABLE;
      // The full description lives in the stored tool-result content; the read
      // tool's native collapse (ctrl+o) handles compactness — collapsed shows
      // the image only, expanded shows the full description + image. Only apply
      // a hard line cap when the user opts in (maxDescriptionLines > 0); by
      // default (0) the description is unbounded so ctrl+o can expand to all
      // of it and the model receives the complete context.
      const { text: final } =
        cfg.maxDescriptionLines && cfg.maxDescriptionLines > 0
          ? truncateDescription(description, cfg.maxDescriptionLines)
          : { text: description };
      return `${IMAGE_PLACEHOLDER_PREFIX}${final}${IMAGE_PLACEHOLDER_SUFFIX}`;
    } catch {
      return UNAVAILABLE;
    } finally {
      if (describeCtx.energyReader) describeCtx.energyReader.catch(() => {});
      uninstallFetchInterceptor();
      clearTimeout(timer);
    }
  })();

  if (visionCache.size >= cfg.cacheMax) {
    const firstKey = visionCache.keys().next().value;
    if (firstKey !== undefined) visionCache.delete(firstKey);
  }
  visionCache.set(key, promise);

  return promise;
}

async function replaceImagesWithDescriptions(
  payload: Record<string, unknown>,
  userPrompt: string,
  visionModel: Model<Api>,
  modelRegistry: ModelRegistry,
  cfg: VisionHandoffConfig,
): Promise<boolean> {
  const messages = payload.messages;
  if (!Array.isArray(messages)) return false;

  let replaced = false;
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const content = (msg as Record<string, unknown>).content;
    if (!Array.isArray(content)) continue;

    for (let i = 0; i < content.length; i++) {
      const img = extractImageFromBlock(content[i]);
      if (!img) continue;
      const description = await describeImage(img.data, img.mimeType, userPrompt, visionModel, modelRegistry, cfg);
      content[i] = makeReplacementText(content[i], description);
      replaced = true;
    }
  }
  return replaced;
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
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (!isConfigured(config)) return;
    if (!isHandoffTarget(ctx.model, config)) return;

    // Capture this turn's user prompt so read-tool images are described in
    // the same context. Fires before any tool_result for the turn.
    pendingTurnPrompt = event.prompt || "";

    const images = event.images;
    if (!images || images.length === 0) return;

    const visionModel = resolveVisionModel(ctx.modelRegistry, config.visionModel!);
    if (!visionModel) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    const userPrompt = event.prompt || "";
    for (const image of images) {
      if (!image || image.type !== "image" || !image.data) continue;
      const mimeType = image.mimeType || "image/png";
      describeImage(image.data, mimeType, userPrompt, visionModel, ctx.modelRegistry, config).catch(
        () => {},
      );
    }
  });

  pi.on("before_provider_request", async (event, ctx) => {
    if (!isConfigured(config)) return;
    if (!isHandoffTarget(ctx.model, config)) return;

    const visionModel = resolveVisionModel(ctx.modelRegistry, config.visionModel!);
    if (!visionModel) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    const payload = event.payload as Record<string, unknown>;
    await replaceImagesWithDescriptions(payload, "", visionModel, ctx.modelRegistry, config);
    return payload;
  });

  pi.on("tool_result", async (event, ctx) => {
    if (!isConfigured(config)) return;
    if (!isHandoffTarget(ctx.model, config)) return;
    if (event.toolName !== "read") return;

    const content = event.content;
    if (!Array.isArray(content)) return;
    // Skip the async work entirely when there is nothing to describe.
    if (!content.some((c) => extractImageFromBlock(c))) return;

    const visionModel = resolveVisionModel(ctx.modelRegistry, config.visionModel!);
    if (!visionModel) {
      notifyUnresolvedVisionModel(ctx, config.visionModel!);
      return;
    }

    const { content: next, changed } = await insertImageDescriptions(content, (img) =>
      describeImage(img.data, img.mimeType, pendingTurnPrompt ?? "", visionModel, ctx.modelRegistry, config),
    );
    if (!changed) return;
    return { content: next };
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
