/**
 * Shared constants, types, and utilities for pi-vision-handoff.
 *
 * Config lives at ~/.pi/agent/extensions/pi-vision-handoff.json — the same
 * convention pi-model-sort uses for picker-backed extensions.
 */

import type { ImageContent, TextContent, ThinkingLevel } from "@earendil-works/pi-ai";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export * from "./usage.js";

/** Subdirectory under the pi agent dir where picker extensions store config. */
const CONFIG_SUBDIR = "extensions";

/** Config file name. */
export const CONFIG_FILENAME = "pi-vision-handoff.json";

/** Full config path: ~/.pi/agent/extensions/pi-vision-handoff.json */
export function getConfigPath(): string {
  return join(getAgentDir(), CONFIG_SUBDIR, CONFIG_FILENAME);
}

/** Description shown in the / commands list. */
export const HANDOFF_COMMAND_DESCRIPTION =
  "Configure vision handoff — pick a vision model to describe images for text-only models";

/** Default system prompt for the vision describer. Mirrors the pi-umans-provider pipeline. */
export const DEFAULT_VISION_PROMPT =
  "You are a vision assistant for a coding agent. Describe this image exhaustively. Cover: all visible text (verbatim if possible), code snippets, UI layout and widgets, diagrams and flow arrows, error messages and stack traces, file trees, terminal output, color and style details, spatial relationships between elements, and anything else a developer would need to act on this image. Do not summarize — be exhaustive.";

/** Prefix prepended to the user's original prompt when describing an image. */
export const DEFAULT_USER_PROMPT_PREFIX = "The user's request about this image: ";

/** Placeholder text block injected in place of an image block. */
export const IMAGE_PLACEHOLDER_PREFIX = "[Image: ";
export const IMAGE_PLACEHOLDER_SUFFIX = "]";

/** Marker appended to a description whose `completeSimple()` call ended with
 *  `stopReason: "length"` — i.e. the vision model hit a token limit (either the
 *  configured `maxTokens` or the provider's hard output cap) before finishing.
 *  A truncated description is still useful, but the agent must not mistake it
 *  for the complete picture (it would reason as if the image contained only
 *  what made it into the text). The marker makes the cut-off visible to both
 *  the agent (it reads the marker in the tool result / context) and the user
 *  (it renders inline). */
export const DESCRIPTION_TRUNCATED_MARKER =
  "\n\n[... description truncated — the vision model reached its output token limit. The above may be incomplete; raise maxTokens or use a higher-output vision model for the full description.]";

/** Append the {@link DESCRIPTION_TRUNCATED_MARKER} to a raw description. */
export function markDescriptionTruncated(text: string): string {
  return text + DESCRIPTION_TRUNCATED_MARKER;
}

/** Default vision cache size (number of described images kept in memory per session). */
export const DEFAULT_CACHE_MAX = 50;

/** The pi thinking levels, ordered from lightest to deepest reasoning.
 *  Mirrors `@earendil-works/pi-ai`'s `ThinkingLevel`; "off" is handled
 *  separately via {@link VisionHandoffConfig.thinking} (a boolean on/off
 *  switch) so the level persists across toggles. */
export const THINKING_LEVELS: readonly ThinkingLevel[] = [
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
];

/** Default thinking effort for the vision describer when thinking is enabled. */
export const DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium";

/** Whether `level` is one of the supported {@link ThinkingLevel} values. */
export function isThinkingLevel(level: unknown): level is ThinkingLevel {
  return typeof level === "string" && (THINKING_LEVELS as readonly string[]).includes(level);
}

/** Per-description request timeout. Generous because the describer generates an
 *  exhaustive multi-paragraph description per image; a single image commonly
 *  takes ~20s, and a batched call over websocket transport scales with image
 *  count. {@link describeTimeoutMs} scales this per batch size. */
export const DESCRIBE_TIMEOUT_MS = 120_000;

/** Per-batch timeout: the base timeout plus a per-image budget so a 5-image
 *  batch isn't held to the same wall-clock budget as a single image. The
 *  describer generates exhaustive prose per image, and websocket transport
 *  adds latency, so the budget scales with the number of images in the call. */
export function describeTimeoutMs(imageCount: number): number {
  const perImage = 45_000;
  return DESCRIBE_TIMEOUT_MS + Math.max(0, imageCount - 1) * perImage;
}

/**
 * Default cap on the number of lines kept from a description before truncation.
 *
 * Mirrors pi core's read-result truncation: tool output is bounded for both the
 * TUI render and the model context. The stored `result.content` is the single
 * source for both surfaces (no decouple point exists at `tool_result` — it's the
 * only hook that still has the raw image before pi-ai strips it), so the cap
 * applies uniformly. 0 = unbounded (default): the read tool's native collapse
 * handles compactness and `ctrl+o` expands to the full description.
 */
export const DEFAULT_MAX_DESCRIPTION_LINES = 0;

/**
 * The note pi core's `read` tool appends to image-result text blocks when the
 * active model lacks image input (see `getNonVisionImageNote` in pi core).
 *
 * Once this extension inserts a description, the note becomes self-
 * contradicting ("image will be omitted" vs. the vivid description that
 * follows) and confuses the model. {@link stripNonVisionImageNote} removes it.
 */
export const NON_VISION_IMAGE_NOTE =
  "[Current model does not support images. The image will be omitted from this request.]";

/**
 * Remove {@link NON_VISION_IMAGE_NOTE} from a text block.
 *
 * The read tool appends the note as `\n${NOTE}` (always the trailing segment of
 * the metadata text block), so this also collapses the orphaned newline it
 * leaves behind. Safe to call on text that does not contain the note (no-op).
 */
export function stripNonVisionImageNote(text: string): string {
  if (!text.includes(NON_VISION_IMAGE_NOTE)) return text;
  return text.split(NON_VISION_IMAGE_NOTE).join("").replace(/\n+$/, "");
}

export interface VisionHandoffConfig {
  /** Master switch. When false, no handoff occurs even if a vision model is configured. */
  enabled: boolean;
  /** The vision-capable model that describes images, as "provider/id". null = not configured. */
  visionModel: string | null;
  /** When true (default), handoff is applied to every model whose input does not include "image". */
  autoHandoff: boolean;
  /** Extra "provider/id" refs that should ALSO receive handoff (e.g. weak vision models). */
  handoffModels: string[];
  /** Max output tokens for a single description. `undefined` (default) = use the
   *  vision model's declared max output (`model.maxTokens`) as the cap — the
   *  highest the model supports — so the "be exhaustive" prompt isn't defeated
   *  by a provider's small omitted-default. Set a number only to cap lower.
   *  A truncation is always surfaced via {@link DESCRIPTION_TRUNCATED_MARKER}
   *  when the model hits the limit. */
  maxTokens?: number;
  /** Max images kept in the in-memory description cache. */
  cacheMax: number;
  /**
   * Max lines kept from a description before truncation (0 = unbounded).
   * Bounds both the TUI render and the model context, like pi core's tool-output
   * truncation.
   */
  maxDescriptionLines: number;
  /** Override the describer system prompt (defaults to DEFAULT_VISION_PROMPT). */
  prompt?: string;
  /** Override the user-prompt prefix (defaults to DEFAULT_USER_PROMPT_PREFIX). */
  userPromptPrefix?: string;
  /** Whether the vision describer should reason (think) before describing.
   *  Off by default — describing images is a perception task, not a reasoning
   *  one, and thinking adds latency + cost. When on, the level below is sent
   *  to the vision model via pi-ai's `reasoning` option (only honoured when
   *  the configured vision model declares `reasoning: true`). */
  thinking: boolean;
  /** Thinking effort for the describer when {@link thinking} is on. One of
   *  {@link THINKING_LEVELS}. Ignored when {@link thinking} is off or the
   *  vision model lacks reasoning support. */
  thinkingLevel: ThinkingLevel;
}

export const DEFAULT_CONFIG: VisionHandoffConfig = {
  enabled: true,
  visionModel: null,
  autoHandoff: true,
  handoffModels: [],
  maxTokens: undefined,
  cacheMax: DEFAULT_CACHE_MAX,
  maxDescriptionLines: DEFAULT_MAX_DESCRIPTION_LINES,
  thinking: false,
  thinkingLevel: DEFAULT_THINKING_LEVEL,
};

/** Parse a "provider/id" reference. Returns null if malformed. */
export function parseModelRef(ref: string): { provider: string; id: string } | null {
  const trimmed = ref.trim();
  if (!trimmed) return null;
  const slashIndex = trimmed.indexOf("/");
  if (slashIndex <= 0) return null; // no slash, or empty provider
  const provider = trimmed.slice(0, slashIndex);
  const id = trimmed.slice(slashIndex + 1);
  if (!provider || !id) return null;
  return { provider, id };
}

/** Format a provider/id reference string. */
export function formatModelRef(provider: string, id: string): string {
  return `${provider}/${id}`;
}

/** Whether a model declares image input. */
export function isVisionModel(model: { input?: ("text" | "image")[] } | undefined | null): boolean {
  return !!model && Array.isArray(model.input) && model.input.includes("image");
}

/** Merge a parsed config object onto defaults, tolerating missing/invalid fields. */
export function normalizeConfig(raw: unknown): VisionHandoffConfig {
  const base: VisionHandoffConfig = { ...DEFAULT_CONFIG };
  if (!raw || typeof raw !== "object") return base;
  const obj = raw as Record<string, unknown>;

  if (typeof obj.enabled === "boolean") base.enabled = obj.enabled;
  if (typeof obj.visionModel === "string" && obj.visionModel.trim()) {
    base.visionModel = parseModelRef(obj.visionModel) ? obj.visionModel.trim() : null;
  } else if (obj.visionModel === null) {
    base.visionModel = null;
  }
  if (typeof obj.autoHandoff === "boolean") base.autoHandoff = obj.autoHandoff;

  if (Array.isArray(obj.handoffModels)) {
    base.handoffModels = obj.handoffModels
      .filter((m): m is string => typeof m === "string")
      .map((m) => m.trim())
      .filter((m) => m && parseModelRef(m));
  }
  // maxTokens: optional. undefined (default) = no artificial cap. Only set when
  // a valid positive finite number is given; any other value leaves it unset.
  if (typeof obj.maxTokens === "number" && Number.isFinite(obj.maxTokens) && obj.maxTokens > 0) {
    base.maxTokens = Math.floor(obj.maxTokens);
  }
  if (typeof obj.cacheMax === "number" && Number.isFinite(obj.cacheMax) && obj.cacheMax > 0) {
    base.cacheMax = Math.floor(obj.cacheMax);
  }
  // maxDescriptionLines: any non-negative finite integer. 0 = unbounded.
  if (
    typeof obj.maxDescriptionLines === "number" &&
    Number.isFinite(obj.maxDescriptionLines) &&
    obj.maxDescriptionLines >= 0
  ) {
    base.maxDescriptionLines = Math.floor(obj.maxDescriptionLines);
  }
  if (typeof obj.prompt === "string" && obj.prompt.trim()) base.prompt = obj.prompt;
  if (typeof obj.userPromptPrefix === "string") base.userPromptPrefix = obj.userPromptPrefix;

  if (typeof obj.thinking === "boolean") base.thinking = obj.thinking;
  if (isThinkingLevel(obj.thinkingLevel)) base.thinkingLevel = obj.thinkingLevel;

  return base;
}

/** Read config from disk (falls back to defaults on missing/corrupt file). */
export function readConfig(): VisionHandoffConfig {
  const path = getConfigPath();
  if (!existsSync(path)) return { ...DEFAULT_CONFIG };
  try {
    const raw = readFileSync(path, "utf8");
    return normalizeConfig(JSON.parse(raw));
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

/** Write config to disk. Creates the directory if needed. Returns the path written. */
export function writeConfig(config: VisionHandoffConfig): string {
  const path = getConfigPath();
  const dir = join(getAgentDir(), CONFIG_SUBDIR);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(path, JSON.stringify(config, null, 2) + "\n", "utf8");
  return path;
}

export interface ExtractedImage {
  data: string;
  mimeType: string;
}

/** Parse a `data:<mime>;base64,<data>` URL into raw base64 + mime. Returns null if not a data URL. */
export function parseDataUrl(url: string): ExtractedImage | null {
  const match = /^data:([^;,]+)?(?:;base64)?,(.*)$/s.exec(url);
  if (!match) return null;
  return { mimeType: match[1] || "image/png", data: match[2] };
}

/**
 * Detect an image block by shape across the three request formats pi uses:
 *   openai-completions: { type: "image_url",  image_url: { url: "data:..." } }
 *   openai-responses:   { type: "input_image", image_url: "data:..." | { url } }
 *   anthropic-messages: { type: "image", source: { type: "base64", media_type, data } }
 */
export function extractImageFromBlock(block: unknown): ExtractedImage | null {
  if (!block || typeof block !== "object") return null;
  const b = block as Record<string, any>;

  if (b.type === "image_url" && typeof b.image_url?.url === "string") {
    return parseDataUrl(b.image_url.url);
  }

  if (b.type === "input_image") {
    const url = typeof b.image_url === "string" ? b.image_url : b.image_url?.url;
    if (typeof url === "string") return parseDataUrl(url);
    return null;
  }

  // anthropic-messages image block (wrapped in `source`).
  if (b.type === "image" && b.source?.type === "base64" && typeof b.source.data === "string") {
    return { data: b.source.data, mimeType: b.source.media_type || "image/png" };
  }

  // pi-ai internal image block — emitted by the `read` tool and carried by
  // ToolResultEvent.content / user-message content blocks (regardless of whether
  // the active model declares image input):
  //   { type: "image", data: "<base64>", mimeType }
  // Distinct from the anthropic shape above: no `source` wrapper, direct fields.
  if (b.type === "image" && typeof b.data === "string") {
    return { data: b.data, mimeType: b.mimeType || "image/png" };
  }

  return null;
}

/** Build a text block that replaces an image block, matching the request format. */
export function makeReplacementText(block: unknown, description: string): Record<string, unknown> {
  const b = (block ?? null) as Record<string, unknown> | null;
  if (b?.type === "input_image") {
    return { type: "input_text", text: description };
  }
  return { type: "text", text: description };
}

/** Outcome of {@link truncateDescription}. */
export interface TruncatedDescription {
  text: string;
  /** True iff lines were dropped. */
  truncated: boolean;
  /** Number of trailing lines removed. */
  hidden: number;
}

/**
 * Head-truncate a description to its first `maxLines` lines with a
 * "... (N more lines)" footer, mirroring pi core's read-result truncation
 * notice (which itself surfaces "... (N more lines, ctrl+o to expand)").
 *
 * Returns the text unchanged when it is at or below the limit, or when
 * `maxLines` is non-positive (0 = unbounded). Head-truncation keeps the leading
 * layout / text-content / structure sections, which are typically the most
 * actionable for a coding agent.
 */
export function truncateDescription(text: string, maxLines: number): TruncatedDescription {
  if (!maxLines || maxLines <= 0) {
    return { text, truncated: false, hidden: 0 };
  }
  const lines = text.split("\n");
  if (lines.length <= maxLines) {
    return { text, truncated: false, hidden: 0 };
  }
  const hidden = lines.length - maxLines;
  const kept = lines.slice(0, maxLines).join("\n");
  return { text: `${kept}\n... (${hidden} more lines)`, truncated: true, hidden };
}

/** Start marker for the k-th image section in a batched describer response. */
export const BATCH_IMAGE_MARKER_START = "<<<IMAGE";
/** End marker closing a single image section in a batched describer response. */
export const BATCH_IMAGE_MARKER_END = "<<<END>>>";

/** Escape a literal string for use in a RegExp. */
function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Section regexes built from the marker constants so the producer
// (batchUserPrompt) and consumer (parseBatchedDescriptions) share one source of
// truth for the delimiter format. A lone `<<<IMAGE k>>> … <<<END>>>` wrapper
// for a single image; a global scan for the multi-image case that stops at the
// next start marker, the end marker, or end-of-text.
const SECTION_SINGLE_RE = new RegExp(
  `${escapeRe(BATCH_IMAGE_MARKER_START)}\\s+1>>>\\s*([\\s\\S]*?)${escapeRe(BATCH_IMAGE_MARKER_END)}`,
);
const SECTION_MULTI_RE = new RegExp(
  `${escapeRe(BATCH_IMAGE_MARKER_START)}\\s+(\\d+)>>>\\s*([\\s\\S]*?)(?=${escapeRe(
    BATCH_IMAGE_MARKER_START,
  )}\\s+\\d+>>>|${escapeRe(BATCH_IMAGE_MARKER_END)}|$)`,
  "g",
);
const SECTION_END_STRIP_RE = new RegExp(`${escapeRe(BATCH_IMAGE_MARKER_END)}\\s*$`);

/** Build the user-message text block for a (possibly batched) describer call.
 *
 * For a single image (count <= 1) this returns the same simple prompt the
 * original per-image pipeline used (no markers), so single-image behaviour is
 * unchanged. For count > 1 it instructs the vision model to emit one delimited
 * `<<<IMAGE k>>> … <<<END>>>` section per image, in order, so the response can
 * be split back into per-image descriptions by {@link parseBatchedDescriptions}.
 *
 * The user's turn prompt (if any) is folded in via {@link DEFAULT_USER_PROMPT_PREFIX}
 * so every image in the batch is described in the context of the same request —
 * one describer call covers the whole prompt's image set, dataloader-style,
 * rather than spinning up one agent call per image. */
export function batchUserPrompt(count: number, userPrompt: string, prefix: string): string {
  const userLine = userPrompt && userPrompt.trim() ? `${prefix}${userPrompt}` : "";
  if (count <= 1) {
    return userLine || "Describe this image.";
  }
  return [
    `Describe each of the ${count} images below exhaustively, in order. For image k (1 through ${count}), emit exactly:`,
    "",
    `${BATCH_IMAGE_MARKER_START} k>>>`,
    "<your exhaustive description>",
    BATCH_IMAGE_MARKER_END,
    ...(userLine ? ["", userLine] : []),
  ].join("\n");
}

/** Parse a batched vision-model response into one description per image, in order.
 *
 * Returns an array of length `count`; each slot is the trimmed description for
 * image k (1-indexed → array index k-1), or `null` when that image's section is
 * missing, empty, or unparseable. Callers treat `null` as "description
 * unavailable" (graceful degradation) — one image failing to parse must not
 * void the rest of the batch.
 *
 * For `count <= 1` the whole response is treated as the single image's
 * description (a lone `<<<IMAGE 1>>> … <<<END>>>` wrapper is stripped if the
 * model emitted one anyway), preserving the original single-image behaviour. */
export function parseBatchedDescriptions(text: string, count: number): (string | null)[] {
  const trimmed = (text ?? "").trim();
  if (count <= 1) {
    const m = SECTION_SINGLE_RE.exec(trimmed);
    const body = (m ? m[1] : trimmed).trim();
    return [body || null];
  }
  const out: (string | null)[] = new Array(count).fill(null);
  let m: RegExpExecArray | null;
  SECTION_MULTI_RE.lastIndex = 0;
  while ((m = SECTION_MULTI_RE.exec(trimmed)) !== null) {
    const idx = parseInt(m[1], 10) - 1;
    const body = m[2].replace(SECTION_END_STRIP_RE, "").trim();
    if (idx >= 0 && idx < count && !out[idx]) out[idx] = body || null;
  }
  return out;
}

/** Truncate (if configured) and wrap a raw description in the `[Image: …]`
 *  placeholder envelope used by every insertion path. Centralises the wrap+
 *  truncate logic shared by the read-tool and context injection paths so both
 *  surfaces stay identical for the same image hash. */
export function wrapDescription(description: string, cfg: VisionHandoffConfig): string {
  const { text: final } =
    cfg.maxDescriptionLines && cfg.maxDescriptionLines > 0
      ? truncateDescription(description, cfg.maxDescriptionLines)
      : { text: description };
  return `${IMAGE_PLACEHOLDER_PREFIX}${final}${IMAGE_PLACEHOLDER_SUFFIX}`;
}

/** Outcome of {@link insertImageDescriptions}. */
export interface ReplacedContent {
  content: (TextContent | ImageContent)[];
  /** True iff at least one image block had a description inserted before it. */
  changed: boolean;
}

/**
 * Insert a description text block before each image block in a tool-result /
 * message content array, KEEPING the image block in place.
 *
 * Why insert (not replace): the stored tool-result content is the single
 * source for both the TUI render (kitty inline images read it via
 * `result.content.filter(c => c.type === "image")`) and the provider payload.
 * Replacing the image would strip it from the terminal render. Keeping the
 * image preserves kitty rendering; the inserted description still reaches
 * non-vision models because pi-ai's `downgradeUnsupportedImages` only rewrites
 * `type: "image"` blocks for non-vision models — text blocks (including this
 * description) pass through untouched to the provider.
 *
 * `describe` is injected (rather than calling the vision model directly) so the
 * extract → describe → insert pipeline is unit-testable without standing up a
 * provider, registry, and API call. The extension wires its real `describeImage`
 * into this helper in its `tool_result` handler.
 *
 * When at least one description is inserted, also strips pi core's
 * {@link NON_VISION_IMAGE_NOTE} from text blocks — the note (appended by the
 * read tool for non-vision models) would otherwise contradict the inserted
 * description. Text blocks are reassigned (not mutated in place); the input
 * array is left untouched.
 *
 * Returns a new array and `changed: false` when there were no images, so
 * callers can short-circuit and avoid mutating pi's stored result unnecessarily.
 */
export async function insertImageDescriptions(
  content: readonly (TextContent | ImageContent)[] | undefined,
  describe: (img: ExtractedImage) => Promise<string>,
): Promise<ReplacedContent> {
  if (!Array.isArray(content)) {
    return { content: [], changed: false };
  }
  const next: (TextContent | ImageContent)[] = [];
  let changed = false;
  for (const block of content) {
    const img = extractImageFromBlock(block);
    if (!img) {
      next.push(block);
      continue;
    }
    const description = await describe(img);
    next.push({ type: "text", text: description } satisfies TextContent);
    next.push(block);
    changed = true;
  }
  if (!changed) {
    return { content: next, changed };
  }
  // We inserted at least one description. Strip the read tool's
  // "[Current model does not support images...]" note from text blocks — it
  // contradicts the description we just inserted ("image will be omitted" vs.
  // the description that follows) and confuses the model.
  for (let i = 0; i < next.length; i++) {
    const block = next[i];
    if (block.type === "text" && typeof block.text === "string" && block.text.includes(NON_VISION_IMAGE_NOTE)) {
      next[i] = { type: "text", text: stripNonVisionImageNote(block.text) } satisfies TextContent;
    }
  }
  return { content: next, changed };
}
