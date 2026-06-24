import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  DEFAULT_CONFIG,
  DESCRIPTION_TRUNCATED_MARKER,
  extractImageFromBlock,
  formatModelRef,
  getConfigPath,
  insertImageDescriptions,
  isVisionModel,
  makeReplacementText,
  markDescriptionTruncated,
  NON_VISION_IMAGE_NOTE,
  normalizeConfig,
  parseDataUrl,
  parseModelRef,
  parseBatchedDescriptions,
  batchUserPrompt,
  readConfig,
  stripNonVisionImageNote,
  truncateDescription,
  wrapDescription,
  writeConfig,
  BATCH_IMAGE_MARKER_END,
  type VisionHandoffConfig,
} from "../../src/index.js";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";

describe("parseModelRef / formatModelRef", () => {
  it("parses provider/id", () => {
    expect(parseModelRef("openai/gpt-4o")).toEqual({ provider: "openai", id: "gpt-4o" });
    expect(parseModelRef("ollama/llava:13b")).toEqual({ provider: "ollama", id: "llava:13b" });
  });

  it("round-trips through formatModelRef", () => {
    expect(formatModelRef("openai", "gpt-4o")).toBe("openai/gpt-4o");
  });

  it("rejects malformed refs", () => {
    expect(parseModelRef("")).toBeNull();
    expect(parseModelRef("   ")).toBeNull();
    expect(parseModelRef("no-slash")).toBeNull();
    expect(parseModelRef("/no-provider")).toBeNull();
    expect(parseModelRef("no-id/")).toBeNull();
  });

  it("accepts provider ids containing slashes (deep ids)", () => {
    // Only the first slash splits provider from id.
    expect(parseModelRef("neuralwatt/moonshotai/Kimi-K2.6")).toEqual({
      provider: "neuralwatt",
      id: "moonshotai/Kimi-K2.6",
    });
  });
});

describe("isVisionModel", () => {
  it("returns true when input includes image", () => {
    expect(isVisionModel({ input: ["text", "image"] })).toBe(true);
  });

  it("returns false for text-only models", () => {
    expect(isVisionModel({ input: ["text"] })).toBe(false);
  });

  it("returns false for missing/odd shapes", () => {
    expect(isVisionModel(undefined)).toBe(false);
    expect(isVisionModel(null)).toBe(false);
    expect(isVisionModel({})).toBe(false);
    expect(isVisionModel({ input: "text" } as any)).toBe(false);
  });
});

describe("normalizeConfig", () => {
  it("returns defaults for non-object input", () => {
    expect(normalizeConfig(null)).toEqual(DEFAULT_CONFIG);
    expect(normalizeConfig(undefined)).toEqual(DEFAULT_CONFIG);
    expect(normalizeConfig("oops")).toEqual(DEFAULT_CONFIG);
    expect(normalizeConfig({})).toEqual(DEFAULT_CONFIG);
  });

  it("applies valid fields and clamps numbers", () => {
    const cfg = normalizeConfig({
      enabled: false,
      visionModel: "openai/gpt-4o",
      autoHandoff: true,
      handoffModels: ["ollama/llava", "bad-ref", "deepseek/deepseek-chat"],
      maxTokens: 2048,
      cacheMax: 5,
      maxDescriptionLines: 12,
    });
    expect(cfg.enabled).toBe(false);
    expect(cfg.visionModel).toBe("openai/gpt-4o");
    expect(cfg.handoffModels).toEqual(["ollama/llava", "deepseek/deepseek-chat"]);
    expect(cfg.maxTokens).toBe(2048);
    expect(cfg.cacheMax).toBe(5);
    expect(cfg.maxDescriptionLines).toBe(12);
  });

  it("drops an invalid visionModel ref", () => {
    expect(normalizeConfig({ visionModel: "no-slash" }).visionModel).toBeNull();
    expect(normalizeConfig({ visionModel: "  " }).visionModel).toBeNull();
  });

  it("keeps a null visionModel", () => {
    expect(normalizeConfig({ visionModel: null }).visionModel).toBeNull();
  });

  it("ignores non-positive / non-finite numbers", () => {
    const cfg = normalizeConfig({ maxTokens: -10, cacheMax: "big" });
    expect(cfg.maxTokens).toBe(DEFAULT_CONFIG.maxTokens);
    expect(cfg.cacheMax).toBe(DEFAULT_CONFIG.cacheMax);
  });

  it("maxTokens is optional: undefined by default (no artificial cap)", () => {
    expect(DEFAULT_CONFIG.maxTokens).toBeUndefined();
    expect(normalizeConfig({}).maxTokens).toBeUndefined();
    expect(normalizeConfig({ maxTokens: null }).maxTokens).toBeUndefined();
    expect(normalizeConfig({ maxTokens: "1024" }).maxTokens).toBeUndefined();
  });

  it("maxTokens accepts a positive finite number and floors it", () => {
    expect(normalizeConfig({ maxTokens: 2048 }).maxTokens).toBe(2048);
    expect(normalizeConfig({ maxTokens: 1024.9 }).maxTokens).toBe(1024);
  });

  it("accepts maxDescriptionLines, including 0 (unbounded)", () => {
    expect(normalizeConfig({ maxDescriptionLines: 0 }).maxDescriptionLines).toBe(0);
    expect(normalizeConfig({ maxDescriptionLines: 7 }).maxDescriptionLines).toBe(7);
  });

  it("rejects negative / non-finite maxDescriptionLines", () => {
    expect(normalizeConfig({ maxDescriptionLines: -3 }).maxDescriptionLines).toBe(
      DEFAULT_CONFIG.maxDescriptionLines,
    );
    expect(normalizeConfig({ maxDescriptionLines: "lots" }).maxDescriptionLines).toBe(
      DEFAULT_CONFIG.maxDescriptionLines,
    );
  });

  it("accepts prompt and userPromptPrefix overrides", () => {
    const cfg = normalizeConfig({ prompt: "be brief", userPromptPrefix: "Q: " });
    expect(cfg.prompt).toBe("be brief");
    expect(cfg.userPromptPrefix).toBe("Q: ");
  });
});

describe("image block extraction across request formats", () => {
  const base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB";

  it("parses openai-completions image_url blocks", () => {
    const block = { type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/png", data: base64 });
  });

  it("parses openai-responses input_image blocks (string image_url)", () => {
    const block = { type: "input_image", detail: "auto", image_url: `data:image/jpeg;base64,${base64}` };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/jpeg", data: base64 });
  });

  it("parses openai-responses input_image blocks ({ url } shape)", () => {
    const block = { type: "input_image", image_url: { url: `data:image/webp;base64,${base64}` } };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/webp", data: base64 });
  });

  it("parses anthropic-messages image source blocks", () => {
    const block = { type: "image", source: { type: "base64", media_type: "image/gif", data: base64 } };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/gif", data: base64 });
  });

  it("defaults mimeType to image/png when missing", () => {
    const block = { type: "image", source: { type: "base64", data: base64 } };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/png", data: base64 });
  });

  it("parses pi-ai internal image blocks (read tool / ToolResultEvent shape)", () => {
    const block = { type: "image", data: base64, mimeType: "image/png" };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/png", data: base64 });
  });

  it("defaults mimeType to image/png for internal blocks when missing", () => {
    const block = { type: "image", data: base64 };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/png", data: base64 });
  });

  it("prefers the anthropic `source` shape over the internal one when both are present", () => {
    const block = {
      type: "image",
      source: { type: "base64", media_type: "image/gif", data: base64 },
    };
    expect(extractImageFromBlock(block)).toEqual({ mimeType: "image/gif", data: base64 });
  });

  it("returns null for an image block with non-string data", () => {
    expect(extractImageFromBlock({ type: "image", data: 123 })).toBeNull();
    expect(extractImageFromBlock({ type: "image", mimeType: "image/png" })).toBeNull();
  });

  it("returns null for non-image blocks and junk", () => {
    expect(extractImageFromBlock({ type: "text", text: "hi" })).toBeNull();
    expect(extractImageFromBlock(null)).toBeNull();
    expect(extractImageFromBlock(undefined)).toBeNull();
    expect(extractImageFromBlock("string")).toBeNull();
    expect(extractImageFromBlock({ type: "image_url", image_url: { url: "https://example.com/x.png" } })).toBeNull();
    // input_image with a remote (non-data) URL is unsupported in v1
    expect(extractImageFromBlock({ type: "input_image", image_url: "https://example.com/x.png" })).toBeNull();
  });
});

describe("parseDataUrl", () => {
  it("extracts mime and base64 body", () => {
    expect(parseDataUrl("data:image/png;base64,ABC")).toEqual({ mimeType: "image/png", data: "ABC" });
  });

  it("returns null for non-data URLs", () => {
    expect(parseDataUrl("https://example.com/x.png")).toBeNull();
    expect(parseDataUrl("not a url")).toBeNull();
  });
});

describe("makeReplacementText", () => {
  it("emits input_text for responses blocks", () => {
    const block = { type: "input_image", image_url: "data:image/png;base64,ABC" };
    expect(makeReplacementText(block, "[Image: desc]")).toEqual({ type: "input_text", text: "[Image: desc]" });
  });

  it("emits text for openai-completions blocks", () => {
    const block = { type: "image_url", image_url: { url: "data:image/png;base64,ABC" } };
    expect(makeReplacementText(block, "[Image: desc]")).toEqual({ type: "text", text: "[Image: desc]" });
  });

  it("emits text for anthropic blocks", () => {
    const block = { type: "image", source: { type: "base64", data: "ABC" } };
    expect(makeReplacementText(block, "[Image: desc]")).toEqual({ type: "text", text: "[Image: desc]" });
  });

  it("emits text for pi-ai internal image blocks", () => {
    const block = { type: "image", data: "ABC", mimeType: "image/png" };
    expect(makeReplacementText(block, "[Image: desc]")).toEqual({ type: "text", text: "[Image: desc]" });
  });
});

describe("config round-trip via PI_CODING_AGENT_DIR", () => {
  let tmpHome: string;
  let savedEnv: string | undefined;

  beforeEach(() => {
    tmpHome = mkdtempSync(join(tmpdir(), "pi-vision-handoff-"));
    savedEnv = process.env.PI_CODING_AGENT_DIR;
    process.env.PI_CODING_AGENT_DIR = tmpHome;
  });

  afterEach(() => {
    rmSync(tmpHome, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = savedEnv;
  });

  it("writes to ~/.pi-equivalent/extensions/pi-vision-handoff.json", () => {
    const path = writeConfig({ ...DEFAULT_CONFIG, visionModel: "openai/gpt-4o" });
    expect(path).toBe(join(tmpHome, "extensions", "pi-vision-handoff.json"));
    expect(existsSync(path)).toBe(true);

    const onDisk = JSON.parse(readFileSync(path, "utf8")) as VisionHandoffConfig;
    expect(onDisk.visionModel).toBe("openai/gpt-4o");
  });

  it("readConfig returns defaults when the file is missing", () => {
    expect(readConfig()).toEqual(DEFAULT_CONFIG);
  });

  it("readConfig round-trips a written config", () => {
    const cfg: VisionHandoffConfig = {
      ...DEFAULT_CONFIG,
      enabled: false,
      visionModel: "ollama/llava:13b",
      handoffModels: ["deepseek/deepseek-chat"],
      autoHandoff: false,
      maxTokens: 512,
      cacheMax: 3,
    };
    writeConfig(cfg);
    expect(readConfig()).toEqual(cfg);
  });

  it("readConfig tolerates a corrupt file", () => {
    const dir = join(tmpHome, "extensions");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "pi-vision-handoff.json"), "{not json", "utf8");
    expect(readConfig()).toEqual(DEFAULT_CONFIG);
  });

  it("getConfigPath points at extensions/pi-vision-handoff.json", () => {
    expect(getConfigPath()).toBe(join(tmpHome, "extensions", "pi-vision-handoff.json"));
  });
});

describe("insertImageDescriptions", () => {
  const base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB";
  const textBlock = { type: "text", text: "hello" } as const;

  it("inserts a description text block before each image, keeping the image", async () => {
    const describe = async (img: { mimeType: string }) => `[Image: ${img.mimeType}]`;
    const result = await insertImageDescriptions(
      [textBlock, { type: "image", data: base64, mimeType: "image/png" }],
      describe,
    );
    expect(result.changed).toBe(true);
    expect(result.content).toEqual([
      { type: "text", text: "hello" },
      { type: "text", text: "[Image: image/png]" },
      { type: "image", data: base64, mimeType: "image/png" },
    ]);
  });

  it("reports changed=false and leaves content unchanged when there are no images", async () => {
    const result = await insertImageDescriptions([textBlock], async () => "never called");
    expect(result.changed).toBe(false);
    expect(result.content).toEqual([textBlock]);
  });

  it("handles empty content", async () => {
    const result = await insertImageDescriptions([], async () => "x");
    expect(result.changed).toBe(false);
    expect(result.content).toEqual([]);
  });

  it("handles undefined / non-array content defensively", async () => {
    const result = await insertImageDescriptions(undefined, async () => "x");
    expect(result.changed).toBe(false);
    expect(result.content).toEqual([]);
  });

  it("inserts a description before every image block across formats, keeping each image", async () => {
    const describe = async (img: { data: string }) => `desc(${img.data})`;
    const input: unknown = [
      { type: "image", data: "AAA", mimeType: "image/png" },
      textBlock,
      { type: "image", source: { type: "base64", media_type: "image/gif", data: "BBB" } },
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,CCC" } },
    ];
    const result = await insertImageDescriptions(
      input as (TextContent | ImageContent)[],
      describe,
    );
    expect(result.changed).toBe(true);
    expect(result.content).toEqual([
      { type: "text", text: "desc(AAA)" },
      { type: "image", data: "AAA", mimeType: "image/png" },
      { type: "text", text: "hello" },
      { type: "text", text: "desc(BBB)" },
      { type: "image", source: { type: "base64", media_type: "image/gif", data: "BBB" } },
      { type: "text", text: "desc(CCC)" },
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,CCC" } },
    ]);
  });

  it("keeps the original image block by reference (for TUI kitty rendering)", async () => {
    const image: ImageContent = { type: "image", data: base64, mimeType: "image/png" };
    const result = await insertImageDescriptions([image], async () => "d");
    expect(result.content[1]).toBe(image);
  });

  it("strips pi core's non-vision image note from text blocks when inserting a description", async () => {
    const metadata: TextContent = {
      type: "text",
      text: `Read image file [image/png]\n[Image: original 2356x964]\n${NON_VISION_IMAGE_NOTE}`,
    };
    const image: ImageContent = { type: "image", data: base64, mimeType: "image/png" };
    const result = await insertImageDescriptions([metadata, image], async () => "a vivid description");
    expect(result.changed).toBe(true);
    // Metadata text block no longer carries the contradictory note...
    expect(result.content[0]).toEqual({
      type: "text",
      text: "Read image file [image/png]\n[Image: original 2356x964]",
    });
    // ...the inserted description follows...
    expect(result.content[1]).toEqual({ type: "text", text: "a vivid description" });
    // ...and the image is kept.
    expect(result.content[2]).toBe(image);
  });

  it("leaves text blocks untouched when there are no images to describe", async () => {
    const metadata: TextContent = { type: "text", text: `note?\n${NON_VISION_IMAGE_NOTE}` };
    const result = await insertImageDescriptions([metadata], async () => "never called");
    expect(result.changed).toBe(false);
    expect(result.content[0]).toBe(metadata);
  });

  it("does not mutate the input array", async () => {
    const inputImage: ImageContent = { type: "image", data: base64, mimeType: "image/png" };
    const input: (TextContent | ImageContent)[] = [textBlock, inputImage];
    await insertImageDescriptions(input, async () => "d");
    expect(input).toEqual([textBlock, inputImage]);
    expect(input).toHaveLength(2);
  });
});

describe("truncateDescription", () => {
  it("returns the text unchanged when at or below the limit", () => {
    const text = "line1\nline2\nline3";
    expect(truncateDescription(text, 5)).toEqual({ text, truncated: false, hidden: 0 });
    expect(truncateDescription(text, 3)).toEqual({ text, truncated: false, hidden: 0 });
  });

  it("head-truncates and reports the hidden count", () => {
    const text = "a\nb\nc\nd\ne";
    const result = truncateDescription(text, 2);
    expect(result.truncated).toBe(true);
    expect(result.hidden).toBe(3);
    expect(result.text).toBe("a\nb\n... (3 more lines)");
  });

  it("treats 0 and negative limits as unbounded", () => {
    const text = "a\nb\nc";
    expect(truncateDescription(text, 0)).toEqual({ text, truncated: false, hidden: 0 });
    expect(truncateDescription(text, -5)).toEqual({ text, truncated: false, hidden: 0 });
  });

  it("handles single-line text at limit 1 without truncating", () => {
    const text = "only line";
    expect(truncateDescription(text, 1)).toEqual({ text, truncated: false, hidden: 0 });
  });

  it("truncates to a single line and counts the rest as hidden", () => {
    const result = truncateDescription("a\nb\nc\nd", 1);
    expect(result.text).toBe("a\n... (3 more lines)");
    expect(result.hidden).toBe(3);
    expect(result.truncated).toBe(true);
  });
});

describe("stripNonVisionImageNote", () => {
  it("removes the note and the orphaned newline before it", () => {
    const text = `Read image file [image/png]\n[Image: original 2356x964]\n${NON_VISION_IMAGE_NOTE}`;
    expect(stripNonVisionImageNote(text)).toBe("Read image file [image/png]\n[Image: original 2356x964]");
  });

  it("returns text unchanged when the note is absent", () => {
    expect(stripNonVisionImageNote("just text")).toBe("just text");
    expect(stripNonVisionImageNote("")).toBe("");
  });

  it("does not match a partial / truncated note", () => {
    const partial = "[Current model does not support images]";
    expect(stripNonVisionImageNote(`prefix\n${partial}`)).toBe(`prefix\n${partial}`);
  });

  it("removes the note even when it is the only content", () => {
    expect(stripNonVisionImageNote(NON_VISION_IMAGE_NOTE)).toBe("");
  });

  it("does not mutate the input string (strings are immutable; returns a new value)", () => {
    const original = `prefix\n${NON_VISION_IMAGE_NOTE}`;
    const stripped = stripNonVisionImageNote(original);
    expect(stripped).toBe("prefix");
    expect(original).toBe(`prefix\n${NON_VISION_IMAGE_NOTE}`);
  });
});

describe("batchUserPrompt", () => {
  it("returns the single-image prompt (no markers) for count <= 1", () => {
    expect(batchUserPrompt(1, "what is this?", "Q: ")).toBe("Q: what is this?");
    // count 0 is treated like a single image
    expect(batchUserPrompt(0, "x", "P: ")).toBe("P: x");
  });

  it("falls back to a bare describe instruction when there is no user prompt (single)", () => {
    expect(batchUserPrompt(1, "", "P: ")).toBe("Describe this image.");
    expect(batchUserPrompt(1, "   ", "P: ")).toBe("Describe this image.");
  });

  it("emits per-image marker instructions for a batch and folds in the user prompt", () => {
    const out = batchUserPrompt(3, "compare these", "The user's request about this image: ");
    expect(out).toContain("Describe each of the 3 images below");
    expect(out).toContain("<<<IMAGE k>>>");
    expect(out).toContain(BATCH_IMAGE_MARKER_END);
    expect(out).toContain("The user's request about this image: compare these");
  });

  it("omits the user-prompt section entirely when there is no user prompt (batch)", () => {
    const out = batchUserPrompt(2, "", "P: ");
    expect(out).toContain("<<<IMAGE k>>>");
    expect(out).not.toContain("P: ");
  });
});

describe("parseBatchedDescriptions", () => {
  it("returns the whole text as the single description for count <= 1 (no markers)", () => {
    expect(parseBatchedDescriptions("a plain description with no markers", 1)).toEqual([
      "a plain description with no markers",
    ]);
  });

  it("strips a lone <<<IMAGE 1>>> wrapper if the model emitted one for a single image", () => {
    const text = `<<<IMAGE 1>>>
inner description
<<<END>>>`;
    expect(parseBatchedDescriptions(text, 1)).toEqual(["inner description"]);
  });

  it("returns [null] for an empty single-image response", () => {
    expect(parseBatchedDescriptions("   ", 1)).toEqual([null]);
    expect(parseBatchedDescriptions("", 1)).toEqual([null]);
  });

  it("parses N delimited sections in order into N slots", () => {
    const text = [
      "<<<IMAGE 1>>>",
      "desc one",
      "<<<END>>>",
      "<<<IMAGE 2>>>",
      "desc two",
      "<<<END>>>",
      "<<<IMAGE 3>>>",
      "desc three",
      "<<<END>>>",
    ].join("\n");
    expect(parseBatchedDescriptions(text, 3)).toEqual(["desc one", "desc two", "desc three"]);
  });

  it("tolerates a missing <<<END>>> on the last section", () => {
    const text = "<<<IMAGE 1>>>\ndesc one\n<<<END>>>\n<<<IMAGE 2>>>\ndesc two";
    expect(parseBatchedDescriptions(text, 2)).toEqual(["desc one", "desc two"]);
  });

  it("degrades missing/unparseable sections to null without voiding the rest", () => {
    // image 2's section is entirely absent; images 1 and 3 parse fine
    const text = "<<<IMAGE 1>>>\ndesc one\n<<<END>>>\n<<<IMAGE 3>>>\ndesc three\n<<<END>>>";
    expect(parseBatchedDescriptions(text, 3)).toEqual(["desc one", null, "desc three"]);
  });

  it("treats an empty section body as null (unavailable)", () => {
    const text = "<<<IMAGE 1>>>\n\n<<<END>>>\n<<<IMAGE 2>>>\ndesc two\n<<<END>>>";
    expect(parseBatchedDescriptions(text, 2)).toEqual([null, "desc two"]);
  });

  it("ignores out-of-range / duplicate section indices (first-wins)", () => {
    const text =
      "<<<IMAGE 1>>>\nfirst\n<<<END>>>\n<<<IMAGE 1>>>\nsecond\n<<<END>>>\n<<<IMAGE 9>>>\nfar\n<<<END>>>";
    expect(parseBatchedDescriptions(text, 2)).toEqual(["first", null]);
  });

  it("returns all-null when the response has no markers for a batch", () => {
    expect(parseBatchedDescriptions("the model ignored the format", 3)).toEqual([null, null, null]);
  });
});

describe("wrapDescription", () => {
  it("wraps a raw description in the [Image: …] envelope", () => {
    expect(wrapDescription("a vivid scene", DEFAULT_CONFIG)).toBe("[Image: a vivid scene]");
  });

  it("applies the configured line cap when maxDescriptionLines > 0", () => {
    const cfg: VisionHandoffConfig = { ...DEFAULT_CONFIG, maxDescriptionLines: 2 };
    expect(wrapDescription("a\nb\nc\nd", cfg)).toBe("[Image: a\nb\n... (2 more lines)]");
  });

  it("leaves the description unbounded when maxDescriptionLines is 0 (default)", () => {
    const long = "line1\nline2\nline3\nline4";
    expect(wrapDescription(long, DEFAULT_CONFIG)).toBe(`[Image: ${long}]`);
  });
});

describe("markDescriptionTruncated", () => {
  it("appends the truncation marker to a raw description", () => {
    expect(markDescriptionTruncated("partial text")).toBe(`partial text${DESCRIPTION_TRUNCATED_MARKER}`);
  });

  it("the marker survives wrapDescription (stays inside the [Image: …] envelope)", () => {
    expect(wrapDescription(markDescriptionTruncated("partial"), DEFAULT_CONFIG)).toBe(
      `[Image: partial${DESCRIPTION_TRUNCATED_MARKER}]`,
    );
  });

  it("the marker mentions the token limit so the agent/user know it's incomplete", () => {
    expect(DESCRIPTION_TRUNCATED_MARKER).toMatch(/truncat/i);
    expect(DESCRIPTION_TRUNCATED_MARKER).toMatch(/token/i);
  });
});

describe("insertImageDescriptions with a batched (pre-resolved) describe callback", () => {
  const base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB";
  const textBlock = { type: "text", text: "hello" } as const;

  it("uses the pre-resolved map keyed by image hash, one describe source for many images", async () => {
    // Simulate the batched path: a single Map<hash, description> feeds
    // insertImageDescriptions via a lookup callback, so the insertion helper
    // stays image-agnostic and testable without a provider.
    const imgA: ImageContent = { type: "image", data: "AAA", mimeType: "image/png" };
    const imgB: ImageContent = { type: "image", data: "BBB", mimeType: "image/png" };
    const map = new Map<string, string>([
      ["AAA\x00image/png", "[Image: desc A]"],
      ["BBB\x00image/png", "[Image: desc B]"],
    ]);
    const lookup = async (img: { data: string; mimeType: string }) =>
      map.get(`${img.data}\x00${img.mimeType}`) ?? "[Image: description unavailable]";
    const result = await insertImageDescriptions([imgA, textBlock, imgB], lookup);
    expect(result.changed).toBe(true);
    expect(result.content).toEqual([
      { type: "text", text: "[Image: desc A]" },
      imgA,
      textBlock,
      { type: "text", text: "[Image: desc B]" },
      imgB,
    ]);
  });
});
