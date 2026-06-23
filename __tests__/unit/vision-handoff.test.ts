import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  DEFAULT_CONFIG,
  extractImageFromBlock,
  formatModelRef,
  getConfigPath,
  isVisionModel,
  makeReplacementText,
  normalizeConfig,
  parseDataUrl,
  parseModelRef,
  readConfig,
  writeConfig,
  type VisionHandoffConfig,
} from "../../src/index.js";

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
    });
    expect(cfg.enabled).toBe(false);
    expect(cfg.visionModel).toBe("openai/gpt-4o");
    expect(cfg.handoffModels).toEqual(["ollama/llava", "deepseek/deepseek-chat"]);
    expect(cfg.maxTokens).toBe(2048);
    expect(cfg.cacheMax).toBe(5);
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
