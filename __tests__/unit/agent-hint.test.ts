import { describe, it, expect, vi, beforeEach } from "vitest";

// Deterministic config: enabled handoff with a resolvable vision model, so the
// before_agent_start handler reaches its end (where the hint is returned)
// regardless of the developer's real ~/.pi config.
vi.mock("../../src/index.js", async (importActual) => {
  const actual = (await importActual()) as Record<string, unknown>;
  return {
    ...actual,
    readConfig: () => ({
      ...(actual.DEFAULT_CONFIG as object),
      enabled: true,
      visionModel: "test/vision",
      autoHandoff: true,
      handoffModels: [],
      prewarmPastedImages: false,
      asyncClipboardHandoff: false,
    }),
  };
});

import factory from "../../vision-handoff.js";
import { AGENT_HINT_MARKER } from "../../vision-handoff.js";
import { DEFAULT_CONFIG, normalizeConfig } from "../../src/index.js";

const makeCtx = (modelInput: string[]) => ({
  mode: "print" as const,
  hasUI: false,
  model: { provider: "agent", id: "target", input: modelInput },
  modelRegistry: {
    find: () => ({ provider: "test", id: "vision", input: ["text", "image"] }),
  },
  ui: { notify: vi.fn() },
});

function setup() {
  const handlers: Record<string, (event: any, ctx: any) => Promise<unknown>> = {};
  factory({
    on: vi.fn((event: string, handler: any) => {
      handlers[event] = handler;
    }),
    registerCommand: vi.fn(),
    registerMessageRenderer: vi.fn(),
    sendMessage: vi.fn(),
    appendEntry: vi.fn(),
    events: { emit: vi.fn() },
    getActiveTools: () => [],
  } as any);
  return handlers;
}

describe("agent capability hint", () => {
  beforeEach(() => {
    // No pasted clipboard paths in any prompt used here.
    process.env.PI_CODING_AGENT_DIR = process.env.PI_CODING_AGENT_DIR || "/tmp/agent-hint-test";
  });

  it("defaults agentHint to on and honors explicit false in config normalization", () => {
    expect(DEFAULT_CONFIG.agentHint).toBe(true);
    expect(normalizeConfig({}).agentHint).toBe(true);
    expect(normalizeConfig({ agentHint: false }).agentHint).toBe(false);
  });

  // Generous timeout: the handler lazily imports the dataloader/describer
  // chain on first call, which alone can exceed vitest's 5s default on
  // constrained machines.
  it("appends the hint section for a non-vision handoff target", { timeout: 30000 }, async () => {
    const handlers = setup();
    const result = (await handlers["before_agent_start"](
      { type: "before_agent_start", prompt: "hello", images: [], systemPrompt: "BASE PROMPT" },
      makeCtx(["text"]),
    )) as { systemPrompt?: string } | undefined;

    expect(typeof result?.systemPrompt).toBe("string");
    expect(result!.systemPrompt!.startsWith("BASE PROMPT")).toBe(true);
    expect(result!.systemPrompt!).toContain(AGENT_HINT_MARKER);
    expect(result!.systemPrompt!).toContain("read tool on its file path");
  });

  it("leaves the system prompt untouched for a vision-capable model", async () => {
    const handlers = setup();
    const result = await handlers["before_agent_start"](
      { type: "before_agent_start", prompt: "hello", images: [], systemPrompt: "BASE PROMPT" },
      makeCtx(["text", "image"]),
    );
    expect(result).toBeUndefined();
  });
});
