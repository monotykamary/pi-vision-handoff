import { describe, it, expect, beforeEach, vi } from "vitest";

// Mock the describer so no provider call is made; the loader wraps the canned
// descriptions with [Image: …] exactly as in production.
const runBatch = vi.fn(async (misses: { hash: string }[]) => {
  const out = new Map<string, string>();
  for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
  return out;
});
const describeSingle = vi.fn(async (img: { data: string }) => `single-${img.data}`);
vi.mock("../../src/describer.js", () => ({
  runBatch: (...args: unknown[]) => runBatch(...(args as Parameters<typeof runBatch>)),
  describeSingle: (...args: unknown[]) => describeSingle(...(args as Parameters<typeof describeSingle>)),
}));

// Keep every real export of src/index.js but force a deterministic config so
// the test does not depend on the user's ~/.pi/agent/extensions config.
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
    }),
  };
});

import factory from "../../vision-handoff.js";
import { extractImageFromBlock } from "../../src/index.js";
import { UNAVAILABLE } from "../../src/dataloader.js";

interface CapturedPi {
  on: ReturnType<typeof vi.fn>;
  registerCommand: ReturnType<typeof vi.fn>;
  appendEntry: ReturnType<typeof vi.fn>;
  events: { emit: ReturnType<typeof vi.fn> };
  getActiveTools: () => string[];
}

const setup = (): { pi: CapturedPi; handlers: Record<string, (event: any, ctx: any) => Promise<unknown>> } => {
  const handlers: Record<string, (event: any, ctx: any) => Promise<unknown>> = {};
  const pi: CapturedPi = {
    on: vi.fn((event: string, handler: any) => {
      handlers[event] = handler;
    }),
    registerCommand: vi.fn(),
    appendEntry: vi.fn(),
    events: { emit: vi.fn() },
    getActiveTools: () => [],
  };
  factory(pi as any);
  return { pi, handlers };
};

const sessionCtx = () => ({
  mode: "print" as const,
  hasUI: false,
  model: { provider: "agent", id: "text-only", input: ["text"] },
  modelRegistry: {
    find: () => ({ provider: "test", id: "vision", input: ["text", "image"] }),
  },
  ui: { notify: vi.fn() },
});

const imageContent = (data: string) => ({
  type: "image",
  data,
  mimeType: "image/png",
});

const readImageResult = (data: string) => [
  {
    type: "text",
    text: "Read image file [image/png]\n[Image: original 10x10.]\n[Current model does not support images. The image will be omitted from this request.]",
  },
  imageContent(data),
];

describe("pi-fabric codemode nested tool_result swap", () => {
  beforeEach(() => {
    runBatch.mockClear();
    describeSingle.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) => {
      const out = new Map<string, string>();
      for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
      return out;
    });
  });

  it("replaces image blocks with descriptions for a fabric_ nested call", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const event = {
      type: "tool_result",
      toolName: "read",
      toolCallId: "fabric_abc-123",
      input: { path: "x.png" },
      content: readImageResult("AAA") as any,
      isError: false,
    };
    const result = (await handlers["tool_result"](event, {
      ...sessionCtx(),
      signal: new AbortController().signal,
    })) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    expect(extractImageFromBlock(result!.content.find((b) => b.type === "image") ?? null)).toBeNull();
    const descBlock = result!.content.find((b) => b.type === "text" && b.text?.startsWith("[Image: desc-for-"));
    expect(descBlock).toBeDefined();
    // The non-vision note is stripped from the surviving text block.
    expect(result!.content.some((b) => b.text?.includes("does not support images"))).toBe(false);
  });

  it("keeps the image block for a normal (non-fabric) toolCallId", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const event = {
      type: "tool_result",
      toolName: "read",
      toolCallId: "call_abc-123",
      input: { path: "x.png" },
      content: readImageResult("BBB") as any,
      isError: false,
    };
    const result = (await handlers["tool_result"](event, {
      ...sessionCtx(),
      signal: new AbortController().signal,
    })) as { content: Array<{ type: string }> } | undefined;

    expect(result).toBeDefined();
    // Normal flow keeps the image (kitty render + /resume); only the note is stripped.
    expect(result!.content.some((b) => b.type === "image")).toBe(true);
    expect(result!.content.some((b) => (b as { text?: string }).text?.includes("does not support images"))).toBe(false);
  });

  it("uses UNAVAILABLE when the describer fails for a nested call", async () => {
    runBatch.mockImplementation(async () => new Map());
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const event = {
      type: "tool_result",
      toolName: "read",
      toolCallId: "fabric_fail-1",
      input: { path: "x.png" },
      content: readImageResult("CCC") as any,
      isError: false,
    };
    const result = (await handlers["tool_result"](event, {
      ...sessionCtx(),
      signal: new AbortController().signal,
    })) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    expect(result!.content.some((b) => b.type === "image")).toBe(false);
    expect(result!.content.some((b) => b.text === UNAVAILABLE)).toBe(true);
  });

  it("does nothing when handoff is not a target for the active model", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const event = {
      type: "tool_result",
      toolName: "read",
      toolCallId: "fabric_abc-123",
      input: { path: "x.png" },
      content: readImageResult("DDD") as any,
      isError: false,
    };
    // Vision-capable active model + autoHandoff => NOT a handoff target.
    const result = (await handlers["tool_result"](event, {
      ...sessionCtx(),
      model: { provider: "agent", id: "vision-model", input: ["text", "image"] },
      signal: new AbortController().signal,
    })) as unknown;

    expect(result).toBeUndefined();
    expect(runBatch).not.toHaveBeenCalled();
  });
});
