import { describe, it, expect, beforeEach, vi } from "vitest";

// Mock the describer so no provider call is made; the loader wraps the canned
// descriptions with [Image: …] exactly as in production.
const runBatch = vi.fn(async (misses: { hash: string }[]) => {
  const out = new Map<string, string>();
  for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
  return out;
});
const describeSingle = vi.fn(async (img: { data: string }) => `single-${img.data}`);
const ASYNC_CLIPBOARD_PATH = "/tmp/pi-clipboard-async.png";
const {
  readImageBufferBoundedMock,
  findPastedImagePathsMock,
  readImageBufferMock,
  resolvePrewarmImageMock,
} = vi.hoisted(() => ({
  readImageBufferBoundedMock: vi.fn((): { buf: Buffer; mimeType: string } | null => ({
    buf: Buffer.from("RECOVERED"),
    mimeType: "image/png",
  })),
  findPastedImagePathsMock: vi.fn((prompt: string) =>
    prompt.includes("clipboard-image") ? ["/tmp/pi-clipboard-async.png"] : [],
  ),
  readImageBufferMock: vi.fn(() => ({ buf: Buffer.from("ASYNC"), mimeType: "image/png" })),
  resolvePrewarmImageMock: vi.fn(async () => ({ data: "ASYNC", mimeType: "image/png" })),
}));
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
      asyncClipboardHandoff: true,
    }),
  };
});

vi.mock("../../src/image.js", async (importActual) => {
  const actual = (await importActual()) as Record<string, unknown>;
  return {
    ...actual,
    readImageBufferBounded: readImageBufferBoundedMock,
    findPastedImagePaths: findPastedImagePathsMock,
    readImageBuffer: readImageBufferMock,
    resolvePrewarmImage: resolvePrewarmImageMock,
  };
});

import factory from "../../vision-handoff.js";
import { UNAVAILABLE } from "../../src/dataloader.js";

interface CapturedPi {
  on: ReturnType<typeof vi.fn>;
  registerCommand: ReturnType<typeof vi.fn>;
  registerMessageRenderer: ReturnType<typeof vi.fn>;
  sendMessage: ReturnType<typeof vi.fn>;
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
    registerMessageRenderer: vi.fn(),
    sendMessage: vi.fn(),
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

const ctxWithSignal = () => ({ ...sessionCtx(), signal: new AbortController().signal });

describe("pi-fabric codemode nested tool_result + context hook", () => {
  beforeEach(() => {
    runBatch.mockClear();
    describeSingle.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) => {
      const out = new Map<string, string>();
      for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
      return out;
    });
  });

  it("keeps the image block for a fabric_ nested call (the context hook swaps for the model)", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "fabric_abc-123",
        input: { path: "x.png" },
        content: readImageResult("AAA") as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    // The nested tool_result keeps the image (kitty render + /resume); the
    // `context` hook swaps it for the description on the LLM-bound clone, so
    // NO description text is injected here.
    expect(result!.content.some((b) => b.type === "image")).toBe(true);
    expect(result!.content.some((b) => b.text?.startsWith("[Image: desc-for-"))).toBe(false);
    // pi's non-vision note is stripped from the surviving text block.
    expect(result!.content.some((b) => b.text?.includes("does not support images"))).toBe(false);
  });

  it("keeps the image block for a normal (non-fabric) toolCallId", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_abc-123",
        input: { path: "x.png" },
        content: readImageResult("BBB") as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string }> } | undefined;

    expect(result).toBeDefined();
    expect(result!.content.some((b) => b.type === "image")).toBe(true);
    expect(result!.content.some((b) => (b as { text?: string }).text?.includes("does not support images"))).toBe(false);
  });

  it("keeps the image when the describer fails for a nested call (UNAVAILABLE flows via the context hook)", async () => {
    runBatch.mockImplementation(async () => new Map());
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "fabric_fail-1",
        input: { path: "x.png" },
        content: readImageResult("CCC") as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    // The nested tool_result keeps the image even on describer failure; the
    // `context` hook substitutes UNAVAILABLE on the LLM-bound clone (below).
    expect(result!.content.some((b) => b.type === "image")).toBe(true);
    expect(result!.content.some((b) => b.text === UNAVAILABLE)).toBe(false);
  });

  it("swaps the re-attached fabric_exec image for the description via the context hook", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    // Warm the cache via the nested tool_result (which keeps the image),
    // mirroring how pi-fabric re-attaches the image to the fabric_exec
    // tool-result content that the `context` hook then scans.
    await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "fabric_abc-123",
        input: { path: "x.png" },
        content: readImageResult("AAA") as any,
        isError: false,
      },
      ctxWithSignal(),
    );

    const messages = [
      {
        role: "toolResult",
        content: [
          { type: "text", text: "Read image file [image/png]" },
          imageContent("AAA"),
        ],
      },
    ];
    const result = (await handlers["context"](
      { type: "context", messages: messages as any },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    expect(result).toBeDefined();
    const blocks = result!.messages[0]!.content;
    // The image is swapped for the cached description on the LLM-bound clone.
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    expect(blocks.some((b) => b.text?.startsWith("[Image: desc-for-"))).toBe(true);
  });

  it("substitutes UNAVAILABLE via the context hook when the describer fails", async () => {
    runBatch.mockImplementation(async () => new Map());
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const messages = [
      {
        role: "toolResult",
        content: [
          { type: "text", text: "Read image file [image/png]" },
          imageContent("ZZZ"),
        ],
      },
    ];
    const result = (await handlers["context"](
      { type: "context", messages: messages as any },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    expect(result).toBeDefined();
    const blocks = result!.messages[0]!.content;
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    expect(blocks.some((b) => b.text === UNAVAILABLE)).toBe(true);
  });

  it("does nothing when handoff is not a target for the active model", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "fabric_abc-123",
        input: { path: "x.png" },
        content: readImageResult("DDD") as any,
        isError: false,
      },
      { ...sessionCtx(), model: { provider: "agent", id: "vision-model", input: ["text", "image"] }, signal: new AbortController().signal },
    )) as unknown;

    expect(result).toBeUndefined();
    expect(runBatch).not.toHaveBeenCalled();
  });
});

const flushAsyncHandoff = async () => {
  for (let i = 0; i < 4; i++) {
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
};

describe("async pasted-path fallback race", () => {
  beforeEach(() => {
    runBatch.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) => {
      const out = new Map<string, string>();
      for (const miss of misses) out.set(miss.hash, `desc-for-${miss.hash}`);
      return out;
    });
    findPastedImagePathsMock.mockClear();
    readImageBufferMock.mockClear();
    resolvePrewarmImageMock.mockClear();
  });

  it("injects a steering message when no matching read wins", async () => {
    const { pi, handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    await handlers["before_agent_start"](
      { type: "before_agent_start", prompt: "inspect clipboard-image", images: [] },
      sessionCtx(),
    );
    await handlers["tool_call"](
      { type: "tool_call", toolName: "fabric_exec", input: { code: "return 1" } },
      ctxWithSignal(),
    );
    await flushAsyncHandoff();

    expect(pi.sendMessage).toHaveBeenCalledTimes(1);
    expect(pi.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        customType: "vision-handoff-async",
        content: expect.stringContaining("[Image: desc-for-"),
        display: true,
        details: { imageCount: 1 },
      }),
      { deliverAs: "steer", triggerTurn: true },
    );
  });

  it("lets a nested pi.read of the pasted path cancel the fallback", async () => {
    const { pi, handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    await handlers["before_agent_start"](
      { type: "before_agent_start", prompt: "inspect clipboard-image", images: [] },
      sessionCtx(),
    );
    await handlers["tool_call"](
      { type: "tool_call", toolName: "fabric_exec", input: { code: "pi.read(...)" } },
      ctxWithSignal(),
    );
    await handlers["tool_call"](
      { type: "tool_call", toolName: "read", input: { path: ASYNC_CLIPBOARD_PATH } },
      ctxWithSignal(),
    );
    await flushAsyncHandoff();

    // Cancellation may beat the setImmediate DataLoader dispatch entirely or
    // leave it running for the read result to reuse; only message injection is
    // required to lose this race.
    expect(pi.sendMessage).not.toHaveBeenCalled();
  });

  it("does not cancel for a read of an unrelated path", async () => {
    const { pi, handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    await handlers["before_agent_start"](
      { type: "before_agent_start", prompt: "inspect clipboard-image", images: [] },
      sessionCtx(),
    );
    await handlers["tool_call"](
      { type: "tool_call", toolName: "pi.read", input: { path: "/tmp/unrelated.txt" } },
      ctxWithSignal(),
    );
    await flushAsyncHandoff();

    expect(pi.sendMessage).toHaveBeenCalledTimes(1);
  });

  it("renders a dim summary until expanded, then dims the expanded body per line", () => {
    const { pi } = setup();
    const renderer = pi.registerMessageRenderer.mock.calls.find(
      ([customType]) => customType === "vision-handoff-async",
    )?.[1];
    const fg = vi.fn((_color: string, text: string) => text);
    const theme = { fg };
    const message = {
      content: "Asynchronous vision handoff\n[Image: detailed description]",
      details: { imageCount: 1 },
    };

    const collapsed = renderer(message, { expanded: false }, theme).render(200).join("\n");
    expect(collapsed).toContain("Ctrl+O to expand");
    expect(collapsed).not.toContain("detailed description");

    fg.mockClear();
    const expanded = renderer(message, { expanded: true }, theme).render(200).join("\n");
    expect(expanded).toContain("detailed description");
    expect(expanded).toContain("Ctrl+O to collapse");
    // The summary line and every content line are themed dim (grey), per line.
    expect(fg).toHaveBeenCalledWith("dim", expect.stringContaining("Vision handoff"));
    expect(fg).toHaveBeenCalledWith("dim", "Asynchronous vision handoff");
    expect(fg).toHaveBeenCalledWith("dim", "[Image: detailed description]");
  });
});

describe("omitted-image recovery (read emitted [Image omitted], no image block)", () => {
  beforeEach(() => {
    runBatch.mockClear();
    describeSingle.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) => {
      const out = new Map<string, string>();
      for (const m of misses) out.set(m.hash, "desc-for-" + m.hash);
      return out;
    });
    readImageBufferBoundedMock.mockClear();
    readImageBufferBoundedMock.mockImplementation(() => ({ buf: Buffer.from("RECOVERED"), mimeType: "image/png" }));
  });

  it("re-reads the raw file and replaces the omitted note with the description", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const omittedContent = [
      {
        type: "text",
        text: "Read image file [image/png]\n[Image omitted: could not be resized below the inline image size limit.]\n[Current model does not support images. The image will be omitted from this request.]",
      },
    ];
    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_omit-1",
        input: { path: "/abs/x.png" },
        content: omittedContent as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    expect(readImageBufferBoundedMock).toHaveBeenCalledWith("/abs/x.png");
    expect(result!.content.some((b) => b.text?.startsWith("[Image: desc-for-"))).toBe(true);
    expect(result!.content.some((b) => b.text?.includes("[Image omitted:"))).toBe(false);
    expect(runBatch).toHaveBeenCalledTimes(1);
  });

  it("resolves a relative read path against ctx.cwd", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const omittedContent = [
      { type: "text", text: "Read image file [image/png]\n[Image omitted: could not be resized below the inline image size limit.]" },
    ];
    await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_omit-2",
        input: { path: "rel/x.png" },
        content: omittedContent as any,
        isError: false,
      },
      { ...ctxWithSignal(), cwd: "/working/dir" },
    );
    expect(readImageBufferBoundedMock).toHaveBeenCalledWith("/working/dir/rel/x.png");
  });

  it("leaves the omitted note when the file can't be re-read (no path)", async () => {
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const omittedContent = [
      { type: "text", text: "Read image file [image/png]\n[Image omitted: could not be resized below the inline image size limit.]" },
    ];
    const result = await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_omit-3",
        input: {},
        content: omittedContent as any,
        isError: false,
      },
      ctxWithSignal(),
    );
    expect(result).toBeUndefined();
    expect(runBatch).not.toHaveBeenCalled();
  });

  it("leaves the omitted note when the re-read returns null (APNG/unsupported/too large)", async () => {
    readImageBufferBoundedMock.mockImplementationOnce(() => null);
    const { handlers } = setup();
    await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());

    const omittedContent = [
      { type: "text", text: "Read image file [image/png]\n[Image omitted: could not be resized below the inline image size limit.]" },
    ];
    const result = await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_omit-4",
        input: { path: "/abs/x.png" },
        content: omittedContent as any,
        isError: false,
      },
      ctxWithSignal(),
    );
    expect(result).toBeUndefined();
    expect(runBatch).not.toHaveBeenCalled();
  });
});
