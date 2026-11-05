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

// Keep every real export of src/index.js but force a deterministic, MUTABLE
// config so each test can toggle persistDescriptions and reload it via
// session_start (mirrors editing the config file between sessions).
const configState = vi.hoisted(() => ({
  current: {
    enabled: true,
    visionModel: "test/vision",
    fallbackModels: [],
    autoHandoff: true,
    handoffModels: [],
    prewarmPastedImages: false,
    asyncClipboardHandoff: false,
    persistDescriptions: false,
    awarePrompt: false,
    maxTokens: undefined,
    cacheMax: 50,
    maxDescriptionLines: 0,
    thinking: false,
    thinkingLevel: "medium",
  },
}));

vi.mock("../../src/index.js", async (importActual) => {
  const actual = (await importActual()) as Record<string, unknown>;
  return {
    ...actual,
    readConfig: () => ({ ...configState.current }),
  };
});

import { tmpdir } from "node:os";
import { join } from "node:path";

vi.mock("../../src/image.js", async (importActual) => {
  const actual = await importActual<typeof import("../../src/image.js")>();
  return { ...actual, readImageBuffer: vi.fn(actual.readImageBuffer), resolvePrewarmImage: vi.fn(actual.resolvePrewarmImage) };
});

import factory from "../../vision-handoff.js";
import { UNAVAILABLE } from "../../src/dataloader.js";
import { imageHash, findPastedImagePaths, readImageBuffer, resolvePrewarmImage } from "../../src/image.js";
import { buildPersistedDescriptionBlock } from "../../src/index.js";

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
  cwd: "/working/dir",
  model: { provider: "agent", id: "text-only", input: ["text"] },
  modelRegistry: {
    find: () => ({ provider: "test", id: "vision", input: ["text", "image"] }),
  },
  ui: { notify: vi.fn() },
});

const ctxWithSignal = () => ({ ...sessionCtx(), signal: new AbortController().signal });

const imageContent = (data: string) => ({
  type: "image",
  data,
  mimeType: "image/png",
});

const hashOf = (data: string) => imageHash("image/png", data);

/** The shape of a read tool result for an image file, with pi's non-vision
 *  note (which the tool_result handler strips). */
const readImageResult = (data: string) => [
  {
    type: "text",
    text: "Read image file [image/png]\n[Image: original 10x10.]\n[Current model does not support images. The image will be omitted from this request.]",
  },
  imageContent(data),
];

/** The persisted-description text block the tool_result handler writes when
 *  persistDescriptions is on. */
const persistedBlock = (data: string, description: string) => ({
  type: "text",
  text: buildPersistedDescriptionBlock(hashOf(data), description),
});

const startSession = async (handlers: Record<string, (event: any, ctx: any) => Promise<unknown>>) => {
  await handlers["session_start"]({ type: "session_start", reason: "startup" }, sessionCtx());
};

describe("eager user attachments", () => {
  beforeEach(() => {
    vi.mocked(readImageBuffer).mockClear();
    vi.mocked(resolvePrewarmImage).mockClear();
    runBatch.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) =>
      new Map(misses.map(({ hash }) => [hash, `desc-for-${hash}`])));
    configState.current.persistDescriptions = false;
    configState.current.asyncClipboardHandoff = true;
    configState.current.awarePrompt = false;
  });

  it("aware prompt prefers supplied attachment descriptions and treats markers as provenance", async () => {
    configState.current.awarePrompt = true;
    const { handlers } = setup();
    await startSession(handlers);
    const result = await handlers.before_agent_start({ prompt: "look", images: [], systemPrompt: "base" }, sessionCtx()) as any;
    expect(result.systemPrompt).toContain("do not reread the attached path");
    expect(result.systemPrompt).toContain("records provenance, not proof");
    expect(result.systemPrompt).toContain("say so explicitly");
    configState.current.awarePrompt = false;
  });

  it("skips only marked paths while preserving the unmarked temp-path flow", () => {
    const attached = join(tmpdir(), "pi-clipboard-attached.png");
    const legacy = join(tmpdir(), "localterm-paste-legacy.png");
    expect(findPastedImagePaths(`[Attached image: ${attached}] ${legacy}`)).toEqual([legacy]);
    expect(findPastedImagePaths(`[Attached image: ${attached}] ${attached}`)).toEqual([attached]);
  });

  // Input hooks finish before ANY before_agent_start hook, independent of
  // extension load order. Simulate Fabric's completed transform in both orders.
  for (const order of ["fabric-first", "handoff-first"]) {
    for (const normalized of ["RAW", "NORMALIZED"]) {
      it(`${order}: ${normalized} bytes reach the first request once, without path fallback`, async () => {
        const { handlers, pi } = setup();
        await startSession(handlers);
        const data = `${order}-${normalized}`;
        const input = { prompt: `inspect [Attached image: ${join(tmpdir(), "pi-clipboard-eager.png")}]`, images: [imageContent(`${order}-RAW`)] };
        await handlers.before_agent_start({ messageType: "user", ...input }, sessionCtx());
        await new Promise<void>((resolve) => setImmediate(resolve));
        expect(runBatch).not.toHaveBeenCalled();
        expect(readImageBuffer).not.toHaveBeenCalled();
        expect(resolvePrewarmImage).not.toHaveBeenCalled();
        const stored = { role: "user", content: [{ type: "text", text: input.prompt }, imageContent(data)] };
        const result = await handlers.context({ messages: structuredClone([stored]) }, ctxWithSignal()) as any;
        expect(runBatch).toHaveBeenCalledTimes(1);
        expect(runBatch.mock.calls[0][0]).toEqual([expect.objectContaining({ hash: hashOf(data) })]);
        expect(JSON.stringify(result.messages)).toContain(`desc-for-${hashOf(data)}`);
        expect(result.messages[0].content.some((b: any) => b.type === "image")).toBe(false);
        expect(stored.content[1]).toEqual(imageContent(data));
        await new Promise<void>((resolve) => setImmediate(resolve));
        expect(pi.sendMessage).not.toHaveBeenCalled();
      });
    }
  }

  it("batches all existing user ImageContent attachments on the first request", async () => {
    const { handlers } = setup();
    await startSession(handlers);
    await handlers.before_agent_start({ prompt: "compare", images: [imageContent("BATCH-A"), imageContent("BATCH-B")] }, sessionCtx());
    expect(runBatch).not.toHaveBeenCalled();
    await handlers.context({ messages: [{ role: "user", content: [imageContent("BATCH-A"), imageContent("BATCH-B")] }] }, ctxWithSignal());
    expect(runBatch).toHaveBeenCalledTimes(1);
    expect(runBatch.mock.calls[0][0]).toHaveLength(2);
  });

  it("persists normalized user attachments without replacing images; resume needs no describer", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);
    const original = { role: "user", timestamp: 123, content: [imageContent("USER-PERSIST-A"), imageContent("USER-PERSIST-B")] };
    const result = await handlers.message_end({ message: original }, ctxWithSignal()) as any;
    expect(result.message.timestamp).toBe(123);
    expect(result.message.content.slice(0, 2)).toEqual(original.content);
    expect(original.content).toHaveLength(2);
    expect(runBatch).toHaveBeenCalledTimes(1);
    expect(runBatch.mock.calls[0][0]).toHaveLength(2);
    // A fresh module instance has an empty loader cache, just as on /resume.
    vi.resetModules();
    const freshFactory = (await import("../../vision-handoff.js")).default;
    const resumed = setup();
    freshFactory(resumed.pi as any);
    await startSession(resumed.handlers);
    runBatch.mockClear();
    const context = await resumed.handlers.context({ messages: structuredClone([result.message]) }, ctxWithSignal()) as any;
    expect(runBatch).not.toHaveBeenCalled();
    expect(context.messages[0].content.filter((b: any) => b.type === "image")).toHaveLength(0);
    expect(context.messages[0].content.filter((b: any) => b.text?.startsWith("[Image: desc-for-"))).toHaveLength(2);
    expect(JSON.stringify(context)).not.toContain("[Image described:");
  });

  it("leaves native vision attachments untouched, including stored originals", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);
    const ctx = { ...ctxWithSignal(), model: { provider: "native", id: "vision", input: ["text", "image"] } };
    const message = { role: "user", content: [imageContent("NATIVE")] };
    expect(await handlers.message_end({ message }, ctx)).toBeUndefined();
    expect(await handlers.context({ messages: [message] }, ctx)).toBeUndefined();
    expect(message.content).toEqual([imageContent("NATIVE")]);
    expect(runBatch).not.toHaveBeenCalled();
  });

  it("does not persist failed user descriptions, and retries on a later request", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);
    runBatch.mockImplementation(async () => new Map());
    const message = { role: "user", content: [imageContent("USER-RETRY")] };
    expect(await handlers.message_end({ message }, ctxWithSignal())).toBeUndefined();
    const failedCalls = runBatch.mock.calls.length;
    runBatch.mockImplementation(async (misses: { hash: string }[]) => new Map(misses.map(({ hash }) => [hash, "recovered"])));
    const result = await handlers.context({ messages: structuredClone([message]) }, ctxWithSignal()) as any;
    expect(runBatch.mock.calls.length).toBeGreaterThan(failedCalls);
    expect(JSON.stringify(result)).toContain("[Image: recovered]");
    expect(message.content).toEqual([imageContent("USER-RETRY")]);
  });
});

describe("persistDescriptions — session-file persistence", () => {
  beforeEach(() => {
    runBatch.mockClear();
    describeSingle.mockClear();
    runBatch.mockImplementation(async (misses: { hash: string }[]) => {
      const out = new Map<string, string>();
      for (const m of misses) out.set(m.hash, `desc-for-${m.hash}`);
      return out;
    });
  });

  it("persist ON: the read tool_result carries a [Image described: <hash>] block next to the blob", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);

    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_1",
        input: { path: "x.png" },
        content: readImageResult("PERSIST-A") as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    // The blob stays (kitty rendering / /resume)...
    expect(result!.content.some((b) => b.type === "image")).toBe(true);
    // ...and the persisted description block sits next to it.
    const desc = result!.content.find((b) => b.text?.includes(`[Image described: ${hashOf("PERSIST-A")}]`));
    expect(desc).toBeDefined();
    expect(desc!.text).toContain("[Image: desc-for-");
    // pi's non-vision note is still stripped.
    expect(result!.content.some((b) => b.text?.includes("does not support images"))).toBe(false);
    expect(runBatch).toHaveBeenCalledTimes(1);
  });

  it("persist ON: describer failure (UNAVAILABLE) is NOT persisted — next turn re-attempts", async () => {
    runBatch.mockImplementation(async () => new Map());
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);

    const result = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_2",
        input: { path: "x.png" },
        content: readImageResult("PERSIST-FAIL") as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(result).toBeDefined();
    // Blob stays, but no persisted marker block (a failure must not be replayed).
    expect(result!.content.some((b) => b.type === "image")).toBe(true);
    expect(result!.content.some((b) => b.text?.includes("[Image described: "))).toBe(false);
  });

  it("persist ON + resume: context reuses the persisted description — NO vision call, blob dropped, marker stripped", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);

    // Simulate a resumed session: the stored read tool-result content carries
    // the blob AND the persisted description block. The in-memory cache is
    // empty — runBatch has never been called.
    const data = "PERSIST-RESUME";
    const messages = [
      {
        role: "toolResult",
        content: [
          { type: "text", text: "Read image file [image/png]" },
          imageContent(data),
          persistedBlock(data, "[Image: desc-for-resume]"),
        ],
      },
    ];
    const result = (await handlers["context"](
      { type: "context", messages: messages as any },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    expect(result).toBeDefined();
    const blocks = result!.messages[0]!.content;
    // No image blob in the LLM-bound payload (swapped away)...
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    // ...the description text is present EXACTLY ONCE...
    const descs = blocks.filter((b) => b.text?.startsWith("[Image: desc-for-resume]"));
    expect(descs).toHaveLength(1);
    // ...and the marker itself is stripped (the model never sees it).
    expect(blocks.some((b) => b.text?.includes("[Image described: "))).toBe(false);
    // THE key assertion: no vision call on resume.
    expect(runBatch).not.toHaveBeenCalled();
  });

  it("persist ON: same-turn tool_result + context produces a single description (no duplicate)", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);

    // Turn: read → tool_result describes (1 call) and writes the persisted block.
    const data = "PERSIST-TURN";
    const stored = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_3",
        input: { path: "x.png" },
        content: readImageResult(data) as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> };

    // context fires with the STORED content (blob + persisted block).
    const result = (await handlers["context"](
      { type: "context", messages: [{ role: "toolResult", content: stored.content }] as any },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    const blocks = result!.messages[0]!.content;
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    const descs = blocks.filter((b) => b.text?.startsWith("[Image: desc-for-"));
    expect(descs).toHaveLength(1);
    // One vision call total (the tool_result describe); the context reuse adds none.
    expect(runBatch).toHaveBeenCalledTimes(1);
  });

  it("persist ON + resume: a mixed payload describes only the uncovered image", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);

    const persistedData = "PERSIST-COVERED";
    const freshData = "PERSIST-FRESH";
    const messages = [
      {
        role: "toolResult",
        content: [
          { type: "text", text: "Read image file [image/png]" },
          imageContent(persistedData),
          persistedBlock(persistedData, "[Image: from session]"),
        ],
      },
      {
        role: "toolResult",
        content: [{ type: "text", text: "Read image file [image/png]" }, imageContent(freshData)],
      },
    ];
    const result = (await handlers["context"](
      { type: "context", messages: messages as any },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    const m0 = result!.messages[0]!.content;
    const m1 = result!.messages[1]!.content;
    expect(m0.some((b) => b.type === "image")).toBe(false);
    expect(m1.some((b) => b.type === "image")).toBe(false);
    expect(m0.some((b) => b.text === "[Image: from session]")).toBe(true);
    expect(m1.some((b) => b.text?.startsWith("[Image: desc-for-"))).toBe(true);
    // Only the uncovered image triggered a vision call (single-image batch).
    expect(runBatch).toHaveBeenCalledTimes(1);
    expect(runBatch.mock.calls[0][0]).toHaveLength(1);
  });

  it("persist OFF: behavior unchanged — no persisted block, context describes via the vision model", async () => {
    configState.current.persistDescriptions = false;
    const { handlers } = setup();
    await startSession(handlers);

    // tool_result: no persisted marker block, image kept.
    const data = "PERSIST-OFF";
    const stored = (await handlers["tool_result"](
      {
        type: "tool_result",
        toolName: "read",
        toolCallId: "call_4",
        input: { path: "x.png" },
        content: readImageResult(data) as any,
        isError: false,
      },
      ctxWithSignal(),
    )) as { content: Array<{ type: string; text?: string }> } | undefined;

    expect(stored).toBeDefined();
    expect(stored!.content.some((b) => b.text?.includes("[Image described: "))).toBe(false);
    expect(stored!.content.some((b) => b.type === "image")).toBe(true);
    const callsAfterRead = runBatch.mock.calls.length;

    // context on a fresh cold payload (no persisted markers): the image is a
    // cache-miss → vision call, image swapped for text, exactly as before the
    // persistence feature.
    const result = (await handlers["context"](
      {
        type: "context",
        messages: [
          {
            role: "toolResult",
            content: [{ type: "text", text: "Read image file" }, imageContent("PERSIST-OFF-COLD")],
          },
        ] as any,
      },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    expect(result).toBeDefined();
    const blocks = result!.messages[0]!.content;
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    expect(blocks.some((b) => b.text?.startsWith("[Image: desc-for-"))).toBe(true);
    expect(runBatch.mock.calls.length).toBeGreaterThan(callsAfterRead);
  });

  it("persist OFF: a pre-existing persisted block is still honored (no duplicate description)", async () => {
    configState.current.persistDescriptions = false;
    const { handlers } = setup();
    await startSession(handlers);

    // A marker written while persist was ON survives an OFF toggle; reusing it
    // prevents a duplicate description in the payload (and wastes no call).
    const data = "PERSIST-OFF-LEGACY";
    const result = (await handlers["context"](
      {
        type: "context",
        messages: [
          {
            role: "toolResult",
            content: [
              { type: "text", text: "Read image file" },
              imageContent(data),
              persistedBlock(data, "[Image: legacy]"),
            ],
          },
        ] as any,
      },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    const blocks = result!.messages[0]!.content;
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    expect(blocks.filter((b) => b.text === "[Image: legacy]")).toHaveLength(1);
    expect(runBatch).not.toHaveBeenCalled();
  });

  it("persist ON: a persisted UNAVAILABLE body is ignored → falls back to the vision model", async () => {
    configState.current.persistDescriptions = true;
    const { handlers } = setup();
    await startSession(handlers);

    const data = "PERSIST-UNAVAIL";
    const result = (await handlers["context"](
      {
        type: "context",
        messages: [
          {
            role: "toolResult",
            content: [
              { type: "text", text: "Read image file" },
              imageContent(data),
              persistedBlock(data, UNAVAILABLE),
            ],
          },
        ] as any,
      },
      ctxWithSignal(),
    )) as { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined;

    const blocks = result!.messages[0]!.content;
    expect(blocks.some((b) => b.type === "image")).toBe(false);
    expect(blocks.some((b) => b.text?.startsWith("[Image: desc-for-"))).toBe(true);
    expect(runBatch).toHaveBeenCalledTimes(1);
  });
});
