import { afterEach, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const loader = vi.hoisted(() => ({
  imported: vi.fn(),
  created: vi.fn(),
  reset: vi.fn(),
  resetTurnAbort: vi.fn(),
  setPendingTurnPrompt: vi.fn(),
  bindTurnContext: vi.fn(),
  loadDescription: vi.fn(async () => "A blue square"),
}));
vi.mock("../../src/dataloader.js", () => {
  loader.imported();
  return {
    DescriptionLoader: class {
      constructor() { loader.created(); }
      reset = loader.reset;
      resetTurnAbort = loader.resetTurnAbort;
      setPendingTurnPrompt = loader.setPendingTurnPrompt;
      bindTurnContext = loader.bindTurnContext;
      loadDescription = loader.loadDescription;
    },
  };
});

let directory: string | undefined;
afterEach(() => {
  vi.unstubAllEnvs();
  vi.clearAllMocks();
  if (directory) rmSync(directory, { recursive: true, force: true });
});

async function launch() {
  directory = mkdtempSync(join(tmpdir(), "pi-vision-lazy-"));
  vi.stubEnv("PI_CODING_AGENT_DIR", directory);
  mkdirSync(join(directory, "extensions"));
  writeFileSync(join(directory, "extensions", "pi-vision-handoff.json"), JSON.stringify({
    enabled: true,
    autoHandoff: true,
    visionModel: "fixture/vision",
  }));
  vi.resetModules();
  const { default: extension } = await import("../../vision-handoff.js");
  const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
  extension({
    on: (name: string, hook: (event: unknown, ctx: unknown) => unknown) => handlers.set(name, hook),
    registerCommand: () => {},
    registerMessageRenderer: () => {},
  } as unknown as ExtensionAPI);
  const ctx = {
    mode: "rpc",
    hasUI: false,
    cwd: directory,
    model: { provider: "fixture", id: "text", input: ["text"] },
    modelRegistry: { find: () => ({ provider: "fixture", id: "vision", input: ["text", "image"] }) },
    ui: { notify: vi.fn() },
  };
  const fire = async (name: string, event: unknown = {}) => {
    const handler = handlers.get(name);
    if (!handler) throw new Error(`Missing hook: ${name}`);
    return handler(event, ctx);
  };
  return { fire, ctx };
}

it("keeps a configured worker cold through text-only turns and initializes the describer on its first image", async () => {
  const { fire, ctx } = await launch();
  await fire("session_start");
  await vi.dynamicImportSettled();
  expect(loader.imported).not.toHaveBeenCalled();
  expect(loader.created).not.toHaveBeenCalled();

  await fire("before_agent_start", { messageType: "user", prompt: "Describe what happens next", systemPrompt: "fixture" });
  await fire("tool_result", { toolName: "read", input: { path: "text.txt" }, content: [{ type: "text", text: "fixture" }] });
  await fire("before_agent_start", { messageType: "custom", systemPrompt: "fixture" });
  await vi.dynamicImportSettled();
  expect(loader.imported).not.toHaveBeenCalled();

  await fire("tool_result", { toolName: "read", input: { path: "image.png" }, content: [{ type: "image", mimeType: "image/png", data: "AAAA" }] });
  expect(loader.imported).toHaveBeenCalledOnce();
  expect(loader.created).toHaveBeenCalledOnce();
  expect(loader.resetTurnAbort).toHaveBeenCalledOnce();
  expect(loader.setPendingTurnPrompt).toHaveBeenCalledWith("Describe what happens next");
  expect(loader.bindTurnContext).toHaveBeenCalledWith(ctx);
  expect(loader.loadDescription).toHaveBeenCalledWith({ mimeType: "image/png", data: "AAAA" });

  await fire("session_shutdown");
  await fire("session_start");
  expect(loader.reset).toHaveBeenCalledOnce();
  expect(loader.created).toHaveBeenCalledOnce();
});