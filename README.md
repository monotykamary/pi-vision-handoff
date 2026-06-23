<div align="center">

# 👁️ pi-vision-handoff

**Give text-only [pi](https://github.com/earendil-works/pi-coding-agent) models vision**

_Describe images with a vision model you pick, then feed the text to models that can't see._

[![pi extension](https://img.shields.io/badge/pi-extension-blueviolet)](https://github.com/earendil-works/pi-coding-agent)
[![npm](https://img.shields.io/npm/v/pi-vision-handoff)](https://www.npmjs.com/package/pi-vision-handoff)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

</div>

---

## The Problem

Some of the best coding models are blind. You paste a screenshot, a UI mock, a stack trace, or a diagram into pi — and a text-only model either silently ignores the image or rejects the request outright. Up to now your only options were to describe the image yourself, or switch to a (often weaker-for-coding) vision model just to read it.

The `pi-umans-provider` extension quietly solved this for GLM 5.1: a hardcoded "vision handoff" pipeline that had `umans-flash` describe each image at prompt time and swapped the text in for the image block before the request left. It worked great — but it was welded to one provider, one describer, and one set of models.

## The Solution

`pi-vision-handoff` extracts that pipeline and makes it **provider-agnostic**:

- **Pick any vision-capable model** from your registry via an interactive picker — OpenAI, Anthropic, Google, Ollama, or any custom provider pi knows about.
- Your choice **persists** to `~/.pi/agent/extensions/pi-vision-handoff.json`.
- For any model that doesn't declare image input (or any model you explicitly target), `pi-vision-handoff` describes the image with your chosen vision model and **inserts the description before the kept image** in the stored `read`-tool result — **before** pi-ai can strip the image for non-vision models. The image stays for kitty rendering; pi-ai later strips only the image block, leaving the description text for the model.
- Works across all four image-block shapes pi uses — the three provider-transformed formats (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) plus pi-ai's internal `{ type: "image", data, mimeType }` emitted by the `read` tool — detected by shape.
- Descriptions are **cached per image hash** (LRU), so the swap is instant by the time the request fires.

No `settings.json` touched. No per-provider glue. Pick a describer once and every text-only model you own can suddenly see.

## Features

- **🎮 Interactive picker** — `/vision-handoff` opens a TUI listing every model, vision-capable ones first (👁), to choose your describer.
- **🖼️ Read-tool hijack** — `read`-tool images are intercepted at `tool_result` time, before pi-ai can strip them for non-vision models. The kept image still renders inline (kitty); the description is inserted as a text block the text-only model consumes.
- **🔌 Provider-agnostic** — uses pi's own model execution machinery (`@earendil-works/pi-ai`'s `complete()`), so it works with any provider/configured model, including custom provider extensions.
- **🧠 Automatic targets** — by default, handoff applies to *every* model that lacks native vision. Opt out with `/vision-handoff auto off`.
- **🗂️ Explicit overrides** — force handoff for specific models (e.g. a weak vision model) with `/vision-handoff add`.
- **⚡ Cache-warmed** — `before_agent_start` describes attached images the moment you submit, so the request is rarely delayed.
- **🛡️ Graceful degradation** — no API key? Describer unreachable? Aborted? The image is replaced with a clean `[Image: description unavailable]` placeholder instead of crashing your turn.
- **📊 Usage reporting** — every real describer call reports model + tokens (and Neuralwatt energy/cost when the vision model is a Neuralwatt model), via `pi.appendEntry` + a `vision-handoff:usage` event for live consumers.
- **🔧 Tunable** — cap description length (`maxDescriptionLines`; unbounded by default, so `read`'s native `ctrl+o` collapse handles compactness) and cache size, in the config file.

## Usage

### Interactive Commands

| Command | What it does |
|---------|-------------|
| `/vision-handoff` | Open the interactive picker to choose the vision model |
| `/vision-handoff select` | Same as `/vision-handoff` |
| `/vision-handoff model openai/gpt-4o` | Set the vision model directly |
| `/vision-handoff status` | Show current config + whether handoff is active for the current model |
| `/vision-handoff enable` / `disable` | Master switch (keeps your configured model) |
| `/vision-handoff auto on` / `off` | Toggle automatic handoff for all non-vision models |
| `/vision-handoff add ollama/llava:13b` | Force handoff for an extra model |
| `/vision-handoff remove ollama/llava:13b` | Stop forcing handoff for a model |
| `/vision-handoff clear` | Clear the configured vision model |
| `/vision-handoff help` | Show usage reference |

### Config File

Created automatically at `~/.pi/agent/extensions/pi-vision-handoff.json` on first change:

```json
{
  "enabled": true,
  "visionModel": "openai/gpt-4o",
  "autoHandoff": true,
  "handoffModels": ["ollama/llava:13b"],
  "maxTokens": 1024,
  "cacheMax": 50,
  "maxDescriptionLines": 0,
  "prompt": "Describe this image exhaustively…",
  "userPromptPrefix": "The user's request about this image: "
}
```

| Field | Default | Effect |
|-------|---------|--------|
| `enabled` | `true` | Master switch. When `false`, no handoff occurs. |
| `visionModel` | `null` | The describer, as `provider/id`. `null` = not configured (handoff inactive). |
| `autoHandoff` | `true` | Apply handoff to every model whose `input` does not include `image`. |
| `handoffModels` | `[]` | Extra `provider/id` refs that should also receive handoff. |
| `maxTokens` | `1024` | Cap on a single description's output. |
| `cacheMax` | `50` | Max described images kept in the in-memory cache per session. |
| `maxDescriptionLines` | `0` | Cap on description lines (`0` = unbounded). Default keeps the full description so the `read` tool's native collapse (`ctrl+o`) handles compactness and the model gets complete context; setting `> 0` applies a lossy head-cap to both the TUI render and the model. |
| `prompt` | _(built-in)_ | Override the describer system prompt. |
| `userPromptPrefix` | _(built-in)_ | Override the prefix prepended to your original prompt. |

> The config path uses pi's `getAgentDir()` — set `PI_CODING_AGENT_DIR` to relocate it.

## Installation

**With `pi install`** (recommended):

```bash
pi install npm:pi-vision-handoff
```

Or install from GitHub:

```bash
pi install https://github.com/monotykamary/pi-vision-handoff
```

**With npm**:

```bash
npm install pi-vision-handoff
```

Or in `~/.pi/agent/settings.json`:

```json
{
  "packages": [
    "npm:pi-vision-handoff"
  ]
}
```

Then `/reload` or restart pi.

For a quick one-off test:

```bash
pi -e ./vision-handoff.ts
```

## How It Works

Read-tool images (the common case — paste a screenshot, `read` an image file):

    read tool returns an image block
      → tool_result (toolName === "read")
          • fires BEFORE pi-ai's transformMessages can strip the image for
            non-vision models (the strip would leave only a "(tool image
            omitted)" placeholder — too late to describe)
          • is this model a handoff target? (non-vision, or in handoffModels)
          • for each {type:"image",...} block → describeImage() with your
            chosen vision model
              - complete() via pi-ai (resolves API key/headers/baseUrl)
              - cached by sha256(mime + base64)
          • AUGMENTS the stored result: inserts "[Image: <description>]" as a
            text block BEFORE the kept image, returns {content} to pi
          • the image stays in content → kitty renders it inline
          • at provider time, pi-ai strips ONLY the image block for non-vision
            models — the inserted description text passes through to the model

User-attached images (pasted into the prompt, not read via a tool):

    → before_agent_start
        • warms the description cache for attached images (fire-and-forget)
    → before_provider_request
        • walks the provider payload, swaps image blocks for "[Image: <desc>]"
        • NOTE: for non-vision models, pi-ai strips image blocks BEFORE this
          hook fires — so user-attached images on text-only models are not yet
          described. Use the `read` tool on the image file instead, which routes
          through the tool_result path above.

Result: the text-only model receives a vivid text description alongside the
(now-stripped) image reference, and your turn proceeds normally — while the
terminal still renders the image inline via kitty.

### Image-block formats handled

| Hook | Image block shape | Replacement |
|------|-------------------|-------------|
| `tool_result` (read) | `{ type: "image", data, mimeType }` (pi-ai internal) | description text block inserted **before** the kept image |
| `before_provider_request` | `{ type: "image_url", image_url: { url: "data:…" } }` (OpenAI Chat Completions) | `{ type: "text", text }` |
| `before_provider_request` | `{ type: "input_image", image_url: "data:…" }` (OpenAI Responses) | `{ type: "input_text", text }` |
| `before_provider_request` | `{ type: "image", source: { type: "base64", media_type, data } }` (Anthropic Messages) | `{ type: "text", text }` |

The describer call itself goes through pi's normal model machinery (`complete()`), **not** the agent event loop — so it never re-triggers `before_provider_request` (no recursion).

### Usage reporting

Every **real** describer call (cache misses only — cache hits emit nothing) reports its model + tokens so pi and other extensions can account for the handoff cost. When the vision model is a **Neuralwatt** model, the response's `: energy` / `: cost` / `: mcr-session` SSE comments are also captured (the OpenAI SDK discards comment lines, so the response body is teed and parsed — the same technique `pi-neuralwatt-provider` uses). For non-Neuralwatt vision models the energy fields are **omitted** (not zeroed), so consumers can distinguish "no energy" from "zero energy".

Each record is published two ways, mirroring `pi-neuralwatt-provider`'s `neuralwatt:turn-energy` pattern:

- **`pi.appendEntry("vision-handoff-usage", record)`** — persisted to the session log, so it replays on `/resume`, fork, and branch navigation.
- **`pi.events.emit("vision-handoff:usage", record)`** — live event-bus channel a consumer (e.g. a `pi-tps`-style extension) can filter on to see tokens **and** energy in one payload.

Record shape:

```ts
{
  imageHash: string,            // sha256(mime + base64), first 32 hex chars
  model: string, provider: string,
  responseModel?: string, responseId?: string,
  usage: Usage,                 // { input, output, cacheRead, cacheWrite, totalTokens, cost }
  // Present ONLY when Neuralwatt SSE energy comments were captured:
  energyJoules?: number, costUsd?: number,
  energyRaw?: object, mcrSessionRaw?: object, costRaw?: object,
}
```

Because `before_agent_start` fires several `describeImage()` calls fire-and-forget, the fetch interception is **refcounted** (installed only while ≥1 describe is in flight) and uses `AsyncLocalStorage` to route each teed response body to the describe call that issued it — so concurrent describes each attribute their own energy correctly without clobbering `globalThis.fetch`. When the vision model is a Neuralwatt model, `pi-neuralwatt-provider`'s own `streamNeuralwatt` tee nests on top and restores back to this interceptor; both tees read the same comment lines independently (the accepted duplication for easy filtering).

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **pi-vision-handoff** (this) | Provider-agnostic; pick any describer; automatic for text-only models; cached; survives across providers | Adds one extra model call per unique image |
| Native vision on every model | Zero overhead | Not all models support it; you may be forced off your preferred coding model |
| Manually describing images | No extension | Tedious; lossy; kills the "paste a screenshot" workflow |
| The original `pi-umans-provider` handoff | Battle-tested | Hardcoded to `umans-flash` + UMANS models only |
| Switching to a vision model to read an image, then back | Works | Context loss across model swaps; worse coding model for the actual work |

## Development

```bash
pnpm install
pnpm test          # Vitest unit tests (77 passing)
pnpm typecheck     # TypeScript validation
pnpm lint:dead     # Dead code detection (knip)
```

### Structure

```
.
├── vision-handoff.ts            # Main extension: hooks, command, describer
├── src/
│   ├── index.ts                  # Config schema, read/write, image-block helpers (barrel)
│   ├── usage.ts                  # Describer usage + Neuralwatt energy capture, fetch interceptor
│   └── vision-model-selector.ts  # Interactive picker TUI component
├── __tests__/unit/
│   ├── config-dir.test.ts        # Ensures getAgentDir() usage
│   ├── usage.test.ts             # Energy parsing, usage records, concurrency-safe fetch routing
│   └── vision-handoff.test.ts    # Config, refs, image-block extraction, insertion, truncation, round-trip
├── package.json
├── tsconfig.json
├── knip.json
└── vitest.config.ts
```

## Acknowledgements

The vision handoff concept and the exhaustive describer prompt originate from
the [pi-umans-provider](https://github.com/monotykamary/pi-umans-provider) GLM 5.1
pipeline. The picker TUI builds on the patterns from
[pi-hide-providers](https://github.com/monotykamary/pi-hide-providers), which in
turn mirror pi core's built-in selectors.

## License

MIT
