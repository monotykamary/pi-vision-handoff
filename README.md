<div align="center">

# 👁️ pi-vision-handoff

**Give text-only [pi](https://github.com/earendil-works/pi-coding-agent) models vision**

_Describe images with a vision model you pick, then feed the text to models that can't see._

[![pi extension](https://img.shields.io/badge/pi-extension-blueviolet)](https://github.com/earendil-works/pi-coding-agent)
[![npm](https://img.shields.io/npm/v/pi-vision-handoff)](https://www.npmjs.com/package/pi-vision-handoff)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

<img src="https://raw.githubusercontent.com/monotykamary/pi-vision-handoff/main/assets/vision-handoff.jpg" alt="Vision Handoff picker — an interactive TUI listing every model, vision-capable ones marked with an eye, to choose the describer for text-only models" width="820">

</div>

---

## The Problem

Some of the best coding models are blind. You paste a screenshot, a UI mock, a stack trace, or a diagram into pi — and a text-only model either silently ignores the image or rejects the request outright. Up to now your only options were to describe the image yourself, or switch to a (often weaker-for-coding) vision model just to read it.

The `pi-umans-provider` extension quietly solved this for GLM 5.1: a hardcoded "vision handoff" pipeline that had `umans-flash` describe each image at prompt time and swapped the text in for the image block before the request left. It worked great — but it was welded to one provider, one describer, and one set of models.

## The Solution

`pi-vision-handoff` extracts that pipeline and makes it **provider-agnostic**:

- **Pick any vision-capable model** from your registry via an interactive picker — OpenAI, Anthropic, Google, Ollama, or any custom provider pi knows about.
- Your choice **persists** to `~/.pi/agent/extensions/pi-vision-handoff.json`.
- For any model that doesn't declare image input (or any model you explicitly target), `pi-vision-handoff` describes the image with your chosen vision model and swaps the image block for its description text at the **`context`** event — which fires *before* pi-ai's `downgradeUnsupportedImages` can strip image blocks for non-vision models. This covers every image source: pasted/attached images, `read`-tool results, and custom extension-injected messages. (Read-tool images additionally keep the description + image in the stored `tool_result` for kitty inline rendering and `/resume`.)
- Works across all four image-block shapes pi uses — the three provider-transformed formats (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) plus pi-ai's internal `{ type: "image", data, mimeType }` emitted by the `read` tool — detected by shape.
- Descriptions are **cached per image hash** (LRU) and produced **one batched vision call per image set** (dataloader-style: a prompt describing N images does not spin up N describer calls), so the swap is instant by the time the request fires.

No `settings.json` touched. No per-provider glue. Pick a describer once and every text-only model you own can suddenly see.

## Features

- **🎮 Interactive picker** — `/vision-handoff` opens a TUI listing every model, vision-capable ones first (👁), to choose your describer.
- **🖼️ DataLoader-batched descriptions** — the `read` tools are `load()` callers: N parallel reads coalesce into ONE batched vision call (dispatched via `setImmediate` after the poll phase, so reads completing together batch instead of splitting), awaited during the tool-result phase (free time) so the agent's next turn never blocks on the describer. Descriptions are ready before `context` fires, so the swap is a non-blocking cache hit.
- **🧹 Hides pi's "model does not support images" note** — on read results the extension strips pi's `[Current model does not support images…]` note from the text block (it's misleading once the handoff delivers the image's content as text), while keeping the image block for kitty inline rendering and `/resume`.
- **🔌 Provider-agnostic** — uses pi's own model execution machinery (`@earendil-works/pi-ai`'s `complete()`), so it works with any provider/configured model, including custom provider extensions.
- **🧠 Automatic targets** — by default, handoff applies to *every* model that lacks native vision. Opt out with `/vision-handoff auto off`.
- **🗂️ Explicit overrides** — force handoff for specific models (e.g. a weak vision model) with `/vision-handoff add`.
- **⚡ Pre-warmed at paste-enter** — the moment you press enter, `before_agent_start` scans the prompt for pasted clipboard image temp-file paths (pi writes pasted images to `<tmpdir>/pi-clipboard-<uuid>.<ext>` and inserts the path as text), reads them, and kicks off the ONE batched vision call concurrent with the agent's first response — so by the time the agent reads the files, they're already cache hits.
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
  "maxTokens": null,
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
| `maxTokens` | _(unset = model max, clamped to context window)_ | Cap on a single description's output. `null`/unset = use the vision model's declared max output (`model.maxTokens`), clamped so `maxTokens + 8192 <= contextWindow` (a model whose declared max equals its full context window would otherwise be rejected with a 400). Set a number only to cap cost/latency. A truncation is always surfaced via a `[... description truncated …]` marker when the model hits the limit, so a cut-off description is never mistaken for complete. |
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

The extension implements the **Facebook DataLoader pattern** for image
descriptions. The `read` tools are the `load()` callers; a per-image cache
memoizes promises; all `load()` calls in the same execution frame coalesce
into ONE batched vision call, dispatched via `setImmediate` after the poll
phase (so reads completing together batch instead of splitting into N calls).

    → before_agent_start
        • captures this turn's user prompt (shared by every image description)
        • binds turn context (vision model + abort signal) to the loader
        • PRE-WARMS at paste-enter via the loader, two sources coalescing into
          ONE batch:
          1. attached image blocks (event.images) — vision-capable targets
          2. pasted clipboard image FILE PATHS in the prompt text — the common
             non-vision flow: pi writes each pasted image to
             `<tmpdir>/pi-clipboard-<uuid>.<ext>` and inserts the PATH as text
             at the cursor, so on a non-vision model they arrive as path tokens
             in `event.prompt`, NOT as `event.images`. The extension scans the
             prompt for those temp paths (confined to the OS temp dir — a
             prompt can't trick it into reading arbitrary files), reads the
             files synchronously (keeps all load() calls in one batch frame →
             one vision call), and `loadDescription()`s them — so the vision
             call starts the INSTANT you press enter, concurrent with the
             agent's first response, not waiting for it to `read` the files
    → read tool / tool_result   (PRIMARY injection point)
        • pi runs `read` calls in parallel (Promise.all); each read's
          tool_result handler calls `loadDescription(img)` for its image
          blocks and AWAITS the shared batch
        • DataLoader: all `load()` calls in one event-loop poll iteration
          land in ONE batch → ONE `complete()` vision call for the whole read
          set. `enqueuePostPromiseJob` schedules dispatch via `setImmediate`
          (the check phase, which runs AFTER the whole poll phase — not
          `process.nextTick`, which would drain between the reads' I/O
          callbacks and split them into N single-image calls)
        • awaits the shared batch — runs the describer during the tool-result
          phase (free time: the agent is just waiting for tool results), so
          the batch is COMPLETE before `context` fires → `context` is a
          non-blocking cache hit, not a cold miss on the critical path
        • does NOT mutate the result: the image stays in storage for kitty
          inline rendering and `/resume`; the image→text swap happens in the
          `context` hook (on the cloned LLM-bound payload only)
    → context   (FALLBACK + swap — fires before each LLM call)
        • catches image blocks that didn't go through `read`'s tool_result —
          user-attached images, custom extension-injected messages — and
          swaps read images too (cache hits from the tool_result priming)
        • `loadDescription()` is a cache hit (warmed above) or queues into the
          loader's current batch; swaps images for text in the cloned LLM-bound
          payload (`emitContext` does a `structuredClone`), leaving storage
          intact for kitty inline rendering and `/resume`

Because the describer runs during the **tool-result phase** (before the
agent's next turn), not during the `context` transform (the critical path
right before the LLM call), the agent gets described text immediately when
its turn starts — it never waits on the describer inline.

### Batching: the DataLoader

A single frame's image set is described with **one** vision-model call, not
one per image. `loadDescription(img)` is synchronous: on a cache miss it
pushes the image's key (hash) into the current batch and returns a memoized
promise; on a cache hit it returns the existing (in-flight or resolved)
promise. `enqueuePostPromiseJob` schedules dispatch via `setImmediate` (the
check phase, after the whole poll phase AND after the microtask queue
drains), so every `load()` caller — whether from sync code, a `.then`
cascade, or a separate I/O callback in the same poll iteration — registers
its key before the single vision call fires. This is why N parallel reads
coalesce into one call rather than splitting into N single-image calls:
`setImmediate` defers past the entire poll phase, whereas DataLoader's
classic `process.nextTick` would drain between the reads' I/O callbacks and
fire a dispatch after the first read but before the second. The batched call
sends every uncached image in a single user message with per-image
`<<<IMAGE k>>> … <<<END>>>` delimiters; the response is parsed back into
per-image descriptions (keyed by `sha256(mime + base64)` in the per-image
cache). If the vision model ignores the delimiter format and the batched
response can't be split, each unparsed image falls back to its own
single-image `complete()` call **in parallel** (no delimiters to cooperate
with) — descriptions still arrive together. Only when a per-image call
itself genuinely fails (auth, timeout, empty) does that image degrade to
`[Image: description unavailable]`; one bad image never voids the rest.

Because pi runs parallel `read` tool calls via `Promise.all` and fires
each read's `tool_result` event as that read's I/O completes (poll phase) —
via `agent.afterToolCall` → `finalizeExecutedToolCall` — the loader's
`setImmediate` dispatch defers to the check phase AFTER the whole poll
iteration, so reads completing together share ONE batch → ONE vision call,
all resolving together. Reads completing in separate poll iterations get
separate calls, but always in parallel, never sequential. (The per-image
cache also dedupes a duplicate `load()` of the same image in one frame:
dispatch resolves every callback by hash, so a second load whose first cache
entry was evicted mid-frame still resolves — it never hangs.)

**Failures are never cached.**

**Failures are never cached.** A genuine describer failure returns
`[Image: description unavailable]` for that turn but is NOT written to the
per-image cache, so the next turn re-attempts (and surfaces the real error
via a `ctx.ui.notify` warning instead of serving a stuck placeholder). This
avoids a transient failure poisoning the cache for the rest of the session.

**Per-image-set warning.** The `context` hook fires before every LLM turn,
and describer failures aren't cached — so without dedup the same broken
vision model would re-warn every turn. Each distinct image is warned about
at most once per session (tracked by image hash, reset on `/resume`/new
session); subsequent turns for the same failing image degrade silently to
the placeholder instead of spamming.

**Timeout scales with batch size.** The describer generates an exhaustive
multi-paragraph description per image, and the call is batched (N images →
1 request). The timeout is `DESCRIBE_TIMEOUT_MS` (120s) plus a per-image
budget (45s × (imageCount − 1)), so a 5-image batch isn't held to the same
wall-clock budget as one image. A timeout surfaces as `describer timed out
after <N>s` rather than a misleading `stopReason "aborted"`.

**No silent truncation.** The describer prompt says "be exhaustive", so the
default `maxTokens` is **unset** — the describer uses the vision model's
declared max output (`model.maxTokens`) as the cap, rather than relying on a
provider's small omitted-default. That value is clamped so `maxTokens + 8192
<= contextWindow`: a model whose declared `maxTokens` equals its full
`contextWindow` (e.g. a custom provider that sets both to the same number)
would otherwise be rejected by the provider with a 400 (you can't request
output tokens equal to the entire context window when you also have input),
so the clamp subtracts a small input reserve. If the model still hits a token
limit (a cap you set, or the provider's hard output maximum), `stopReason`
becomes `"length"`; the describer appends a visible
`[... description truncated …]` marker to the (still useful) partial text
rather than letting a cut-off description pass as complete. For a batched call
the marker lands on the last image being emitted when the cap hit (the one cut
off mid-stream); earlier sections had `<<<END>>>` delimiters and are complete.

**Aborts propagate.** The `context` hook runs in the foreground (pi awaits
it before the LLM call), so a slow describer would also make aborting a turn
slow — pi has to wait for the transform to return. The hook therefore wires
the turn's abort signal (`ctx.signal`, the active run's `AbortController`)
into the describer's `complete()` call, so a user cancel kills the in-flight
vision request immediately. A deliberate abort is not warned about (it's
not a vision-model failure) and the LLM-bound payload is left untouched since
the turn is being torn down anyway.

### Image-block formats handled

| Hook | Image block shape | Handling |
|------|-------------------|----------|
| `context` (all messages) | `{ type: "image", data, mimeType }` (pi-ai internal) | undescribed → replaced with description text; already-described → dropped |
| `context` (all messages) | `{ type: "image_url", image_url: { url: "data:…" } }` (OpenAI Chat Completions) | detected by shape → replaced with `{ type: "text", text }` |
| `context` (all messages) | `{ type: "input_image", image_url: "data:…" }` (OpenAI Responses) | detected by shape → replaced with `{ type: "input_text", text }` |
| `context` (all messages) | `{ type: "image", source: { type: "base64", media_type, data } }` (Anthropic Messages) | detected by shape → replaced with `{ type: "text", text }` |

The describer call itself goes through pi's normal model machinery (`complete()`),
**not** the agent event loop — so it never re-triggers `context` (no recursion).
The `read` tool result keeps its image block untouched (kitty inline + `/resume`);
only the `context`-cloned LLM-bound payload has images swapped for text.

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
  imageHash: string,            // representative (first) member of the batch; sha256(mime + base64), first 32 hex chars
  imageHashes?: string[],        // present only for batched calls (length > 1): every image the call covered
  model: string, provider: string,
  responseModel?: string, responseId?: string,
  usage: Usage,                 // { input, output, cacheRead, cacheWrite, totalTokens, cost }
  // Present ONLY when Neuralwatt SSE energy comments were captured:
  energyJoules?: number, costUsd?: number,
  energyRaw?: object, mcrSessionRaw?: object, costRaw?: object,
}
```

One record is emitted per **real** describer call (a batched call describing several images still emits a single record, with `imageHashes` listing every member so consumers can attribute the call's tokens/energy per image without double-counting). Because `before_agent_start` fires a batched describe fire-and-forget, the fetch interception is **refcounted** (installed only while ≥1 describe is in flight) and uses `AsyncLocalStorage` to route each teed response body to the describe call that issued it — so concurrent describes each attribute their own energy correctly without clobbering `globalThis.fetch`. When the vision model is a Neuralwatt model, `pi-neuralwatt-provider`'s own `streamNeuralwatt` tee nests on top and restores back to this interceptor; both tees read the same comment lines independently (the accepted duplication for easy filtering).

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **pi-vision-handoff** (this) | Provider-agnostic; pick any describer; automatic for text-only models; cached; batched (one call per image set); survives across providers | Adds one extra model call per image set per turn |
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
├── vision-handoff.ts            # Wiring layer: pi hooks, /vision-handoff command
├── src/
│   ├── index.ts                  # Config schema, read/write, image-block helpers, batching (barrel)
│   ├── dataloader.ts              # DescriptionLoader — DataLoader-batched descriptions (Disposable)
│   ├── describer.ts              # Vision describer calls (runBatch / describeSingle) with `using` resource guards
│   ├── image.ts                  # Image hashing, MIME sniffing, clipboard-path reading
│   ├── dispose.ts                 # `Disposable` guard factories for `using` (fetch interceptor, timer, abort wire)
│   ├── usage.ts                  # Describer usage + Neuralwatt energy capture, fetch interceptor
│   └── vision-model-selector.ts  # Interactive picker TUI component
├── __tests__/unit/
│   ├── config-dir.test.ts        # Ensures getAgentDir() usage
│   ├── usage.test.ts             # Energy parsing, usage records, concurrency-safe fetch routing
│   ├── vision-handoff.test.ts    # Config, refs, image-block extraction, insertion, truncation, round-trip
│   ├── dataloader.test.ts        # Batch coalescing, memoization, failure eviction, Disposable reset
│   ├── describer.test.ts        # stopReason handling (length → truncation marker, aborted/error)
│   ├── image.test.ts             # MIME sniffing, clipboard-path confinement, file reading
│   └── dispose.test.ts           # `using` guards: fetch refcount, timeout, abort-wire propagation
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
