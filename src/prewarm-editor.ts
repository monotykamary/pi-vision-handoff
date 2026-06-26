/**
 * Paste-time prewarm editor wrapper.
 *
 * pi has no "image pasted into the prompt" event — the extension event bus
 * only surfaces input at submit time (`input` / `before_agent_start`). The
 * earliest observable signal that a clipboard image has landed in the prompt
 * is the editor's own `onChange(text)`, which fires when pi's
 * `handleClipboardImagePaste` inserts the temp-file path
 * (`<tmpdir>/pi-clipboard-<uuid>.<ext>`) via `insertTextAtCursor` — still
 * pre-submit. This wrapper installs a `CustomEditor` that observes `onChange`
 * to prewarm the describer for newly-pasted clipboard image paths the instant
 * they appear, so the vision call starts before the user hits enter (true
 * paste-time) instead of at `before_agent_start` (submit-time).
 *
 * It does NOT override `handleInput`, so every keybinding and pi's clipboard
 * paste flow (`ctrl+v` → `onPasteImage` → `handleClipboardImagePaste`) is
 * untouched. We only observe the result via `onChange`.
 *
 * Why chain `onChange` via an accessor: pi's `setCustomEditorComponent` does
 * `newEditor.onChange = this.defaultEditor.onChange` after construction,
 * clobbering any `onChange` set in the constructor — and that default
 * `onChange` tracks bash-mode for the editor border, so it must keep running.
 * The accessor captures pi's assignment and runs it alongside our observer.
 * (The `Editor` base never reassigns `onChange` itself, so the captured
 * reference is stable for the editor's lifetime.)
 *
 * Opt-in via `VisionHandoffConfig.prewarmPastedImages`. When off (or the
 * active model isn't a handoff target), the observer short-circuits before
 * the path regex runs, so overhead is one boolean check per text change.
 */

import { CustomEditor, type KeybindingsManager, type ModelRegistry } from "@earendil-works/pi-coding-agent";
import type { EditorTheme, TUI } from "@earendil-works/pi-tui";
import { diffPrewarmPaths } from "./image.js";

/** Dependencies the editor can't own itself (held by the engine, captured at
 *  session start). `modelRegistry` is fixed for the editor's (session)
 *  lifetime; `shouldPrewarm` and `prewarmPath` read live config/model state so
 *  toggling the opt-in or switching models takes effect without reinstalling. */
export interface PrewarmEditorDeps {
  modelRegistry: ModelRegistry;
  /** Live gate: paste-time prewarm runs only when this returns true. */
  shouldPrewarm: () => boolean;
  /** Prewarm one clipboard image path (read + resize-match + loadDescription). */
  prewarmPath: (path: string, modelRegistry: ModelRegistry) => void;
}

/** A `CustomEditor` that observes `onChange` to prewarm pasted clipboard image
 *  paths at paste-time. See the module doc for the `onChange` chaining. */
export class PrewarmEditor extends CustomEditor {
  private readonly knownPaths = new Set<string>();
  private piOnChange: ((text: string) => void) | undefined;
  private readonly shouldPrewarm: () => boolean;
  private readonly prewarmPath: (path: string, modelRegistry: ModelRegistry) => void;
  private readonly modelRegistry: ModelRegistry;

  constructor(tui: TUI, theme: EditorTheme, keybindings: KeybindingsManager, deps: PrewarmEditorDeps) {
    super(tui, theme, keybindings);
    this.shouldPrewarm = deps.shouldPrewarm;
    this.prewarmPath = deps.prewarmPath;
    this.modelRegistry = deps.modelRegistry;

    // Stable wrapper: calls pi's onChange (bash-mode border tracking) then our
    // observer. Returned on every read so the Editor's `this.onChange(...)` calls
    // all hit the same function.
    const wrapper = (text: string): void => {
      this.piOnChange?.(text);
      this.handleTextChange(text);
    };
    // pi assigns defaultEditor.onChange to this property after construction;
    // capture it via the setter, expose the wrapper via the getter.
    Object.defineProperty(this, "onChange", {
      configurable: true,
      enumerable: true,
      get: () => wrapper,
      set: (fn: ((text: string) => void) | undefined) => {
        this.piOnChange = fn;
      },
    });
  }

  /** Scan the editor text for clipboard image paths not yet seen this prompt
   *  and prewarm each. Clears the known set when the editor empties (after
   *  submit) so a later paste (fresh uuid) re-prewarms and the set can't grow
   *  unbounded. */
  private handleTextChange(text: string): void {
    if (!this.shouldPrewarm()) return;
    const newPaths = diffPrewarmPaths(text, this.knownPaths);
    for (const p of newPaths) {
      this.knownPaths.add(p);
      this.prewarmPath(p, this.modelRegistry);
    }
    if (text.length === 0) this.knownPaths.clear();
  }
}
