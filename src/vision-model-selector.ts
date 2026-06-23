/**
 * VisionModelSelectorComponent — an interactive TUI for choosing which model
 * describes images during vision handoff.
 *
 * Uses the same patterns as pi's built-in selectors and pi-hide-providers:
 * - Lists all models, vision-capable ones first (👁 badge)
 * - A leading "None" row clears the configured vision model
 * - Search/filter via Input component
 * - Enter or Ctrl+S confirms the highlighted model and saves
 * - Esc / Ctrl+C cancels
 * - The currently configured vision model is marked ✓
 */

import {
  Container,
  type Component,
  fuzzyFilter,
  getKeybindings,
  Input,
  Key,
  matchesKey,
  Spacer,
  Text,
} from "@earendil-works/pi-tui";
import type { Theme } from "@earendil-works/pi-coding-agent";
import { DynamicBorder, keyText } from "@earendil-works/pi-coding-agent";
import { formatModelRef, isVisionModel } from "./index.js";

interface DisplayItem {
  /** "provider/id", or null for the synthetic "None" row. */
  ref: string | null;
  provider: string;
  modelId: string;
  modelName: string;
  vision: boolean;
  none?: boolean;
}

export interface VisionModelSelectorResult {
  /** The selected "provider/id", or null if the user picked "None" / cancelled. */
  ref: string | null;
  /** True if the user cancelled (esc) — config should not change. */
  cancelled: boolean;
}

export class VisionModelSelectorComponent implements Component {
  private theme: Theme;
  private done: (result: VisionModelSelectorResult) => void;

  private allItems: DisplayItem[];
  private filteredItems: DisplayItem[];
  private selectedIndex = 0;
  private readonly maxVisible = 10;
  private searchInput: Input;
  private listContainer: Container;
  private footerText: Text;

  private currentRef: string | null;

  private _focused = false;
  get focused(): boolean {
    return this._focused;
  }
  set focused(value: boolean) {
    this._focused = value;
    this.searchInput.focused = value;
  }

  constructor(
    theme: Theme,
    allModels: Array<{ provider: string; id: string; name: string; input?: ("text" | "image")[] }>,
    currentRef: string | null,
    done: (result: VisionModelSelectorResult) => void,
  ) {
    this.theme = theme;
    this.done = done;
    this.currentRef = currentRef;
    this.allItems = this.buildItems(allModels);
    this.filteredItems = this.allItems;

    const startIdx = this.allItems.findIndex((i) => i.ref === currentRef);
    this.selectedIndex = startIdx >= 0 ? startIdx : 0;

    this.searchInput = new Input();
    this.listContainer = new Container();
    this.footerText = new Text(this.getFooterText(), 0, 0);

    this.searchInput.onSubmit = () => {
      const item = this.filteredItems[this.selectedIndex];
      if (item) this.confirm(item);
    };

    this.updateList();
  }

  render(width: number): string[] {
    const lines: string[] = [];
    lines.push(...new DynamicBorder((s) => this.theme.fg("accent", s)).render(width));
    lines.push("");
    lines.push(this.theme.fg("accent", this.theme.bold("Vision Handoff")));
    lines.push(
      this.theme.fg("muted", "Pick a vision-capable model to describe images for text-only models."),
    );
    lines.push("");
    lines.push(...this.searchInput.render(width));
    lines.push("");
    lines.push(...this.listContainer.render(width));
    lines.push("");
    lines.push(...this.footerText.render(width));
    lines.push(...new DynamicBorder((s) => this.theme.fg("accent", s)).render(width));
    return lines;
  }

  handleInput(data: string): void {
    const kb = getKeybindings();

    if (kb.matches(data, "tui.select.up")) {
      if (this.filteredItems.length === 0) return;
      this.selectedIndex =
        this.selectedIndex === 0
          ? this.filteredItems.length - 1
          : this.selectedIndex - 1;
      this.updateList();
      return;
    }

    if (kb.matches(data, "tui.select.down")) {
      if (this.filteredItems.length === 0) return;
      this.selectedIndex =
        this.selectedIndex === this.filteredItems.length - 1
          ? 0
          : this.selectedIndex + 1;
      this.updateList();
      return;
    }

    if (kb.matches(data, "tui.select.confirm")) {
      const item = this.filteredItems[this.selectedIndex];
      if (item) this.confirm(item);
      return;
    }

    if (matchesKey(data, Key.ctrl("s"))) {
      const item = this.filteredItems[this.selectedIndex];
      if (item) this.confirm(item);
      return;
    }

    if (matchesKey(data, Key.escape)) {
      this.finish(true);
      return;
    }

    if (matchesKey(data, Key.ctrl("c"))) {
      if (this.searchInput.getValue()) {
        this.searchInput.setValue("");
        this.refresh();
      } else {
        this.finish(true);
      }
      return;
    }

    this.searchInput.handleInput(data);
    this.refresh();
  }

  invalidate(): void {
    this.searchInput.invalidate();
    this.listContainer.invalidate();
    this.footerText.invalidate();
  }

  // Internal helpers

  private buildItems(
    allModels: Array<{ provider: string; id: string; name: string; input?: ("text" | "image")[] }>,
  ): DisplayItem[] {
    const items: DisplayItem[] = [
      {
        ref: null,
        provider: "",
        modelId: "none",
        modelName: "None — disable vision handoff",
        vision: false,
        none: true,
      },
    ];

    const make = (m: {
      provider: string;
      id: string;
      name: string;
      input?: ("text" | "image")[];
    }): DisplayItem => ({
      ref: formatModelRef(m.provider, m.id),
      provider: m.provider,
      modelId: m.id,
      modelName: m.name || m.id,
      vision: isVisionModel(m),
    });

    // Vision-capable first (registry order), then the rest (registry order).
    const visionModels = allModels.filter((m) => isVisionModel(m)).map(make);
    const textModels = allModels.filter((m) => !isVisionModel(m)).map(make);
    return [...items, ...visionModels, ...textModels];
  }

  private getFooterText(): string {
    const totalCount = this.allItems.length - 1; // exclude the None row
    const visionCount = this.allItems.filter((i) => i.vision).length;

    const current = this.currentRef
      ? `current: ${this.currentRef}`
      : "current: none";

    const parts: string[] = [
      `${keyText("tui.select.confirm")} select`,
      `ctrl+s done`,
      `esc cancel`,
      this.searchInput.getValue() ? `${this.filteredItems.length - 1} match` : `${totalCount} models · ${visionCount} vision`,
    ];

    return this.theme.fg("dim", `  ${parts.join(" · ")} · ${current} `);
  }

  private refresh(): void {
    const query = this.searchInput.getValue();
    this.filteredItems = query
      ? fuzzyFilter(
          this.allItems,
          query,
          (i) => `${i.provider} ${i.modelId} ${i.ref ?? "none"} ${i.modelName}`,
        )
      : this.allItems;
    this.selectedIndex = Math.min(
      this.selectedIndex,
      Math.max(0, this.filteredItems.length - 1),
    );
    this.updateList();
  }

  private updateList(): void {
    this.listContainer.clear();

    if (this.filteredItems.length === 0) {
      this.listContainer.addChild(
        new Text(this.theme.fg("muted", "  No matching models"), 0, 0),
      );
      this.footerText.setText(this.getFooterText());
      return;
    }

    const startIndex = Math.max(
      0,
      Math.min(
        this.selectedIndex - Math.floor(this.maxVisible / 2),
        this.filteredItems.length - this.maxVisible,
      ),
    );
    const endIndex = Math.min(startIndex + this.maxVisible, this.filteredItems.length);

    for (let i = startIndex; i < endIndex; i++) {
      const item = this.filteredItems[i];
      if (!item) continue;

      const isSelected = i === this.selectedIndex;
      const prefix = isSelected ? this.theme.fg("accent", "→ ") : "  ";

      let label: string;
      if (item.none) {
        label = this.theme.fg("warning", item.modelName);
      } else {
        const labelled = isSelected
          ? this.theme.fg("accent", item.modelId)
          : item.modelId;
        const badge = item.vision ? this.theme.fg("success", " 👁") : this.theme.fg("muted", " ·");
        const providerBadge = this.theme.fg("muted", ` [${item.provider}]`);
        label = `${labelled}${providerBadge}${badge}`;
      }

      const current = item.ref === this.currentRef && item.ref !== null
        ? this.theme.fg("success", " ✓")
        : item.none && this.currentRef === null
          ? this.theme.fg("success", " ✓")
          : "";

      this.listContainer.addChild(new Text(`${prefix}${label}${current}`, 0, 0));
    }

    if (startIndex > 0 || endIndex < this.filteredItems.length) {
      this.listContainer.addChild(
        new Text(
          this.theme.fg("muted", `  (${this.selectedIndex + 1}/${this.filteredItems.length})`),
          0, 0,
        ),
      );
    }

    const selected = this.filteredItems[this.selectedIndex];
    if (selected) {
      this.listContainer.addChild(new Spacer(1));
      if (selected.none) {
        this.listContainer.addChild(
          new Text(this.theme.fg("muted", `  ${selected.modelName}`), 0, 0),
        );
      } else {
        this.listContainer.addChild(
          new Text(this.theme.fg("muted", `  Model Name: ${selected.modelName}`), 0, 0),
        );
        this.listContainer.addChild(
          new Text(
            this.theme.fg(
              "dim",
              `${selected.vision ? "👁 vision-capable — recommended describer" : "no native vision — not a good describer"}`,
            ),
            0, 0,
          ),
        );
      }
    }

    this.footerText.setText(this.getFooterText());
  }

  private confirm(item: DisplayItem): void {
    this.done({ ref: item.ref, cancelled: false });
  }

  private finish(cancelled: boolean): void {
    this.done({ ref: null, cancelled });
  }
}
