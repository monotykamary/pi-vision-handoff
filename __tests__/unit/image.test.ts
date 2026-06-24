import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { sniffImageMime, readImageFile, findClipboardImagePaths, imageHash } from "../../src/image.js";

describe("sniffImageMime", () => {
  it("detects PNG from magic bytes", () => {
    const buf = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0]);
    expect(sniffImageMime(buf)).toBe("image/png");
  });

  it("detects JPEG", () => {
    const buf = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0]);
    expect(sniffImageMime(buf)).toBe("image/jpeg");
  });

  it("detects GIF", () => {
    expect(sniffImageMime(Buffer.from("GIF89a", "ascii"))).toBe("image/gif");
  });

  it("detects WEBP (RIFF/WEBP)", () => {
    const buf = Buffer.alloc(14);
    buf.write("RIFF", 0, "ascii");
    buf.write("WEBP", 8, "ascii");
    expect(sniffImageMime(buf)).toBe("image/webp");
  });

  it("returns null for an animated JPEG (0xf7 marker)", () => {
    expect(sniffImageMime(Buffer.from([0xff, 0xd8, 0xff, 0xf7, 0, 0]))).toBeNull();
  });

  it("returns null for unsupported / too-short buffers", () => {
    expect(sniffImageMime(Buffer.from([1, 2, 3]))).toBeNull();
    expect(sniffImageMime(Buffer.alloc(0))).toBeNull();
  });
});

describe("imageHash", () => {
  it("is stable and differs by mime or data", () => {
    const a = imageHash("image/png", "AAA");
    expect(imageHash("image/png", "AAA")).toBe(a);
    expect(imageHash("image/jpeg", "AAA")).not.toBe(a);
    expect(imageHash("image/png", "BBB")).not.toBe(a);
  });
});

describe("readImageFile", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "pi-vh-image-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  it("reads a PNG file into base64 + sniffed mime", () => {
    const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0]);
    const path = join(dir, "img.png");
    writeFileSync(path, png);
    const img = readImageFile(path);
    expect(img).toEqual({ mimeType: "image/png", data: png.toString("base64") });
  });

  it("returns null for a non-image file", () => {
    const path = join(dir, "not.png");
    writeFileSync(path, "plain text, not an image");
    expect(readImageFile(path)).toBeNull();
  });

  it("returns null for a missing file", () => {
    expect(readImageFile(join(dir, "nope.png"))).toBeNull();
  });
});

describe("findClipboardImagePaths", () => {
  const tmp = tmpdir();

  it("collects pi-clipboard temp paths inside the OS temp dir", () => {
    const prompt = `Look at ${tmp}/pi-clipboard-abc-1.png and ${tmp}/pi-clipboard-def-2.jpg please`;
    expect(findClipboardImagePaths(prompt).sort()).toEqual(
      [`${tmp}/pi-clipboard-abc-1.png`, `${tmp}/pi-clipboard-def-2.jpg`].sort(),
    );
  });

  it("rejects a clipboard-shaped path outside the temp dir (confinement)", () => {
    const prompt = `evil: /etc/pi-clipboard-evil.png`;
    expect(findClipboardImagePaths(prompt)).toEqual([]);
  });

  it("resolves a relative clipboard path against the temp dir", () => {
    const prompt = `see pi-clipboard-rel-3.gif`;
    expect(findClipboardImagePaths(prompt)).toEqual([join(tmp, "pi-clipboard-rel-3.gif")]);
  });

  it("dedupes repeated paths", () => {
    const prompt = `${tmp}/pi-clipboard-dup.png ${tmp}/pi-clipboard-dup.png`;
    expect(findClipboardImagePaths(prompt)).toEqual([`${tmp}/pi-clipboard-dup.png`]);
  });

  it("supports webp and jpeg extensions", () => {
    const prompt = `${tmp}/pi-clipboard-a.webp ${tmp}/pi-clipboard-b.jpeg`;
    expect(findClipboardImagePaths(prompt).sort()).toEqual(
      [`${tmp}/pi-clipboard-a.webp`, `${tmp}/pi-clipboard-b.jpeg`].sort(),
    );
  });
});
