import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  sniffImageMime,
  imageDimensions,
  willBeResized,
  resolvePrewarmImage,
  readImageBuffer,
  findClipboardImagePaths,
  imageHash,
} from "../../src/image.js";

// ── header crafters (minimal valid headers with known dimensions) ──────────

function pngHeader(width: number, height: number): Buffer {
  const b = Buffer.alloc(24);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(b, 0); // signature
  b.writeUInt32BE(13, 8); // IHDR length
  b.write("IHDR", 12, "ascii");
  b.writeUInt32BE(width, 16);
  b.writeUInt32BE(height, 20);
  return b;
}

/** A PNG with an acTL chunk before IDAT (animated). */
function animatedPngHeader(): Buffer {
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(25);
  ihdr.writeUInt32BE(13, 0);
  ihdr.write("IHDR", 4, "ascii");
  ihdr.writeUInt32BE(100, 8);
  ihdr.writeUInt32BE(100, 12);
  // actl chunk: length 8, "acTL", 8 bytes data
  const actl = Buffer.alloc(20);
  actl.writeUInt32BE(8, 0);
  actl.write("acTL", 4, "ascii");
  return Buffer.concat([sig, ihdr, actl]);
}

function gifHeader(width: number, height: number): Buffer {
  const b = Buffer.alloc(13);
  b.write("GIF89a", 0, "ascii");
  b.writeUInt16LE(width, 6);
  b.writeUInt16LE(height, 8);
  return b;
}

/** JPEG with an optional APP0 segment then an SOF0 carrying width/height. */
function jpegHeader(width: number, height: number, withApp0 = true): Buffer {
  const soi = Buffer.from([0xff, 0xd8]);
  const parts: Buffer[] = [soi];
  if (withApp0) {
    const app0 = Buffer.alloc(18);
    app0.writeUInt8(0xff, 0);
    app0.writeUInt8(0xe0, 1);
    app0.writeUInt16BE(16, 2); // length includes the 2 length bytes
    app0.write("JFIF", 4, "ascii");
    parts.push(app0);
  }
  // SOF0: marker, length=11 (1 precision + 2 h + 2 w + 1 ncomp + 3*ncomp), precision 8
  const sof = Buffer.alloc(13);
  sof.writeUInt8(0xff, 0);
  sof.writeUInt8(0xc0, 1);
  sof.writeUInt16BE(11, 2);
  sof.writeUInt8(8, 4); // precision
  sof.writeUInt16BE(height, 5);
  sof.writeUInt16BE(width, 7);
  sof.writeUInt8(1, 9); // 1 component
  parts.push(sof);
  return Buffer.concat(parts);
}

function webpVp8xHeader(width: number, height: number): Buffer {
  const b = Buffer.alloc(30);
  b.write("RIFF", 0, "ascii");
  b.writeUInt32LE(0, 4);
  b.write("WEBP", 8, "ascii");
  b.write("VP8X", 12, "ascii");
  b.writeUInt32LE(10, 16);
  b.writeUInt8(0, 20); // flags
  b.writeUInt32LE(0, 21); // reserved
  b.writeUInt8((width - 1) & 0xff, 24);
  b.writeUInt8(((width - 1) >> 8) & 0xff, 25);
  b.writeUInt8(((width - 1) >> 16) & 0xff, 26);
  b.writeUInt8((height - 1) & 0xff, 27);
  b.writeUInt8(((height - 1) >> 8) & 0xff, 28);
  b.writeUInt8(((height - 1) >> 16) & 0xff, 29);
  return b;
}

function webpVp8lHeader(width: number, height: number): Buffer {
  const b = Buffer.alloc(30);
  b.write("RIFF", 0, "ascii");
  b.writeUInt32LE(0, 4);
  b.write("WEBP", 8, "ascii");
  b.write("VP8L", 12, "ascii");
  b.writeUInt32LE(10, 16);
  b.writeUInt8(0x2f, 20); // signature
  const val = (width - 1) | ((height - 1) << 14);
  b.writeUInt32LE(val, 21);
  return b;
}

function webpVp8Header(width: number, height: number): Buffer {
  const b = Buffer.alloc(30);
  b.write("RIFF", 0, "ascii");
  b.writeUInt32LE(0, 4);
  b.write("WEBP", 8, "ascii");
  b.write("VP8 ", 12, "ascii");
  b.writeUInt32LE(10, 16);
  b.writeUInt8(0x9d, 23); // start code
  b.writeUInt8(0x01, 24);
  b.writeUInt8(0x2a, 25);
  b.writeUInt16LE(width & 0x3fff, 26);
  b.writeUInt16LE(height & 0x3fff, 28);
  return b;
}

// ── sniffImageMime ───────────────────────────────────────────────────────────

describe("sniffImageMime", () => {
  it("detects PNG", () => {
    expect(sniffImageMime(pngHeader(10, 10))).toBe("image/png");
  });

  it("detects JPEG", () => {
    const buf = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0]);
    expect(sniffImageMime(buf)).toBe("image/jpeg");
  });

  it("detects GIF", () => {
    expect(sniffImageMime(gifHeader(10, 10))).toBe("image/gif");
  });

  it("detects WEBP", () => {
    expect(sniffImageMime(webpVp8xHeader(10, 10))).toBe("image/webp");
  });

  it("rejects animated PNG (aligned with pi — pi reads it as text)", () => {
    expect(sniffImageMime(animatedPngHeader())).toBeNull();
  });

  it("rejects the animated-JPEG marker 0xf7", () => {
    expect(sniffImageMime(Buffer.from([0xff, 0xd8, 0xff, 0xf7, 0, 0]))).toBeNull();
  });

  it("returns null for unsupported / too-short buffers", () => {
    expect(sniffImageMime(Buffer.from([1, 2, 3]))).toBeNull();
    expect(sniffImageMime(Buffer.alloc(0))).toBeNull();
    // PNG signature but no IHDR chunk → not a (static) PNG
    const bad = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(sniffImageMime(bad)).toBeNull();
  });
});

// ── imageDimensions ──────────────────────────────────────────────────────────

describe("imageDimensions", () => {
  it("parses PNG width/height", () => {
    expect(imageDimensions(pngHeader(2500, 1500), "image/png")).toEqual({ width: 2500, height: 1500 });
    expect(imageDimensions(pngHeader(16, 8), "image/png")).toEqual({ width: 16, height: 8 });
  });

  it("parses GIF width/height", () => {
    expect(imageDimensions(gifHeader(2500, 1500), "image/gif")).toEqual({ width: 2500, height: 1500 });
  });

  it("parses JPEG width/height, skipping the APP0 segment", () => {
    expect(imageDimensions(jpegHeader(2500, 1500, true), "image/jpeg")).toEqual({ width: 2500, height: 1500 });
    expect(imageDimensions(jpegHeader(16, 8, false), "image/jpeg")).toEqual({ width: 16, height: 8 });
  });

  it("parses WEBP VP8X (extended)", () => {
    expect(imageDimensions(webpVp8xHeader(2500, 1500), "image/webp")).toEqual({ width: 2500, height: 1500 });
  });

  it("parses WEBP VP8L (lossless)", () => {
    expect(imageDimensions(webpVp8lHeader(16, 8), "image/webp")).toEqual({ width: 16, height: 8 });
    expect(imageDimensions(webpVp8lHeader(2500, 1500), "image/webp")).toEqual({ width: 2500, height: 1500 });
  });

  it("parses WEBP VP8 (lossy)", () => {
    expect(imageDimensions(webpVp8Header(16, 8), "image/webp")).toEqual({ width: 16, height: 8 });
    expect(imageDimensions(webpVp8Header(2500, 1500), "image/webp")).toEqual({ width: 2500, height: 1500 });
  });

  it("returns null for an unsupported mime", () => {
    expect(imageDimensions(Buffer.alloc(30), "image/bmp")).toBeNull();
  });

  it("returns null for a too-short buffer", () => {
    expect(imageDimensions(Buffer.alloc(5), "image/png")).toBeNull();
  });
});

// ── willBeResized ────────────────────────────────────────────────────────────

describe("willBeResized", () => {
  it("returns false for an image within both dimension and size limits", () => {
    expect(willBeResized(pngHeader(2000, 2000), "image/png")).toBe(false);
    expect(willBeResized(pngHeader(1000, 500), "image/png")).toBe(false);
  });

  it("returns true when width exceeds 2000", () => {
    expect(willBeResized(pngHeader(2001, 500), "image/png")).toBe(true);
  });

  it("returns true when height exceeds 2000", () => {
    expect(willBeResized(pngHeader(500, 2001), "image/png")).toBe(true);
  });

  it("returns true when the base64 payload reaches 4.5MB even if dimensions fit", () => {
    // 3_600_000 raw bytes → ceil(/3)*4 = 4_800_000 base64 bytes > 4.5MB (4_718_592)
    const buf = Buffer.alloc(3_600_000);
    pngHeader(1000, 500).copy(buf);
    expect(willBeResized(buf, "image/png")).toBe(true);
  });

  it("returns true (default-to-resize) when dimensions can't be parsed", () => {
    expect(willBeResized(Buffer.alloc(30), "image/bmp")).toBe(true);
  });

  it("is orientation-invariant: swapping dimensions does not change the result", () => {
    // EXIF rotation swaps w/h; max(w,h) > 2000 is the deciding factor either way.
    expect(willBeResized(pngHeader(2500, 1000), "image/png")).toBe(true);
    expect(willBeResized(pngHeader(1000, 2500), "image/png")).toBe(true);
    expect(willBeResized(pngHeader(1000, 500), "image/png")).toBe(false);
    expect(willBeResized(pngHeader(500, 1000), "image/png")).toBe(false);
  });
});

// ── resolvePrewarmImage ──────────────────────────────────────────────────────

describe("resolvePrewarmImage", () => {
  it("returns the raw base64 (no resize) when pi wouldn't resize", async () => {
    const buf = pngHeader(1000, 500);
    const resize = vi.fn();
    const img = await resolvePrewarmImage(buf, "image/png", resize);
    expect(img).toEqual({ data: buf.toString("base64"), mimeType: "image/png" });
    expect(resize).not.toHaveBeenCalled();
  });

  it("runs the injected resize and returns its output when pi would resize", async () => {
    const buf = pngHeader(2500, 1500);
    const resized = { data: "resized-base64", mimeType: "image/jpeg" };
    const resize = vi.fn().mockResolvedValue(resized);
    const img = await resolvePrewarmImage(buf, "image/png", resize);
    expect(resize).toHaveBeenCalledWith(buf, "image/png");
    expect(img).toEqual(resized);
  });

  it("returns null (skip pre-warm) when resize fails (pi emits no image block)", async () => {
    const buf = pngHeader(2500, 1500);
    const resize = vi.fn().mockResolvedValue(null);
    const img = await resolvePrewarmImage(buf, "image/png", resize);
    expect(img).toBeNull();
  });

  it("propagates a resize rejection as a skipped pre-warm via the caller's catch", async () => {
    const buf = pngHeader(2500, 1500);
    const resize = vi.fn().mockRejectedValue(new Error("worker boom"));
    await expect(resolvePrewarmImage(buf, "image/png", resize)).rejects.toThrow("worker boom");
  });
});

// ── imageHash ────────────────────────────────────────────────────────────────

describe("imageHash", () => {
  it("is stable and differs by mime or data", () => {
    const a = imageHash("image/png", "AAA");
    expect(imageHash("image/png", "AAA")).toBe(a);
    expect(imageHash("image/jpeg", "AAA")).not.toBe(a);
    expect(imageHash("image/png", "BBB")).not.toBe(a);
  });
});

// ── readImageBuffer ──────────────────────────────────────────────────────────

describe("readImageBuffer", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "pi-vh-image-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  it("reads a PNG file into its raw buffer + sniffed mime", () => {
    const png = pngHeader(10, 10);
    const path = join(dir, "img.png");
    writeFileSync(path, png);
    const out = readImageBuffer(path);
    expect(out).toEqual({ buf: png, mimeType: "image/png" });
  });

  it("returns null for a non-image file (sniff fails)", () => {
    const path = join(dir, "not.png");
    writeFileSync(path, "plain text, not an image");
    expect(readImageBuffer(path)).toBeNull();
  });

  it("returns null for an animated PNG (aligned with pi)", () => {
    const path = join(dir, "anim.png");
    writeFileSync(path, animatedPngHeader());
    expect(readImageBuffer(path)).toBeNull();
  });

  it("returns null for a missing file", () => {
    expect(readImageBuffer(join(dir, "nope.png"))).toBeNull();
  });
});

// ── findClipboardImagePaths ──────────────────────────────────────────────────

describe("findClipboardImagePaths", () => {
  const tmp = tmpdir();

  it("collects pi-clipboard temp paths inside the OS temp dir", () => {
    const prompt = `Look at ${tmp}/pi-clipboard-abc-1.png and ${tmp}/pi-clipboard-def-2.jpg please`;
    expect(findClipboardImagePaths(prompt).sort()).toEqual(
      [`${tmp}/pi-clipboard-abc-1.png`, `${tmp}/pi-clipboard-def-2.jpg`].sort(),
    );
  });

  it("rejects a clipboard-shaped path outside the temp dir (confinement)", () => {
    expect(findClipboardImagePaths(`evil: /etc/pi-clipboard-evil.png`)).toEqual([]);
  });

  it("resolves a relative clipboard path against the temp dir", () => {
    expect(findClipboardImagePaths(`see pi-clipboard-rel-3.gif`)).toEqual([
      join(tmp, "pi-clipboard-rel-3.gif"),
    ]);
  });

  it("dedupes repeated paths", () => {
    expect(findClipboardImagePaths(`${tmp}/pi-clipboard-dup.png ${tmp}/pi-clipboard-dup.png`)).toEqual([
      `${tmp}/pi-clipboard-dup.png`,
    ]);
  });

  it("supports webp and jpeg extensions", () => {
    expect(findClipboardImagePaths(`${tmp}/pi-clipboard-a.webp ${tmp}/pi-clipboard-b.jpeg`).sort()).toEqual(
      [`${tmp}/pi-clipboard-a.webp`, `${tmp}/pi-clipboard-b.jpeg`].sort(),
    );
  });
});
