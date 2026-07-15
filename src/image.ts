/**
 * Pure image IO utilities: MIME sniffing, dimension parsing, and resolving a
 * pasted clipboard image file to the SAME {@link ExtractedImage} pi's `read`
 * tool will emit — so a pre-warm's cache key matches the later `tool_result`'s
 * key (no wasted vision call).
 *
 * Why this matters: pi's `read` tool runs `resizeImage` (on by default). For a
 * small image (≤2000×2000 AND <4.5MB base64) the resize is a no-op that returns
 * the raw input bytes unchanged — so our raw read matches. For an oversized
 * image pi RE-ENCODES (Photon resize + possible JPEG@80), producing different
 * bytes; pre-warming the raw file would then cache-miss at `tool_result` time
 * and waste a vision call. {@link willBeResized} mirrors pi's threshold so we
 * run the same {@link resolvePrewarmImage} pipeline and match the key in both
 * cases.
 *
 * MIME sniffing is aligned with pi's `detectSupportedImageMimeType` (incl.
 * rejecting animated PNG, which pi treats as a non-image → text) so our sniff
 * agrees with pi on whether a file is an image at all.
 */

import { readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, sep } from "node:path";
import crypto from "node:crypto";
import type { ExtractedImage } from "./index.js";

// pi's resize defaults (see @earendil-works/pi-coding-agent utils/image-resize).
// The `read` tool returns raw input bytes unchanged when the image fits BOTH
// the dimension limit AND the base64-payload size limit; otherwise it resizes.
const RESIZE_MAX_WIDTH = 2000;
const RESIZE_MAX_HEIGHT = 2000;
const RESIZE_MAX_BYTES = 4.5 * 1024 * 1024; // base64 payload size

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** Stable hash of an image's MIME + base64 data, used as the dataloader key. */
export function imageHash(mimeType: string, data: string): string {
  return crypto.createHash("sha256").update(`${mimeType}\x00${data}`).digest("hex").slice(0, 32);
}

function readUint32BE(buf: Buffer, offset: number): number {
  return (
    (buf[offset] ?? 0) * 0x1000000 +
    ((buf[offset + 1] ?? 0) << 16) +
    ((buf[offset + 2] ?? 0) << 8) +
    (buf[offset + 3] ?? 0)
  );
}

function readUint16BE(buf: Buffer, offset: number): number {
  return ((buf[offset] ?? 0) << 8) + (buf[offset + 1] ?? 0);
}

function readUint16LE(buf: Buffer, offset: number): number {
  return (buf[offset] ?? 0) + ((buf[offset + 1] ?? 0) << 8);
}

function startsWithAscii(buf: Buffer, offset: number, text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    if (buf[offset + i] !== text.charCodeAt(i)) return false;
  }
  return true;
}

function isPng(buf: Buffer): boolean {
  // IHDR chunk: length == 13 at offset 8, "IHDR" at offset 12.
  return buf.length >= 16 && readUint32BE(buf, PNG_SIGNATURE.length) === 13 && startsWithAscii(buf, 12, "IHDR");
}

function isAnimatedPng(buf: Buffer): boolean {
  // Scan chunks for "acTL" (animation control) before "IDAT". Matches pi's
  // detectSupportedImageMimeType so an animated PNG is treated as a non-image
  // (pi reads it as text) and we skip pre-warming it.
  let offset = PNG_SIGNATURE.length;
  while (offset + 8 <= buf.length) {
    const chunkLength = readUint32BE(buf, offset);
    const typeOffset = offset + 4;
    if (startsWithAscii(buf, typeOffset, "acTL")) return true;
    if (startsWithAscii(buf, typeOffset, "IDAT")) return false;
    const next = offset + 8 + chunkLength + 4;
    if (next <= offset || next > buf.length) return false;
    offset = next;
  }
  return false;
}

/** Sniff an image MIME type from magic bytes, aligned with pi's
 *  detectSupportedImageMimeType. Returns null for unsupported/animated PNG. */
export function sniffImageMime(buf: Buffer): string | null {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    return buf[3] === 0xf7 ? null : "image/jpeg";
  }
  if (buf.length >= 8 && PNG_SIGNATURE.every((b, i) => buf[i] === b)) {
    return isPng(buf) && !isAnimatedPng(buf) ? "image/png" : null;
  }
  if (buf.length >= 3 && startsWithAscii(buf, 0, "GIF")) return "image/gif";
  if (buf.length >= 12 && startsWithAscii(buf, 0, "RIFF") && startsWithAscii(buf, 8, "WEBP")) {
    return "image/webp";
  }
  return null;
}

/** Parse an image's pixel dimensions from its headers (no decode). Returns null
 *  for an unsupported/unparseable format. The dimension check is orientation-
 *  invariant (`max(w,h) > 2000` is unchanged by EXIF rotation swaps), so it
 *  agrees with pi's resize decision even for EXIF-oriented images. */
export function imageDimensions(buf: Buffer, mimeType: string): { width: number; height: number } | null {
  switch (mimeType) {
    case "image/png": {
      // IHDR: width (4 BE) at offset 16, height (4 BE) at offset 20.
      if (buf.length < 24) return null;
      return { width: readUint32BE(buf, 16), height: readUint32BE(buf, 20) };
    }
    case "image/gif": {
      // Logical screen descriptor: width (2 LE) at 6, height (2 LE) at 8.
      if (buf.length < 10) return null;
      return { width: readUint16LE(buf, 6), height: readUint16LE(buf, 8) };
    }
    case "image/jpeg": {
      return jpegDimensions(buf);
    }
    case "image/webp": {
      return webpDimensions(buf);
    }
    default:
      return null;
  }
}

/** Scan JPEG segments for a SOF marker and read width/height. */
function jpegDimensions(buf: Buffer): { width: number; height: number } | null {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  let offset = 2;
  while (offset + 9 <= buf.length) {
    if (buf[offset] !== 0xff) return null;
    // Skip fill bytes.
    let marker = buf[offset + 1];
    while (marker === 0xff && offset + 2 < buf.length) {
      offset++;
      marker = buf[offset + 1];
    }
    // Standalone markers (RSTn, SOI, EOI) have no length payload.
    if (marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7)) {
      offset += 2;
      continue;
    }
    if (offset + 4 > buf.length) return null;
    const segLen = readUint16BE(buf, offset + 2);
    // SOF0–SOF15 (excluding RST/sof-defined non-SOF): C0–CF except C4 (DHT),
    // C8 (JPG), CC (DAC).
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      if (offset + 9 > buf.length) return null;
      const height = readUint16BE(buf, offset + 5);
      const width = readUint16BE(buf, offset + 7);
      return { width, height };
    }
    // SOS: image data follows, no more SOF.
    if (marker === 0xda) return null;
    offset += 2 + segLen;
  }
  return null;
}

/** Parse WEBP dimensions across VP8 (lossy), VP8L (lossless), VP8X (extended). */
function webpDimensions(buf: Buffer): { width: number; height: number } | null {
  if (buf.length < 30 || !startsWithAscii(buf, 0, "RIFF") || !startsWithAscii(buf, 8, "WEBP")) return null;
  const fourcc = buf.subarray(12, 16).toString("ascii");
  if (fourcc === "VP8 ") {
    // Lossy: 3-byte frame tag at 20, 3-byte start code at 23, width (2 LE &
    // 0x3FFF) at 26, height (2 LE & 0x3FFF) at 28.
    return { width: readUint16LE(buf, 26) & 0x3fff, height: readUint16LE(buf, 28) & 0x3fff };
  }
  if (fourcc === "VP8L") {
    // Lossless: 0x2F signature at 20, then 14-bit (width-1) + 14-bit (height-1)
    // packed LSB-first across bytes 21–24.
    if (buf[20] !== 0x2f) return null;
    const val = (buf[21] ?? 0) | ((buf[22] ?? 0) << 8) | ((buf[23] ?? 0) << 16) | ((buf[24] ?? 0) << 24);
    return { width: 1 + (val & 0x3fff), height: 1 + ((val >> 14) & 0x3fff) };
  }
  if (fourcc === "VP8X") {
    // Extended: canvas width-1 (24-bit LE) at 24, height-1 (24-bit LE) at 27.
    const w = 1 + ((buf[24] ?? 0) | ((buf[25] ?? 0) << 8) | ((buf[26] ?? 0) << 16));
    const h = 1 + ((buf[27] ?? 0) | ((buf[28] ?? 0) << 8) | ((buf[29] ?? 0) << 16));
    return { width: w, height: h };
  }
  return null;
}

/** Base64-encoded size of `byteLength` raw bytes (4/3 ratio, ceil). */
function base64Size(byteLength: number): number {
  return Math.ceil(byteLength / 3) * 4;
}

/** Whether pi's `read` tool would RESIZE (re-encode) this image. When false,
 *  pi returns the raw input bytes unchanged — so a raw pre-warm matches the
 *  `tool_result` key. When true, pi re-encodes — the caller must run the same
 *  resize pipeline to match. Mirrors pi's resize threshold exactly. Unknown
 *  dimensions default to `true` (force the resize path, which always matches). */
export function willBeResized(buf: Buffer, mimeType: string): boolean {
  const dims = imageDimensions(buf, mimeType);
  if (!dims) return true;
  return (
    dims.width > RESIZE_MAX_WIDTH ||
    dims.height > RESIZE_MAX_HEIGHT ||
    base64Size(buf.length) >= RESIZE_MAX_BYTES
  );
}

/** A resize function shaped like pi's `resizeImage` (injected so this module
 *  stays free of the pi-coding-agent dependency and fully unit-testable). */
export type ResizeFn = (
  inputBytes: Uint8Array,
  mimeType: string,
) => Promise<{ data: string; mimeType: string } | null>;

/** Resolve a clipboard image buffer to the SAME {@link ExtractedImage} pi's
 *  `read` tool will emit, so the pre-warm's cache key matches the later
 *  `tool_result`'s key (no wasted vision call):
 *  - no-resize case (≤2000² & <4.5MB base64): return the raw bytes — pi's
 *    no-resize path returns `inputBytes` unchanged.
 *  - resize case: run the same `resize` pipeline (pi's `resizeImage`) and
 *    return the re-encoded data — matches pi's resize path.
 *  Returns null if resize failed (pi then emits no image block, so there is
 *  nothing to pre-warm) or if the bytes aren't a supported image. */
export async function resolvePrewarmImage(
  buf: Buffer,
  mimeType: string,
  resize: ResizeFn,
): Promise<ExtractedImage | null> {
  if (!willBeResized(buf, mimeType)) {
    return { data: buf.toString("base64"), mimeType };
  }
  const resized = await resize(buf, mimeType);
  if (!resized) return null;
  return { data: resized.data, mimeType: resized.mimeType };
}

/** Read an image file into its raw buffer + sniffed MIME. Returns null if the
 *  file can't be read or isn't a supported image (aligned with pi's sniff). */
export function readImageBuffer(filePath: string): { buf: Buffer; mimeType: string } | null {
  try {
    const buf = readFileSync(filePath);
    const mimeType = sniffImageMime(buf);
    if (!mimeType) return null;
    return { buf, mimeType };
  } catch {
    return null;
  }
}

/** Max raw file size the omitted-image recovery will re-read. pi's `read`
 *  already read the file; this bounds the handoff's re-read so a pathologically
 *  huge file doesn't double the memory. 20MB covers the most generous vision
 *  model inline-image limit; a larger file would be rejected by the vision
 *  model anyway. */
export const MAX_RECOVER_IMAGE_BYTES = 20 * 1024 * 1024;

/** pi core's `read` tool emits this text note (with NO image block) when it
 *  detected an image but `processImage` failed — Photon/WASM unavailable,
 *  decode failure, convert-to-PNG failure, or couldn't resize below the inline
 *  size limit. The handoff's image-block path never sees these (no image
 *  block), so the image goes undescribed and the model is told the image was
 *  "omitted". This detects that note so the tool_result handler can re-read the
 *  raw file and describe its bytes directly (the vision model decodes them —
 *  no Photon needed). */
export function isOmittedImageNote(text: string): boolean {
  return text.includes("Read image file [") && text.includes("[Image omitted:");
}

/** Read an image file for the omitted-image recovery, bounded by
 *  {@link MAX_RECOVER_IMAGE_BYTES}. Returns null if the file can't be read, is
 *  too large, or isn't a supported image (aligned with pi's sniff — APNG and
 *  unsupported formats are rejected, since the vision model can't decode them
 *  either). */
export function readImageBufferBounded(filePath: string): { buf: Buffer; mimeType: string } | null {
  try {
    const stat = statSync(filePath);
    if (stat.size > MAX_RECOVER_IMAGE_BYTES) return null;
  } catch {
    return null;
  }
  return readImageBuffer(filePath);
}

/** Regex matching pi's pasted-clipboard temp image file paths anywhere in the
 *  prompt text. pi writes pasted clipboard images to
 *  `<tmpdir>/pi-clipboard-<uuid>.<ext>` and inserts the path as text at the
 *  cursor, so on a non-vision model these arrive as path tokens in the user
 *  prompt — NOT as `event.images`. Matching them lets the extension pre-warm
 *  the describer at paste-enter (concurrent with the agent's first response)
 *  instead of waiting for the agent to `read` them. */
const CLIPBOARD_IMAGE_PATH_RE = /(\S*pi-clipboard-[^\s]+\.(?:png|jpe?g|gif|webp))/gi;

/** Extract pasted clipboard image file paths from a prompt. Confined to the
 *  OS temp directory so an attacker-crafted prompt can't trick the extension
 *  into reading arbitrary files — only pi's own clipboard temp files qualify. */
export function findClipboardImagePaths(prompt: string): string[] {
  const tmp = tmpdir();
  const paths = new Set<string>();
  for (const m of prompt.matchAll(CLIPBOARD_IMAGE_PATH_RE)) {
    const p = m[1];
    if (!p) continue;
    const abs = isAbsolute(p) ? p : join(tmp, p);
    // Ensure the resolved candidate stays inside the temp directory.
    if (abs.startsWith(tmp + sep) || abs === tmp) paths.add(abs);
  }
  return [...paths];
}

/** Diff the clipboard image paths in `text` against `known`, returning those
 *  that are newly appeared (not yet seen). Used by the paste-time prewarm
 *  editor to prewarm only newly-pasted paths on each text change, without
 *  re-reading already-seen ones. `findClipboardImagePaths` already dedups
 *  within a single text, so each returned path is unique and seen at most
 *  once. Returns [] when `text` holds no clipboard paths (e.g. ordinary
 *  typing) — so the editor's per-keystroke cost when the opt-in is on is one
 *  regex scan that almost always yields nothing. */
export function diffPrewarmPaths(text: string, known: Set<string>): string[] {
  const paths = findClipboardImagePaths(text);
  const newPaths: string[] = [];
  for (const p of paths) {
    if (!known.has(p)) newPaths.push(p);
  }
  return newPaths;
}
