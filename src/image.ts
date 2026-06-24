/**
 * Pure image IO utilities: hashing, MIME sniffing, and reading pasted
 * clipboard image files from the prompt text.
 *
 * `readImageFile` is SYNCHRONOUS by design — the dataloader requires every
 * `loadDescription()` call in a frame to land in the SAME batch (dispatched
 * after the microtask cascade settles). An async read would suspend the
 * handler on I/O, letting the dispatch fire mid-read and split one batch into
 * two vision calls. Clipboard images are small (a few MB), so a sync read is
 * sub-millisecond.
 */

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, sep } from "node:path";
import crypto from "node:crypto";
import type { ExtractedImage } from "./index.js";

/** Stable hash of an image's MIME + base64 data, used as the dataloader key. */
export function imageHash(mimeType: string, data: string): string {
  return crypto.createHash("sha256").update(`${mimeType}\x00${data}`).digest("hex").slice(0, 32);
}

/** Sniff an image MIME type from magic bytes (mirrors pi's
 *  detectSupportedImageMimeType). Returns null for unsupported/animated. */
export function sniffImageMime(buf: Buffer): string | null {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    return buf[3] === 0xf7 ? null : "image/jpeg";
  }
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (buf.length >= 8 && png.every((b, i) => buf[i] === b)) return "image/png";
  if (buf.length >= 3 && buf.subarray(0, 3).toString("ascii") === "GIF") return "image/gif";
  if (buf.length >= 12 && buf.subarray(0, 4).toString("ascii") === "RIFF" && buf.subarray(8, 12).toString("ascii") === "WEBP") {
    return "image/webp";
  }
  return null;
}

/** Read an image file from disk into an {@link ExtractedImage} (base64 + sniffed
 *  MIME). Synchronous — see the module doc for why. Returns null if the file
 *  can't be read or isn't a supported image. */
export function readImageFile(filePath: string): ExtractedImage | null {
  try {
    const buf = readFileSync(filePath);
    const mimeType = sniffImageMime(buf);
    if (!mimeType) return null;
    return { data: buf.toString("base64"), mimeType };
  } catch {
    return null;
  }
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
