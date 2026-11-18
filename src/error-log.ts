/**
 * Best-effort structured error logging for pi-vision-handoff.
 *
 * Every describer failure (auth error, network error, abort, empty response,
 * stopReason error/aborted, timeout) and every user-facing "image description
 * failed" warning is appended as one JSONL line to
 *   ~/.pi/agent/logs/pi-vision-handoff/errors.log
 * (resolvable via $PI_CODING_AGENT_DIR, like every other pi path).
 *
 * This is the troubleshooting surface for the "image description failed —
 * unknown error" warning. That warning fires with reason "unknown error" when
 * the engine's shared last-error string was already null — typically because a
 * concurrent batch reset it between the failure and the warn — so the detailed
 * reason lives only here, captured at the describer's failure source where the
 * real exception/stopReason is still in hand. A `warn` entry with reason
 * "unknown error" carries the failing image hashes; correlate them (and the
 * timestamp) with the matching `batch`/`single` entry to recover the real
 * cause.
 *
 * Best-effort: logging never throws and never breaks a describer turn. The
 * directory is created lazily; a single size-based rotation (errors.log →
 * errors.log.1) bounds growth to ~2× the cap.
 */

import { appendFileSync, closeSync, fchmodSync, mkdirSync, openSync, renameSync, statSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { resolveAgentDir } from "./agent-dir.js";
import { isFsError, resolveWritableTarget, withFileLock } from "./shared-file.js";

/** Subdirectory under the pi agent dir holding this extension's logs. */
const LOG_SUBDIR = "logs/pi-vision-handoff";
/** Active log file name. */
const LOG_FILENAME = "errors.log";

/** A single error log is capped at this many bytes before it rotates to the
 *  single `.1` backup, bounding total on-disk log to ~2× this. Generous enough
 *  to retain a long troubleshooting history; small enough that a runaway
 *  broken vision model (failures aren't cached, so they re-fire per turn)
 *  can't fill the disk. */
export const MAX_LOG_BYTES = 10 * 1024 * 1024;

/** One structured error-log record (one JSONL line). */
export interface VisionErrorLogEntry {
  /** ISO 8601 timestamp (filled by {@link appendVisionError}). */
  timestamp: string;
  /** Where the entry originated: "batch"/"single" = describer failure source
   *  (rich detail), "warn" = the user-facing warning (may carry reason
   *  "unknown error" when the engine's shared error was already reset). */
  phase: "batch" | "single" | "warn";
  /** Human-readable failure reason (what `setLastError` received, or the warn
   *  reason). "unknown error" in a `warn` entry means no describer error was
   *  captured — correlate via {@link imageHashes} + {@link timestamp} with the
   *  matching batch/single entry. */
  reason: string;
  /** Configured vision model ref ("provider/id"), or null if unset. */
  visionModel: string | null;
  /** Image hashes the failure covered (batch/single) or the warning named. */
  imageHashes: string[];
  /** Number of images involved. */
  imageCount: number;
  /** Describer stopReason, when the failure came from a completed response. */
  stopReason?: string;
  /** True when the describer's timeout fired. */
  timedOut?: boolean;
  /** The timeout budget in ms, when timedOut applies. */
  timeoutMs?: number;
  /** Provider error message from the response (response.errorMessage). */
  errorMessage?: string;
  /** Stack trace of a thrown error, when available. */
  errorStack?: string;
  /** Small config snapshot relevant to troubleshooting. */
  config?: { maxTokens?: number; thinking: boolean; thinkingLevel: string };
  /** The model being handed off to ("provider/id"), at the warn point. */
  activeModel?: string;
}

/** Directory holding this extension's logs:
 *  ~/.pi/agent/logs/pi-vision-handoff (overridable via $PI_CODING_AGENT_DIR). */
export function getErrorLogDir(): string {
  return join(resolveAgentDir(), LOG_SUBDIR);
}

/** Path to the active error log file. */
export function getErrorLogPath(): string {
  return join(getErrorLogDir(), LOG_FILENAME);
}

/** Serialize an entry to one JSONL line (JSON + terminating newline). Pure and
 *  unit-testable independent of the disk. */
export function formatErrorLogLine(entry: VisionErrorLogEntry): string {
  return JSON.stringify(entry) + "\n";
}

/** Rotate the log when it has grown to at least `maxSize`: the current file
 *  becomes the `.1` backup (overwriting any prior backup) and the active path
 *  is cleared for a fresh log. Bounds total on-disk log to ~2× `maxSize`.
 *  No-op when the file doesn't exist or is under the cap. Never throws. */
export function rotateLogIfNeeded(path: string = getErrorLogPath(), maxSize: number = MAX_LOG_BYTES): void {
  try {
    withLogLock(path, (target) => rotateLockedLog(target, maxSize));
  } catch {
    // Logging must not throw, including when another process owns the lock.
  }
}

function withLogLock<T>(path: string, action: (target: string) => T): T {
  const target = resolveWritableTarget(path);
  if (basename(target).toLowerCase() === "settings.json") throw new Error("Not an error log");
  return withFileLock(target, 75, () => {
    if (resolveWritableTarget(path) !== target) throw new Error("Log target changed");
    return action(target);
  });
}

/** Called only with the active log's lock held. Returns its old mode on rotation. */
function rotateLockedLog(path: string, maxSize: number): number | undefined {
  try {
    const info = statSync(path);
    if (!info.isFile()) throw new Error("Not a regular log file");
    if (info.size >= maxSize) {
      try {
        renameSync(path, path + ".1");
        return info.mode & 0o7777;
      } catch {
        // If rotation fails, continue appending to the existing file.
      }
    }
  } catch (error) {
    if (!isFsError(error, "ENOENT")) throw error;
  }
  return undefined;
}

/**
 * Append one structured error entry to the log. Fills `timestamp`, ensures the
 * log directory exists, rotates on size, and writes a JSONL line. Best-effort:
 * swallows every error so a logging failure can NEVER break a describer turn
 * (the describer calls this from its failure paths, where throwing would mask
 * the real error and abort the batch).
 */
export function appendVisionError(input: Omit<VisionErrorLogEntry, "timestamp">): void {
  try {
    mkdirSync(getErrorLogDir(), { recursive: true });
    const entry: VisionErrorLogEntry = { timestamp: new Date().toISOString(), ...input };
    const line = formatErrorLogLine(entry);
    withLogLock(getErrorLogPath(), (target) => {
      const rotatedMode = rotateLockedLog(target, MAX_LOG_BYTES);
      if (rotatedMode === undefined) {
        appendFileSync(target, line, "utf8");
      } else {
        const fd = openSync(target, "wx", 0o600);
        try {
          fchmodSync(fd, rotatedMode);
          writeFileSync(fd, line, "utf8");
        } finally {
          closeSync(fd);
        }
      }
    });
  } catch {
    // never break the describer on a logging failure
  }
}
