import { randomUUID } from "node:crypto";
import { closeSync, fchmodSync, lstatSync, mkdirSync, openSync, realpathSync, renameSync, rmdirSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";

export function isFsError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}

export function resolveWritableTarget(path: string): string {
  try {
    return realpathSync(path);
  } catch (error) {
    if (!isFsError(error, "ENOENT")) throw error;
    try {
      lstatSync(path);
    } catch (statError) {
      if (isFsError(statError, "ENOENT")) return join(realpathSync(dirname(path)), basename(path));
      throw statError;
    }
    // Re-check in case the file appeared; a dangling symlink must not be replaced.
    return realpathSync(path);
  }
}

export function withFileLock<T>(target: string, waitMs: number, action: () => T): T {
  const lock = `${target}.lock`;
  const deadline = performance.now() + waitMs;
  let sleeper: Int32Array | undefined;
  for (;;) {
    try {
      // Never steal on age. A crash can leave a lock that needs manual removal.
      mkdirSync(lock, { mode: 0o700 });
      break;
    } catch (error) {
      if (!isFsError(error, "EEXIST")) throw error;
      const remaining = deadline - performance.now();
      if (remaining <= 0) throw new Error(`File is locked: ${lock}`);
      sleeper ??= new Int32Array(new SharedArrayBuffer(4));
      Atomics.wait(sleeper, 0, 0, Math.min(10, remaining));
    }
  }
  try {
    return action();
  } finally {
    rmdirSync(lock);
  }
}

export function replaceFile(target: string, content: string, mode?: number): void {
  const temporary = join(dirname(target), `.vision-handoff-${randomUUID()}.tmp`);
  const fd = openSync(temporary, "wx", mode === undefined ? 0o666 : 0o600);
  try {
    try {
      writeFileSync(fd, content, "utf8");
      if (mode !== undefined) fchmodSync(fd, mode);
    } finally {
      closeSync(fd);
    }
    renameSync(temporary, target);
  } catch (error) {
    try { unlinkSync(temporary); } catch { /* keep the original failure */ }
    throw error;
  }
}
