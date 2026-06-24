/**
 * `Disposable` guard factories for the `using` keyword (Explicit Resource
 * Management). Each factory acquires a resource and returns a `Disposable`
 * whose `[Symbol.dispose]` releases it, so a `using` binding replaces a
 * manual acquire/`try`/`finally`/release pair.
 *
 * Used by the describer to bundle the fetch interceptor, the timeout timer,
 * and the turn-abort wire into lexical scopes whose cleanup is automatic.
 */

import { installFetchInterceptor, uninstallFetchInterceptor } from "./usage.js";

/** A `Disposable` that uninstalls the refcounted fetch interceptor on release. */
export function fetchInterceptorGuard(): Disposable {
  installFetchInterceptor();
  return {
    [Symbol.dispose]() {
      uninstallFetchInterceptor();
    },
  };
}

/** A `Disposable` that clears a `setTimeout` handle on release. */
export function timeoutGuard(ms: number, onTimeout: () => void): Disposable {
  const handle = setTimeout(onTimeout, ms);
  return {
    [Symbol.dispose]() {
      clearTimeout(handle);
    },
  };
}

/** A `Disposable` abort-wire: propagates a turn abort signal into a
 *  describer's `AbortController`, and detaches the listener on release.
 *  Also exposes `userAborted()` so the describer can tell a deliberate user
 *  cancel apart from a provider/timeout abort (to suppress spurious warnings).
 *
 *  Always returns a `Disposable` (a no-op when there is no turn signal) so a
 *  `using` binding never has to null-check. */
export interface AbortWire extends Disposable {
  /** True iff the abort originated from the turn signal (a user cancel). */
  userAborted(): boolean;
}

export function abortWireGuard(turnSignal: AbortSignal | undefined, controller: AbortController): AbortWire {
  if (!turnSignal) {
    return {
      [Symbol.dispose]() {},
      userAborted: () => false,
    };
  }
  const onAbort = () => controller.abort();
  if (turnSignal.aborted) controller.abort();
  else turnSignal.addEventListener("abort", onAbort, { once: true });
  return {
    [Symbol.dispose]() {
      turnSignal.removeEventListener("abort", onAbort);
    },
    userAborted: () => turnSignal.aborted,
  };
}
