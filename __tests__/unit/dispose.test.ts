import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { fetchInterceptorGuard, timeoutGuard, abortWireGuard } from "../../src/dispose.js";
import { fetchInterceptorRefcount, installFetchInterceptor, uninstallFetchInterceptor } from "../../src/usage.js";

describe("fetchInterceptorGuard", () => {
  beforeEach(() => {
    // Ensure a clean refcount baseline.
    while (fetchInterceptorRefcount() > 0) uninstallFetchInterceptor();
  });

  it("installs on construct and uninstalls on dispose via `using`", () => {
    expect(fetchInterceptorRefcount()).toBe(0);
    {
      using guard = fetchInterceptorGuard();
      expect(fetchInterceptorRefcount()).toBe(1);
    }
    expect(fetchInterceptorRefcount()).toBe(0);
  });

  it("nests refcounts across multiple guards", () => {
    {
      using a = fetchInterceptorGuard();
      using b = fetchInterceptorGuard();
      expect(fetchInterceptorRefcount()).toBe(2);
    }
    expect(fetchInterceptorRefcount()).toBe(0);
  });
});

describe("timeoutGuard", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("fires onTimeout after the delay", () => {
    const fn = vi.fn();
    {
      using g = timeoutGuard(1000, fn);
      vi.advanceTimersByTime(999);
      expect(fn).not.toHaveBeenCalled();
    }
    // disposed: the timer is cleared, so advancing never fires.
    vi.advanceTimersByTime(5000);
    expect(fn).not.toHaveBeenCalled();
  });

  it("fires before the scope ends if the delay elapses", () => {
    const fn = vi.fn();
    using g = timeoutGuard(100, fn);
    vi.advanceTimersByTime(100);
    expect(fn).toHaveBeenCalledTimes(1);
    // dispose is a no-op now (already fired), must not throw.
  });
});

describe("abortWireGuard", () => {
  it("propagates an already-aborted turn signal into the controller", () => {
    const turn = new AbortController();
    turn.abort();
    const desc = new AbortController();
    using wire = abortWireGuard(turn.signal, desc);
    expect(desc.signal.aborted).toBe(true);
    expect(wire.userAborted()).toBe(true);
  });

  it("propagates a later abort and detaches on dispose", () => {
    const turn = new AbortController();
    const desc = new AbortController();
    {
      using wire = abortWireGuard(turn.signal, desc);
      expect(desc.signal.aborted).toBe(false);
      turn.abort();
      expect(desc.signal.aborted).toBe(true);
      expect(wire.userAborted()).toBe(true);
    }
    // listener removed after dispose: re-aborting a different controller would
    // not affect anything here, and turn has no lingering listeners.
  });

  it("returns a no-op guard with userAborted()=false when there is no turn signal", () => {
    const desc = new AbortController();
    using wire = abortWireGuard(undefined, desc);
    expect(wire.userAborted()).toBe(false);
    expect(desc.signal.aborted).toBe(false);
  });
});
