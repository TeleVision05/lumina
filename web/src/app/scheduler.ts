/**
 * Timing helpers for the render loop: when to start the next inference, and rolling statistics
 * for the readout (inference time, frame time, achieved rate).
 */

/** Target estimator rate. The brief asks for 10 to 15 inferences per second. */
export const TARGET_HZ = 12;

/**
 * Starts work at most `hz` times per second and never while a previous run is in flight. If a run
 * takes longer than the period, the next one starts as soon as it finishes.
 */
export class RateGate {
  private lastStart = Number.NEGATIVE_INFINITY;
  private busy = false;

  constructor(public hz: number = TARGET_HZ) {}

  /** True (and marks the start) when a new run may begin at time `nowMs`. */
  tryStart(nowMs: number): boolean {
    if (this.busy || nowMs - this.lastStart < 1000 / this.hz) return false;
    this.busy = true;
    this.lastStart = nowMs;
    return true;
  }

  finish(): void {
    this.busy = false;
  }

  get inFlight(): boolean {
    return this.busy;
  }
}

/** Fixed-size window of recent samples with median, p95 and rate. */
export class Rolling {
  private readonly values: number[] = [];
  private readonly times: number[] = [];

  constructor(private readonly size = 120) {}

  push(value: number, atMs: number): void {
    this.values.push(value);
    this.times.push(atMs);
    if (this.values.length > this.size) {
      this.values.shift();
      this.times.shift();
    }
  }

  get count(): number {
    return this.values.length;
  }

  quantile(q: number): number | null {
    if (this.values.length === 0) return null;
    const s = [...this.values].sort((a, b) => a - b);
    const i = Math.min(s.length - 1, Math.max(0, Math.ceil(q * s.length) - 1));
    return s[i] ?? null;
  }

  get median(): number | null {
    return this.quantile(0.5);
  }

  /**
   * Samples per second over the most recent `windowMs` (default 3 s) of the window, or null with
   * fewer than two samples in that span.
   */
  rate(windowMs = 3000): number | null {
    const last = this.times[this.times.length - 1];
    if (last === undefined) return null;
    const recent = this.times.filter((t) => t >= last - windowMs);
    const first = recent[0];
    if (recent.length < 2 || first === undefined || last <= first) return null;
    return ((recent.length - 1) * 1000) / (last - first);
  }

  clear(): void {
    this.values.length = 0;
    this.times.length = 0;
  }
}
