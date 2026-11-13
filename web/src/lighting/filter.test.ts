import { describe, expect, it } from 'vitest';
import { paramsForResponsiveness, VectorOneEuro } from './filter';

/** Deterministic standard normal noise (mulberry32 + Box-Muller). */
function gaussian(seed: number): () => number {
  let s = seed >>> 0;
  const uniform = (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
}

/** Seconds until a 1-D step from 0 to 1 (scale 1) settles within 10% of its final value. */
function settleTime(responsiveness: number, rateHz: number): number {
  const f = new VectorOneEuro(paramsForResponsiveness(responsiveness));
  const dt = 1 / rateHz;
  for (let k = 0; k < rateHz; k++) f.filter([0], k * dt, 1);
  const t0 = rateHz * dt;
  for (let k = 0; k < rateHz * 10; k++) {
    const t = t0 + k * dt;
    const y = f.filter([1], t, 1)[0] ?? 0;
    if (y >= 0.9) return t - t0;
  }
  return Number.POSITIVE_INFINITY;
}

function std(xs: number[]): number {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
}

/**
 * Output std / input std for a constant level of 1 plus 5% gaussian noise, filtered at `rateHz`.
 * `hold` > 1 keeps each noisy value for that many frames (a 12 Hz estimator under a 60 Hz renderer).
 */
function jitterRatio(responsiveness: number, rateHz: number, hold = 1): number {
  const f = new VectorOneEuro(paramsForResponsiveness(responsiveness));
  const noise = gaussian(20261113);
  const ins: number[] = [];
  const outs: number[] = [];
  let x = 1;
  for (let k = 0; k < rateHz * 20; k++) {
    if (k % hold === 0) x = 1 + 0.05 * noise();
    const y = f.filter([x], k / rateHz, 1)[0] ?? 0;
    if (k > rateHz * 5) {
      ins.push(x);
      outs.push(y);
    }
  }
  return std(outs) / std(ins);
}

describe('VectorOneEuro step response', () => {
  it('settles a full step within 10% in under 0.35 s at the default responsiveness', () => {
    const t = settleTime(0.5, 60);
    console.log(`step settle (r=0.5, 60 Hz): ${(t * 1000).toFixed(0)} ms`);
    expect(t).toBeLessThan(0.35);
  });

  it('is faster when more responsive and slower when steadier', () => {
    const slow = settleTime(0, 60);
    const mid = settleTime(0.5, 60);
    const fast = settleTime(1, 60);
    console.log(
      `step settle 60 Hz: r=0 ${(slow * 1000).toFixed(0)} ms, r=0.5 ${(mid * 1000).toFixed(0)} ms, r=1 ${(fast * 1000).toFixed(0)} ms`,
    );
    expect(fast).toBeLessThan(mid);
    expect(mid).toBeLessThan(slow);
    expect(fast).toBeLessThan(0.15);
  });

  it('never overshoots and rises monotonically', () => {
    const f = new VectorOneEuro(paramsForResponsiveness(1));
    let prev = 0;
    f.filter([0], 0, 1);
    for (let k = 1; k < 120; k++) {
      const y = f.filter([1], k / 60, 1)[0] ?? 0;
      expect(y).toBeGreaterThanOrEqual(prev);
      expect(y).toBeLessThanOrEqual(1);
      prev = y;
    }
    expect(prev).toBeGreaterThan(0.999);
  });

  it('settles in about the same time at 30 Hz and 60 Hz (rate independent)', () => {
    const a = settleTime(0.5, 30);
    const b = settleTime(0.5, 60);
    console.log(
      `step settle r=0.5: 30 Hz ${(a * 1000).toFixed(0)} ms, 60 Hz ${(b * 1000).toFixed(0)} ms`,
    );
    // Four 30 Hz frames: the derivative estimate sees fewer, larger steps at the lower rate.
    expect(Math.abs(a - b)).toBeLessThanOrEqual(4 / 30 + 1e-9);
  });

  it('measures speed relative to scale, so a 10x brighter scene behaves the same', () => {
    const run = (k: number): number => {
      const f = new VectorOneEuro(paramsForResponsiveness(0.5));
      f.filter([0], 0, k);
      return (f.filter([k], 1 / 60, k)[0] ?? 0) / k;
    };
    expect(run(10)).toBeCloseTo(run(1), 12);
  });
});

describe('VectorOneEuro jitter', () => {
  it('removes at least 80% of steady-state jitter at the default responsiveness', () => {
    const ratio = jitterRatio(0.5, 60);
    console.log(`jitter ratio (r=0.5, 60 Hz, 5% noise): ${ratio.toFixed(3)}`);
    expect(ratio).toBeLessThan(0.2);
  });

  it('removes over half the jitter of a 12 Hz estimator held across 60 Hz frames', () => {
    const ratio = jitterRatio(0.5, 60, 5);
    console.log(`jitter ratio (r=0.5, 12 Hz held at 60 Hz, 5% noise): ${ratio.toFixed(3)}`);
    expect(ratio).toBeLessThan(0.5);
  });

  it('smooths more at the steady end than at the responsive end', () => {
    const steady = jitterRatio(0, 60);
    const responsive = jitterRatio(1, 60);
    console.log(`jitter ratio 60 Hz: r=0 ${steady.toFixed(3)}, r=1 ${responsive.toFixed(3)}`);
    expect(steady).toBeLessThan(responsive);
    expect(responsive).toBeLessThan(1);
  });
});

describe('VectorOneEuro bookkeeping', () => {
  it('passes the first sample through and restarts after a long gap', () => {
    const f = new VectorOneEuro(paramsForResponsiveness(0.5));
    expect(Array.from(f.filter([1, 2, 3], 0, 1))).toEqual([1, 2, 3]);
    f.filter([1, 2, 3], 0.016, 1);
    expect(Array.from(f.filter([5, 5, 5], 3, 1))).toEqual([5, 5, 5]);
  });

  it('applies one shared cutoff, so every coefficient moves by the same fraction', () => {
    const f = new VectorOneEuro(paramsForResponsiveness(0.5));
    f.filter([0, 0, 0], 0, 1);
    const y = f.filter([1, 2, 4], 1 / 60, 1);
    expect((y[1] ?? 0) / 2).toBeCloseTo(y[0] ?? 0, 12);
    expect((y[2] ?? 0) / 4).toBeCloseTo(y[0] ?? 0, 12);
  });
});
