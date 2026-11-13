/**
 * Temporal filter for the 27 SH coefficients: a One Euro filter (Casiez, Roussel and Vogel, CHI
 * 2012, https://gery.casiez.net/1euro/) applied to the whole vector at once.
 *
 * Why one shared cutoff instead of 27 independent filters: independent cutoffs would smooth the red,
 * green and blue coefficients by different amounts during a change, so the colour of the light
 * would drift while it moves. Here the adaptive cutoff comes from the speed of the whole vector,
 * divided by a brightness scale so the same settings work for dim and bright scenes.
 *
 *   cutoff = minCutoff + beta * |d(sh)/dt| / scale        (Hz; scale = luminance of L_0)
 *   alpha  = 1 / (1 + 1 / (2 pi cutoff dt))
 *
 * Low minCutoff removes jitter when the light is still; high beta removes lag when it moves.
 */

export interface OneEuroParams {
  /** Cutoff frequency in Hz when the signal is still. */
  minCutoff: number;
  /** How fast the cutoff rises with relative speed (per unit of relative change per second). */
  beta: number;
  /** Cutoff in Hz for the derivative estimate. */
  dCutoff: number;
}

// End points chosen by a grid search over step settle time and jitter (filter.test.ts measures
// both): the steady end settles a full step in about 0.9 s and keeps about 5% of white jitter; the
// responsive end settles in about 0.07 s and keeps about a third.
const SMOOTH: OneEuroParams = { minCutoff: 0.05, beta: 0.02, dCutoff: 0.3 };
const SNAPPY: OneEuroParams = { minCutoff: 2, beta: 0.4, dCutoff: 1 };

/**
 * Map the responsiveness control (0 = steadiest, 1 = most responsive) to filter parameters.
 * Interpolates geometrically, because both parameters act on a log scale.
 */
export function paramsForResponsiveness(r: number): OneEuroParams {
  const t = Math.min(Math.max(r, 0), 1);
  const geo = (a: number, b: number): number => a * (b / a) ** t;
  return {
    minCutoff: geo(SMOOTH.minCutoff, SNAPPY.minCutoff),
    beta: geo(SMOOTH.beta, SNAPPY.beta),
    dCutoff: geo(SMOOTH.dCutoff, SNAPPY.dCutoff),
  };
}

function alpha(cutoffHz: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dt);
}

/** Gaps longer than this restart the filter instead of smoothing across them. */
const MAX_GAP_S = 1;

export class VectorOneEuro {
  private x: Float64Array | null = null;
  private dx: Float64Array | null = null;
  private t = 0;

  constructor(public params: OneEuroParams) {}

  reset(): void {
    this.x = null;
    this.dx = null;
  }

  /**
   * Filter one sample taken at `tSeconds`. `scale` is a positive brightness reference (for SH, the
   * luminance of L_0) that makes the speed relative. Returns the filtered vector (a fresh copy).
   */
  filter(values: ArrayLike<number>, tSeconds: number, scale: number): Float64Array {
    const n = values.length;
    const x = this.x;
    const dt = tSeconds - this.t;
    if (x === null || x.length !== n || dt > MAX_GAP_S || dt < 0) {
      this.x = Float64Array.from(values);
      this.dx = new Float64Array(n);
      this.t = tSeconds;
      return Float64Array.from(this.x);
    }
    if (dt === 0) return Float64Array.from(x);
    const dx = this.dx ?? new Float64Array(n);
    const ad = alpha(this.params.dCutoff, dt);
    let speed2 = 0;
    for (let i = 0; i < n; i++) {
      const raw = ((values[i] ?? 0) - (x[i] ?? 0)) / dt;
      const d = (dx[i] ?? 0) + ad * (raw - (dx[i] ?? 0));
      dx[i] = d;
      speed2 += d * d;
    }
    const relSpeed = Math.sqrt(speed2) / Math.max(scale, 1e-6);
    const a = alpha(this.params.minCutoff + this.params.beta * relSpeed, dt);
    for (let i = 0; i < n; i++) x[i] = (x[i] ?? 0) + a * ((values[i] ?? 0) - (x[i] ?? 0));
    this.dx = dx;
    this.t = tSeconds;
    return Float64Array.from(x);
  }
}
