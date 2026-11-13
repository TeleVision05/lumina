/**
 * Small constructors for radiance SH9 (docs/CONVENTIONS.md, section 3). Used by the fixed studio
 * rig and the image-statistics estimator, and by tests that need a known environment.
 */
import { LUMA, shBasis, type RGB, type SH9, type Vec3 } from './sh';

/** Y_0 = 0.282095; a uniform radiance a has L_0 = 4 pi Y_0 a. */
const Y0 = 0.282095;

export function zeroSH(): RGB[] {
  return Array.from({ length: 9 }, (): RGB => [0, 0, 0]);
}

/** SH of a uniform environment of radiance `a` (linear RGB). A white sphere then shows `a`. */
export function uniformSH(a: RGB): RGB[] {
  const out = zeroSH();
  const k = 4 * Math.PI * Y0;
  out[0] = [k * a[0], k * a[1], k * a[2]];
  return out;
}

/**
 * Add a distant light from unit direction `d` whose irradiance on a surface facing it is `e`
 * (linear RGB): L_i += e Y_i(d). `soften` in [0, 1) damps band 2 (and band 1 by half as much),
 * turning the delta into a broad lobe; 0 keeps the exact delta projection.
 */
export function addDirectional(sh: SH9, d: Vec3, e: RGB, soften = 0): RGB[] {
  const n = Math.hypot(d[0], d[1], d[2]);
  if (!(n > 0)) throw new RangeError('direction must be non-zero');
  const basis = shBasis([d[0] / n, d[1] / n, d[2] / n]);
  const damp = [1, 1 - soften / 2, 1 - soften];
  return sh.map((c, i): RGB => {
    const w = (basis[i] ?? 0) * (damp[i === 0 ? 0 : i < 4 ? 1 : 2] ?? 1);
    return [c[0] + w * e[0], c[1] + w * e[1], c[2] + w * e[2]];
  });
}

export function scaleSH(sh: SH9, s: number): RGB[] {
  return sh.map((c): RGB => [c[0] * s, c[1] * s, c[2] * s]);
}

/** Linear luminance of an RGB triple (Rec. 709 weights). */
export function luminance(c: RGB): number {
  return LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2];
}

/** Average radiance of the environment (L_0 / (4 pi Y_0)), linear RGB. */
export function meanRadiance(sh: SH9): RGB {
  const c = sh[0] ?? [0, 0, 0];
  const k = 1 / (4 * Math.PI * Y0);
  return [c[0] * k, c[1] * k, c[2] * k];
}

/** True when all 27 values are finite numbers. */
export function isFiniteSH(sh: SH9): boolean {
  return sh.length === 9 && sh.every((c) => c.every((v) => Number.isFinite(v)));
}
