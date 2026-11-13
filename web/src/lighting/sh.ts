/**
 * Real spherical harmonics up to band 2 (SH9). Mirror of training/lumina_train/sh.py; both are
 * checked against shared/sh_vectors.json. Read docs/CONVENTIONS.md before changing anything here.
 *
 * - Frame: camera space, right-handed, +X right, +Y up, -Z forward (three.js camera default).
 * - Basis and ordering: exactly three.js SphericalHarmonics3.getBasisAt.
 * - Coefficients: incident RADIANCE, linear RGB, layout sh[coefficient] = [r, g, b].
 */
import { SphericalHarmonics3, Vector3 } from 'three';

export type Vec3 = readonly [number, number, number];
export type RGB = readonly [number, number, number];
/** Nine RGB coefficients of incident radiance. */
export type SH9 = readonly RGB[];

export const SH_COEFFS = 9;
/** Band of each coefficient. */
export const BAND_OF_COEFF = [0, 1, 1, 1, 2, 2, 2, 2, 2] as const;
/** Cosine-lobe convolution factor per band (Ramamoorthi and Hanrahan 2001). */
export const BAND_FACTORS = [Math.PI, (2 * Math.PI) / 3, Math.PI / 4] as const;
/** Rec. 709 / sRGB linear luminance weights. */
export const LUMA: RGB = [0.2126, 0.7152, 0.0722];
/** For non-negative radiance |L1| / L0 <= sqrt(3) (= 0.488603 / 0.282095). */
export const MAX_L1_OVER_L0 = 0.488603 / 0.282095;
/** Coefficients odd in x: Y3 ~ x, Y4 ~ xy, Y7 ~ xz. */
const MIRROR_X_SIGNS = [1, 1, 1, -1, -1, 1, 1, -1, 1] as const;

function coeff(sh: SH9, i: number): RGB {
  const c = sh[i];
  if (c === undefined) throw new RangeError(`SH9 needs 9 coefficients, missing index ${i}`);
  return c;
}

/** The nine basis functions at a unit direction, three.js order and constants. */
export function shBasis(d: Vec3): number[] {
  const [x, y, z] = d;
  return [
    0.282095,
    0.488603 * y,
    0.488603 * z,
    0.488603 * x,
    1.092548 * x * y,
    1.092548 * y * z,
    0.315392 * (3 * z * z - 1),
    1.092548 * x * z,
    0.546274 * (x * x - y * y),
  ];
}

/** Irradiance E(n) = sum_i A_band(i) L_i Y_i(n) at a unit normal. */
export function irradiance(sh: SH9, n: Vec3): RGB {
  const basis = shBasis(n);
  const out = [0, 0, 0];
  for (let i = 0; i < SH_COEFFS; i++) {
    const w = BAND_FACTORS[BAND_OF_COEFF[i] ?? 0] * (basis[i] ?? 0);
    const c = coeff(sh, i);
    out[0] = (out[0] ?? 0) + w * c[0];
    out[1] = (out[1] ?? 0) + w * c[1];
    out[2] = (out[2] ?? 0) + w * c[2];
  }
  return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
}

export interface DominantLight {
  /** Unit direction toward the light in the camera frame, or null when there is none. */
  direction: Vec3 | null;
  /** |L1| / (sqrt(3) L0) of the luminance SH: 0 uniform light, 1 a single point light. */
  directionality: number;
}

/** Dominant light direction from the luminance-weighted band-1 coefficients. */
export function dominantDirection(sh: SH9, minRatio = 1e-6): DominantLight {
  const lum = (i: number): number => {
    const c = coeff(sh, i);
    return LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2];
  };
  const l0 = lum(0);
  const v: Vec3 = [lum(3), lum(1), lum(2)]; // (x, y, z) from (Y3 ~ x, Y1 ~ y, Y2 ~ z)
  const norm = Math.hypot(v[0], v[1], v[2]);
  if (!(l0 > 0)) return { direction: null, directionality: 0 };
  const directionality = Math.min(norm / (MAX_L1_OVER_L0 * l0), 1);
  if (directionality < minRatio) return { direction: null, directionality };
  return { direction: [v[0] / norm, v[1] / norm, v[2] / norm], directionality };
}

/**
 * SH of the environment mirrored left-right (x -> -x). Apply exactly once when the UI shows a
 * horizontally mirrored (selfie) video, so the light stays on the same side as in the picture.
 */
export function mirrorX(sh: SH9): SH9 {
  return MIRROR_X_SIGNS.map((s, i) => {
    const c = coeff(sh, i);
    return [s * c[0], s * c[1], s * c[2]] as const;
  });
}

/** Build SH9 from the model output tensor data, laid out [1, 9, 3] row-major. */
export function shFromFlat(data: ArrayLike<number>): SH9 {
  if (data.length !== SH_COEFFS * 3) {
    throw new RangeError(`expected ${SH_COEFFS * 3} values, got ${data.length}`);
  }
  const out: RGB[] = [];
  for (let i = 0; i < SH_COEFFS; i++) {
    out.push([data[3 * i] ?? 0, data[3 * i + 1] ?? 0, data[3 * i + 2] ?? 0]);
  }
  return out;
}

/**
 * Convert Lumina SH (incident radiance) to the coefficients a three.js LightProbe expects.
 *
 * Verified 2026-11-13 against three r187 by the Playwright parity test (web/e2e/parity.spec.ts):
 * three.js LightProbe coefficients are RADIANCE in this exact basis and order; the shader applies
 * the cosine convolution (shGetIrradianceAt) and MeshLambertMaterial divides by pi. So the
 * conversion is the identity: no pi, no band factors. If a three.js upgrade changes this, the
 * parity test fails and the fix belongs here and only here.
 */
export function toThreeSH(sh: SH9, target = new SphericalHarmonics3()): SphericalHarmonics3 {
  for (let i = 0; i < SH_COEFFS; i++) {
    const c = coeff(sh, i);
    const dst = target.coefficients[i];
    if (dst === undefined) throw new RangeError('SphericalHarmonics3 must have 9 coefficients');
    dst.set(c[0], c[1], c[2]);
  }
  return target;
}

/** Dominant direction as a three.js vector (for a DirectionalLight position), or null. */
export function dominantDirectionVector(sh: SH9): Vector3 | null {
  const { direction } = dominantDirection(sh);
  return direction === null ? null : new Vector3(direction[0], direction[1], direction[2]);
}
