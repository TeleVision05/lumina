/** IEC 61966-2-1 sRGB transfer functions (the same ones docs/CONVENTIONS.md uses). */

/** sRGB-encoded value in [0, 1] to linear. */
export function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

/** Linear value to sRGB-encoded, clamped to [0, 1]. */
export function linearToSrgb(v: number): number {
  const x = Math.min(Math.max(v, 0), 1);
  return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
}

/** Linear value to an 8-bit sRGB level, rounded like the Python reference (round half up). */
export function linearToSrgb8(v: number): number {
  return Math.floor(linearToSrgb(v) * 255 + 0.5);
}

/** Lookup table from 8-bit sRGB levels to linear values. */
export const SRGB8_TO_LINEAR: Float32Array = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) t[i] = srgbToLinear(i / 255);
  return t;
})();
