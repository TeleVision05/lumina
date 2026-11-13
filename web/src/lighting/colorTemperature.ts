/**
 * Correlated colour temperature of a linear sRGB colour, for display only ("about 4,300 K").
 * Linear sRGB -> CIE XYZ (D65) -> xy chromaticity -> McCamy's cubic (1992). McCamy is accurate to a
 * few kelvin between about 2,800 K and 6,500 K and degrades outside it, which is fine for a label.
 */
import type { RGB } from './sh';

export const CCT_MIN = 1500;
export const CCT_MAX = 15000;

export function chromaticity(c: RGB): { x: number; y: number } | null {
  const [r, g, b] = c;
  const X = 0.4124 * r + 0.3576 * g + 0.1805 * b;
  const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const Z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
  const s = X + Y + Z;
  if (!(s > 1e-9)) return null;
  return { x: X / s, y: Y / s };
}

/** Kelvin, clamped to [CCT_MIN, CCT_MAX], or null for black or non-finite input. */
export function colorTemperature(c: RGB): number | null {
  const xy = chromaticity(c);
  if (xy === null) return null;
  const n = (xy.x - 0.332) / (0.1858 - xy.y);
  const cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33;
  if (!Number.isFinite(cct)) return null;
  return Math.min(Math.max(cct, CCT_MIN), CCT_MAX);
}
