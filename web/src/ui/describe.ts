/** Plain-language labels for the light reading. Pure functions, tested in ui.test.ts. */
import { CCT_MAX, CCT_MIN } from '../lighting/colorTemperature';
import type { Vec3 } from '../lighting/sh';

/**
 * Where the light comes from, as seen on screen (display frame: x right, y up, z toward the
 * viewer). "Camera side" means between the camera and the object; it lights the visible face.
 */
export function describeDirection(d: Vec3 | null): string {
  if (d === null) return 'Even light, no main direction';
  const [x, y, z] = d;
  const vert = y > 0.45 ? 'upper' : y < -0.35 ? 'lower' : '';
  const side = x > 0.35 ? 'right' : x < -0.35 ? 'left' : '';
  let where: string;
  if (vert !== '' && side !== '') where = `the ${vert} ${side}`;
  else if (vert === 'upper') where = 'above';
  else if (vert === 'lower') where = 'below';
  else if (side !== '') where = `the ${side}`;
  else where = z >= 0 ? 'straight ahead' : 'straight behind';
  const depth = z > 0.35 ? ', camera side' : z < -0.35 ? ', behind the object' : '';
  if (where === 'straight ahead' || where === 'straight behind') {
    return z >= 0 ? 'Light from the camera' : 'Light from behind the object';
  }
  return `Light from ${where}${depth}`;
}

export function describeQuality(directionality: number): string {
  const pct = Math.round(Math.min(Math.max(directionality, 0), 1) * 100);
  if (directionality < 0.12) return `Flat, even light (${pct}% directional)`;
  if (directionality < 0.4) return `Soft light (${pct}% directional)`;
  return `Directional light (${pct}% directional)`;
}

/** Position of a colour temperature on the warm-to-cool scale, 0..1 (log scale, 2,000-10,000 K). */
export function kelvinPosition(k: number): number {
  const lo = Math.log(2000);
  const hi = Math.log(10000);
  const v = Math.log(Math.min(Math.max(k, CCT_MIN), CCT_MAX));
  return Math.min(Math.max((v - lo) / (hi - lo), 0), 1);
}

/** "4,300 K", rounded to 100 K (the estimate is not more precise than that). */
export function formatKelvin(k: number | null): string {
  if (k === null) return '–';
  return `${(Math.round(k / 100) * 100).toLocaleString('en-US')} K`;
}

export function formatMs(ms: number | null): string {
  if (ms === null) return '–';
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}
