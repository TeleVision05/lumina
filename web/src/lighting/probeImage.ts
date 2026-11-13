/**
 * CPU render of a white Lambertian sphere under an SH estimate: the "grey ball" in the readout
 * card. Same camera and maths as shared/sphere_reference.png (orthographic, looking down -Z, pixel
 * = srgb(albedo * E(n) / pi)), so a unit test can hold it to the Python reference.
 */
import { irradiance, type SH9 } from './sh';
import { linearToSrgb8 } from './srgb';

/** RGBA bytes, row 0 at the top. Outside the disc alpha is 0; the rim is anti-aliased. */
export function renderProbe(sh: SH9, size: number, albedo = 1): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(new ArrayBuffer(size * size * 4));
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const x = ((col + 0.5) / size) * 2 - 1;
      const y = 1 - ((row + 0.5) / size) * 2;
      const r2 = x * x + y * y;
      const k = (row * size + col) * 4;
      const coverage = Math.min(Math.max((1 - Math.sqrt(r2)) * size * 0.5 + 0.5, 0), 1);
      if (coverage <= 0) continue;
      const z = Math.sqrt(Math.max(1 - r2, 0));
      const e = irradiance(sh, [x, y, z]);
      out[k] = linearToSrgb8((albedo * e[0]) / Math.PI);
      out[k + 1] = linearToSrgb8((albedo * e[1]) / Math.PI);
      out[k + 2] = linearToSrgb8((albedo * e[2]) / Math.PI);
      out[k + 3] = Math.round(coverage * 255);
    }
  }
  return out;
}
