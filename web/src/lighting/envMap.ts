/**
 * A small equirectangular environment synthesised from SH9, for the reflections of glossy and
 * metal materials. HEURISTIC, stated plainly: SH9 holds only the blurry, low-frequency part of the
 * light, so the reflection of a chrome ball shows soft gradients, not the room. Optionally the
 * camera frame is pasted onto the forward hemisphere (the part of the world the camera sees), which
 * gives real detail at the rim of a chrome ball where the forward view is reflected. Nothing behind
 * the camera is invented.
 *
 * Layout matches three.js equirect sampling (common.glsl equirectUv) for a DataTexture with
 * flipY = false: u = atan2(z, x) / (2 pi) + 0.5, v = asin(y) / pi + 0.5, data row 0 at v = 0 (-Y).
 */
import { shBasis, type SH9, type Vec3 } from './sh';
import { luminance, meanRadiance } from './shBuild';
import { SRGB8_TO_LINEAR } from './srgb';

export interface CameraPatch {
  /** Un-mirrored sensor image, RGBA sRGB bytes, row 0 at the top. */
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  /** True when the display (and therefore the SH) is mirrored. */
  mirrored: boolean;
  /** Assumed vertical field of view of the camera, degrees. */
  vfovDeg: number;
  /** 0..1 blend of the camera picture over the SH radiance. */
  weight: number;
}

/** Direction at the centre of equirect texel (i, j). */
export function envDirection(i: number, j: number, width: number, height: number): Vec3 {
  const phi = ((i + 0.5) / width - 0.5) * 2 * Math.PI;
  const lat = ((j + 0.5) / height - 0.5) * Math.PI;
  const c = Math.cos(lat);
  return [c * Math.cos(phi), Math.sin(lat), c * Math.sin(phi)];
}

/** Texel coordinates (fractional) for a unit direction, inverse of envDirection. */
export function envTexel(d: Vec3, width: number, height: number): [number, number] {
  const u = Math.atan2(d[2], d[0]) / (2 * Math.PI) + 0.5;
  const v = Math.asin(Math.min(Math.max(d[1], -1), 1)) / Math.PI + 0.5;
  return [u * width - 0.5, v * height - 0.5];
}

/** Radiance of the SH environment in direction d (may ring below 0; callers clamp). */
export function radianceAt(sh: SH9, d: Vec3): [number, number, number] {
  const b = shBasis(d);
  const out: [number, number, number] = [0, 0, 0];
  sh.forEach((c, i) => {
    const w = b[i] ?? 0;
    out[0] += w * c[0];
    out[1] += w * c[1];
    out[2] += w * c[2];
  });
  return out;
}

interface Sampler {
  sample: (d: Vec3) => { rgb: [number, number, number]; w: number } | null;
}

function cameraSampler(cam: CameraPatch, sh: SH9): Sampler {
  const tanV = Math.tan(((cam.vfovDeg / 2) * Math.PI) / 180);
  const tanH = (tanV * cam.width) / cam.height;
  let sum = 0;
  for (let k = 0; k < cam.rgba.length; k += 4) {
    sum += luminance([
      SRGB8_TO_LINEAR[cam.rgba[k] ?? 0] ?? 0,
      SRGB8_TO_LINEAR[cam.rgba[k + 1] ?? 0] ?? 0,
      SRGB8_TO_LINEAR[cam.rgba[k + 2] ?? 0] ?? 0,
    ]);
  }
  const camMean = sum / Math.max(cam.width * cam.height, 1);
  // Bring the picture to the brightness level of the estimate.
  const gain = Math.min(Math.max(luminance(meanRadiance(sh)) / Math.max(camMean, 1e-4), 0.1), 10);
  return {
    sample: (d) => {
      if (d[2] >= -1e-3) return null;
      const u = 0.5 + d[0] / -d[2] / (2 * tanH);
      const v = 0.5 - d[1] / -d[2] / (2 * tanV);
      if (u <= 0 || u >= 1 || v <= 0 || v >= 1) return null;
      const edge = Math.min(u, 1 - u, v, 1 - v);
      const w = cam.weight * Math.min(edge / 0.08, 1);
      const su = cam.mirrored ? 1 - u : u;
      const x = Math.min(Math.floor(su * cam.width), cam.width - 1);
      const y = Math.min(Math.floor(v * cam.height), cam.height - 1);
      const k = (y * cam.width + x) * 4;
      const lin = (b: number | undefined): number => (SRGB8_TO_LINEAR[b ?? 0] ?? 0) * gain;
      return { rgb: [lin(cam.rgba[k]), lin(cam.rgba[k + 1]), lin(cam.rgba[k + 2])], w };
    },
  };
}

/** RGBA float texels (alpha 1), width * height * 4, ready for a FloatType DataTexture. */
export function buildEnvironment(
  sh: SH9,
  width: number,
  height: number,
  camera: CameraPatch | null = null,
  out: Float32Array = new Float32Array(width * height * 4),
): Float32Array {
  const cam = camera !== null && camera.weight > 0 ? cameraSampler(camera, sh) : null;
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const d = envDirection(i, j, width, height);
      const rgb = radianceAt(sh, d);
      const s = cam?.sample(d) ?? null;
      const k = (j * width + i) * 4;
      for (let c = 0; c < 3; c++) {
        const base = Math.max(rgb[c] ?? 0, 0);
        out[k + c] = s === null ? base : base + s.w * ((s.rgb[c] ?? 0) - base);
      }
      out[k + 3] = 1;
    }
  }
  return out;
}
