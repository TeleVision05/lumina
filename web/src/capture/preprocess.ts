/**
 * Frame preprocessing for the model contract (docs/CONVENTIONS.md, section 7): centre-crop the
 * un-mirrored camera frame to 4:3, resize to 128 x 96, and pack as NCHW float32 RGB holding the
 * sRGB-encoded values in [0, 1]. Normalisation lives inside the ONNX graph, not here.
 *
 * Resizing is done in two steps: the browser scales the crop to twice the model size (drawImage),
 * then boxDownsample averages exact 2 x 2 blocks. The second step is deterministic and tested.
 */
import { INPUT_SHAPE } from '../lighting/modelContract';
import { SRGB8_TO_LINEAR } from '../lighting/srgb';

export const MODEL_H = INPUT_SHAPE[2];
export const MODEL_W = INPUT_SHAPE[3];
/** drawImage target is this many times the model size in each axis. */
export const OVERSAMPLE = 2;

export interface Rect {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

/** Largest centred 4:3 rectangle inside a width x height frame, in whole pixels. */
export function cropRect43(width: number, height: number): Rect {
  if (!(width > 0 && height > 0)) throw new RangeError('frame has no pixels');
  if (width * 3 >= height * 4) {
    const sw = Math.round((height * 4) / 3);
    return { sx: Math.floor((width - sw) / 2), sy: 0, sw, sh: height };
  }
  const sh = Math.round((width * 3) / 4);
  return { sx: 0, sy: Math.floor((height - sh) / 2), sw: width, sh };
}

/** Average non-overlapping factor x factor blocks of an RGBA image (alpha averaged too). */
export function boxDownsample(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  factor: number,
): Uint8ClampedArray {
  if (width % factor !== 0 || height % factor !== 0) {
    throw new RangeError(`${width}x${height} is not divisible by ${factor}`);
  }
  const w = width / factor;
  const h = height / factor;
  const out = new Uint8ClampedArray(w * h * 4);
  const n = factor * factor;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let dy = 0; dy < factor; dy++) {
          const row = (y * factor + dy) * width;
          for (let dx = 0; dx < factor; dx++) sum += rgba[(row + x * factor + dx) * 4 + c] ?? 0;
        }
        out[(y * w + x) * 4 + c] = Math.round(sum / n);
      }
    }
  }
  return out;
}

/** RGBA bytes (row 0 at the top) to NCHW float32: plane R, then G, then B, values byte / 255. */
export function rgbaToNCHW(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  out: Float32Array = new Float32Array(3 * width * height),
): Float32Array {
  const plane = width * height;
  if (rgba.length !== plane * 4) throw new RangeError('RGBA length does not match the size');
  if (out.length !== 3 * plane) throw new RangeError('output length does not match the size');
  for (let p = 0; p < plane; p++) {
    out[p] = (rgba[4 * p] ?? 0) / 255;
    out[plane + p] = (rgba[4 * p + 1] ?? 0) / 255;
    out[2 * plane + p] = (rgba[4 * p + 2] ?? 0) / 255;
  }
  return out;
}

export interface ExposureStats {
  /** Mean linear luminance. */
  meanLuminance: number;
  /** Fraction of pixels with any channel at 250 or above. */
  clipped: number;
  /** Fraction of pixels with every channel at 8 or below. */
  crushed: number;
}

export function exposureStats(rgba: Uint8ClampedArray): ExposureStats {
  const n = rgba.length / 4;
  let lum = 0;
  let clipped = 0;
  let crushed = 0;
  for (let k = 0; k < rgba.length; k += 4) {
    const r = rgba[k] ?? 0;
    const g = rgba[k + 1] ?? 0;
    const b = rgba[k + 2] ?? 0;
    lum +=
      0.2126 * (SRGB8_TO_LINEAR[r] ?? 0) +
      0.7152 * (SRGB8_TO_LINEAR[g] ?? 0) +
      0.0722 * (SRGB8_TO_LINEAR[b] ?? 0);
    if (r >= 250 || g >= 250 || b >= 250) clipped++;
    if (r <= 8 && g <= 8 && b <= 8) crushed++;
  }
  return { meanLuminance: lum / n, clipped: clipped / n, crushed: crushed / n };
}

/**
 * 0..1 trust in any image-based estimate given how well exposed the frame is: a frame that is
 * mostly black or mostly blown out carries little information about the light.
 */
export function exposureQuality(s: ExposureStats): number {
  const bad = Math.max(s.clipped, s.crushed);
  const dark = Math.min(s.meanLuminance / 0.02, 1);
  return Math.max(0, Math.min(1, (1 - bad) * dark));
}
