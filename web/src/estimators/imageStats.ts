/**
 * Classical baseline: lighting from image statistics, no learning.
 *
 * - Ambient: the mean linear colour of the frame. A white matte ball in a uniform environment of
 *   radiance m shows exactly m, so with no directional part the ball has the frame's average
 *   brightness and colour cast.
 * - Direction: the luminance-weighted centroid of the brightest tenth of a coarse 16 x 12 grid.
 *   HEURISTIC: a bright region on the right of the picture is taken to mean light from the right,
 *   tilted up and toward the camera (most bright regions are lit surfaces, not the source itself,
 *   and the light reaching a desk object usually comes from above and in front).
 * - Strength: how much brighter that region is than the frame, mapped to a directional share of
 *   at most 60%.
 *
 * Confidence is deliberately low (at most 0.4): this is the baseline the network has to beat.
 */
import type { FrameSample } from '../capture/frame';
import { exposureQuality, exposureStats } from '../capture/preprocess';
import { makeEstimate, type LightingEstimate } from '../lighting/estimate';
import type { RGB, SH9, Vec3 } from '../lighting/sh';
import { addDirectional, luminance, uniformSH } from '../lighting/shBuild';
import { SRGB8_TO_LINEAR } from '../lighting/srgb';
import type { Estimator } from './types';

const GRID_W = 16;
const GRID_H = 12;
const MAX_SHARE = 0.6;
const MAX_CONFIDENCE = 0.4;

interface Cell {
  rgb: RGB;
  lum: number;
  /** Cell centre, x right and y up, both in [-1, 1]. */
  x: number;
  y: number;
}

function gridCells(f: FrameSample): Cell[] {
  const cw = f.width / GRID_W;
  const ch = f.height / GRID_H;
  const cells: Cell[] = [];
  for (let gy = 0; gy < GRID_H; gy++) {
    for (let gx = 0; gx < GRID_W; gx++) {
      const acc = [0, 0, 0];
      let n = 0;
      for (let y = Math.floor(gy * ch); y < Math.floor((gy + 1) * ch); y++) {
        for (let x = Math.floor(gx * cw); x < Math.floor((gx + 1) * cw); x++) {
          const k = (y * f.width + x) * 4;
          for (let c = 0; c < 3; c++)
            acc[c] = (acc[c] ?? 0) + (SRGB8_TO_LINEAR[f.rgba[k + c] ?? 0] ?? 0);
          n++;
        }
      }
      const rgb: RGB = [(acc[0] ?? 0) / n, (acc[1] ?? 0) / n, (acc[2] ?? 0) / n];
      const x = ((gx + 0.5) / GRID_W) * 2 - 1;
      const y = 1 - ((gy + 0.5) / GRID_H) * 2;
      cells.push({ rgb, lum: luminance(rgb), x, y });
    }
  }
  return cells;
}

function average(cells: Cell[]): RGB {
  const s = cells.reduce<[number, number, number]>(
    (a, c) => [a[0] + c.rgb[0], a[1] + c.rgb[1], a[2] + c.rgb[2]],
    [0, 0, 0],
  );
  const n = Math.max(cells.length, 1);
  return [s[0] / n, s[1] / n, s[2] / n];
}

/** The pure estimate, exported for tests. */
export function statsEstimate(frame: FrameSample): LightingEstimate {
  const cells = gridCells(frame);
  const mean = average(cells);
  const meanLum = luminance(mean);
  const sorted = [...cells].sort((a, b) => b.lum - a.lum);
  const bright = sorted.slice(0, Math.max(1, Math.round(cells.length / 10)));
  const brightRgb = average(bright);
  const brightLum = luminance(brightRgb);
  const quality = exposureQuality(exposureStats(frame.rgba));
  if (!(meanLum > 1e-5)) return makeEstimate(uniformSH([0, 0, 0]), 0);

  const share = MAX_SHARE * Math.min(Math.max((brightLum / meanLum - 1.2) / 3, 0), 1);
  const floor = bright[bright.length - 1]?.lum ?? 0;
  let wsum = 0;
  let cx = 0;
  let cy = 0;
  for (const c of bright) {
    const w = c.lum - floor + 1e-6;
    wsum += w;
    cx += w * c.x;
    cy += w * c.y;
  }
  cx /= wsum;
  cy /= wsum;
  const d: Vec3 = [0.8 * cx, 0.35 + 0.6 * cy, 0.6];
  const chroma: RGB = [
    brightRgb[0] / brightLum,
    brightRgb[1] / brightLum,
    brightRgb[2] / brightLum,
  ];
  const eLum = 2 * Math.PI * share * meanLum;
  const ambient: RGB = [mean[0] * (1 - share), mean[1] * (1 - share), mean[2] * (1 - share)];
  let sh: SH9 = uniformSH(ambient);
  if (share > 0) {
    sh = addDirectional(sh, d, [chroma[0] * eLum, chroma[1] * eLum, chroma[2] * eLum], 0.3);
  }
  return makeEstimate(sh, MAX_CONFIDENCE * quality);
}

export class ImageStatsEstimator implements Estimator {
  readonly id = 'stats';
  readonly displayFrame = false;

  estimate(frame: FrameSample | null): Promise<LightingEstimate> {
    if (frame === null) return Promise.reject(new Error('no frame yet'));
    return Promise.resolve(statsEstimate(frame));
  }

  dispose(): void {
    // Nothing to release.
  }
}
