/** Helpers for truth.spec.ts: the sample clip as a fake camera, reference frames and circular maths. */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { FIXTURE_DIR, SAMPLES_DIR } from './paths';

export const CLIP_NAME = process.env.LUMINA_CLIP ?? 'rotate-environment';
export const CLIP_MP4 = `${SAMPLES_DIR}${CLIP_NAME}.mp4`;
export const TRUTH_Y4M = `${FIXTURE_DIR}truth-${CLIP_NAME}.y4m`;
export const MODEL_FILE = `${SAMPLES_DIR}../models/lumina-sh.onnx`;
export const MODEL_W = 128;
export const MODEL_H = 96;
const FRAME_BYTES = MODEL_W * MODEL_H * 3;

export interface TruthFrame {
  dominant_direction: number[];
  directionality: number;
  model_sh: number[][];
}

export function readTruth(): { fps: number; frames: TruthFrame[] } {
  const j = JSON.parse(readFileSync(`${SAMPLES_DIR}${CLIP_NAME}.json`, 'utf8')) as {
    fps: number;
    per_frame: TruthFrame[];
  };
  return { fps: j.fps, frames: j.per_frame };
}

function ffmpeg(args: string[], maxBuffer = 1 << 28): Buffer {
  const r = spawnSync('ffmpeg', ['-v', 'error', '-nostdin', ...args], { maxBuffer });
  if (r.status !== 0) {
    throw new Error(`ffmpeg failed (is it installed?): ${String(r.stderr)}`);
  }
  return r.stdout;
}

/** Convert the sample clip to the Y4M that Chromium's fake camera reads (idempotent). */
export function ensureTruthY4m(): void {
  mkdirSync(FIXTURE_DIR, { recursive: true });
  if (existsSync(TRUTH_Y4M)) return;
  ffmpeg(['-y', '-i', CLIP_MP4, '-vf', 'scale=320:240', '-pix_fmt', 'yuv420p', TRUTH_Y4M]);
}

/** Every clip frame at model size (128 x 96, RGB bytes), the reference for frame matching. */
export function referenceFrames(): Uint8Array[] {
  const raw = ffmpeg([
    '-i',
    CLIP_MP4,
    '-vf',
    `scale=${MODEL_W}:${MODEL_H}:flags=area`,
    '-pix_fmt',
    'rgb24',
    '-f',
    'rawvideo',
    '-',
  ]);
  const out: Uint8Array[] = [];
  for (let o = 0; o + FRAME_BYTES <= raw.length; o += FRAME_BYTES) {
    out.push(new Uint8Array(raw.subarray(o, o + FRAME_BYTES)));
  }
  return out;
}

/** Mean absolute byte difference between a browser RGBA frame and a reference RGB frame. */
export function meanAbsDiff(rgba: readonly number[], rgb: Uint8Array): number {
  let sum = 0;
  for (let p = 0; p < MODEL_W * MODEL_H; p++) {
    for (let c = 0; c < 3; c++) sum += Math.abs((rgba[4 * p + c] ?? 0) - (rgb[3 * p + c] ?? 0));
  }
  return sum / FRAME_BYTES;
}

export function nearestFrame(
  rgba: readonly number[],
  refs: Uint8Array[],
): { index: number; diff: number } {
  let best = { index: -1, diff: Number.POSITIVE_INFINITY };
  refs.forEach((ref, index) => {
    const diff = meanAbsDiff(rgba, ref);
    if (diff < best.diff) best = { index, diff };
  });
  return best;
}

/** Azimuth about +Y in the camera frame: 0 straight ahead (-Z), positive toward +X (right). */
export const azimuth = (d: readonly number[]): number => Math.atan2(d[0] ?? 0, -(d[2] ?? 0));

/** Smallest signed angle a - b in (-pi, pi]. */
export function angleDiff(a: number, b: number): number {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}

export function angleBetweenDeg(a: readonly number[], b: readonly number[]): number {
  const dot = (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0) + (a[2] ?? 0) * (b[2] ?? 0);
  const n = Math.hypot(...a) * Math.hypot(...b);
  return (Math.acos(Math.min(1, Math.max(-1, dot / n))) * 180) / Math.PI;
}

/** Length of the mean unit vector of the angles (1 = all equal, 0 = spread evenly). */
export function resultant(angles: number[]): number {
  const c = angles.reduce((s, a) => s + Math.cos(a), 0) / angles.length;
  const s = angles.reduce((s2, a) => s2 + Math.sin(a), 0) / angles.length;
  return Math.hypot(c, s);
}

export const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? Number.NaN;
};
