/**
 * Truth test: the sample clip with known per-frame lighting plays as the fake camera in the built
 * app. Checks, against ground truth and against Python onnxruntime:
 *  1. browser preprocessing: the frame the network sees equals an independent ffmpeg resize;
 *  2. the network's output in the browser equals Python onnxruntime on the very same frames;
 *  3. the estimated light direction follows the true one as the environment turns 360 degrees,
 *     with a bounded lag, and the rendered probe ball's bright side follows the estimate.
 * Honest scope: the model is weak (see docs/EVAL.md), so the direction thresholds are deliberately
 * modest and only assert "follows", never "accurate".
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import { CHROMIUM_ARGS } from '../playwright.config';
import type { NeuralCapture } from '../src/app/debug';
import { dominantDirection, shFromFlat } from '../src/lighting/sh';
import { startCamera, state } from './helpers';
import { FIXTURE_DIR } from './paths';
import {
  angleBetweenDeg,
  angleDiff,
  azimuth,
  ensureTruthY4m,
  median,
  MODEL_FILE,
  nearestFrame,
  readTruth,
  referenceFrames,
  resultant,
  TRUTH_Y4M,
} from './truthFixtures';

ensureTruthY4m();
test.use({
  launchOptions: {
    args: CHROMIUM_ARGS.map((a) =>
      a.startsWith('--use-file-for-fake-video-capture=')
        ? `--use-file-for-fake-video-capture=${TRUTH_Y4M}`
        : a,
    ),
  },
});

interface Sample {
  capture: NeuralCapture;
  filtered: number[] | null;
  probeBrightX: number;
  mirrored: boolean;
}

/**
 * Record every new network result inside the page (no Playwright round trips, which would starve
 * the app under software rendering): the frame, the raw output, the filtered direction and the
 * x of the lit probe ball's brightness centroid (-1 left edge, 1 right edge).
 */
async function collect(page: Page, maxSeconds: number, minSamples: number): Promise<Sample[]> {
  return page.evaluate(
    async ({ ms, min }) => {
      const dbg = window.__lumina;
      if (dbg === undefined) throw new Error('debug handle missing');
      const probe = document.getElementById('probe') as HTMLCanvasElement;
      const probeX = (): number => {
        const px = probe.getContext('2d')?.getImageData(0, 0, probe.width, probe.height).data;
        if (px === undefined) return 0;
        let sum = 0;
        let wx = 0;
        for (let y = 0; y < probe.height; y++) {
          for (let x = 0; x < probe.width; x++) {
            const k = (y * probe.width + x) * 4;
            if ((px[k + 3] ?? 0) < 250) continue;
            const l = (px[k] ?? 0) + (px[k + 1] ?? 0) + (px[k + 2] ?? 0);
            sum += l;
            wx += l * (((x + 0.5) / probe.width) * 2 - 1);
          }
        }
        return sum > 0 ? wx / sum : 0;
      };
      const out: Sample[] = [];
      let last = -1;
      const start = performance.now();
      const end = start + ms;
      while (performance.now() < end && (out.length < min || performance.now() < start + 16_000)) {
        const s = dbg.state();
        const cap = dbg.lastNeural();
        if (cap !== null && s.inferenceCount !== last) {
          last = s.inferenceCount;
          out.push({
            capture: cap,
            filtered: s.estimate?.direction ?? null,
            probeBrightX: probeX(),
            mirrored: s.mirrored === true,
          });
        }
        await new Promise((r) => setTimeout(r, 20));
      }
      return out;
    },
    { ms: maxSeconds * 1000, min: minSamples },
  );
}

function ortReference(samples: Sample[]): number[][][] {
  mkdirSync(FIXTURE_DIR, { recursive: true });
  const inp = `${FIXTURE_DIR}ort-in.json`;
  const outp = `${FIXTURE_DIR}ort-out.json`;
  writeFileSync(
    inp,
    JSON.stringify({ model: MODEL_FILE, frames: samples.map((s) => s.capture.rgba) }),
  );
  const r = spawnSync(
    'uv',
    [
      'run',
      '--no-project',
      '--with',
      'onnxruntime==1.31.0',
      '--with',
      'numpy',
      'python',
      '-I',
      '../scripts/ort_reference.py',
      inp,
      outp,
    ],
    { encoding: 'utf8' },
  );
  if (r.status !== 0) throw new Error(`Python onnxruntime reference failed: ${r.stderr}`);
  return (JSON.parse(readFileSync(outp, 'utf8')) as { sh: number[][][] }).sh;
}

interface Ctx {
  samples: Sample[];
  /** Clip frame index each sample's network frame was matched to. */
  idx: number[];
  truth: ReturnType<typeof readTruth>;
}

const flatSh = (rows: number[][]): ReturnType<typeof shFromFlat> => shFromFlat(rows.flat());
const wrap = (i: number): number => ((i % 240) + 240) % 240;
/** Signed distance between two frame indices on the 240-frame loop. */
const idxDistance = (a: number, b: number): number => wrap(a - b + 120) - 120;
const maxAbsDiff = (a: number[], b: number[]): number =>
  Math.max(...a.map((v, k) => Math.abs(v - (b[k] ?? Number.NaN))));

/** 1. Each browser frame is a frame of the clip and close to an independent ffmpeg resize. */
function checkPreprocessing(c: Ctx, diffs: number[]): void {
  console.log(
    `TRUTH preprocessing: median |browser - ffmpeg| = ${median(diffs).toFixed(2)} levels, max ${Math.max(...diffs).toFixed(2)}`,
  );
  expect(median(diffs)).toBeLessThan(6);
  expect(Math.max(...c.idx) - Math.min(...c.idx)).toBeGreaterThan(200); // a full turn was seen
}

/** 2. The network's output in the browser equals Python onnxruntime on the same frames. */
function checkOrtParity(c: Ctx): void {
  const ort = ortReference(c.samples);
  const dev = c.samples.map((s, i) => maxAbsDiff(s.capture.sh.flat(), ort[i]?.flat() ?? []));
  const scale = Math.max(...ort.flat(2).map(Math.abs));
  console.log(
    `TRUTH ort parity: max |browser - python| = ${Math.max(...dev).toExponential(2)} (largest coefficient ${scale.toFixed(2)})`,
  );
  expect(Math.max(...dev) / scale).toBeLessThan(1e-3);
  const vsFile = c.samples.map(
    (s, i) =>
      maxAbsDiff(s.capture.sh.flat(), c.truth.frames[c.idx[i] ?? 0]?.model_sh.flat() ?? []) / scale,
  );
  console.log(
    `TRUTH vs clip-time python prediction: median relative deviation ${median(vsFile).toFixed(3)}`,
  );
}

/** 3. The estimated direction sweeps the same way as the true one. */
function checkDirection(c: Ctx): void {
  const pairs = c.samples.flatMap((s, i) => {
    const d = dominantDirection(flatSh(s.capture.sh)).direction;
    const t = c.truth.frames[c.idx[i] ?? 0];
    return d === null || t === undefined ? [] : [{ est: d, truth: t.dominant_direction }];
  });
  const err = pairs.map((p) => angleBetweenDeg(p.est, p.truth));
  const same = resultant(pairs.map((p) => angleDiff(azimuth(p.est), azimuth(p.truth))));
  const opposite = resultant(pairs.map((p) => angleDiff(azimuth(p.est), -azimuth(p.truth))));
  console.log(
    `TRUTH direction: median error ${median(err).toFixed(1)} deg; azimuth agreement R=${same.toFixed(2)} (same sweep) vs ${opposite.toFixed(2)} (reversed)`,
  );
  expect(same).toBeGreaterThan(0.3);
  expect(same).toBeGreaterThan(opposite + 0.1);
}

/** Mean |azimuth difference| between the filtered estimate and the raw output `lag` frames earlier. */
function lagError(c: Ctx, raw: (number | null)[], lag: number): number | null {
  const errs: number[] = [];
  c.samples.forEach((s, i) => {
    const f = s.filtered;
    if (f === null) return;
    const want = wrap((c.idx[i] ?? 0) - lag);
    const j = c.idx.findIndex((k) => Math.abs(idxDistance(k, want)) <= 3);
    const r = j < 0 ? null : raw[j];
    if (r === null || r === undefined) return;
    const x = s.mirrored ? -(f[0] ?? 0) : (f[0] ?? 0);
    errs.push(Math.abs(angleDiff(Math.atan2(x, -(f[2] ?? 0)), r)));
  });
  return errs.length > 8 ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
}

/**
 * 3b. Lag of the displayed (filtered) estimate behind the network's own estimate of the same
 * moment. The truth cannot give this: the model's bias of tens of degrees on a steady 1.5 deg per
 * frame sweep would read as tens of frames of lag.
 */
function checkLag(c: Ctx): void {
  const raw = c.samples.map((s) => {
    const d = dominantDirection(flatSh(s.capture.sh)).direction;
    return d === null ? null : azimuth(d);
  });
  const scores: { lag: number; err: number }[] = [];
  for (let lag = -6; lag <= 48; lag += 2) {
    const err = lagError(c, raw, lag);
    if (err !== null) scores.push({ lag, err });
  }
  expect(scores.length).toBeGreaterThan(5);
  const best = scores.reduce((a, b) => (b.err < a.err ? b : a));
  console.log(
    `TRUTH lag curve (frames:mean abs azimuth diff deg) ${scores.map((l) => `${l.lag}:${((l.err * 180) / Math.PI).toFixed(0)}`).join(' ')}`,
  );
  console.log(
    `TRUTH lag: the displayed estimate trails the network output by ${best.lag} clip frames (${((best.lag / c.truth.fps) * 1000).toFixed(0)} ms)`,
  );
  expect(best.lag).toBeGreaterThanOrEqual(0);
  expect(best.lag / c.truth.fps).toBeLessThan(1.5);
}

/** 3c. The rendered probe ball's bright side follows the filtered estimate and the truth. */
function checkProbe(c: Ctx): void {
  const est = c.samples.flatMap((s) =>
    s.filtered === null || Math.abs(s.filtered[0] ?? 0) < 0.25
      ? []
      : [Math.sign(s.filtered[0] ?? 0) === Math.sign(s.probeBrightX)],
  );
  const agreeEst = est.filter(Boolean).length / est.length;
  console.log(
    `TRUTH probe ball: bright side agrees with the estimate in ${(agreeEst * 100).toFixed(0)}% of ${est.length} samples`,
  );
  expect(agreeEst).toBeGreaterThan(0.9);
  const tru = c.samples.flatMap((s, i) => {
    const tx = c.truth.frames[c.idx[i] ?? 0]?.dominant_direction[0] ?? 0;
    return Math.abs(tx) < 0.4
      ? []
      : [Math.sign(s.mirrored ? -tx : tx) === Math.sign(s.probeBrightX)];
  });
  const agreeTruth = tru.filter(Boolean).length / tru.length;
  console.log(
    `TRUTH probe ball: bright side agrees with the true light side in ${(agreeTruth * 100).toFixed(0)}% of ${tru.length} samples`,
  );
  expect(agreeTruth).toBeGreaterThan(0.6);
}

test('estimate follows the true lighting on the rotating-environment clip', async ({ page }) => {
  test.setTimeout(180_000);
  const refs = referenceFrames();
  await startCamera(page);
  // Record for at least 16 s and until 150 results (cap 90 s): the direction statistics are noisy
  // and a slow runner gives few results per second, so collect more rather than loosen anything.
  const samples = await collect(page, 90, 150);
  // Local runs give 150+ samples; a slow 2-core CI runner gives 40 to 60 (24 ms per inference).
  expect(samples.length).toBeGreaterThan(25);
  expect((await state(page)).placeholder).toBe(false);
  const matches = samples.map((s) => nearestFrame(s.capture.rgba, refs));
  const c: Ctx = { samples, idx: matches.map((m) => m.index), truth: readTruth() };
  checkPreprocessing(
    c,
    matches.map((m) => m.diff),
  );
  checkOrtParity(c);
  checkDirection(c);
  checkLag(c);
  checkProbe(c);
});
