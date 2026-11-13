import { describe, expect, it } from 'vitest';
import { INPUT_SHAPE } from '../lighting/modelContract';
import { classifyCaptureError, PROBLEM_COPY, preflightProblem } from './captureErrors';
import {
  boxDownsample,
  cropRect43,
  exposureQuality,
  exposureStats,
  MODEL_H,
  MODEL_W,
  OVERSAMPLE,
  rgbaToNCHW,
} from './preprocess';
import { shouldMirror } from './source';

describe('cropRect43 (centre crop to 4:3)', () => {
  it.each([
    // width, height -> sx, sy, sw, sh (hand computed)
    [1280, 720, { sx: 160, sy: 0, sw: 960, sh: 720 }],
    [640, 480, { sx: 0, sy: 0, sw: 640, sh: 480 }],
    [480, 640, { sx: 0, sy: 140, sw: 480, sh: 360 }],
    [1920, 1080, { sx: 240, sy: 0, sw: 1440, sh: 1080 }],
    [720, 1280, { sx: 0, sy: 370, sw: 720, sh: 540 }],
  ] as const)('%i x %i', (w, h, want) => {
    expect(cropRect43(w, h)).toEqual(want);
  });

  it('rejects an empty frame', () => {
    expect(() => cropRect43(0, 480)).toThrow(RangeError);
  });
});

describe('boxDownsample', () => {
  it('averages exact 2x2 blocks (hand computed)', () => {
    // 4x2 image; block A = pixels (0,0),(1,0),(0,1),(1,1); block B = the right half.
    const px = [
      [0, 10, 20, 255],
      [4, 14, 24, 255],
      [100, 0, 0, 255],
      [200, 0, 0, 255],
      [8, 18, 28, 255],
      [12, 22, 32, 255],
      [0, 100, 0, 255],
      [0, 200, 1, 255],
    ];
    const out = boxDownsample(Uint8ClampedArray.from(px.flat()), 4, 2, 2);
    expect(Array.from(out)).toEqual([6, 16, 26, 255, 75, 75, 0, 255]);
  });

  it('rejects sizes that do not divide', () => {
    expect(() => boxDownsample(new Uint8ClampedArray(3 * 2 * 4), 3, 2, 2)).toThrow(RangeError);
  });
});

describe('rgbaToNCHW (model input layout)', () => {
  it('writes R, G, B planes in row-major order with values byte / 255, ignoring alpha', () => {
    // 2x2: (0,0) = 255,0,0 ; (1,0) = 0,255,0 ; (0,1) = 0,0,255 ; (1,1) = 51,102,204
    const rgba = Uint8ClampedArray.from([
      255, 0, 0, 7, 0, 255, 0, 7, 0, 0, 255, 7, 51, 102, 204, 7,
    ]);
    const t = rgbaToNCHW(rgba, 2, 2);
    const want = Float32Array.from([1, 0, 0, 0.2, 0, 1, 0, 0.4, 0, 0, 1, 0.8]);
    expect(Array.from(t)).toEqual(Array.from(want));
  });

  it('matches the contract size and keeps sRGB values (no linearisation)', () => {
    expect([1, 3, MODEL_H, MODEL_W]).toEqual([...INPUT_SHAPE]);
    const rgba = new Uint8ClampedArray(MODEL_W * MODEL_H * 4).fill(128);
    const t = rgbaToNCHW(rgba, MODEL_W, MODEL_H);
    expect(t.length).toBe(3 * MODEL_W * MODEL_H);
    expect(t[0]).toBeCloseTo(128 / 255, 7); // sRGB-encoded, so 0.502, not linear 0.216
    expect(Math.min(...t)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...t)).toBeLessThanOrEqual(1);
  });

  it('keeps left/right and top/bottom through the whole crop + downsample + pack chain', () => {
    // Oversampled 256x192 frame: left half red, right half blue, top row band green.
    const w = MODEL_W * OVERSAMPLE;
    const h = MODEL_H * OVERSAMPLE;
    const big = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const c = y < 2 ? [0, 255, 0] : x < w / 2 ? [255, 0, 0] : [0, 0, 255];
        big.set([...c, 255], (y * w + x) * 4);
      }
    }
    const t = rgbaToNCHW(boxDownsample(big, w, h, OVERSAMPLE), MODEL_W, MODEL_H);
    const plane = MODEL_W * MODEL_H;
    const at = (c: number, x: number, y: number): number => t[c * plane + y * MODEL_W + x] ?? -1;
    expect(at(0, 5, 50)).toBe(1); // left is red
    expect(at(2, 5, 50)).toBe(0);
    expect(at(2, MODEL_W - 5, 50)).toBe(1); // right is blue
    expect(at(1, 60, 0)).toBe(1); // top row green
    expect(at(1, 60, MODEL_H - 1)).toBe(0);
  });
});

describe('exposure quality', () => {
  it('trusts a mid-grey frame and distrusts black or blown-out frames', () => {
    const fill = (v: number): Uint8ClampedArray => new Uint8ClampedArray(64 * 4).fill(v);
    expect(exposureQuality(exposureStats(fill(128)))).toBeGreaterThan(0.95);
    expect(exposureQuality(exposureStats(fill(0)))).toBe(0);
    expect(exposureQuality(exposureStats(fill(255)))).toBe(0);
  });
});

describe('mirroring rule', () => {
  it('mirrors front and unknown-facing cameras only', () => {
    expect(shouldMirror('camera', 'user')).toBe(true);
    expect(shouldMirror('camera', undefined)).toBe(true);
    expect(shouldMirror('camera', 'environment')).toBe(false);
    expect(shouldMirror('sample', undefined)).toBe(false);
    expect(shouldMirror('file', 'user')).toBe(false);
  });
});

describe('camera problems in plain language', () => {
  const named = (name: string): Error => Object.assign(new Error('raw'), { name });

  it.each([
    ['NotAllowedError', 'denied'],
    ['SecurityError', 'denied'],
    ['NotFoundError', 'not-found'],
    ['OverconstrainedError', 'not-found'],
    ['NotReadableError', 'in-use'],
    ['AbortError', 'in-use'],
    ['TypeError', 'unsupported'],
    ['WeirdError', 'unknown'],
  ] as const)('%s -> %s', (name, problem) => {
    expect(classifyCaptureError(named(name))).toBe(problem);
  });

  it('spots an insecure page or a missing API before asking', () => {
    expect(preflightProblem({ isSecureContext: false, hasGetUserMedia: true })).toBe('insecure');
    expect(preflightProblem({ isSecureContext: true, hasGetUserMedia: false })).toBe('unsupported');
    expect(preflightProblem({ isSecureContext: true, hasGetUserMedia: true })).toBeNull();
  });

  it('never shows error names or codes to the visitor', () => {
    for (const copy of Object.values(PROBLEM_COPY)) {
      expect(`${copy.title} ${copy.body}`).not.toMatch(/Error|DOMException|getUserMedia|\d{3}/);
    }
  });
});
