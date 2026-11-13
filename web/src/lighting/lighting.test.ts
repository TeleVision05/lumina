import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import { describe, expect, it } from 'vitest';
import { colorTemperature } from './colorTemperature';
import { buildEnvironment, envDirection, envTexel, radianceAt } from './envMap';
import { forDisplay, makeEstimate } from './estimate';
import { renderProbe } from './probeImage';
import { dominantDirection, type SH9 } from './sh';
import { addDirectional, meanRadiance, uniformSH } from './shBuild';
import { linearToSrgb8, srgbToLinear } from './srgb';
import { environmentByName, parseShVectors } from './vectors';

const shared = (name: string): string =>
  fileURLToPath(new URL(`../../../shared/${name}`, import.meta.url));
const vectors = parseShVectors(readFileSync(shared('sh_vectors.json'), 'utf8'));
const lobeX = environmentByName(vectors, 'lobe_pos_x').sh;

describe('forDisplay (the one place an estimate is mirrored)', () => {
  it('leaves an un-mirrored display alone', () => {
    const e = makeEstimate(lobeX, 0.8);
    expect(forDisplay(e, false)).toBe(e);
  });

  it('moves a light on the right of the sensor image to the left of a mirrored display', () => {
    const e = forDisplay(makeEstimate(lobeX, 0.8), true);
    expect(e.direction?.[0]).toBeCloseTo(-1, 9);
    expect(e.confidence).toBe(0.8);
  });

  it('is its own inverse (mirroring twice is the identity)', () => {
    const once = forDisplay(makeEstimate(lobeX, 1), true);
    const twice = forDisplay(once, true);
    expect(twice.sh.flat()).toEqual(lobeX.flat().map((v) => v + 0));
  });
});

describe('renderProbe (readout grey ball)', () => {
  it('matches shared/sphere_reference.png inside the comparison disc', () => {
    const ref = vectors.sphere_reference;
    const png = PNG.sync.read(readFileSync(shared(ref.file)));
    const env = environmentByName(vectors, ref.environment);
    const img = renderProbe(env.sh, ref.size, ref.albedo);
    let sum = 0;
    let max = 0;
    let n = 0;
    for (let row = 0; row < ref.size; row++) {
      for (let col = 0; col < ref.size; col++) {
        const x = ((col + 0.5) / ref.size) * 2 - 1;
        const y = 1 - ((row + 0.5) / ref.size) * 2;
        if (x * x + y * y >= ref.compare_mask_radius ** 2) continue;
        n++;
        for (let c = 0; c < 3; c++) {
          const k = (row * ref.size + col) * 4 + c;
          const d = Math.abs((img[k] ?? 0) - (png.data[k] ?? 0));
          sum += d;
          max = Math.max(max, d);
        }
      }
    }
    console.log(
      `renderProbe vs reference: mean ${(sum / (3 * n)).toFixed(4)} max ${max} (${n} px)`,
    );
    expect(sum / (3 * n)).toBeLessThan(0.05);
    expect(max).toBeLessThanOrEqual(1);
  });
});

describe('colorTemperature', () => {
  it('reads linear white (D65) as about 6,500 K', () => {
    expect(colorTemperature([1, 1, 1])).toBeGreaterThan(6400);
    expect(colorTemperature([1, 1, 1])).toBeLessThan(6600);
  });

  it('orders warm < neutral < cool and returns null for black', () => {
    // Linear sRGB of a tungsten-like source; McCamy puts (1, 0.62, 0.3) at about 3,700 K.
    const warm = colorTemperature([1, 0.5, 0.15]) ?? 0;
    const cool = colorTemperature([0.75, 0.88, 1.2]) ?? 0;
    expect(warm).toBeLessThan(3200);
    expect(cool).toBeGreaterThan(8000);
    expect(colorTemperature([0, 0, 0])).toBeNull();
  });
});

describe('equirect environment', () => {
  it('maps texels to directions and back (three.js equirectUv convention)', () => {
    const [w, h] = [64, 32];
    for (const [i, j] of [
      [0, 0],
      [17, 5],
      [40, 30],
      [63, 31],
    ] as const) {
      const d = envDirection(i, j, w, h);
      const [u, v] = envTexel(d, w, h);
      expect(u).toBeCloseTo(i, 9);
      expect(v).toBeCloseTo(j, 9);
    }
    // Centre column looks along +X, u = 0.25 along -Z, top rows (high v) look up.
    expect(envDirection(31.5, 15.5, w, h)[0]).toBeCloseTo(1, 9);
    expect(envDirection(15.5, 15.5, w, h)[2]).toBeCloseTo(-1, 9);
    expect(envDirection(0, h - 1, w, h)[1]).toBeGreaterThan(0.99);
  });

  it('puts the brightest texel of a +X lobe at the centre column', () => {
    const data = buildEnvironment(lobeX, 64, 32);
    let best = -1;
    let at = 0;
    for (let k = 0; k < 64 * 32; k++) {
      if ((data[4 * k] ?? 0) > best) [best, at] = [data[4 * k] ?? 0, k];
    }
    expect(Math.abs((at % 64) - 31.5)).toBeLessThanOrEqual(0.5);
  });

  it('is never negative and equals the SH radiance without a camera patch', () => {
    const data = buildEnvironment(lobeX, 32, 16);
    expect(Math.min(...data)).toBeGreaterThanOrEqual(0);
    const d = envDirection(3, 4, 32, 16);
    expect(data[(4 * 32 + 3) * 4]).toBeCloseTo(Math.max(radianceAt(lobeX, d)[0], 0), 5);
  });

  it('pastes the camera picture on the forward hemisphere only, flipped when mirrored', () => {
    // 4x3 picture: left half bright red, right half black.
    const rgba = new Uint8ClampedArray(4 * 3 * 4);
    for (let k = 0; k < 12; k++) if (k % 4 < 2) rgba.set([255, 0, 0, 255], 4 * k);
    const sh = uniformSH([0.5, 0.5, 0.5]);
    const cam = { rgba, width: 4, height: 3, mirrored: false, vfovDeg: 60, weight: 1 };
    const [w, h] = [128, 64];
    const at = (data: Float32Array, d: [number, number, number]): number => {
      const [u, v] = envTexel(d, w, h);
      return data[(Math.round(v) * w + Math.round(u)) * 4] ?? 0;
    };
    const plain = buildEnvironment(sh, w, h, cam);
    const mirrored = buildEnvironment(sh, w, h, { ...cam, mirrored: true });
    const leftAhead: [number, number, number] = [-0.25, 0, -1];
    const rightAhead: [number, number, number] = [0.25, 0, -1];
    expect(at(plain, leftAhead)).toBeGreaterThan(at(plain, rightAhead));
    expect(at(mirrored, rightAhead)).toBeGreaterThan(at(mirrored, leftAhead));
    // Behind the camera nothing is pasted: radiance stays the uniform 0.5.
    expect(at(plain, [0, 0, 1])).toBeCloseTo(meanRadiance(sh)[0], 3);
  });
});

describe('sRGB and SH builders', () => {
  it('round-trips the transfer function and quantises like the Python reference', () => {
    for (const v of [0, 0.002, 0.04, 0.2, 0.5, 1]) {
      expect(srgbToLinear(linearToSrgb8(v) / 255)).toBeCloseTo(v, 2);
    }
    expect(linearToSrgb8(1)).toBe(255);
    expect(linearToSrgb8(0.7231)).toBe(221); // CONVENTIONS.md section 9 brightest pixel.
  });

  it('builds a directional light whose dominant direction is the input direction', () => {
    const sh: SH9 = addDirectional(uniformSH([0.1, 0.1, 0.1]), [1, 2, -2], [2, 2, 2]);
    const d = dominantDirection(sh).direction ?? [0, 0, 0];
    expect(d[0]).toBeCloseTo(1 / 3, 9);
    expect(d[1]).toBeCloseTo(2 / 3, 9);
    expect(d[2]).toBeCloseTo(-2 / 3, 9);
  });

  it('gives a uniform environment the radiance it was built from', () => {
    expect(meanRadiance(uniformSH([0.25, 0.5, 1]))).toEqual([0.25, 0.5, 1].map((v) => v));
  });
});
