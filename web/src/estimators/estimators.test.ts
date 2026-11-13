import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { RateGate, Rolling } from '../app/scheduler';
import type { FrameSample } from '../capture/frame';
import { CONTRACT_VERSION } from '../lighting/modelContract';
import { irradiance } from '../lighting/sh';
import { meanRadiance } from '../lighting/shBuild';
import { statsEstimate } from './imageStats';
import { readOnnxMetadata } from './onnxMetadata';
import { STUDIO_SH } from './studio';

function frame(pixel: (x: number, y: number) => [number, number, number]): FrameSample {
  const width = 128;
  const height = 96;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) rgba.set([...pixel(x, y), 255], (y * width + x) * 4);
  }
  return { rgba, width, height, time: 0 };
}

describe('image statistics estimator', () => {
  it('turns a flat grey frame into flat light of the frame brightness', () => {
    const e = statsEstimate(frame(() => [128, 128, 128]));
    expect(e.direction).toBeNull();
    // A white matte ball in uniform light shows the mean radiance: sRGB 128 is linear 0.2158.
    expect(meanRadiance(e.sh)[1]).toBeCloseTo(0.2158, 3);
    expect(e.confidence).toBeGreaterThan(0.3);
    expect(e.confidence).toBeLessThanOrEqual(0.4);
  });

  it('puts the light on the right and above when the bright region is upper right', () => {
    const e = statsEstimate(frame((x, y) => (x > 96 && y < 24 ? [255, 250, 240] : [60, 60, 60])));
    expect(e.direction?.[0] ?? 0).toBeGreaterThan(0.3);
    expect(e.direction?.[1] ?? 0).toBeGreaterThan(0.3);
    expect(e.directionality).toBeGreaterThan(0.05);
  });

  it('picks up a warm colour cast', () => {
    const sh = statsEstimate(frame(() => [200, 140, 90])).sh;
    const m = meanRadiance(sh);
    expect(m[0]).toBeGreaterThan(m[1]);
    expect(m[1]).toBeGreaterThan(m[2]);
  });

  it('has no confidence in a black frame', () => {
    expect(statsEstimate(frame(() => [0, 0, 0])).confidence).toBe(0);
  });
});

describe('studio rig', () => {
  it('lights a white matte ball brightly without clipping, key from upper left', () => {
    const peak = irradiance(STUDIO_SH, [-0.5 / 0.995, 0.7 / 0.995, 0.5 / 0.995]);
    const out = peak.map((v) => v / Math.PI);
    expect(Math.max(...out)).toBeLessThan(1);
    expect(Math.max(...out)).toBeGreaterThan(0.6);
  });
});

describe('readOnnxMetadata', () => {
  it('reads lumina.contract and lumina.placeholder from the committed model', () => {
    const path = fileURLToPath(new URL('../../public/models/lumina-sh.onnx', import.meta.url));
    const meta = readOnnxMetadata(new Uint8Array(readFileSync(path)));
    expect(meta['lumina.contract']).toBe(CONTRACT_VERSION);
    expect(['true', 'false']).toContain(meta['lumina.placeholder']);
  });

  it('decodes a hand-built ModelProto with skipped fields', () => {
    const enc = new TextEncoder();
    const str = (field: number, s: string): number[] => {
      const b = [...enc.encode(s)];
      return [(field << 3) | 2, b.length, ...b];
    };
    const entry = [...str(1, 'k'), ...str(2, 'v')];
    const bytes = Uint8Array.from([
      0x08,
      0x96,
      0x01, // field 1 varint 150 (ir_version)
      ...str(2, 'producer'),
      (14 << 3) | 2,
      entry.length,
      ...entry,
    ]);
    expect(readOnnxMetadata(bytes)).toEqual({ k: 'v' });
  });
});

describe('RateGate', () => {
  it('caps a fast estimator at the target rate', () => {
    const gate = new RateGate(12);
    let runs = 0;
    for (let t = 0; t < 10_000; t += 1000 / 60) {
      if (gate.tryStart(t)) {
        runs++;
        gate.finish(); // instant estimator
      }
    }
    expect(runs).toBeGreaterThanOrEqual(110);
    expect(runs).toBeLessThanOrEqual(121);
  });

  it('reaches the target rate when polled by a 10 ms timer', () => {
    const gate = new RateGate(12);
    let runs = 0;
    for (let t = 0; t < 10_000; t += 10) {
      if (gate.tryStart(t)) {
        runs++;
        gate.finish();
      }
    }
    expect(runs).toBeGreaterThanOrEqual(118);
    expect(runs).toBeLessThanOrEqual(126);
  });

  it('never overlaps runs: a 150 ms estimator gets at most one start per 150 ms', () => {
    const gate = new RateGate(12);
    let runs = 0;
    let doneAt = Number.POSITIVE_INFINITY;
    for (let t = 0; t < 3000; t += 1000 / 60) {
      if (t >= doneAt) {
        gate.finish();
        doneAt = Number.POSITIVE_INFINITY;
      }
      if (gate.tryStart(t)) {
        runs++;
        doneAt = t + 150;
      }
    }
    expect(runs).toBeLessThanOrEqual(3000 / 150 + 1);
    expect(runs).toBeGreaterThan(15);
  });
});

describe('Rolling', () => {
  it('reports median, p95 and rate over its window', () => {
    const r = new Rolling(100);
    for (let i = 1; i <= 100; i++) r.push(i, i * 100);
    expect(r.median).toBe(50);
    expect(r.quantile(0.95)).toBe(95);
    expect(r.rate()).toBeCloseTo(10, 9);
    r.push(1000, 10_100);
    expect(r.count).toBe(100);
  });

  it('measures the rate over the recent span only, not a slow start', () => {
    const r = new Rolling(200);
    for (let i = 0; i < 10; i++) r.push(1, i * 1000); // 1 per second for 10 s
    for (let i = 1; i <= 60; i++) r.push(1, 9000 + i * 50); // then 20 per second for 3 s
    expect(r.rate()).toBeCloseTo(20, 6);
  });
});
