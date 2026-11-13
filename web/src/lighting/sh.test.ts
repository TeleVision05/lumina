import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import {
  dominantDirection,
  irradiance,
  mirrorX,
  shBasis,
  shFromFlat,
  toThreeSH,
  type RGB,
  type SH9,
  type Vec3,
} from './sh';
import { parseShVectors } from './vectors';

const vectorsPath = fileURLToPath(new URL('../../../shared/sh_vectors.json', import.meta.url));
const vectors = parseShVectors(readFileSync(vectorsPath, 'utf8'));

function expectClose(actual: readonly number[], expected: readonly number[], tol: number): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((a, i) => {
    expect(Math.abs(a - (expected[i] ?? Number.NaN))).toBeLessThanOrEqual(tol);
  });
}

function evalSH(sh: SH9, d: Vec3): RGB {
  const b = shBasis(d);
  const out = [0, 1, 2].map((k) => sh.reduce((acc, c, i) => acc + (c[k] ?? 0) * (b[i] ?? 0), 0));
  return [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
}

describe('shBasis', () => {
  it('matches the vectors at every normal', () => {
    vectors.normals.forEach((n, i) => {
      expectClose(shBasis(n), vectors.basis_at_normals[i] ?? [], 1e-12);
    });
  });
});

describe.each(vectors.environments.map((e) => [e.name, e] as const))('%s', (_name, env) => {
  it('irradiance matches the vectors', () => {
    vectors.normals.forEach((n, i) => {
      expectClose(irradiance(env.sh, n), env.irradiance[i] ?? [], 1e-9);
    });
  });

  it('three.js SphericalHarmonics3.getIrradianceAt agrees after toThreeSH', () => {
    const three = toThreeSH(env.sh);
    vectors.normals.forEach((n) => {
      const e = three.getIrradianceAt(new Vector3(n[0], n[1], n[2]), new Vector3());
      // three.js rounds its constants to 6 digits; relative agreement ~1e-5.
      expectClose([e.x, e.y, e.z], irradiance(env.sh, n), 1e-5 * Math.max(1, env.sh[0]?.[0] ?? 1));
    });
  });

  it('dominant direction and directionality match the vectors', () => {
    const { direction, directionality } = dominantDirection(env.sh);
    expect(directionality).toBeCloseTo(env.directionality, 9);
    if (env.dominant_direction === null) expect(direction).toBeNull();
    else expectClose(direction ?? [], env.dominant_direction, 1e-9);
  });

  it('mirrorX matches the vectors and equals evaluating at (-x, y, z)', () => {
    const mirrored = mirrorX(env.sh);
    expectClose(mirrored.flat(), env.sh_mirrored_x.flat(), 1e-12);
    vectors.normals.forEach(([x, y, z]) => {
      expectClose(evalSH(mirrored, [x, y, z]), evalSH(env.sh, [-x, y, z]), 1e-12);
    });
  });
});

describe('mirroring a lobe on +X', () => {
  it('moves the dominant light to -X', () => {
    const lobe = vectors.environments.find((e) => e.name === 'lobe_pos_x');
    expect(lobe).toBeDefined();
    const { direction } = dominantDirection(mirrorX(lobe?.sh ?? []));
    expectClose(direction ?? [], [-1, 0, 0], 1e-9);
  });
});

describe('shFromFlat', () => {
  it('reads the [1, 9, 3] model layout as coefficient-major RGB', () => {
    const flat = Float32Array.from({ length: 27 }, (_, i) => i);
    const sh = shFromFlat(flat);
    expect(sh[0]).toEqual([0, 1, 2]);
    expect(sh[8]).toEqual([24, 25, 26]);
  });

  it('rejects the wrong length', () => {
    expect(() => shFromFlat(new Float32Array(26))).toThrow(RangeError);
  });
});
