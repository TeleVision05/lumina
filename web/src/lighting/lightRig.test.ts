import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { deriveLightRig, shadowOpacity } from './lightRig';
import { irradiance, LUMA, mirrorX, type RGB, type SH9, type Vec3 } from './sh';
import { addDirectional, uniformSH } from './shBuild';
import { parseShVectors } from './vectors';

const vectorsPath = fileURLToPath(new URL('../../../shared/sh_vectors.json', import.meta.url));
const vectors = parseShVectors(readFileSync(vectorsPath, 'utf8'));
const lum = (c: RGB): number => LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2];

/** Irradiance the rig renders: probe SH plus the exact clamped-cosine directional light. */
function rigIrradiance(sh: SH9, n: Vec3): RGB {
  const rig = deriveLightRig(sh);
  const e = irradiance(rig.probe, n);
  if (rig.light === null) return e;
  const [dx, dy, dz] = rig.light.direction;
  const cos = Math.max(dx * n[0] + dy * n[1] + dz * n[2], 0);
  const l = rig.light.irradiance;
  return [e[0] + l[0] * cos, e[1] + l[1] * cos, e[2] + l[2] * cos];
}

describe('deriveLightRig on the shared analytic environments', () => {
  it.each(vectors.environments.map((e) => [e.name, e] as const))(
    '%s: probe + light reproduces the estimate irradiance within 10% of its peak',
    (_name, env) => {
      const peak = Math.max(...env.irradiance.map((e) => lum(e)));
      let worst = 0;
      vectors.normals.forEach((n) => {
        const want = irradiance(env.sh, n);
        const got = rigIrradiance(env.sh, n);
        for (let c = 0; c < 3; c++) {
          worst = Math.max(worst, Math.abs((got[c] ?? 0) - (want[c] ?? 0)) / peak);
        }
      });
      console.log(`rig energy error ${_name}: ${(worst * 100).toFixed(2)}% of peak`);
      expect(worst).toBeLessThan(0.1);
    },
  );

  it('gives no directional light for constant white', () => {
    const env = vectors.environments.find((e) => e.name === 'constant_white');
    const rig = deriveLightRig(env?.sh ?? []);
    expect(rig.light).toBeNull();
    expect(rig.share).toBe(0);
  });

  it('points the light along the lobe for lobe_pos_y and takes most of band 1', () => {
    const env = vectors.environments.find((e) => e.name === 'lobe_pos_y');
    const rig = deriveLightRig(env?.sh ?? []);
    expect(rig.light?.direction[1]).toBeCloseTo(1, 9);
    expect(rig.share).toBeGreaterThan(0.8);
    // Band 1 of the probe shrinks by the share.
    expect(rig.probe[1]?.[0] ?? 0).toBeCloseTo((env?.sh[1]?.[0] ?? 0) * (1 - rig.share), 9);
  });

  it('keeps the light colour of the warm lobe (red > green > blue)', () => {
    const env = vectors.environments.find((e) => e.name === 'lobe_oblique_warm');
    const l = deriveLightRig(env?.sh ?? []).light?.irradiance ?? [0, 0, 0];
    expect(l[0]).toBeGreaterThan(l[1]);
    expect(l[1]).toBeGreaterThan(l[2]);
  });

  it('commutes with mirroring: mirrored SH gives the mirrored light direction', () => {
    const env = vectors.environments.find((e) => e.name === 'lobe_oblique_warm');
    const a = deriveLightRig(env?.sh ?? []).light?.direction ?? [0, 0, 0];
    const b = deriveLightRig(mirrorX(env?.sh ?? [])).light?.direction ?? [0, 0, 0];
    expect(b[0]).toBeCloseTo(-a[0], 9);
    expect(b[1]).toBeCloseTo(a[1], 9);
    expect(b[2]).toBeCloseTo(a[2], 9);
  });

  it('never leaves a negative ambient term, even for implausible network output', () => {
    const bad: SH9 = [
      [0.1, 0.1, 0.1],
      [5, 5, 5],
      ...Array.from({ length: 7 }, (): RGB => [0, 0, 0]),
    ];
    const rig = deriveLightRig(bad);
    rig.probe[0]?.forEach((v) => {
      expect(v).toBeGreaterThanOrEqual(0);
    });
  });
});

describe('shadowOpacity', () => {
  const up: Vec3 = [0, 1, 0];
  const keyAbove = addDirectional(uniformSH([0.2, 0.2, 0.2]), [0.3, 0.9, 0.3], [3, 3, 3]);

  it('is zero in flat light', () => {
    const sh = uniformSH([1, 1, 1]);
    expect(shadowOpacity(sh, deriveLightRig(sh), up)).toBe(0);
  });

  it('is strong for a bright key light above a dim ambient, and at most 1', () => {
    const o = shadowOpacity(keyAbove, deriveLightRig(keyAbove), up);
    expect(o).toBeGreaterThan(0.5);
    expect(o).toBeLessThanOrEqual(1);
  });

  it('is zero when the light is below the ground', () => {
    const below = addDirectional(uniformSH([0.2, 0.2, 0.2]), [0, -1, 0], [3, 3, 3]);
    expect(shadowOpacity(below, deriveLightRig(below), up)).toBe(0);
  });

  it('fades as the ambient part grows', () => {
    const dim = addDirectional(uniformSH([1.5, 1.5, 1.5]), [0.3, 0.9, 0.3], [3, 3, 3]);
    expect(shadowOpacity(dim, deriveLightRig(dim), up)).toBeLessThan(
      shadowOpacity(keyAbove, deriveLightRig(keyAbove), up),
    );
  });
});
