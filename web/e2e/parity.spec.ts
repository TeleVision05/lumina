/**
 * three.js parity: a white Lambertian sphere lit by a LightProbe set through toThreeSH() must match
 * shared/sphere_reference.png, which Python rendered as albedo / pi * E(n) from radiance SH.
 * Two wrong conventions are rendered as controls and must NOT match, so the test discriminates.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';
import { BAND_FACTORS, BAND_OF_COEFF } from '../src/lighting/sh';
import { environmentByName, parseShVectors } from '../src/lighting/vectors';
import './pages/harness';

const shared = (name: string): string =>
  fileURLToPath(new URL(`../../shared/${name}`, import.meta.url));
const vectors = parseShVectors(readFileSync(shared('sh_vectors.json'), 'utf8'));
const ref = vectors.sphere_reference;
const reference = PNG.sync.read(readFileSync(shared(ref.file)));
const env = environmentByName(vectors, ref.environment);

/** Tolerances in 8-bit sRGB levels inside the comparison disc. */
const MAX_MEAN_ABS = 0.5;
const MAX_ABS = 2;

interface Diff {
  meanAbs: number;
  maxAbs: number;
  pixels: number;
}

function compare(rgba: number[]): Diff {
  const size = ref.size;
  let sum = 0;
  let max = 0;
  let pixels = 0;
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const x = ((col + 0.5) / size) * 2 - 1;
      const y = 1 - ((row + 0.5) / size) * 2;
      if (x * x + y * y >= ref.compare_mask_radius ** 2) continue;
      pixels++;
      for (let k = 0; k < 3; k++) {
        const i = (row * size + col) * 4 + k;
        const d = Math.abs((rgba[i] ?? 0) - (reference.data[i] ?? 0));
        sum += d;
        max = Math.max(max, d);
      }
    }
  }
  return { meanAbs: sum / (pixels * 3), maxAbs: max, pixels };
}

async function render(page: Page, sh: number[][]): Promise<{ diff: Diff; gl: string }> {
  const out = await page.evaluate(
    ({ coeffs, size }) => {
      const fn = window.__renderSphere;
      if (fn === undefined) throw new Error('sphere harness did not load');
      return fn(coeffs, size);
    },
    { coeffs: sh, size: ref.size },
  );
  return { diff: compare(out.rgba), gl: out.glRenderer };
}

const radiance = env.sh.map((c) => [...c]);
// Wrong convention A: pre-convolved irradiance / pi passed to the probe (would double-convolve).
const preconvolved = env.sh.map((c, i) => {
  const a = BAND_FACTORS[BAND_OF_COEFF[i] ?? 0] / Math.PI;
  return c.map((v) => v * a);
});
// Wrong convention B: a factor of pi too much (irradiance-scaled radiance).
const timesPi = env.sh.map((c) => c.map((v) => v * Math.PI));

test('three.js LightProbe takes radiance SH: white sphere matches the Python reference', async ({
  page,
}) => {
  expect(reference.width).toBe(ref.size);
  await page.goto('e2e/pages/sphere.html');
  await page.waitForFunction(() => window.__renderSphere !== undefined);

  const ok = await render(page, radiance);
  const wrongA = await render(page, preconvolved);
  const wrongB = await render(page, timesPi);
  console.log(
    `parity gl="${ok.gl}" pixels=${ok.diff.pixels} radiance mean=${ok.diff.meanAbs.toFixed(3)}` +
      ` max=${ok.diff.maxAbs}; preconvolved mean=${wrongA.diff.meanAbs.toFixed(3)}` +
      ` max=${wrongA.diff.maxAbs}; times-pi mean=${wrongB.diff.meanAbs.toFixed(3)}` +
      ` max=${wrongB.diff.maxAbs}`,
  );

  expect(ok.diff.pixels).toBeGreaterThan(10000);
  expect(ok.diff.meanAbs).toBeLessThanOrEqual(MAX_MEAN_ABS);
  expect(ok.diff.maxAbs).toBeLessThanOrEqual(MAX_ABS);
  expect(wrongA.diff.meanAbs).toBeGreaterThan(5 * MAX_MEAN_ABS);
  expect(wrongB.diff.meanAbs).toBeGreaterThan(5 * MAX_MEAN_ABS);
});
