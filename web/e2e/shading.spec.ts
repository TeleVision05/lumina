/**
 * Measures Lumina's rendering pieces in three.js against shared/sphere_reference.png (white
 * Lambertian sphere under lobe_oblique_warm, rendered by Python from radiance SH):
 * - the light rig (residual probe + derived DirectionalLight) reproduces the reference up to the
 *   band-2 truncation of the delta light;
 * - MeshStandardMaterial (roughness 1) under the probe: the size of the diffuse energy split that
 *   docs/CONVENTIONS.md lists as UNVERIFIED;
 * - the probe-diffuse patch keeps an SH env map from double counting diffuse light, and the
 *   unpatched control does double count.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';
import { environmentByName, parseShVectors } from '../src/lighting/vectors';
import type { RigMode } from './pages/rig';

const shared = (name: string): string =>
  fileURLToPath(new URL(`../../shared/${name}`, import.meta.url));
const vectors = parseShVectors(readFileSync(shared('sh_vectors.json'), 'utf8'));
const ref = vectors.sphere_reference;
const reference = PNG.sync.read(readFileSync(shared(ref.file)));
const env = environmentByName(vectors, ref.environment);

interface Diff {
  /** Mean of (render - reference) over channels and pixels: positive = brighter. */
  bias: number;
  meanAbs: number;
  maxAbs: number;
}

function compare(rgba: number[]): Diff {
  let sum = 0;
  let signed = 0;
  let max = 0;
  let n = 0;
  for (let row = 0; row < ref.size; row++) {
    for (let col = 0; col < ref.size; col++) {
      const x = ((col + 0.5) / ref.size) * 2 - 1;
      const y = 1 - ((row + 0.5) / ref.size) * 2;
      if (x * x + y * y >= ref.compare_mask_radius ** 2) continue;
      for (let k = 0; k < 3; k++) {
        const i = (row * ref.size + col) * 4 + k;
        const d = (rgba[i] ?? 0) - (reference.data[i] ?? 0);
        signed += d;
        sum += Math.abs(d);
        max = Math.max(max, Math.abs(d));
        n++;
      }
    }
  }
  return { bias: signed / n, meanAbs: sum / n, maxAbs: max };
}

async function render(page: Page, mode: RigMode): Promise<Diff> {
  const rgba = await page.evaluate(
    ({ sh, size, m }) => {
      const fn = window.__renderRig;
      if (fn === undefined) throw new Error('rig harness did not load');
      return fn(sh, size, m);
    },
    { sh: env.sh.map((c) => [...c]), size: ref.size, m: mode },
  );
  const d = compare(rgba);
  console.log(
    `${mode}: bias ${d.bias.toFixed(2)} mean |d| ${d.meanAbs.toFixed(2)} max ${d.maxAbs} (8-bit levels)`,
  );
  return d;
}

test('light rig and material shading against the Python reference sphere', async ({ page }) => {
  await page.goto('e2e/pages/rig.html');
  await page.waitForFunction(() => window.__renderRig !== undefined);
  const rig = await render(page, 'rig-lambert');
  const probe = await render(page, 'standard-probe');
  const patched = await render(page, 'standard-env-patched');
  const unpatched = await render(page, 'standard-env-unpatched');

  // Rig: the directional light replaces part of the band-limited probe with an exact clamped
  // cosine, so it differs from the SH9 reference by the band-2 truncation, a few levels on average.
  expect(rig.meanAbs).toBeLessThan(6);
  // Patched env adds only specular to a rough dielectric: close to probe-only.
  expect(Math.abs(patched.bias - probe.bias)).toBeLessThan(4);
  // Unpatched env double counts diffuse light: clearly brighter than the patched render.
  expect(unpatched.bias - patched.bias).toBeGreaterThan(10);
});
