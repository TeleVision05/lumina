/**
 * The app end to end in headless Chromium with a fake camera (a generated Y4M room clip). Runs
 * against the Vite dev server and, with LUMINA_E2E_TARGET=dist, against the production build
 * served under /lumina/.
 */
import { expect, test } from '@playwright/test';
import {
  colourDistance,
  sampleObject,
  startCamera,
  waitForEstimates,
  watchProblems,
} from './helpers';

test('starts from the camera, runs the network in a worker and renders the object', async ({
  page,
}) => {
  const problems = watchProblems(page);
  await startCamera(page);
  const s = await waitForEstimates(page, 'neural', 8);
  console.log(
    `app start: backend=${s.backend} placeholder=${String(s.placeholder)} mirrored=${String(s.mirrored)} inferences=${s.inferenceCount} median=${s.inferenceMsMedian?.toFixed(2)} ms`,
  );
  expect(s.neural).toBe('ready');
  expect(['wasm', 'webgpu']).toContain(s.backend);
  expect(s.mirrored).toBe(true); // the fake camera reports no facingMode: treated as a webcam
  await expect(page.locator('#media')).toHaveClass(/is-mirrored/);
  const obj = await sampleObject(page);
  console.log(
    `object coverage ${obj.coverage.toFixed(3)} mean ${obj.mean.map((v) => v.toFixed(1)).join(',')}`,
  );
  expect(obj.coverage).toBeGreaterThan(0.4);
  await expect(page.locator('#reading-direction')).not.toHaveText('Waiting for a picture');
  expect(problems.errors).toEqual([]);
  expect(problems.failed).toEqual([]);
});

test('switching estimators and views changes the object shading', async ({ page }) => {
  await startCamera(page);
  await page.getByRole('radio', { name: 'Lumina' }).check();
  await waitForEstimates(page, 'neural', 6);
  await page.waitForTimeout(1500); // let the filter settle on the network's estimate
  const neural = await sampleObject(page);

  await page.getByRole('radio', { name: 'Image statistics' }).check();
  await waitForEstimates(page, 'stats', 6);
  await page.waitForTimeout(1500);
  const stats = await sampleObject(page);

  await page.getByRole('radio', { name: 'Fixed light' }).check();
  const fixed = await sampleObject(page);
  const dNS = colourDistance(neural.mean, stats.mean);
  const dSF = colourDistance(stats.mean, fixed.mean);
  console.log(
    `shading: neural ${neural.mean.map((v) => v.toFixed(1)).join(',')} | stats ${stats.mean.map((v) => v.toFixed(1)).join(',')} | fixed ${fixed.mean.map((v) => v.toFixed(1)).join(',')} | d(neural,stats)=${dNS.toFixed(1)} d(stats,fixed)=${dSF.toFixed(1)}`,
  );
  expect(dNS).toBeGreaterThan(3);
  expect(dSF).toBeGreaterThan(3);
});
