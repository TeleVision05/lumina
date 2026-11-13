/**
 * Performance probe: prints inference time (worker-side session.run), achieved inference rate,
 * frame interval and main-thread frame work for each view, after warm-up. Numbers depend on the
 * machine and its load; the assertions only guard against the pipeline stalling.
 *
 * Default: headless Chromium with SwiftShader (software WebGL), single-threaded WASM.
 * GPU-backed headless (WebGPU inference): LUMINA_GPU=1 npx playwright test e2e/perf.spec.ts
 * GPU rendering with WASM inference: LUMINA_GPU=1 LUMINA_FORCE_WASM=1 npx playwright test e2e/perf.spec.ts
 */
import { expect, test } from '@playwright/test';
import type { LuminaDebugState } from '../src/app/debug';
import { FAKE_CAMERA_Y4M } from './paths';
import { startCamera, state, waitForEstimates } from './helpers';

const GPU = process.env.LUMINA_GPU === '1';
if (GPU) {
  test.use({
    launchOptions: {
      args: [
        '--use-fake-device-for-media-stream',
        '--use-fake-ui-for-media-stream',
        `--use-file-for-fake-video-capture=${FAKE_CAMERA_Y4M}`,
        '--mute-audio',
        '--use-angle=metal',
        '--ignore-gpu-blocklist',
        '--enable-gpu',
        '--enable-unsafe-webgpu',
      ],
    },
  });
}

const fmt = (v: number | null, d = 1): string => (v === null ? 'n/a' : v.toFixed(d));

function line(label: string, s: LuminaDebugState): string {
  return (
    `PERF ${label} | gl="${s.glRenderer}" software=${String(s.softwareRenderer)} ` +
    `backend=${s.backend} scale=${fmt(s.renderScale, 2)} | inference median ${fmt(s.inferenceMsMedian, 2)} ms ` +
    `p95 ${fmt(s.inferenceMsP95, 2)} ms, ${fmt(s.inferencePerSecond)} /s | frame interval median ` +
    `${fmt(s.frameIntervalMedian)} ms p95 ${fmt(s.frameIntervalP95)} ms | main-thread work median ` +
    `${fmt(s.frameWorkMedian)} ms p95 ${fmt(s.frameWorkP95)} ms`
  );
}

test('performance probe', async ({ page }) => {
  test.setTimeout(120_000);
  await startCamera(page, GPU && process.env.LUMINA_FORCE_WASM !== '1' ? '?debug' : '?debug&wasm');
  await waitForEstimates(page, 'neural', 10);
  for (const view of ['Lumina', 'Compare'] as const) {
    await page.getByRole('radio', { name: view }).check();
    await page.waitForTimeout(8000); // the rolling windows hold about the last 2 s of frames
    const s = await state(page);
    console.log(line(view.toLowerCase(), s));
    expect(s.inferencePerSecond ?? 0).toBeGreaterThan(1);
  }
  await page.getByRole('radio', { name: 'Chrome ball' }).check();
  await page.getByRole('radio', { name: 'Lumina' }).check();
  await page.waitForTimeout(8000);
  console.log(line('chrome-lumina', await state(page)));
});
