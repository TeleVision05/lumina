/**
 * Records the demo video: the sample clip as the fake camera, the Compare view (fixed studio light
 * on the left, Lumina on the right), headless. Only runs with LUMINA_RECORD=<output.webm>:
 *   LUMINA_RECORD=/path/raw.webm LUMINA_CLIP=pan-outdoor-sun LUMINA_GPU=1 \
 *     npx playwright test e2e/record.spec.ts
 * LUMINA_GPU=1 uses the Metal GPU for smooth frames (still no window); without it SwiftShader is used.
 */
import { expect, test } from '@playwright/test';
import { CHROMIUM_ARGS } from '../playwright.config';
import { waitForEstimates } from './helpers';
import { ensureTruthY4m, TRUTH_Y4M } from './truthFixtures';

const OUT = process.env.LUMINA_RECORD ?? '';
test.skip(OUT === '', 'set LUMINA_RECORD=<output.webm> to record the demo');
ensureTruthY4m();

const GPU_ARGS = [
  '--use-angle=metal',
  '--ignore-gpu-blocklist',
  '--enable-gpu',
  '--enable-unsafe-webgpu',
];
const args = CHROMIUM_ARGS.filter(
  (a) => process.env.LUMINA_GPU !== '1' || !a.includes('swiftshader'),
)
  .map((a) =>
    a.startsWith('--use-file-for-fake-video-capture=')
      ? `--use-file-for-fake-video-capture=${TRUTH_Y4M}`
      : a,
  )
  .concat(process.env.LUMINA_GPU === '1' ? GPU_ARGS : []);

test.use({
  launchOptions: { args },
  viewport: { width: 1280, height: 720 },
  video: { mode: 'on', size: { width: 1280, height: 720 } },
});

test('record the compare view', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto('?debug');
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', { timeout: 20_000 });
  await waitForEstimates(page, 'neural', 6);
  await page.getByRole('radio', { name: 'Compare' }).check();
  await page.waitForTimeout(Number(process.env.LUMINA_RECORD_MS ?? 16_000));
  await page.close();
  await page.video()?.saveAs(OUT);
});
