/** Frame sources other than the default camera start: files, sample clips, lamp tracking. */
import { existsSync, readdirSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import { startCamera, state, waitForEstimates, watchProblems } from './helpers';
import { FIXTURE_PNG_LEFT, FIXTURE_PNG_RIGHT, PLACEHOLDER_CLIP, SAMPLES_DIR } from './paths';

async function openFile(page: Page, path: string): Promise<void> {
  await page.locator('#file-input').setInputFiles(path);
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', { timeout: 20_000 });
}

async function statsDirectionX(page: Page): Promise<number> {
  await page.getByRole('radio', { name: 'Image statistics' }).check();
  await waitForEstimates(page, 'stats', 4);
  await page.waitForTimeout(1200); // filter settles
  const s = await state(page);
  return s.estimate?.direction?.[0] ?? 0;
}

test('a dropped photo lights the object from the side its lamp is on (not mirrored)', async ({
  page,
}) => {
  await page.goto('?debug');
  await openFile(page, FIXTURE_PNG_LEFT);
  expect((await state(page)).mirrored).toBe(false);
  const left = await statsDirectionX(page);
  await page.getByRole('button', { name: 'Change source' }).click();
  await openFile(page, FIXTURE_PNG_RIGHT);
  const right = await statsDirectionX(page);
  console.log(
    `image statistics direction x: lamp left ${left.toFixed(3)}, lamp right ${right.toFixed(3)}`,
  );
  expect(left).toBeLessThan(-0.2);
  expect(right).toBeGreaterThan(0.2);
});

test('a dropped video file plays as a source and drives the estimate', async ({ page }) => {
  const problems = watchProblems(page);
  await page.goto('?debug');
  await openFile(page, PLACEHOLDER_CLIP);
  const s = await waitForEstimates(page, 'neural', 4);
  expect(s.mirrored).toBe(false);
  expect(problems.errors).toEqual([]);
});

test('the sample clip button works, or says plainly that no clip is bundled', async ({ page }) => {
  const clips = existsSync(SAMPLES_DIR)
    ? readdirSync(SAMPLES_DIR).filter((f) => /\.(mp4|webm)$/i.test(f))
    : [];
  await page.goto('?debug');
  const button = page.getByRole('button', { name: 'Play the sample clip' });
  if (clips.length === 0) {
    await expect(button).toBeHidden();
    await expect(page.locator('#intro-note')).toContainText('No sample clip is included');
  } else {
    await button.click();
    await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', {
      timeout: 20_000,
    });
  }
});

test('live camera: the estimate follows the lamp as it moves across the room', async ({ page }) => {
  await startCamera(page);
  await page.getByRole('radio', { name: 'Image statistics' }).check();
  await waitForEstimates(page, 'stats', 4);
  const xs: number[] = [];
  for (let i = 0; i < 24; i++) {
    await page.waitForTimeout(400);
    xs.push((await state(page)).estimate?.direction?.[0] ?? 0);
  }
  console.log(`camera direction x over 9.6 s: ${xs.map((x) => x.toFixed(2)).join(' ')}`);
  expect(Math.max(...xs)).toBeGreaterThan(0.2);
  expect(Math.min(...xs)).toBeLessThan(-0.2);
});
