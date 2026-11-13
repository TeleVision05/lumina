/** Shared steps for the app specs. */
import { expect, type Page } from '@playwright/test';
import type { LuminaDebugState, ObjectSample } from '../src/app/debug';
import '../src/app/debug';

export interface Problems {
  errors: string[];
  failed: string[];
}

/** Collect page errors, console errors and failed or 4xx/5xx requests. */
export function watchProblems(page: Page): Problems {
  const p: Problems = { errors: [], failed: [] };
  page.on('pageerror', (e) => p.errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') p.errors.push(`console: ${m.text()}`);
  });
  page.on('requestfailed', (r) => p.failed.push(`${r.url()} ${r.failure()?.errorText ?? ''}`));
  page.on('response', (r) => {
    if (r.status() >= 400) p.failed.push(`${r.status()} ${r.url()}`);
  });
  return p;
}

export async function state(page: Page): Promise<LuminaDebugState> {
  return page.evaluate(() => {
    const d = window.__lumina;
    if (d === undefined) throw new Error('debug handle missing (open the page with ?debug)');
    return d.state();
  });
}

export async function sampleObject(page: Page): Promise<ObjectSample> {
  return page.evaluate(() => {
    const d = window.__lumina;
    if (d === undefined) throw new Error('debug handle missing');
    return d.sampleObject();
  });
}

export async function startCamera(page: Page, query = '?debug'): Promise<void> {
  await page.goto(query);
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', { timeout: 20_000 });
}

/** Wait until `count` more estimates have arrived from `estimator`. */
export async function waitForEstimates(
  page: Page,
  estimator: 'neural' | 'stats',
  count = 6,
  timeout = 30_000,
): Promise<LuminaDebugState> {
  const start = (await state(page)).inferenceCount;
  await expect
    .poll(
      async () => {
        const s = await state(page);
        return s.activeEstimator === estimator && s.inferenceCount >= start + count;
      },
      { timeout, intervals: [200] },
    )
    .toBe(true);
  return state(page);
}

export function colourDistance(a: readonly number[], b: readonly number[]): number {
  return Math.hypot(
    (a[0] ?? 0) - (b[0] ?? 0),
    (a[1] ?? 0) - (b[1] ?? 0),
    (a[2] ?? 0) - (b[2] ?? 0),
  );
}
