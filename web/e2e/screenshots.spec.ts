/**
 * Screenshots of the interface for docs/media. Only runs with LUMINA_SCREENSHOTS=1, so ordinary
 * test runs never rewrite committed images:
 *   LUMINA_SCREENSHOTS=1 npx playwright test e2e/screenshots.spec.ts
 */
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { waitForEstimates } from './helpers';

const OUT = fileURLToPath(new URL('../../docs/media/', import.meta.url));
test.skip(process.env.LUMINA_SCREENSHOTS !== '1', 'set LUMINA_SCREENSHOTS=1 to write docs/media');

async function running(page: Page): Promise<void> {
  await page.goto('?debug');
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', { timeout: 20_000 });
  await waitForEstimates(page, 'neural', 6);
}

test('desktop', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('?debug');
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}web-intro.png` });
  await running(page);
  await page.getByRole('radio', { name: 'Image statistics' }).check();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${OUT}web-compare.png` });
  await page.getByRole('radio', { name: 'Chrome ball' }).check();
  await page.getByRole('radio', { name: 'Lumina' }).check();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}web-chrome.png` });
  await page.getByRole('radio', { name: 'Glazed vase' }).check();
  await page.locator('#settings-toggle').click();
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}web-vase-settings.png` });
  await page.locator('#settings-toggle').click();
  await page.locator('#how-open').click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}web-how.png` });
});

test.describe('phone', () => {
  test.use({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });

  test('portrait', async ({ page }) => {
    await page.goto('?debug');
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}web-phone-intro.png` });
    await running(page);
    await page.getByRole('button', { name: 'Details' }).click();
    await page.getByRole('radio', { name: 'Image statistics' }).check();
    await page.getByRole('radio', { name: 'Brass knot' }).check();
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `${OUT}web-phone-details.png` });
    await page.getByRole('button', { name: 'Less' }).click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${OUT}web-phone.png` });
  });
});
