/** Keyboard use, reduced motion and the phone layout. */
import { expect, test, type Page } from '@playwright/test';
import { state, waitForEstimates } from './helpers';

test('the whole flow works from the keyboard', async ({ page }) => {
  await page.goto('?debug');
  // Tab from the top of the page to the camera button.
  for (let i = 0; i < 5; i++) {
    await page.keyboard.press('Tab');
    const label = await page.evaluate(() => document.activeElement?.textContent.trim());
    if (label === 'Use my camera') break;
  }
  await page.keyboard.press('Enter');
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', { timeout: 20_000 });
  await expect(page.locator('#scene')).toBeFocused();

  const before = (await state(page)).objectRect;
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('+');
  const after = (await state(page)).objectRect;
  expect(after.x + after.w / 2).toBeLessThan(before.x + before.w / 2);
  expect(after.w).toBeGreaterThan(before.w);

  const handle = page.locator('#split-handle');
  await handle.focus();
  await page.keyboard.press('ArrowRight');
  await expect(handle).toHaveAttribute('aria-valuenow', '55');

  await page.locator('#how-open').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#how')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#how')).toBeHidden();
  await expect(page.locator('#how-open')).toBeFocused();

  await page.locator('#settings-toggle').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#settings')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#settings')).toBeHidden();
});

async function objectRenderJitter(page: Page): Promise<number> {
  await page.goto('?debug');
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await waitForEstimates(page, 'neural', 3);
  // Fixed light is a constant estimate, so the grain is the only thing that can differ between two
  // renders in a row (with the live estimate the lamp and the filter moved the mean by up to 1.6
  // levels and made the old mean-colour check flaky under load).
  await page.getByRole('radio', { name: 'Fixed light' }).check();
  return page.evaluate(() => window.__lumina?.renderJitter() ?? Number.NaN);
}

test('reduced motion: the grain does not animate', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  expect(await objectRenderJitter(page)).toBeLessThan(0.5);
});

test('control: without reduced motion the grain does animate (the jitter measure can see it)', async ({
  page,
}) => {
  expect(await objectRenderJitter(page)).toBeGreaterThan(1);
});

test.describe('phone portrait', () => {
  test.use({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    hasTouch: true,
    isMobile: true,
  });

  test('fits the screen with no sideways scrolling, and the controls are reachable', async ({
    page,
  }) => {
    await page.goto('?debug');
    await page.getByRole('button', { name: 'Use my camera' }).click();
    await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', {
      timeout: 20_000,
    });
    await waitForEstimates(page, 'neural', 2);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
    for (const id of ['#dock', '#reading']) {
      const box = await page.locator(id).boundingBox();
      expect(box).not.toBeNull();
      expect((box?.x ?? -1) >= 0 && (box?.x ?? 0) + (box?.width ?? 0) <= 390).toBe(true);
      expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(844);
    }
    await page.getByRole('button', { name: 'Details' }).click();
    await expect(page.locator('.kelvin')).toBeVisible();
  });
});
