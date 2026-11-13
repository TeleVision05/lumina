/** Keyboard use, reduced motion and the phone layout. */
import { expect, test } from '@playwright/test';
import { sampleObject, state, waitForEstimates } from './helpers';

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

test('reduced motion: the app runs and the grain does not animate', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('?debug');
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await waitForEstimates(page, 'neural', 3);
  const a = await sampleObject(page);
  const b = await sampleObject(page);
  expect(a.coverage).toBeGreaterThan(0.4);
  // Same estimate, same static grain: two renders in a row match closely.
  expect(Math.abs(a.mean[0] - b.mean[0])).toBeLessThan(1.5);
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
