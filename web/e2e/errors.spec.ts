/** What a visitor sees when something fails: plain language, never raw error names. */
import { expect, test, type Page } from '@playwright/test';
import { state } from './helpers';

const RAW = /Error|DOMException|getUserMedia|undefined|null|\b[45]\d\d\b/;

async function failCamera(page: Page, name: string): Promise<void> {
  await page.addInitScript((errName) => {
    navigator.mediaDevices.getUserMedia = () =>
      Promise.reject(new DOMException('raw detail for the console', errName));
  }, name);
}

async function expectProblem(page: Page, title: string): Promise<void> {
  await page.goto('?debug');
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await expect(page.locator('#problem-title')).toHaveText(title);
  await expect(page.locator('#problem')).toBeVisible();
  expect(await page.locator('#problem').innerText()).not.toMatch(RAW);
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'intro');
}

test('camera permission turned down', async ({ page }) => {
  await failCamera(page, 'NotAllowedError');
  await expectProblem(page, 'Camera access was turned down');
});

test('no camera connected', async ({ page }) => {
  await failCamera(page, 'NotFoundError');
  await expectProblem(page, 'No camera found');
});

test('camera busy in another app', async ({ page }) => {
  await failCamera(page, 'NotReadableError');
  await expectProblem(page, 'The camera is busy');
});

test('insecure page (http)', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'isSecureContext', { value: false });
  });
  await expectProblem(page, 'The camera needs a secure page');
});

test('a file that is not an image or video', async ({ page }) => {
  await page.goto('?debug');
  await page.locator('#file-input').setInputFiles({
    name: 'notes.txt',
    mimeType: 'text/plain',
    buffer: Buffer.from('hello'),
  });
  await expect(page.locator('#problem-title')).toHaveText('That file did not open');
  expect(await page.locator('#problem').innerText()).not.toMatch(RAW);
});

test('the network fails to load: Lumina switches to image statistics and says so', async ({
  page,
}) => {
  await page.route('**/models/lumina-sh.onnx', (route) => route.fulfill({ status: 404, body: '' }));
  await page.goto('?debug');
  await page.getByRole('button', { name: 'Use my camera' }).click();
  await expect(page.locator('#toast')).toContainText('switched to image statistics', {
    timeout: 30_000,
  });
  expect(await page.locator('#toast').innerText()).not.toMatch(RAW);
  await expect(page.getByRole('radio', { name: 'Image statistics' })).toBeChecked();
  await expect
    .poll(async () => (await state(page)).activeEstimator, { timeout: 15_000 })
    .toBe('stats');
});
