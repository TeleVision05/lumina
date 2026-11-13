/** Mirroring end to end, real drag-and-drop, and a camera switch that fails. */
import { expect, test } from '@playwright/test';
import { startCamera, state, waitForEstimates } from './helpers';

test('mirrored camera: a lamp on the sensor left lights the object from the display right', async ({
  page,
}) => {
  await startCamera(page);
  expect((await state(page)).mirrored).toBe(true);
  await page.getByRole('radio', { name: 'Image statistics' }).check();
  await waitForEstimates(page, 'stats', 4);
  let agree = 0;
  let total = 0;
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(300);
    const s = await state(page);
    const x = s.estimate?.direction?.[0] ?? 0;
    if (Math.abs(x) < 0.2 || s.sensorBrightSide === 0) continue;
    total++;
    // Mirrored display: the sensor's left (-1) is the display's right (+x).
    if (Math.sign(x) === -s.sensorBrightSide) agree++;
  }
  console.log(`mirroring: ${agree}/${total} samples on the expected side`);
  expect(total).toBeGreaterThanOrEqual(10);
  expect(agree / total).toBeGreaterThanOrEqual(0.75);
});

test('a file dropped onto the page becomes the source', async ({ page }) => {
  await page.goto('?debug');
  await page.evaluate(async () => {
    const c = document.createElement('canvas');
    c.width = 160;
    c.height = 120;
    const ctx = c.getContext('2d');
    if (ctx === null) throw new Error('no 2d context');
    ctx.fillStyle = '#5a554f';
    ctx.fillRect(0, 0, 160, 120);
    ctx.fillStyle = '#fff3dc';
    ctx.beginPath();
    ctx.arc(30, 25, 14, 0, Math.PI * 2);
    ctx.fill();
    const blob = await new Promise<Blob | null>((r) => {
      c.toBlob(r, 'image/png');
    });
    if (blob === null) throw new Error('no blob');
    const dt = new DataTransfer();
    dt.items.add(new File([blob], 'dropped.png', { type: 'image/png' }));
    window.dispatchEvent(new DragEvent('dragenter', { dataTransfer: dt }));
    window.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, cancelable: true }));
  });
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running', { timeout: 20_000 });
  await expect(page.locator('#app')).not.toHaveClass(/is-dragging/);
  expect((await state(page)).mirrored).toBe(false);
});

test('switching to a camera that fails falls back to the one in use and says so', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const md = navigator.mediaDevices;
    const enumerate = md.enumerateDevices.bind(md);
    const open = md.getUserMedia.bind(md);
    let calls = 0;
    md.enumerateDevices = async () => {
      const list = await enumerate();
      const extra = { kind: 'videoinput', deviceId: 'second-camera', groupId: '', label: '' };
      return [...list, extra as MediaDeviceInfo];
    };
    md.getUserMedia = (c) => {
      calls++;
      if (calls === 2) return Promise.reject(new DOMException('raw', 'NotReadableError'));
      return open(c);
    };
  });
  await startCamera(page);
  const button = page.getByRole('button', { name: 'Switch camera' });
  await expect(button).toBeVisible();
  await button.click();
  await expect(page.locator('#toast')).toContainText('went back to the one you were using');
  await expect(page.locator('#app')).toHaveAttribute('data-phase', 'running');
  const before = (await state(page)).inferenceCount;
  // Software WebGL on a shared CI runner can drop below 1 inference per second.
  await expect
    .poll(async () => (await state(page)).inferenceCount, { timeout: 20_000 })
    .toBeGreaterThan(before + 3);
});
