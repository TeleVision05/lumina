/**
 * Privacy: while the camera runs and Lumina estimates, every request (page and worker) is a plain
 * GET to the app's own origin, and none carries a body. No camera frame can leave the page.
 */
import { expect, test } from '@playwright/test';
import { startCamera, waitForEstimates } from './helpers';

test('no request leaves the origin or carries a body while the camera runs', async ({
  page,
  context,
  baseURL,
}) => {
  const origin = new URL(baseURL ?? 'http://localhost').origin;
  const seen: { method: string; url: string; body: boolean }[] = [];
  context.on('request', (r) => {
    seen.push({ method: r.method(), url: r.url(), body: r.postDataBuffer() !== null });
  });
  await startCamera(page);
  await waitForEstimates(page, 'neural', 10);
  console.log(`privacy: ${seen.length} requests, all to ${origin}`);
  const urls = seen
    .map((r) => r.url)
    .filter((u) => !u.startsWith('blob:') && !u.startsWith('data:'));
  expect(urls.length).toBeGreaterThan(0);
  expect(urls.filter((u) => new URL(u).origin !== origin)).toEqual([]);
  expect(seen.filter((r) => r.method !== 'GET' || r.body)).toEqual([]);
  expect(urls.some((u) => u.endsWith('/models/lumina-sh.onnx'))).toBe(true);
});
