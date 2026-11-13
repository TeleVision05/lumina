/** The committed ONNX model loads in onnxruntime-web (wasm, headless Chromium) with the contract. */
import { expect, test } from '@playwright/test';
import { INPUT_NAME, OUTPUT_NAME, OUTPUT_SHAPE } from '../src/lighting/modelContract';
import './pages/harness';

test('lumina-sh.onnx runs in onnxruntime-web with the contract shapes', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('e2e/pages/model.html');
  await page.waitForFunction(() => window.__runModel !== undefined);
  const result = await page.evaluate(async () => {
    const fn = window.__runModel;
    if (fn === undefined) throw new Error('model harness did not load');
    return fn();
  });
  console.log(`model smoke ${JSON.stringify(result)}`);
  expect(errors).toEqual([]);
  expect(result.inputNames).toEqual([INPUT_NAME]);
  expect(result.outputNames).toEqual([OUTPUT_NAME]);
  expect(result.outputDims).toEqual([...OUTPUT_SHAPE]);
  expect(result.outputType).toBe('float32');
  expect(result.allFinite).toBe(true);
});
