/** Loads the committed ONNX model with onnxruntime-web (wasm) and runs one inference. */
import * as ort from 'onnxruntime-web';
import { INPUT_NAME, INPUT_SHAPE, modelUrl, OUTPUT_NAME } from '../../src/lighting/modelContract';
import type { ModelSmoke } from './harness';

// GitHub Pages cannot send COOP/COEP headers, so the site runs single-threaded wasm.
ort.env.wasm.numThreads = 1;

async function runModel(): Promise<ModelSmoke> {
  const session = await ort.InferenceSession.create(modelUrl(), { executionProviders: ['wasm'] });
  const size = INPUT_SHAPE.reduce<number>((a, b) => a * b, 1);
  const image = new ort.Tensor('float32', new Float32Array(size).fill(0.5), [...INPUT_SHAPE]);
  const outputs = await session.run({ [INPUT_NAME]: image });
  const sh = outputs[OUTPUT_NAME];
  if (sh === undefined) throw new Error(`model has no output named ${OUTPUT_NAME}`);
  const data = sh.data as Float32Array;
  const result: ModelSmoke = {
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    outputDims: sh.dims,
    outputType: sh.type,
    allFinite: data.every((v) => Number.isFinite(v)),
  };
  await session.release();
  return result;
}

window.__runModel = runModel;
