/**
 * Inference worker: runs the Lumina ONNX model with onnxruntime-web off the main thread, so the
 * page never stalls. WebGPU when the browser offers an adapter, otherwise single-threaded WASM
 * (GitHub Pages cannot send the COOP/COEP headers that wasm threads need). The ORT wasm binaries
 * are same-origin assets emitted by Vite (see vite/plugins.ts); nothing is fetched from a CDN.
 */
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { rgbaToNCHW } from '../capture/preprocess';
import { INPUT_NAME, INPUT_SHAPE, OUTPUT_NAME } from '../lighting/modelContract';
import { readOnnxMetadata } from './onnxMetadata';
import type { Backend, FromWorker, InitRequest, RunRequest, ToWorker } from './workerProtocol';

type Ort = typeof import('onnxruntime-web');

interface Loaded {
  ort: Ort;
  session: InferenceSession;
  input: Float32Array;
}

let loaded: Loaded | null = null;
const post = (msg: FromWorker, transfer: Transferable[] = []): void => {
  self.postMessage(msg, { transfer });
};

async function webgpuAvailable(): Promise<boolean> {
  // navigator.gpu is missing in browsers without WebGPU, whatever the DOM types say.
  const nav: { gpu?: { requestAdapter: () => Promise<unknown> } } = navigator;
  const gpu = nav.gpu;
  if (gpu === undefined) return false;
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function createSession(
  ort: Ort,
  model: Uint8Array,
  gpu: boolean,
): Promise<{ session: InferenceSession; backend: Backend }> {
  if (gpu) {
    try {
      return {
        session: await ort.InferenceSession.create(model, { executionProviders: ['webgpu'] }),
        backend: 'webgpu',
      };
    } catch (err) {
      console.warn('WebGPU session failed; falling back to WASM', err);
    }
  }
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
  return { session, backend: 'wasm' };
}

async function init(req: InitRequest): Promise<void> {
  const t0 = performance.now();
  const gpu = req.preferGpu && (await webgpuAvailable());
  const ort: Ort = gpu
    ? await import('onnxruntime-web/webgpu')
    : await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const res = await fetch(req.modelUrl);
  if (!res.ok) throw new Error(`model fetch ${res.status} ${req.modelUrl}`);
  const model = new Uint8Array(await res.arrayBuffer());
  const meta = readOnnxMetadata(model);
  const { session, backend } = await createSession(ort, model, gpu);
  const input = new Float32Array(INPUT_SHAPE.reduce<number>((a, b) => a * b, 1));
  loaded = { ort, session, input };
  // Warm up: the first runs compile shaders / allocate; keep them out of the timings.
  for (let i = 0; i < 3; i++) await runModel(input.fill(0.5));
  const placeholder = meta['lumina.placeholder'];
  post({
    type: 'ready',
    backend,
    placeholder: placeholder === undefined ? null : placeholder === 'true',
    contract: meta['lumina.contract'] ?? null,
    loadMs: performance.now() - t0,
  });
}

async function runModel(data: Float32Array): Promise<Float32Array> {
  if (loaded === null) throw new Error('model not loaded');
  const tensor: Tensor = new loaded.ort.Tensor('float32', data, [...INPUT_SHAPE]);
  const out = await loaded.session.run({ [INPUT_NAME]: tensor });
  const sh = out[OUTPUT_NAME];
  if (sh === undefined) throw new Error(`model has no output ${OUTPUT_NAME}`);
  return Float32Array.from(sh.data as Float32Array);
}

async function run(req: RunRequest): Promise<void> {
  if (loaded === null) throw new Error('run before init');
  const rgba = new Uint8ClampedArray(req.rgba);
  rgbaToNCHW(rgba, req.width, req.height, loaded.input);
  const t0 = performance.now();
  const sh = await runModel(loaded.input);
  post({ type: 'result', id: req.id, sh, ms: performance.now() - t0 }, [sh.buffer]);
}

self.onmessage = (e: MessageEvent<ToWorker>): void => {
  const msg = e.data;
  const task = msg.type === 'init' ? init(msg) : run(msg);
  task.catch((err: unknown) => {
    const detail = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    post(
      msg.type === 'run'
        ? { type: 'error', stage: 'run', id: msg.id, detail }
        : { type: 'error', stage: 'init', detail },
    );
  });
};
