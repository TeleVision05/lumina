/**
 * The neural estimator, main-thread side. Owns the worker, sends frames, turns replies into
 * estimates. One frame is in flight at a time; the app's scheduler sets the rate.
 */
import type { FrameSample } from '../capture/frame';
import { exposureQuality, exposureStats } from '../capture/preprocess';
import { makeEstimate, type LightingEstimate } from '../lighting/estimate';
import { modelUrl } from '../lighting/modelContract';
import { shFromFlat } from '../lighting/sh';
import { isFiniteSH } from '../lighting/shBuild';
import type { Estimator } from './types';
import type { Backend, FromWorker, ReadyReply } from './workerProtocol';

export type NeuralStatus =
  | { state: 'loading' }
  | { state: 'ready'; backend: Backend; placeholder: boolean | null; loadMs: number }
  | { state: 'failed' };

export const NEURAL_COPY = {
  failed:
    'The neural network could not start in this browser, so Lumina switched to image statistics. Everything else still works.',
  placeholder:
    'This build ships a placeholder network with untrained weights, so its estimate is not meaningful yet.',
} as const;

/** The placeholder model is honest about itself; its estimates get half the trust. */
const PLACEHOLDER_TRUST = 0.5;

interface Pending {
  resolve: (e: LightingEstimate) => void;
  reject: (err: Error) => void;
  stats: number;
}

export class NeuralEstimator implements Estimator {
  readonly id = 'neural';
  readonly displayFrame = false;
  status: NeuralStatus = { state: 'loading' };
  /** Worker-side session.run time of the last inference, ms. */
  lastRunMs = 0;
  private readonly worker: Worker;
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor(private readonly onStatus: (s: NeuralStatus) => void) {
    this.worker = new Worker(new URL('./neural.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => {
      this.receive(e.data);
    };
    this.worker.onerror = (e) => {
      console.error('inference worker crashed', e.message);
      this.fail();
    };
    this.worker.postMessage({
      type: 'init',
      modelUrl: modelUrl(),
      preferGpu: !new URLSearchParams(location.search).has('wasm'),
    });
  }

  private ready(msg: ReadyReply): void {
    this.status = {
      state: 'ready',
      backend: msg.backend,
      placeholder: msg.placeholder,
      loadMs: msg.loadMs,
    };
    console.info(
      `Lumina model ready on ${msg.backend} in ${msg.loadMs.toFixed(0)} ms (contract ${msg.contract ?? '?'})`,
    );
    this.onStatus(this.status);
  }

  private fail(): void {
    this.status = { state: 'failed' };
    for (const p of this.pending.values()) p.reject(new Error('neural estimator failed'));
    this.pending.clear();
    this.onStatus(this.status);
  }

  private receive(msg: FromWorker): void {
    if (msg.type === 'ready') {
      this.ready(msg);
      return;
    }
    if (msg.type === 'error') {
      console.error(`inference worker ${msg.stage} error: ${msg.detail}`);
      if (msg.stage === 'init') this.fail();
      else if (msg.id !== undefined) this.pending.get(msg.id)?.reject(new Error(msg.detail));
      if (msg.id !== undefined) this.pending.delete(msg.id);
      return;
    }
    const p = this.pending.get(msg.id);
    this.pending.delete(msg.id);
    if (p === undefined) return;
    this.lastRunMs = msg.ms;
    const sh = shFromFlat(msg.sh);
    if (!isFiniteSH(sh)) {
      p.reject(new Error('model returned non-finite values'));
      return;
    }
    const trust =
      this.status.state === 'ready' && this.status.placeholder === true ? PLACEHOLDER_TRUST : 1;
    p.resolve(makeEstimate(sh, p.stats * trust));
  }

  estimate(frame: FrameSample | null): Promise<LightingEstimate> {
    if (frame === null) return Promise.reject(new Error('no frame yet'));
    if (this.status.state !== 'ready')
      return Promise.reject(new Error(`model ${this.status.state}`));
    const id = this.nextId++;
    const rgba = frame.rgba.slice().buffer;
    const stats = exposureQuality(exposureStats(frame.rgba));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, stats });
      this.worker.postMessage({ type: 'run', id, rgba, width: frame.width, height: frame.height }, [
        rgba,
      ]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}
