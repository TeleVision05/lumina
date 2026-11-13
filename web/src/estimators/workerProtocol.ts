/** Messages between the page and the inference worker (estimators/neural.worker.ts). */

export type Backend = 'webgpu' | 'wasm';

export interface InitRequest {
  type: 'init';
  modelUrl: string;
  /** Try WebGPU first when the browser offers it. */
  preferGpu: boolean;
}

export interface RunRequest {
  type: 'run';
  id: number;
  /** RGBA bytes of the model-size frame (transferred). */
  rgba: ArrayBuffer;
  width: number;
  height: number;
}

export type ToWorker = InitRequest | RunRequest;

export interface ReadyReply {
  type: 'ready';
  backend: Backend;
  /** From the model's metadata (lumina.placeholder); null when the model does not say. */
  placeholder: boolean | null;
  contract: string | null;
  loadMs: number;
}

export interface ResultReply {
  type: 'result';
  id: number;
  /** 27 values, [1, 9, 3] row-major (transferred). */
  sh: Float32Array;
  /** session.run time inside the worker, ms. */
  ms: number;
}

export interface ErrorReply {
  type: 'error';
  stage: 'init' | 'run';
  id?: number;
  /** Developer detail for the console only; never shown to the visitor. */
  detail: string;
}

export type FromWorker = ReadyReply | ResultReply | ErrorReply;
