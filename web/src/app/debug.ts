/**
 * Read-only test handle, attached only when the URL has ?debug. Playwright reads the app state
 * and samples the rendered object's pixels through it. Nothing here changes the app's behaviour.
 */
import type { FrameSample } from '../capture/frame';
import { rendererName } from '../render/quality';
import type { App } from './app';

export interface ObjectSample {
  /** Mean sRGB of covered pixels inside the object's screen rectangle (premultiplied removed). */
  mean: [number, number, number];
  /** Fraction of pixels in the rectangle that the object covers. */
  coverage: number;
}

export interface LuminaDebugState {
  running: boolean;
  phase: string;
  view: string;
  selectedEstimator: string;
  activeEstimator: string | null;
  neural: string;
  backend: string;
  placeholder: boolean | null;
  mirrored: boolean | null;
  inferenceCount: number;
  inferenceMsMedian: number | null;
  inferenceMsP95: number | null;
  inferencePerSecond: number | null;
  frameIntervalMedian: number | null;
  frameWorkMedian: number | null;
  frameWorkP95: number | null;
  estimate: {
    sh: number[][];
    direction: number[] | null;
    directionality: number;
    confidence: number;
  } | null;
  objectRect: { x: number; y: number; w: number; h: number };
  renderScale: number;
  softwareRenderer: boolean;
  frameIntervalP95: number | null;
  glRenderer: string;
  /** Brighter half of the last un-mirrored frame the estimator saw: -1 left, 1 right, 0 even. */
  sensorBrightSide: number;
}

function brightSide(frame: FrameSample | null): number {
  if (frame === null) return 0;
  let left = 0;
  let right = 0;
  for (let y = 0; y < frame.height; y++) {
    for (let x = 0; x < frame.width; x++) {
      const k = (y * frame.width + x) * 4;
      const v = (frame.rgba[k] ?? 0) + (frame.rgba[k + 1] ?? 0) + (frame.rgba[k + 2] ?? 0);
      if (x < frame.width / 2) left += v;
      else right += v;
    }
  }
  const d = (right - left) / Math.max(right + left, 1);
  return Math.abs(d) < 0.02 ? 0 : Math.sign(d);
}

export interface NeuralCapture {
  /** RGBA bytes of the 128 x 96 frame the network saw. */
  rgba: number[];
  /** The network's raw output for that frame (sensor frame, unfiltered). */
  sh: number[][];
}

export interface LuminaDebug {
  state: () => LuminaDebugState;
  lastNeural: () => NeuralCapture | null;
  sampleObject: () => ObjectSample;
}

declare global {
  interface Window {
    __lumina?: LuminaDebug;
  }
}

function state(app: App): LuminaDebugState {
  const p = app.pipeline;
  const st = p.neural.status;
  const est = p.latest;
  return {
    running: app.source !== null,
    phase: document.getElementById('app')?.dataset.phase ?? '',
    view: app.settings.view,
    selectedEstimator: app.settings.estimator,
    activeEstimator: p.activeId,
    neural: st.state,
    backend: app.backendLabel(),
    placeholder: st.state === 'ready' ? st.placeholder : null,
    mirrored: app.source?.mirrored ?? null,
    inferenceCount: p.inferenceCount,
    inferenceMsMedian: p.inferenceMs.median,
    inferenceMsP95: p.inferenceMs.quantile(0.95),
    inferencePerSecond: p.inferenceMs.rate(),
    frameIntervalMedian: app.frameIntervals.median,
    frameWorkMedian: app.frameWork.median,
    frameWorkP95: app.frameWork.quantile(0.95),
    estimate:
      est === null
        ? null
        : {
            sh: est.sh.map((c) => [...c]),
            direction: est.direction === null ? null : [...est.direction],
            directionality: est.directionality,
            confidence: est.confidence,
          },
    objectRect: app.stage.objectScreenRect(window.innerWidth, window.innerHeight),
    renderScale: app.stage.governor.scale,
    softwareRenderer: app.stage.quality.software,
    frameIntervalP95: app.frameIntervals.quantile(0.95),
    glRenderer: rendererName(app.stage.renderer.getContext()),
    sensorBrightSide: brightSide(p.lastFrame),
  };
}

function sampleObject(app: App): ObjectSample {
  const source = app.source;
  if (source === null) throw new Error('no source running');
  app.renderFrame(performance.now(), source);
  const gl = app.stage.renderer.getContext();
  const dpr = app.stage.renderer.getPixelRatio();
  const r = app.stage.objectScreenRect(window.innerWidth, window.innerHeight);
  const x = Math.max(0, Math.round(r.x * dpr));
  const w = Math.max(1, Math.round(r.w * dpr));
  const h = Math.max(1, Math.round(r.h * dpr));
  const y = Math.max(0, gl.drawingBufferHeight - Math.round((r.y + r.h) * dpr));
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const sum = [0, 0, 0];
  let covered = 0;
  for (let k = 0; k < px.length; k += 4) {
    const a = px[k + 3] ?? 0;
    if (a < 250) continue;
    covered++;
    for (let c = 0; c < 3; c++) sum[c] = (sum[c] ?? 0) + (px[k + c] ?? 0);
  }
  const n = Math.max(covered, 1);
  return {
    mean: [(sum[0] ?? 0) / n, (sum[1] ?? 0) / n, (sum[2] ?? 0) / n],
    coverage: covered / (w * h),
  };
}

export function attachDebug(app: App): void {
  if (!new URLSearchParams(location.search).has('debug')) return;
  window.__lumina = {
    state: () => state(app),
    sampleObject: () => sampleObject(app),
    lastNeural: () => {
      const n = app.pipeline.lastNeural;
      return n === null ? null : { rgba: Array.from(n.rgba), sh: n.sh };
    },
  };
}
