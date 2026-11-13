/**
 * Read-only test handle, attached only when the URL has ?debug. Playwright reads the app state
 * and samples the rendered object's pixels through it. Nothing here changes the app's behaviour.
 */
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
}

export interface LuminaDebug {
  state: () => LuminaDebugState;
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
  window.__lumina = { state: () => state(app), sampleObject: () => sampleObject(app) };
}
