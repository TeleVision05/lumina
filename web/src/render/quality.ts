/**
 * Render quality. Two levers, both measured under headless SwiftShader at 1280 x 720
 * (e2e/perf.spec.ts prints the live numbers):
 * - Multisampling on a float target is the most expensive part of a software-rendered frame, so
 *   software renderers (SwiftShader, llvmpipe: what Chrome uses when the GPU is blocklisted) get
 *   no MSAA and a smaller shadow map.
 * - A resolution governor lowers the render scale while frames are slow and raises it again when
 *   there is headroom, so a slow phone stays responsive instead of stuttering.
 */

export interface Quality {
  software: boolean;
  samples: number;
  shadowMapSize: number;
  /** Starting render scale (fraction of the device pixel ratio). */
  initialScale: number;
  /**
   * Minimum time between environment-map rebuilds, ms. three.js PMREM never goes below a 256 px
   * cube; that took roughly 300 ms per rebuild under SwiftShader on a loaded laptop.
   */
  envIntervalMs: number;
}

const SOFTWARE = /swiftshader|llvmpipe|softpipe|software|basic render/i;

export function qualityFor(rendererName: string): Quality {
  const software = SOFTWARE.test(rendererName);
  return software
    ? { software, samples: 0, shadowMapSize: 512, initialScale: 0.5, envIntervalMs: 2000 }
    : { software, samples: 4, shadowMapSize: 1024, initialScale: 1, envIntervalMs: 150 };
}

export function rendererName(gl: WebGLRenderingContext | WebGL2RenderingContext): string {
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const name: unknown = ext
    ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
  return typeof name === 'string' ? name : 'unknown';
}

export const MIN_SCALE = 0.4;
export const MAX_SCALE = 1;
/** Frames slower than this (ms) push the scale down. About 40 frames per second. */
const SLOW_MS = 25;
/** Frame work below this (ms) for a while lets the scale come back up. */
const FAST_WORK_MS = 7;
const DECIDE_EVERY_MS = 1000;
const HOLD_UP_MS = 3000;

export class ResolutionGovernor {
  private costs: number[] = [];
  private works: number[] = [];
  private lastDecision = 0;
  private fastSince: number | null = null;

  constructor(public scale: number) {}

  /**
   * Record one frame: `intervalMs` since the previous frame and `workMs` spent in it. Returns the
   * new scale when it changes, otherwise null.
   */
  record(nowMs: number, intervalMs: number, workMs: number): number | null {
    this.costs.push(Math.max(intervalMs, workMs));
    this.works.push(workMs);
    if (nowMs - this.lastDecision < DECIDE_EVERY_MS) return null;
    this.lastDecision = nowMs;
    // Mean, not median: occasional long stalls (a GPU-bound frame) must count against the scale.
    const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
    const cost = mean(this.costs);
    const work = mean(this.works);
    this.costs = [];
    this.works = [];
    if (cost > SLOW_MS && this.scale > MIN_SCALE) {
      this.fastSince = null;
      return this.set(this.scale * 0.8);
    }
    if (work < FAST_WORK_MS && cost < SLOW_MS && this.scale < MAX_SCALE) {
      this.fastSince ??= nowMs;
      if (nowMs - this.fastSince >= HOLD_UP_MS) {
        this.fastSince = nowMs;
        return this.set(this.scale * 1.15);
      }
    } else {
      this.fastSince = null;
    }
    return null;
  }

  private set(s: number): number {
    this.scale = Math.min(Math.max(s, MIN_SCALE), MAX_SCALE);
    return this.scale;
  }
}
