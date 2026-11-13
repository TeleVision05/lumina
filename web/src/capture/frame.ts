/**
 * A downsampled, un-mirrored frame shared by every estimator. Grabbed from whatever element the
 * current source drives (camera video, sample video, dropped video or image).
 */
import { boxDownsample, cropRect43, MODEL_H, MODEL_W, OVERSAMPLE } from './preprocess';

export interface FrameSample {
  /** RGBA sRGB bytes, MODEL_W x MODEL_H, row 0 at the top, never mirrored. */
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  /** performance.now() when grabbed, ms. */
  time: number;
}

export type FrameElement = HTMLVideoElement | HTMLImageElement;

export function elementSize(el: FrameElement): { width: number; height: number } {
  if (el instanceof HTMLVideoElement) return { width: el.videoWidth, height: el.videoHeight };
  return { width: el.naturalWidth, height: el.naturalHeight };
}

export function elementReady(el: FrameElement): boolean {
  const { width, height } = elementSize(el);
  if (width === 0 || height === 0) return false;
  if (el instanceof HTMLVideoElement) return el.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
  return el.complete;
}

export class FrameGrabber {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = MODEL_W * OVERSAMPLE;
    this.canvas.height = MODEL_H * OVERSAMPLE;
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
    if (ctx === null) throw new Error('2D canvas is not available');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    this.ctx = ctx;
  }

  /** Centre-cropped 4:3 frame at model resolution, or null while the element has no picture. */
  grab(el: FrameElement): FrameSample | null {
    if (!elementReady(el)) return null;
    const { width, height } = elementSize(el);
    const r = cropRect43(width, height);
    const { width: cw, height: ch } = this.canvas;
    this.ctx.drawImage(el, r.sx, r.sy, r.sw, r.sh, 0, 0, cw, ch);
    const big = this.ctx.getImageData(0, 0, cw, ch).data;
    const rgba = boxDownsample(big, cw, ch, OVERSAMPLE);
    return { rgba, width: MODEL_W, height: MODEL_H, time: performance.now() };
  }
}
