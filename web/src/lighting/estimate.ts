/**
 * What every estimator returns, and the single place where an estimate is mirrored for a mirrored
 * (selfie) display. Estimators always work on the un-mirrored sensor image, so their output is in
 * the sensor's camera frame (docs/CONVENTIONS.md, sections 1 and 2).
 */
import { dominantDirection, mirrorX, type SH9, type Vec3 } from './sh';

export interface LightingEstimate {
  /** Linear radiance SH9 in the camera frame, sh[i] = [r, g, b]. */
  sh: SH9;
  /** Unit direction toward the dominant light, or null when the light is flat. */
  direction: Vec3 | null;
  /** 0 for uniform light, 1 for a single point light (luminance band 1 over band 0). */
  directionality: number;
  /** 0..1: how far the estimator trusts this estimate. Shadows fade with it. */
  confidence: number;
}

export function makeEstimate(sh: SH9, confidence: number): LightingEstimate {
  const { direction, directionality } = dominantDirection(sh);
  return { sh, direction, directionality, confidence: Math.min(Math.max(confidence, 0), 1) };
}

/**
 * The estimate in the frame of what the visitor sees. When the video is displayed mirrored, the
 * left and right of the scene swap, so the lighting must swap too. Apply exactly once, here.
 */
export function forDisplay(estimate: LightingEstimate, mirrored: boolean): LightingEstimate {
  if (!mirrored) return estimate;
  return makeEstimate(mirrorX(estimate.sh), estimate.confidence);
}
