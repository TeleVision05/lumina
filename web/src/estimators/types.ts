/** The one interface every lighting estimator implements. */
import type { FrameSample } from '../capture/frame';
import type { LightingEstimate } from '../lighting/estimate';

export type EstimatorId = 'studio' | 'stats' | 'neural';

export interface Estimator {
  readonly id: EstimatorId;
  /**
   * True when the estimate is already in the display frame and must not be mirrored (the fixed
   * studio rig). False for estimators that read the un-mirrored sensor image.
   */
  readonly displayFrame: boolean;
  /**
   * Given a frame (un-mirrored, model resolution; null when no picture is available), return SH9
   * radiance, the dominant-light direction and a confidence. Rejects when the estimator cannot run.
   */
  estimate(frame: FrameSample | null): Promise<LightingEstimate>;
  dispose(): void;
}

export const ESTIMATOR_LABELS: Record<EstimatorId, string> = {
  studio: 'Fixed studio light',
  stats: 'Image statistics',
  neural: 'Neural network',
};
