/**
 * Fixed studio lighting: the "before" of the comparison. A neutral key light from the upper left
 * of the picture over a soft, slightly cool fill, the way a product shot is lit. It ignores the
 * camera entirely, which is exactly the point.
 */
import { makeEstimate, type LightingEstimate } from '../lighting/estimate';
import type { SH9 } from '../lighting/sh';
import { addDirectional, uniformSH } from '../lighting/shBuild';
import type { Estimator } from './types';

/** Key from the upper left, slightly in front of the object (display frame). */
export const STUDIO_KEY_DIRECTION = [-0.5, 0.7, 0.5] as const;

/** A white matte ball under this rig peaks near 0.85 linear: bright but not clipped. */
export const STUDIO_SH: SH9 = addDirectional(
  uniformSH([0.27, 0.28, 0.3]),
  STUDIO_KEY_DIRECTION,
  [1.78, 1.74, 1.68],
);

export class StudioEstimator implements Estimator {
  readonly id = 'studio';
  readonly displayFrame = true;
  private readonly fixed: LightingEstimate = makeEstimate(STUDIO_SH, 1);

  estimate(): Promise<LightingEstimate> {
    return Promise.resolve(this.fixed);
  }

  dispose(): void {
    // Nothing to release.
  }
}
