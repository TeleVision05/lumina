import { describe, expect, it } from 'vitest';
import { MAX_SCALE, MIN_SCALE, qualityFor, ResolutionGovernor } from './quality';

describe('qualityFor', () => {
  it('treats SwiftShader and llvmpipe as software renderers', () => {
    expect(
      qualityFor('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0)))').software,
    ).toBe(true);
    expect(qualityFor('llvmpipe (LLVM 15.0.7, 256 bits)').samples).toBe(0);
  });

  it('keeps MSAA on real GPUs', () => {
    const q = qualityFor('ANGLE (Apple, ANGLE Metal Renderer: Apple M2 Max, Unspecified Version)');
    expect(q.software).toBe(false);
    expect(q.samples).toBe(4);
    expect(q.initialScale).toBe(1);
  });
});

/** Feed `seconds` of 60 Hz frames with the given interval and work, return the final scale. */
function run(
  g: ResolutionGovernor,
  seconds: number,
  intervalMs: number,
  workMs: number,
  t0 = 0,
): number {
  for (let t = t0; t < t0 + seconds * 1000; t += 1000 / 60) g.record(t, intervalMs, workMs);
  return g.scale;
}

describe('ResolutionGovernor', () => {
  it('steps the scale down while frames are slow, never below the minimum', () => {
    const g = new ResolutionGovernor(1);
    expect(run(g, 2.5, 50, 40)).toBeLessThan(1);
    expect(run(g, 30, 50, 40, 2500)).toBe(MIN_SCALE);
  });

  it('holds steady at 60 fps with moderate work', () => {
    const g = new ResolutionGovernor(0.8);
    expect(run(g, 10, 16.7, 10)).toBe(0.8);
  });

  it('climbs back only after sustained headroom, never above the maximum', () => {
    const g = new ResolutionGovernor(0.5);
    expect(run(g, 2.5, 16.7, 2)).toBe(0.5); // not yet: needs 3 s of headroom
    expect(run(g, 60, 16.7, 2, 2500)).toBe(MAX_SCALE);
  });
});
