/**
 * Split an SH9 lighting estimate into a three.js light rig: a DirectionalLight (which can cast a
 * shadow and gives a sharp terminator) plus a LightProbe holding everything else.
 *
 * The directional light is a distant delta light from the dominant direction d whose irradiance
 * on a surface facing it is e (linear RGB). Its own SH projection is e * Y_i(d) for all nine
 * coefficients (docs/CONVENTIONS.md, section 3), so that is what is subtracted from the probe:
 * light is never counted twice. The rendered irradiance becomes
 *
 *   E(n) = E_probe(n) + e * max(n . d, 0)  ~=  E_estimate(n)
 *
 * up to the band-2 truncation of the clamped cosine (measured in lightRig.test.ts).
 *
 * How much of band 1 goes to the light ("share") rises smoothly with directionality, so flat,
 * overcast light gets no directional light and no hard shadow.
 */
import { irradiance, LUMA, shBasis, dominantDirection, type RGB, type SH9, type Vec3 } from './sh';

const Y1 = 0.488603;

export interface RigOptions {
  /** Largest fraction of the band-1 energy given to the directional light. */
  maxShare: number;
  /** Directionality below which there is no directional light. */
  onset: number;
  /** Directionality at which the share reaches maxShare. */
  full: number;
}

export const DEFAULT_RIG_OPTIONS: RigOptions = { maxShare: 0.85, onset: 0.06, full: 0.45 };

export interface DirectionalPart {
  /** Unit vector toward the light, camera frame. */
  direction: Vec3;
  /** Irradiance on a surface facing the light, linear RGB (three.js color * intensity). */
  irradiance: RGB;
}

export interface LightRig {
  /** What remains for the LightProbe (radiance SH). */
  probe: SH9;
  light: DirectionalPart | null;
  /** Fraction of the band-1 energy assigned to the light, 0..maxShare. */
  share: number;
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
}

export function deriveLightRig(sh: SH9, opts: RigOptions = DEFAULT_RIG_OPTIONS): LightRig {
  const { direction: d, directionality } = dominantDirection(sh);
  const share = d === null ? 0 : opts.maxShare * smoothstep(opts.onset, opts.full, directionality);
  if (d === null || share <= 0) return { probe: sh, light: null, share: 0 };

  const basis = shBasis(d);
  const l0 = sh[0] ?? [0, 0, 0];
  const e = [0, 1, 2].map((c) => {
    // Band-1 vector of this channel projected on d; a delta light of irradiance e has e*Y1*d.
    const v = (sh[3]?.[c] ?? 0) * d[0] + (sh[1]?.[c] ?? 0) * d[1] + (sh[2]?.[c] ?? 0) * d[2];
    const full = Math.max(v, 0) / Y1;
    // Never take more than 90% of the ambient term of a channel (guards noisy network output).
    const cap = (0.9 * Math.max(l0[c] ?? 0, 0)) / (basis[0] ?? 1);
    return Math.min(share * full, cap);
  });
  const light: RGB = [e[0] ?? 0, e[1] ?? 0, e[2] ?? 0];
  const probe = sh.map((c, i): RGB => {
    const y = basis[i] ?? 0;
    return [c[0] - light[0] * y, c[1] - light[1] * y, c[2] - light[2] * y];
  });
  return { probe, light: { direction: d, irradiance: light }, share };
}

const lum = (c: RGB): number => LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2];

/**
 * Opacity of the shadow on a ground plane with unit normal `groundNormal`: the fraction of the
 * ground's irradiance that the directional light supplies, i.e. how much darker the ground gets
 * where the object blocks that light. 0 in flat light or when the light is below the ground.
 */
export function shadowOpacity(sh: SH9, rig: LightRig, groundNormal: Vec3): number {
  if (rig.light === null) return 0;
  const [dx, dy, dz] = rig.light.direction;
  const cos = dx * groundNormal[0] + dy * groundNormal[1] + dz * groundNormal[2];
  if (cos <= 0) return 0;
  const total = lum(irradiance(sh, groundNormal));
  if (!(total > 0)) return 0;
  return Math.min(Math.max((lum(rig.light.irradiance) * cos) / total, 0), 1);
}
