/**
 * One complete three.js light rig built from a lighting estimate: LightProbe (residual SH),
 * DirectionalLight (dominant light, casts the shadow), a shadow opacity for the ground, and an
 * environment map for glossy materials. The app keeps two rigs, "fixed" and "Lumina", and shows
 * one, the other, or each on one side of the split.
 */
import {
  DataTexture,
  DirectionalLight,
  EquirectangularReflectionMapping,
  FloatType,
  LightProbe,
  LinearFilter,
  Object3D,
  PMREMGenerator,
  RGBAFormat,
  Vector3,
  type Texture,
  type WebGLRenderer,
  type WebGLCubeRenderTarget,
} from 'three';
import { buildEnvironment, type CameraPatch } from '../lighting/envMap';
import type { LightingEstimate } from '../lighting/estimate';
import { deriveLightRig, shadowOpacity, type LightRig } from '../lighting/lightRig';
import { toThreeSH, type Vec3 } from '../lighting/sh';

const ENV_W = 128;
const ENV_H = 64;
const LIGHT_DISTANCE = 8;

export interface RigContext {
  groundNormal: Vec3;
  /** World position of the object's centre and its radius (for the shadow camera). */
  center: Vector3;
  radius: number;
  /** 0..1 user scale on shadow darkness. */
  shadowScale: number;
  /** Camera picture for the forward hemisphere of the env map, or null. */
  camera: CameraPatch | null;
  /** Changes whenever the object, its placement or the ground changes (shadow map cache key). */
  sceneKey: string;
  /** False when no visible material reflects the environment: skip the costly PMREM rebuild. */
  needsEnv: boolean;
  nowMs: number;
}

export class RigNode {
  readonly root = new Object3D();
  readonly probe = new LightProbe();
  readonly light = new DirectionalLight(0xffffff, 0);
  rig: LightRig | null = null;
  /** Opacity the ground's ShadowMaterial should use while this rig renders. */
  shadow = 0;
  private readonly envData = new Float32Array(ENV_W * ENV_H * 4);
  private readonly envTexture: DataTexture;
  private pmremTarget: WebGLCubeRenderTarget | null = null;
  private lastEnvMs = Number.NEGATIVE_INFINITY;
  private lastSh: number[] = [];
  private shadowKey = '';

  constructor(
    name: string,
    private readonly pmrem: PMREMGenerator,
    private readonly opts: { shadowMapSize: number; envIntervalMs: number },
  ) {
    this.root.name = name;
    this.light.castShadow = true;
    this.light.shadow.mapSize.set(opts.shadowMapSize, opts.shadowMapSize);
    this.light.shadow.bias = -0.0005;
    this.light.shadow.normalBias = 0.02;
    this.root.add(this.probe, this.light, this.light.target);
    this.envTexture = new DataTexture(this.envData, ENV_W, ENV_H, RGBAFormat, FloatType);
    this.envTexture.mapping = EquirectangularReflectionMapping;
    this.envTexture.magFilter = LinearFilter;
    this.envTexture.minFilter = LinearFilter;
  }

  set visible(v: boolean) {
    this.root.visible = v;
  }

  /** The PMREM environment for env-mapped materials, or null before the first build. */
  get environment(): Texture | null {
    return this.pmremTarget?.texture ?? null;
  }

  update(estimate: LightingEstimate, ctx: RigContext): void {
    const rig = deriveLightRig(estimate.sh);
    this.rig = rig;
    toThreeSH(rig.probe, this.probe.sh);
    this.probe.intensity = 1;
    this.updateLight(rig, ctx);
    const strength = shadowOpacity(estimate.sh, rig, ctx.groundNormal);
    this.shadow = Math.min(1, strength * estimate.confidence * ctx.shadowScale);
    const flat = estimate.sh.flat();
    const same = ctx.camera === null && flat.every((v, i) => v === this.lastSh[i]);
    if (ctx.needsEnv && !same && ctx.nowMs - this.lastEnvMs >= this.opts.envIntervalMs) {
      this.lastEnvMs = ctx.nowMs;
      this.lastSh = flat;
      buildEnvironment(estimate.sh, ENV_W, ENV_H, ctx.camera, this.envData);
      this.envTexture.needsUpdate = true;
      this.pmremTarget = this.pmrem.fromEquirectangular(this.envTexture, this.pmremTarget);
    }
  }

  private updateLight(rig: LightRig, ctx: RigContext): void {
    const part = rig.light;
    if (part === null) {
      this.light.intensity = 0;
      return;
    }
    const [r, g, b] = part.irradiance;
    const peak = Math.max(r, g, b, 1e-6);
    this.light.color.setRGB(r / peak, g / peak, b / peak);
    this.light.intensity = peak;
    const [dx, dy, dz] = part.direction;
    this.light.target.position.copy(ctx.center);
    this.light.position.set(
      ctx.center.x + dx * LIGHT_DISTANCE,
      ctx.center.y + dy * LIGHT_DISTANCE,
      ctx.center.z + dz * LIGHT_DISTANCE,
    );
    // Re-render the shadow map only when the light, the object or the ground moved (directions
    // quantised to about 0.3 degrees). The fixed rig then renders its shadow once.
    const q = (v: number): number => Math.round(v * 200);
    const key = `${ctx.sceneKey}|${q(dx)},${q(dy)},${q(dz)}`;
    this.light.shadow.autoUpdate = false;
    if (key !== this.shadowKey) {
      this.shadowKey = key;
      this.light.shadow.needsUpdate = true;
    }
    const cam = this.light.shadow.camera;
    const half = ctx.radius * 2.2;
    cam.left = -half;
    cam.right = half;
    cam.top = half;
    cam.bottom = -half;
    cam.near = LIGHT_DISTANCE - ctx.radius * 4;
    cam.far = LIGHT_DISTANCE + ctx.radius * 6;
    cam.updateProjectionMatrix();
    // Softer edges for broader light: wider PCF kernel as directionality falls.
    this.light.shadow.radius = 2 + 6 * (1 - Math.min(rig.share / 0.85, 1));
  }

  /** Re-render the environment on the next update (after a renderer context change). */
  invalidate(): void {
    this.lastEnvMs = Number.NEGATIVE_INFINITY;
    this.lastSh = [];
    this.shadowKey = '';
  }

  dispose(): void {
    this.envTexture.dispose();
    this.pmremTarget?.dispose();
    this.light.dispose();
  }
}

export function createPmrem(renderer: WebGLRenderer): PMREMGenerator {
  return new PMREMGenerator(renderer);
}
