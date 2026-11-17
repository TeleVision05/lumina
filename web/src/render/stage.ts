/**
 * The three.js stage: a transparent canvas over the video, a camera at the origin looking down -Z
 * (world frame = camera frame, docs/CONVENTIONS.md section 1), the ground, the object and two
 * light rigs. Renders one rig, the other, or each on one side of a vertical split.
 *
 * Tone mapping: none. The contract says the estimate is "the lighting as this camera exposure sees
 * it", so linear -> sRGB with no curve is what makes a white matte ball match the footage. The
 * three.js parity test measured exactly this pipeline (NoToneMapping, SRGBColorSpace).
 */
import {
  HalfFloatType,
  NoToneMapping,
  PCFShadowMap,
  PerspectiveCamera,
  Plane,
  Raycaster,
  Scene,
  SRGBColorSpace,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { CAMERA_HEIGHT, Ground } from './ground';
import { createObject, type ObjectId, type SceneObject } from './objects';
import { PostPass, type PostSettings } from './post';
import { qualityFor, rendererName, ResolutionGovernor, type Quality } from './quality';
import { createPmrem, RigNode } from './rigNode';

/** Assumed vertical field of view of a laptop or phone camera, degrees (not measured). */
export const CAMERA_VFOV_DEG = 55;

export type View = 'fixed' | 'lumina' | 'split';

export interface Placement {
  /** Ground-local position (x right, z toward the camera) of the object's foot. */
  x: number;
  z: number;
  scale: number;
}

export const MIN_DISTANCE = 1.4;
export const MAX_DISTANCE = 9;
/**
 * Object scale in AR. Scene units become metres there; the objects are about one unit tall, so
 * 0.3 gives a 30 cm ball, a believable size to stand on a floor or a table.
 */
export const AR_OBJECT_SCALE = 0.3;

/** Keep the object in front of the camera and within a sensible distance range. */
export function clampPlacement(p: Placement): Placement {
  const z = Math.min(Math.max(p.z, -MAX_DISTANCE), -MIN_DISTANCE);
  const xMax = Math.abs(z) * 1.2;
  const x = Math.min(Math.max(p.x, -xMax), xMax);
  return { x, z, scale: Math.min(Math.max(p.scale, 0.3), 3) };
}

export class Stage {
  readonly renderer: WebGLRenderer;
  readonly camera = new PerspectiveCamera(CAMERA_VFOV_DEG, 4 / 3, 0.05, 100);
  readonly scene = new Scene();
  readonly ground = new Ground();
  readonly fixedRig: RigNode;
  readonly luminaRig: RigNode;
  object: SceneObject;
  placement: Placement = { x: 0, z: -3.2, scale: 1 };
  readonly quality: Quality;
  readonly governor: ResolutionGovernor;
  private readonly target: WebGLRenderTarget;
  private readonly post = new PostPass();
  private readonly raycaster = new Raycaster();
  private size = { width: 1, height: 1, dpr: 1 };
  private arMode = false;
  private objectVisible = true;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.renderer = new WebGLRenderer({
      canvas,
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
    });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = NoToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.renderer.setClearColor(0x000000, 0);
    this.quality = qualityFor(rendererName(this.renderer.getContext()));
    this.governor = new ResolutionGovernor(this.quality.initialScale);
    this.target = new WebGLRenderTarget(1, 1, {
      type: HalfFloatType,
      samples: this.quality.samples,
    });
    const pmrem = createPmrem(this.renderer);
    this.fixedRig = new RigNode('fixed-rig', pmrem, this.quality);
    this.luminaRig = new RigNode('lumina-rig', pmrem, this.quality);
    this.scene.add(this.ground.group, this.fixedRig.root, this.luminaRig.root);
    this.object = createObject('matte');
    this.ground.group.add(this.object.group);
    this.applyPlacement();
  }

  /** Viewport size in CSS pixels and the device pixel ratio (before the render scale). */
  resize(width: number, height: number, dpr: number): void {
    this.size = { width, height, dpr };
    // three.js owns the drawing buffer size while an XR session presents.
    if (this.renderer.xr.isPresenting) return;
    const ratio = dpr * this.governor.scale;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.target.setSize(
      Math.max(1, Math.round(width * ratio)),
      Math.max(1, Math.round(height * ratio)),
    );
  }

  /** Feed one frame's timing to the resolution governor; re-sizes when the scale changes. */
  recordFrame(nowMs: number, intervalMs: number, workMs: number): void {
    if (this.arMode) return;
    if (this.governor.record(nowMs, intervalMs, workMs) !== null) {
      this.resize(this.size.width, this.size.height, this.size.dpr);
    }
  }

  setObject(id: ObjectId): void {
    if (id === this.object.id) return;
    this.ground.group.remove(this.object.group);
    this.object.dispose();
    this.object = createObject(id);
    this.object.group.visible = this.objectVisible;
    this.ground.group.add(this.object.group);
    this.applyPlacement();
  }

  setPlacement(p: Placement): void {
    this.placement = clampPlacement(p);
    this.applyPlacement();
  }

  /**
   * AR mode: the ground follows the tracked floor (set by the WebXR session) and the object
   * stands at the ground origin at a real-world size; it stays hidden until it is placed.
   */
  setArMode(on: boolean): void {
    this.arMode = on;
    this.setObjectVisible(!on);
    if (!on) this.ground.setTilt(this.ground.tiltRad);
    this.applyPlacement();
  }

  setObjectVisible(v: boolean): void {
    this.objectVisible = v;
    this.object.group.visible = v;
    this.ground.group.visible = v;
  }

  private applyPlacement(): void {
    if (this.arMode) {
      const s = AR_OBJECT_SCALE;
      this.object.group.position.set(0, 0, 0);
      this.object.group.scale.setScalar(s);
      this.ground.placeContact(0, 0, this.object.radius * 2.8 * s, 0.32);
      return;
    }
    const { x, z, scale } = this.placement;
    this.object.group.position.set(x, 0, z);
    this.object.group.scale.setScalar(scale);
    this.ground.placeContact(x, z, this.object.radius * 2.8 * scale, 0.32);
  }

  /** Restore the non-XR camera-frame ground tilt and placement after exiting WebXR. */
  restoreFromXR(tiltDeg: number): void {
    this.setArMode(false);
    this.ground.setTilt((tiltDeg * Math.PI) / 180);
    this.applyPlacement();
  }

  /** Changes whenever anything that casts or receives the shadow changes. */
  sceneKey(): string {
    const p = this.object.group.position;
    const g = this.ground.group.position;
    const scale = this.object.group.scale.x;
    const f = (v: number): string => v.toFixed(3);
    return [
      this.object.id,
      `${f(p.x)},${f(p.z)},${f(scale)}`,
      `${f(g.x)},${f(g.y)},${f(g.z)},${this.ground.tiltRad.toFixed(4)}`,
      this.objectVisible ? 'v' : 'h',
    ].join('|');
  }

  /** World centre and radius of the object (for light rigs). */
  objectBounds(): { center: Vector3; radius: number } {
    const p = this.object.group.position;
    const scale = this.object.group.scale.x;
    const center = this.ground.toWorld(p.x, this.object.centerHeight * scale, p.z);
    return { center, radius: this.object.radius * scale };
  }

  /** Ground-local point under normalised device coordinates, or null above the horizon. */
  pickGround(ndcX: number, ndcY: number): { x: number; z: number } | null {
    this.raycaster.setFromCamera(new Vector2(ndcX, ndcY), this.camera);
    const n = new Vector3(...this.ground.normal);
    const hit = this.raycaster.ray.intersectPlane(new Plane(n, CAMERA_HEIGHT), new Vector3());
    if (hit === null) return null;
    const local = this.ground.group.worldToLocal(hit);
    return { x: local.x, z: local.z };
  }

  /** Screen rectangle (CSS pixels) around the object, for tests and focus outlines. */
  objectScreenRect(width: number, height: number): { x: number; y: number; w: number; h: number } {
    const { center, radius } = this.objectBounds();
    const p = center.clone().project(this.camera);
    const dist = center.length();
    const halfH = (radius / (dist * Math.tan((CAMERA_VFOV_DEG * Math.PI) / 360))) * (height / 2);
    const cx = ((p.x + 1) / 2) * width;
    const cy = ((1 - p.y) / 2) * height;
    return { x: cx - halfH, y: cy - halfH, w: 2 * halfH, h: 2 * halfH };
  }

  private renderRig(rig: RigNode, other: RigNode, x: number, w: number, h: number): void {
    rig.visible = true;
    other.visible = false;
    this.ground.shadowMaterial.opacity = rig.shadow;
    for (const m of this.object.envMaterials) m.envMap = rig.environment;
    this.target.scissor.set(x, 0, w, h);
    this.target.scissorTest = true;
    this.renderer.setRenderTarget(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  /** Render the frame. `splitX` is the split position in [0, 1] of the width (split view only). */
  render(view: View, splitX: number, post: PostSettings): void {
    const w = this.target.width;
    const h = this.target.height;
    if (view === 'fixed') this.renderRig(this.fixedRig, this.luminaRig, 0, w, h);
    else if (view === 'lumina') this.renderRig(this.luminaRig, this.fixedRig, 0, w, h);
    else {
      const cut = Math.round(Math.min(Math.max(splitX, 0), 1) * w);
      this.renderRig(this.fixedRig, this.luminaRig, 0, cut, h);
      this.renderRig(this.luminaRig, this.fixedRig, cut, w - cut, h);
    }
    this.post.render(this.renderer, this.target.texture, w, h, post);
  }

  /** Direct WebXR render (bypasses the 2D post-processing target so stereo/XR projection works). */
  renderXR(view: View): void {
    const rig = view === 'fixed' ? this.fixedRig : this.luminaRig;
    const other = view === 'fixed' ? this.luminaRig : this.fixedRig;
    rig.visible = true;
    other.visible = false;
    this.ground.shadowMaterial.opacity = rig.shadow;
    for (const m of this.object.envMaterials) m.envMap = rig.environment;
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.fixedRig.dispose();
    this.luminaRig.dispose();
    this.object.dispose();
    this.post.dispose();
    this.target.dispose();
    this.renderer.dispose();
  }
}
