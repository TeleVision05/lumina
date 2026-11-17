/**
 * Small pieces of the WebXR placement: the floor reticle, and the ray tests that decide whether
 * a screen tap is aimed at the reticle or at the placed object (taps anywhere else are ignored).
 */
import {
  CircleGeometry,
  Group,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  Quaternion,
  Ray,
  RingGeometry,
  Sphere,
  Vector3,
} from 'three';

/** Ring radius (metres) and how far outside it a tap still counts as "on the ring". */
export const RETICLE_RADIUS = 0.175;
export const RETICLE_TAP_RADIUS = 0.3;
/** Hits whose surface normal is within ~35 degrees of straight up count as floor. */
export const MIN_FLOOR_UP = 0.82;

export interface Reticle {
  group: Group;
  dispose: () => void;
}

export function createReticle(): Reticle {
  const group = new Group();
  const ringGeo = new RingGeometry(RETICLE_RADIUS * 0.8, RETICLE_RADIUS, 48).rotateX(-Math.PI / 2);
  const dotGeo = new CircleGeometry(0.022, 24).rotateX(-Math.PI / 2);
  const mat = new MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });
  const ring = new Mesh(ringGeo, mat);
  const dot = new Mesh(dotGeo, mat);
  ring.renderOrder = dot.renderOrder = 3;
  group.add(ring, dot);
  group.matrixAutoUpdate = false;
  group.visible = false;
  return {
    group,
    dispose: () => {
      ringGeo.dispose();
      dotGeo.dispose();
      mat.dispose();
    },
  };
}

/** World position and up-ness of a hit pose matrix (column-major 4x4, as WebXR gives it). */
export function poseInfo(matrix: ArrayLike<number>): { position: Vector3; up: number } {
  const m = new Matrix4().fromArray(Array.from(matrix));
  const position = new Vector3();
  const q = new Quaternion();
  m.decompose(position, q, new Vector3());
  const up = new Vector3(0, 1, 0).applyQuaternion(q).y;
  return { position, up };
}

/** A ray from an XR pose (origin at the pose, pointing down its -Z axis). */
export function rayFromPose(transform: XRRigidTransform): Ray {
  const { position: p, orientation: o } = transform;
  const q = new Quaternion(o.x, o.y, o.z, o.w);
  return new Ray(new Vector3(p.x, p.y, p.z), new Vector3(0, 0, -1).applyQuaternion(q).normalize());
}

/** True when the ray meets the horizontal plane through `center` within `radius` of it. */
export function rayHitsDisc(ray: Ray, center: Vector3, radius: number): boolean {
  const dy = ray.direction.y;
  if (Math.abs(dy) < 1e-6) return false;
  const t = (center.y - ray.origin.y) / dy;
  if (t <= 0) return false;
  const p = ray.at(t, new Vector3());
  return Math.hypot(p.x - center.x, p.z - center.z) <= radius;
}

export function rayHitsSphere(ray: Ray, center: Vector3, radius: number): boolean {
  return ray.intersectsSphere(new Sphere(center, radius));
}
