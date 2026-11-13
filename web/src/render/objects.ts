/**
 * The virtual objects. All procedural (no third-party assets):
 * - matte: white Lambertian ball, the classic grey-ball light probe. Uses MeshLambertMaterial, the
 *   material the three.js parity test measured, so it shows the estimate exactly.
 * - chrome: mirror ball, the classic reflection probe. Shows the synthesised environment.
 * - vase: glazed ceramic (diffuse colour under a clear glaze), a real-world dielectric.
 * - knot: brushed brass, a rough metal.
 * Each object's local origin is the point where it touches the ground; local +Y is up.
 */
import {
  Color,
  Group,
  LatheGeometry,
  Mesh,
  MeshLambertMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  SphereGeometry,
  TorusKnotGeometry,
  Vector2,
  type BufferGeometry,
  type Material,
} from 'three';
import { useProbeDiffuse } from './pbrPatch';

export type ObjectId = 'matte' | 'chrome' | 'vase' | 'knot';

export const OBJECT_LABELS: Record<ObjectId, string> = {
  matte: 'Matte ball',
  chrome: 'Chrome ball',
  vase: 'Glazed vase',
  knot: 'Brass knot',
};

export const OBJECT_IDS: readonly ObjectId[] = ['matte', 'chrome', 'vase', 'knot'];

export interface SceneObject {
  id: ObjectId;
  group: Group;
  /** Radius of a sphere around the object, for the shadow camera and placement. */
  radius: number;
  /** Height of the object's centre above the ground, unscaled. */
  centerHeight: number;
  /** Materials that reflect the environment map (set per light rig). */
  envMaterials: MeshStandardMaterial[];
  dispose(): void;
}

function vaseProfile(): Vector2[] {
  const pts: Vector2[] = [];
  const n = 48;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const y = t * 1.1;
    // Foot, round belly, narrow neck, small flared lip.
    const belly = 0.34 * Math.sin(Math.PI * Math.min(t / 0.78, 1)) ** 0.8;
    const neck = 0.1 + 0.06 * Math.max(0, (t - 0.86) / 0.14) ** 2;
    const r = t < 0.04 ? 0.18 + t * 2 : Math.max(belly, t > 0.7 ? neck : 0.12);
    pts.push(new Vector2(r, y));
  }
  return pts;
}

interface Spec {
  geometry: BufferGeometry;
  material: Material;
  centerHeight: number;
  radius: number;
  yOffset: number;
}

function spec(id: ObjectId): Spec {
  switch (id) {
    case 'matte':
      return {
        geometry: new SphereGeometry(0.5, 128, 64),
        material: new MeshLambertMaterial({ color: new Color(0.8, 0.8, 0.8) }),
        centerHeight: 0.5,
        radius: 0.5,
        yOffset: 0.5,
      };
    case 'chrome':
      return {
        geometry: new SphereGeometry(0.5, 128, 64),
        material: new MeshStandardMaterial({ color: 0xffffff, metalness: 1, roughness: 0.05 }),
        centerHeight: 0.5,
        radius: 0.5,
        yOffset: 0.5,
      };
    case 'vase':
      return {
        geometry: new LatheGeometry(vaseProfile(), 96),
        material: new MeshPhysicalMaterial({
          color: new Color().setRGB(0.05, 0.2, 0.26),
          roughness: 0.5,
          clearcoat: 1,
          clearcoatRoughness: 0.06,
        }),
        centerHeight: 0.55,
        radius: 0.62,
        yOffset: 0,
      };
    case 'knot':
      return {
        geometry: new TorusKnotGeometry(0.3, 0.1, 256, 32),
        material: new MeshStandardMaterial({
          color: new Color().setRGB(0.83, 0.62, 0.3),
          metalness: 1,
          roughness: 0.32,
        }),
        centerHeight: 0.45,
        radius: 0.48,
        yOffset: 0.45,
      };
  }
}

export function createObject(id: ObjectId): SceneObject {
  const s = spec(id);
  const mesh = new Mesh(s.geometry, s.material);
  mesh.castShadow = true;
  mesh.position.y = s.yOffset;
  const group = new Group();
  group.name = `object-${id}`;
  group.add(mesh);
  const envMaterials: MeshStandardMaterial[] = [];
  if (s.material instanceof MeshStandardMaterial) {
    useProbeDiffuse(s.material);
    envMaterials.push(s.material);
  }
  return {
    id,
    group,
    radius: s.radius,
    centerHeight: s.centerHeight,
    envMaterials,
    dispose: () => {
      s.geometry.dispose();
      s.material.dispose();
    },
  };
}
