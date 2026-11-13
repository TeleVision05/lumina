/**
 * The ground the object stands on: an invisible plane that only shows shadows (ShadowMaterial),
 * plus a soft contact shadow so the object sits down even in flat light, where the directional
 * shadow fades out. In the camera frame the plane is tilted by the camera's downward pitch and
 * sits a fixed height below the camera.
 */
import {
  CanvasTexture,
  Group,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  ShadowMaterial,
  Vector3,
} from 'three';
import type { Vec3 } from '../lighting/sh';

/** Camera height above the ground, scene units (the objects are about one unit tall). */
export const CAMERA_HEIGHT = 1.1;

/** Unit normal of the ground in the camera frame for a camera pitched down by `tiltRad`. */
export function groundNormal(tiltRad: number): Vec3 {
  return [0, Math.cos(tiltRad), Math.sin(tiltRad)];
}

function radialTexture(): CanvasTexture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (ctx !== null) {
    // alphaMap reads the GREEN channel, so the falloff must be an opaque grey ramp: white with
    // varying alpha would read as 1 wherever alpha > 0 and give a hard-edged disc.
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, '#fff');
    g.addColorStop(0.3, '#8c8c8c');
    g.addColorStop(0.65, '#262626');
    g.addColorStop(1, '#000');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
  }
  return new CanvasTexture(canvas);
}

export class Ground {
  /** Ground frame: origin on the plane below the camera, local +Y = ground normal. */
  readonly group = new Group();
  readonly shadowMaterial = new ShadowMaterial({ opacity: 0.5, transparent: true });
  private readonly contactMaterial: MeshBasicMaterial;
  private readonly contact: Mesh;
  private tilt = 0;

  constructor() {
    const plane = new Mesh(new PlaneGeometry(60, 60), this.shadowMaterial);
    plane.rotation.x = -Math.PI / 2;
    plane.receiveShadow = true;
    plane.renderOrder = 1;
    this.contactMaterial = new MeshBasicMaterial({
      color: 0x000000,
      alphaMap: radialTexture(),
      transparent: true,
      depthWrite: false,
      opacity: 0.35,
    });
    this.contact = new Mesh(new PlaneGeometry(1, 1), this.contactMaterial);
    this.contact.rotation.x = -Math.PI / 2;
    this.contact.position.y = 0.002;
    this.contact.renderOrder = 2;
    this.group.add(plane, this.contact);
    this.setTilt(0.26);
  }

  setTilt(rad: number): void {
    this.tilt = rad;
    const [nx, ny, nz] = groundNormal(rad);
    this.group.rotation.set(rad, 0, 0);
    this.group.position.set(-nx * CAMERA_HEIGHT, -ny * CAMERA_HEIGHT, -nz * CAMERA_HEIGHT);
    this.group.updateMatrixWorld(true);
  }

  get tiltRad(): number {
    return this.tilt;
  }

  get normal(): Vec3 {
    return groundNormal(this.tilt);
  }

  /** Put the contact shadow under a point on the ground (ground-local x, z) with this size. */
  placeContact(x: number, z: number, footprint: number, strength: number): void {
    this.contact.position.set(x, 0.002, z);
    this.contact.scale.setScalar(footprint);
    this.contactMaterial.opacity = strength;
  }

  /** World position of a ground-local point. */
  toWorld(x: number, y: number, z: number, target = new Vector3()): Vector3 {
    return this.group.localToWorld(target.set(x, y, z));
  }
}
