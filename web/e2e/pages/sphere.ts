/**
 * Renders a white Lambertian unit sphere lit only by a three.js LightProbe, with the camera and
 * pixel layout of shared/sphere_reference.png (orthographic, looking down -Z, x and y in [-1, 1]).
 */
import {
  LightProbe,
  Mesh,
  MeshLambertMaterial,
  NoToneMapping,
  OrthographicCamera,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  WebGLRenderer,
} from 'three';
import { toThreeSH, type RGB } from '../../src/lighting/sh';
import type { SphereRender } from './harness';

function glRendererName(gl: WebGLRenderingContext | WebGL2RenderingContext): string {
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const name: unknown = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null;
  return typeof name === 'string' ? name : 'unknown';
}

function renderSphere(sh: number[][], size: number): SphereRender {
  const canvas = document.createElement('canvas');
  const renderer = new WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(size, size, false);
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = NoToneMapping;
  renderer.setClearColor(0x000000, 1);

  const scene = new Scene();
  const coeffs = sh.map((c): RGB => [c[0] ?? 0, c[1] ?? 0, c[2] ?? 0]);
  scene.add(new LightProbe(toThreeSH(coeffs), 1));
  const material = new MeshLambertMaterial({ color: 0xffffff });
  scene.add(new Mesh(new SphereGeometry(1, 256, 128), material));

  // Camera frame == world frame (identity rotation), as in Lumina v1.
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 5);
  renderer.render(scene, camera);

  const gl = renderer.getContext();
  const bottomUp = new Uint8Array(size * size * 4);
  gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, bottomUp);
  const rgba: number[] = new Array<number>(size * size * 4);
  for (let row = 0; row < size; row++) {
    const src = (size - 1 - row) * size * 4;
    for (let i = 0; i < size * 4; i++) rgba[row * size * 4 + i] = bottomUp[src + i] ?? 0;
  }
  const result = { rgba, glRenderer: glRendererName(gl) };
  renderer.dispose();
  return result;
}

window.__renderSphere = renderSphere;
