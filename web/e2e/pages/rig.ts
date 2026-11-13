/**
 * Renders the reference sphere (orthographic, looking down -Z, 128 px, white) with Lumina's real
 * rendering pieces, so e2e/shading.spec.ts can compare against shared/sphere_reference.png:
 * - rig-lambert: MeshLambertMaterial, residual LightProbe + derived DirectionalLight (lightRig.ts)
 * - standard-probe: MeshStandardMaterial roughness 1, probe only
 * - standard-env-patched: + SH env map (PMREM) with the probe-diffuse patch (pbrPatch.ts)
 * - standard-env-unpatched: + SH env map without the patch (control: double counts diffuse)
 */
import {
  DataTexture,
  DirectionalLight,
  EquirectangularReflectionMapping,
  FloatType,
  LightProbe,
  LinearFilter,
  Mesh,
  MeshLambertMaterial,
  MeshStandardMaterial,
  NoToneMapping,
  OrthographicCamera,
  PMREMGenerator,
  RGBAFormat,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  WebGLRenderer,
  type Material,
} from 'three';
import { buildEnvironment } from '../../src/lighting/envMap';
import { deriveLightRig } from '../../src/lighting/lightRig';
import { toThreeSH, type RGB } from '../../src/lighting/sh';
import { useProbeDiffuse } from '../../src/render/pbrPatch';

export type RigMode =
  'rig-lambert' | 'standard-probe' | 'standard-env-patched' | 'standard-env-unpatched';

declare global {
  interface Window {
    __renderRig?: (sh: number[][], size: number, mode: RigMode) => number[];
  }
}

function material(mode: RigMode, renderer: WebGLRenderer, sh: RGB[]): Material {
  if (mode === 'rig-lambert') return new MeshLambertMaterial({ color: 0xffffff });
  const m = new MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
  if (mode === 'standard-probe') return m;
  const data = buildEnvironment(sh, 128, 64);
  const tex = new DataTexture(data, 128, 64, RGBAFormat, FloatType);
  tex.mapping = EquirectangularReflectionMapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.needsUpdate = true;
  m.envMap = new PMREMGenerator(renderer).fromEquirectangular(tex).texture;
  if (mode === 'standard-env-patched') useProbeDiffuse(m);
  return m;
}

function renderRig(shIn: number[][], size: number, mode: RigMode): number[] {
  const sh = shIn.map((c): RGB => [c[0] ?? 0, c[1] ?? 0, c[2] ?? 0]);
  const renderer = new WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(size, size, false);
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = NoToneMapping;
  renderer.setClearColor(0x000000, 1);
  const scene = new Scene();
  if (mode === 'rig-lambert') {
    const rig = deriveLightRig(sh);
    scene.add(new LightProbe(toThreeSH(rig.probe), 1));
    if (rig.light !== null) {
      const [r, g, b] = rig.light.irradiance;
      const peak = Math.max(r, g, b);
      const light = new DirectionalLight(0xffffff, peak);
      light.color.setRGB(r / peak, g / peak, b / peak);
      light.position.set(...rig.light.direction);
      scene.add(light, light.target);
    }
  } else {
    scene.add(new LightProbe(toThreeSH(sh), 1));
  }
  scene.add(new Mesh(new SphereGeometry(1, 256, 128), material(mode, renderer, sh)));
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 5);
  renderer.render(scene, camera);
  const gl = renderer.getContext();
  const bottomUp = new Uint8Array(size * size * 4);
  gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, bottomUp);
  const rgba = new Array<number>(size * size * 4);
  for (let row = 0; row < size; row++) {
    const src = (size - 1 - row) * size * 4;
    for (let i = 0; i < size * 4; i++) rgba[row * size * 4 + i] = bottomUp[src + i] ?? 0;
  }
  renderer.dispose();
  return rgba;
}

window.__renderRig = renderRig;
