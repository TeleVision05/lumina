/**
 * Final composite of the rendered object over the (CSS) video: an optional slight blur and film
 * grain so the CG object does not look cleaner than the footage around it. The scene is rendered
 * linear into a multisampled float target; this pass converts to the display's sRGB, adds grain in
 * that space (where sensor noise lives), and writes premultiplied alpha for the transparent canvas.
 * Only covered pixels change: the video itself is never touched.
 */
import {
  Mesh,
  NoBlending,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  Vector2,
  type Texture,
  type WebGLRenderer,
} from 'three';

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const fragmentShader = /* glsl */ `
uniform sampler2D tScene;
uniform vec2 texel;
uniform float blur;
uniform float grain;
uniform float seed;
varying vec2 vUv;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21) + seed);
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

vec4 tap(vec2 o) { return texture2D(tScene, vUv + o * texel * blur); }

void main() {
  vec4 c = texture2D(tScene, vUv);
  if (blur > 0.0) {
    // 3x3 tent kernel (1 2 1 / 2 4 2 / 1 2 1) / 16 on premultiplied colour.
    c = c * 4.0
      + 2.0 * (tap(vec2(1.0, 0.0)) + tap(vec2(-1.0, 0.0)) + tap(vec2(0.0, 1.0)) + tap(vec2(0.0, -1.0)))
      + tap(vec2(1.0, 1.0)) + tap(vec2(-1.0, 1.0)) + tap(vec2(1.0, -1.0)) + tap(vec2(-1.0, -1.0));
    c /= 16.0;
  }
  float a = clamp(c.a, 0.0, 1.0);
  if (a <= 0.0) { gl_FragColor = vec4(0.0); return; }
  vec4 display = linearToOutputTexel(vec4(c.rgb / a, 1.0));
  float n = (hash(gl_FragCoord.xy) + hash(gl_FragCoord.yx + 17.0)) - 1.0;
  vec3 rgb = clamp(display.rgb + n * grain, 0.0, 1.0);
  gl_FragColor = vec4(rgb * a, a);
}`;

export interface PostSettings {
  /** Blur radius in device pixels, 0 to about 1.5. */
  blur: number;
  /** Grain amplitude in sRGB units, 0 to about 0.06. */
  grain: number;
  /** Changes every frame for moving grain; constant when motion is reduced. */
  seed: number;
}

export class PostPass {
  private readonly scene = new Scene();
  private readonly camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly material: ShaderMaterial;

  constructor() {
    this.material = new ShaderMaterial({
      uniforms: {
        tScene: { value: null },
        texel: { value: new Vector2(1, 1) },
        blur: { value: 0 },
        grain: { value: 0 },
        seed: { value: 0 },
      },
      vertexShader,
      fragmentShader,
      blending: NoBlending,
      depthTest: false,
      depthWrite: false,
    });
    this.scene.add(new Mesh(new PlaneGeometry(2, 2), this.material));
  }

  render(
    renderer: WebGLRenderer,
    input: Texture,
    width: number,
    height: number,
    s: PostSettings,
  ): void {
    const u = this.material.uniforms;
    if (u.tScene) u.tScene.value = input;
    (u.texel?.value as Vector2 | undefined)?.set(1 / width, 1 / height);
    if (u.blur) u.blur.value = s.blur;
    if (u.grain) u.grain.value = s.grain;
    if (u.seed) u.seed.value = s.seed;
    renderer.setRenderTarget(null);
    renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.material.dispose();
  }
}
