/**
 * Reads a downsampled 128x96 (MODEL_W x MODEL_H) centre-4:3 FrameSample from a WebXR
 * `camera-access` texture (`XRWebGLBinding.getCameraImage`). Calls `renderer.resetState()`
 * afterward so Three.js's cached GL state stays consistent.
 */
import type { WebGLRenderer } from 'three';
import type { FrameSample } from '../capture/frame';
import { cropRect43, MODEL_H, MODEL_W } from '../capture/preprocess';

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec4 uCrop; // (u0, v0, uSpan, vSpan)
in vec2 vUv;
out vec4 outColor;
void main() {
  // Flip Y so FBO row 0 (returned first by readPixels) holds the top of the camera image.
  vec2 uv = vec2(uCrop.x + vUv.x * uCrop.z, uCrop.y + (1.0 - vUv.y) * uCrop.w);
  outColor = vec4(texture(uTex, uv).rgb, 1.0);
}`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type);
  if (s === null) throw new Error('Failed to create WebGL shader');
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (gl.getShaderParameter(s, gl.COMPILE_STATUS) !== true) {
    const log = gl.getShaderInfoLog(s) ?? 'unknown error';
    gl.deleteShader(s);
    throw new Error(`WebXR camera shader error: ${log}`);
  }
  return s;
}

function createProgram(gl: WebGL2RenderingContext): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, VERT);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (gl.getProgramParameter(prog, gl.LINK_STATUS) === false) {
    const log = gl.getProgramInfoLog(prog) ?? 'unknown error';
    gl.deleteProgram(prog);
    throw new Error(`WebXR camera program link error: ${log}`);
  }
  return prog;
}

export class XrCameraGrabber {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly uTex: WebGLUniformLocation | null;
  private readonly uCrop: WebGLUniformLocation | null;
  private readonly vao: WebGLVertexArrayObject;
  private readonly vbo: WebGLBuffer;
  private readonly fbo: WebGLFramebuffer;
  private readonly targetTex: WebGLTexture;
  private readonly pixels = new Uint8Array(MODEL_W * MODEL_H * 4);

  constructor(private readonly renderer: WebGLRenderer) {
    const ctx = renderer.getContext();
    if (!(ctx instanceof WebGL2RenderingContext)) {
      throw new Error('WebGL2 is required for WebXR camera-access');
    }
    this.gl = ctx;
    this.program = createProgram(ctx);
    this.uTex = ctx.getUniformLocation(this.program, 'uTex');
    this.uCrop = ctx.getUniformLocation(this.program, 'uCrop');

    const vao = ctx.createVertexArray();
    const vbo = ctx.createBuffer();
    const fbo = ctx.createFramebuffer();
    const tex = ctx.createTexture();
    this.vao = vao;
    this.vbo = vbo;
    this.fbo = fbo;
    this.targetTex = tex;

    ctx.bindVertexArray(vao);
    ctx.bindBuffer(ctx.ARRAY_BUFFER, vbo);
    ctx.bufferData(
      ctx.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      ctx.STATIC_DRAW,
    );
    const loc = ctx.getAttribLocation(this.program, 'aPos');
    ctx.enableVertexAttribArray(loc);
    ctx.vertexAttribPointer(loc, 2, ctx.FLOAT, false, 0, 0);
    ctx.bindVertexArray(null);

    ctx.bindTexture(ctx.TEXTURE_2D, tex);
    ctx.texImage2D(
      ctx.TEXTURE_2D,
      0,
      ctx.RGBA8,
      MODEL_W,
      MODEL_H,
      0,
      ctx.RGBA,
      ctx.UNSIGNED_BYTE,
      null,
    );
    ctx.texParameteri(ctx.TEXTURE_2D, ctx.TEXTURE_MIN_FILTER, ctx.LINEAR);
    ctx.texParameteri(ctx.TEXTURE_2D, ctx.TEXTURE_MAG_FILTER, ctx.LINEAR);

    ctx.bindFramebuffer(ctx.FRAMEBUFFER, fbo);
    ctx.framebufferTexture2D(ctx.FRAMEBUFFER, ctx.COLOR_ATTACHMENT0, ctx.TEXTURE_2D, tex, 0);
    ctx.bindFramebuffer(ctx.FRAMEBUFFER, null);
    this.renderer.resetState();
  }

  grab(cameraTexture: WebGLTexture, width: number, height: number): FrameSample | null {
    if (width <= 0 || height <= 0) return null;
    const gl = this.gl;
    const r = cropRect43(width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, MODEL_W, MODEL_H);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cameraTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(this.uTex, 0);
    gl.uniform4f(this.uCrop, r.sx / width, r.sy / height, r.sw / width, r.sh / height);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.readPixels(0, 0, MODEL_W, MODEL_H, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.renderer.resetState();
    return {
      rgba: new Uint8ClampedArray(this.pixels),
      width: MODEL_W,
      height: MODEL_H,
      time: performance.now(),
    };
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteFramebuffer(this.fbo);
    gl.deleteTexture(this.targetTex);
    gl.deleteBuffer(this.vbo);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
  }
}
