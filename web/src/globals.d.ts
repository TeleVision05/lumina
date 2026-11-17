/** File names of the sample clips in public/samples/, injected at build time (vite/plugins.ts). */
declare const __LUMINA_SAMPLES__: readonly string[];

/** WebXR Raw Camera Access & Lighting Estimation APIs (not yet in TypeScript's lib.dom.d.ts). */
interface XRCamera {
  readonly width: number;
  readonly height: number;
}

type XRLightProbe = EventTarget;

interface XRLightEstimate {
  readonly sphericalHarmonicsCoefficients: Float32Array;
}

interface XRView {
  readonly camera?: XRCamera | null;
}

interface XRWebGLBinding {
  getCameraImage(camera: XRCamera): WebGLTexture | null;
}

interface XRSession {
  requestLightProbe?: (options?: { reflectionFormat?: string }) => Promise<XRLightProbe>;
}

interface XRFrame {
  getLightEstimate?: (lightProbe: XRLightProbe) => XRLightEstimate | null;
}
