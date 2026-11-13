/** Globals that the e2e harness pages expose to Playwright specs. */
export interface SphereRender {
  /** RGBA bytes, row 0 at the top, length size * size * 4. */
  rgba: number[];
  glRenderer: string;
}

export interface ModelSmoke {
  inputNames: readonly string[];
  outputNames: readonly string[];
  outputDims: readonly number[];
  outputType: string;
  allFinite: boolean;
}

declare global {
  interface Window {
    __renderSphere?: (sh: number[][], size: number) => SphereRender;
    __runModel?: () => Promise<ModelSmoke>;
  }
}
