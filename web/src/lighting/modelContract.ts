/**
 * The model file contract (docs/CONVENTIONS.md, Model file). Mirror of
 * training/lumina_train/model_contract.py.
 */
export const MODEL_FILE = 'models/lumina-sh.onnx';
/** Model URL under the deploy base path (/lumina/ on GitHub Pages). Call it in the browser only. */
export function modelUrl(base: string = import.meta.env.BASE_URL): string {
  return `${base}${MODEL_FILE}`;
}
export const INPUT_NAME = 'image';
/** NCHW, RGB, sRGB-encoded values in [0, 1]; centre-crop to 4:3, then resize. */
export const INPUT_SHAPE = [1, 3, 96, 128] as const;
export const OUTPUT_NAME = 'sh';
/** (batch, coefficient, RGB): linear incident-radiance SH9 in the camera frame. */
export const OUTPUT_SHAPE = [1, 9, 3] as const;
export const CONTRACT_VERSION = '1';
