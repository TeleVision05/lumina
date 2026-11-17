/**
 * Browser AR capability detection. Uses feature detection only (no User-Agent sniffing):
 * - WebXR `immersive-ar` for Android Chrome / WebXR-capable browsers (hit-test floor tracking).
 * - `<a rel="ar">` for iOS / iPadOS Safari AR Quick Look (USDZ handoff).
 */
export type ArSupport = 'webxr' | 'quicklook' | 'none';

export async function isWebXRArSupported(): Promise<boolean> {
  if (typeof navigator === 'undefined') return false;
  const xr = navigator.xr;
  if (xr === undefined || typeof xr.isSessionSupported !== 'function') return false;
  try {
    return await xr.isSessionSupported('immersive-ar');
  } catch {
    return false;
  }
}

export function isQuickLookSupported(): boolean {
  if (typeof document === 'undefined') return false;
  const anchor = document.createElement('a');
  return typeof anchor.relList.supports === 'function' && anchor.relList.supports('ar');
}

export async function detectArSupport(): Promise<ArSupport> {
  if (await isWebXRArSupported()) return 'webxr';
  if (isQuickLookSupported()) return 'quicklook';
  return 'none';
}
