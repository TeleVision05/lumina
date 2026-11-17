/**
 * iOS Safari AR Quick Look launcher: exports the current Three.js object to USDZ in-browser and
 * opens Apple's native ARKit floor-tracking viewer via `<a rel="ar">`.
 *
 * Note: AR Quick Look hands off to Apple's system viewer, which uses ARKit's built-in lighting
 * rather than Lumina's live neural lighting pipeline.
 */
import type { Object3D } from 'three';

export async function launchQuickLook(object: Object3D, filename = 'lumina.usdz'): Promise<void> {
  const { USDZExporter } = await import('three/examples/jsm/exporters/USDZExporter.js');
  const exporter = new USDZExporter();
  const bytes = await exporter.parseAsync(object);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const blob = new Blob([copy], { type: 'model/vnd.usdz+zip' });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement('a');
  anchor.setAttribute('rel', 'ar');
  anchor.setAttribute('download', filename);
  anchor.href = url;
  // iOS Safari requires an <img> child inside <a rel="ar"> to launch AR Quick Look directly.
  const img = document.createElement('img');
  img.alt = '';
  img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  anchor.appendChild(img);
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 60_000);
}
