/**
 * Build plugins for Lumina.
 *
 * luminaSamples lists public/samples/*.mp4|webm at build (or dev-server start) time and injects
 * the names as __LUMINA_SAMPLES__, so the app knows which sample clips exist without probing
 * URLs. The folder may be empty or missing; then the list is empty and the UI says so.
 *
 * The onnxruntime-web wasm binaries need no plugin: the ORT bundles reference them with
 * new URL('...wasm', import.meta.url), which Vite emits as hashed same-origin assets under
 * <base>assets/. So the runtime is self-hosted (no CDN) and cache-busted by content hash.
 */
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import type { Plugin } from 'vite';

const SAMPLE_EXT = /\.(mp4|webm)$/i;

export function listSamples(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => SAMPLE_EXT.test(f) && !f.startsWith('.'))
    .sort();
}

export function luminaSamples(publicDir: string): Plugin {
  return {
    name: 'lumina-samples',
    config() {
      const samples = listSamples(join(publicDir, 'samples'));
      return { define: { __LUMINA_SAMPLES__: JSON.stringify(samples) } };
    },
  };
}
