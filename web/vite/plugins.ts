/**
 * Build plugins for Lumina.
 *
 * - luminaSamples: lists public/samples/*.mp4|webm at build (or dev-server start) time and injects
 *   the names as __LUMINA_SAMPLES__, so the app knows which sample clips exist without probing
 *   URLs. The folder may be empty or missing; then the list is empty and the UI says so.
 * - ortRuntime: self-hosts the onnxruntime-web wasm binaries under <base>ort/. GitHub Pages
 *   cannot send cross-origin-isolation headers and the site loads nothing from third-party CDNs,
 *   so the single-threaded wasm files are served from the site itself.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
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

/** The wasm binaries the two ORT entry points load (ort.wasm.* and ort.webgpu.*). */
export const ORT_WASM_FILES = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.wasm',
] as const;

function ortDistDir(): string {
  const require = createRequire(import.meta.url);
  // The package's "exports" map lists each wasm file, so this resolves without deep paths.
  return dirname(require.resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'));
}

export function ortRuntime(base: string): Plugin {
  const dist = ortDistDir();
  const prefix = `${base}ort/`;
  return {
    name: 'lumina-ort-runtime',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0] ?? '';
        const file = url.startsWith(prefix) ? url.slice(prefix.length) : '';
        if (!(ORT_WASM_FILES as readonly string[]).includes(file)) {
          next();
          return;
        }
        res.setHeader('Content-Type', 'application/wasm');
        res.end(readFileSync(join(dist, file)));
      });
    },
    generateBundle() {
      for (const file of ORT_WASM_FILES) {
        this.emitFile({
          type: 'asset',
          fileName: `ort/${file}`,
          source: readFileSync(join(dist, file)),
        });
      }
    },
  };
}
