import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { luminaSamples } from './vite/plugins.ts';

// GitHub Pages serves the site at https://television05.github.io/lumina/, so every asset URL is
// prefixed with /lumina/. Use import.meta.env.BASE_URL for runtime URLs (models, media).
const BASE = '/lumina/';
const publicDir = fileURLToPath(new URL('./public', import.meta.url));

export default defineConfig({
  base: BASE,
  plugins: [luminaSamples(publicDir)],
  // onnxruntime-web ships its own .wasm/.mjs loader; pre-bundling it breaks the loader's
  // relative URL resolution.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  // The main chunk is about 600 kB minified (150 kB gzip), almost all three.js; it is one
  // download for a one-page app, so the default 500 kB warning is raised rather than split.
  build: { target: 'es2022', sourcemap: true, chunkSizeWarningLimit: 700 },
  server: { port: 5179, strictPort: true },
  preview: { port: 5180, strictPort: true },
  test: {
    include: ['src/**/*.test.ts', 'vite/**/*.test.ts'],
    environment: 'node',
  },
});
