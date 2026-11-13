import { defineConfig } from 'vitest/config';

// GitHub Pages serves the site at https://television05.github.io/lumina/, so every asset URL is
// prefixed with /lumina/. Use import.meta.env.BASE_URL for runtime URLs (models, media).
export default defineConfig({
  base: '/lumina/',
  // onnxruntime-web ships its own .wasm/.mjs loader; pre-bundling it breaks the loader's
  // relative URL resolution.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', sourcemap: true },
  server: { port: 5179, strictPort: true },
  preview: { port: 5180, strictPort: true },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
