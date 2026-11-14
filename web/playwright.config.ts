import { defineConfig, devices } from '@playwright/test';
import { FAKE_CAMERA_Y4M } from './e2e/paths';

/**
 * Targets (LUMINA_E2E_TARGET):
 * - dev (default): Vite dev server on 5179, all specs including the foundation's harness pages.
 * - dist: the production build served under /lumina/ by scripts/serve-pages.mjs (Pages-like, no
 *   COOP/COEP), app specs only. Run `npm run build` first, or use `npm run e2e:dist`.
 */
const TARGET = process.env.LUMINA_E2E_TARGET === 'dist' ? 'dist' : 'dev';
const PORT = TARGET === 'dist' ? 5181 : 5179;

// Headless Chromium only. The fake-media flags guarantee no test can ever open a real camera or
// show a permission prompt; the camera is a generated Y4M clip (e2e/fixtures/scene.ts).
// SwiftShader makes WebGL output identical on laptops and CI.
export const CHROMIUM_ARGS = [
  '--use-fake-device-for-media-stream',
  '--use-fake-ui-for-media-stream',
  `--use-file-for-fake-video-capture=${FAKE_CAMERA_Y4M}`,
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--mute-audio',
];

export default defineConfig({
  testDir: 'e2e',
  globalSetup: './e2e/global-setup.ts',
  fullyParallel: true,
  workers: 2,
  timeout: 60_000,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: [['list']],
  // truth.spec.ts needs ffmpeg and uv; it runs against the production build (e2e:dist).
  testIgnore:
    TARGET === 'dist' ? ['parity.spec.ts', 'model.spec.ts', 'shading.spec.ts'] : ['truth.spec.ts'],
  use: {
    baseURL: `http://localhost:${PORT}/lumina/`,
    headless: true,
    launchOptions: { args: CHROMIUM_ARGS },
    permissions: ['camera'],
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], headless: true } }],
  webServer: {
    command:
      TARGET === 'dist'
        ? `node scripts/serve-pages.mjs ${PORT}`
        : `npx vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/lumina/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
