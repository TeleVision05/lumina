import { defineConfig, devices } from '@playwright/test';

const PORT = 5179;

// Headless Chromium only. The fake-media flags guarantee no test can ever open a real camera or
// show a permission prompt; SwiftShader makes WebGL output identical on laptops and CI.
export const CHROMIUM_ARGS = [
  '--use-fake-device-for-media-stream',
  '--use-fake-ui-for-media-stream',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
];

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  workers: 2,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${PORT}/lumina/`,
    headless: true,
    launchOptions: { args: CHROMIUM_ARGS },
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], headless: true } }],
  webServer: {
    command: `npx vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/lumina/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
