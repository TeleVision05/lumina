/** Paths shared by the Playwright config, global setup and specs. */
import { fileURLToPath } from 'node:url';

const here = (p: string): string => fileURLToPath(new URL(p, import.meta.url));

/** Generated fixtures (gitignored with the rest of test-results/). */
export const FIXTURE_DIR = here('../test-results/fixtures/');
export const FAKE_CAMERA_Y4M = `${FIXTURE_DIR}room.y4m`;
export const FIXTURE_PNG_LEFT = `${FIXTURE_DIR}room-lamp-left.png`;
export const FIXTURE_PNG_RIGHT = `${FIXTURE_DIR}room-lamp-right.png`;
/** Committed tiny clip (VP9 in MP4) for the dropped-video test; see e2e/fixtures/README.md. */
export const PLACEHOLDER_CLIP = here('./fixtures/placeholder-clip.mp4');
export const SAMPLES_DIR = here('../public/samples/');
