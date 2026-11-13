/** Writes the fake-camera clip and an image fixture before the browser launches. */
import { mkdirSync } from 'node:fs';
import { FAKE_CAMERA_Y4M, FIXTURE_DIR, FIXTURE_PNG_LEFT, FIXTURE_PNG_RIGHT } from './paths';
import { CLIP, writePng, writeY4m } from './fixtures/scene';

export default function globalSetup(): void {
  mkdirSync(FIXTURE_DIR, { recursive: true });
  writeY4m(FAKE_CAMERA_Y4M);
  writePng(FIXTURE_PNG_LEFT, 0);
  writePng(FIXTURE_PNG_RIGHT, CLIP.fps * CLIP.seconds - 1);
}
