/**
 * Synthetic "room" footage for the fake camera, generated in pure TypeScript so CI needs no
 * ffmpeg: a warm grey wall, a darker table, and a bright warm lamp that sits at the upper LEFT for
 * the first half of the loop and the upper RIGHT for the second half (sensor image, un-mirrored).
 */
import { writeFileSync } from 'node:fs';
import { PNG } from 'pngjs';

export const CLIP = { width: 320, height: 240, fps: 15, seconds: 4 } as const;

export type Side = 'left' | 'right';

/** Which side the lamp is on at frame index `i`. */
export function lampSide(i: number): Side {
  return i < (CLIP.fps * CLIP.seconds) / 2 ? 'left' : 'right';
}

/** RGB bytes (row-major, 3 per pixel) of frame `i`. */
export function frameRgb(i: number, w: number = CLIP.width, h: number = CLIP.height): Uint8Array {
  const out = new Uint8Array(w * h * 3);
  const lx = (lampSide(i) === 'left' ? 0.2 : 0.8) * w;
  const ly = 0.2 * h;
  const lr = 0.09 * w;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const table = y > 0.7 * h;
      const shade = table ? 0.55 : 1 - 0.25 * (y / h);
      let r = (table ? 92 : 128) * shade;
      let g = (table ? 80 : 120) * shade;
      let b = (table ? 70 : 112) * shade;
      const d = Math.hypot(x - lx, y - ly);
      const glow = Math.exp(-((d / (3.5 * lr)) ** 2)) * 70;
      r += glow;
      g += glow * 0.85;
      b += glow * 0.6;
      if (d < lr) [r, g, b] = [255, 238, 205];
      const k = (y * w + x) * 3;
      out[k] = Math.min(255, Math.round(r));
      out[k + 1] = Math.min(255, Math.round(g));
      out[k + 2] = Math.min(255, Math.round(b));
    }
  }
  return out;
}

/** Write the clip as Y4M (I420, BT.601 limited range), the format Chrome's fake camera reads. */
export function writeY4m(path: string): void {
  const { width: w, height: h, fps, seconds } = CLIP;
  const header = Buffer.from(`YUV4MPEG2 W${w} H${h} F${fps}:1 Ip A1:1 C420\n`);
  const frameHeader = Buffer.from('FRAME\n');
  const ySize = w * h;
  const cSize = (w / 2) * (h / 2);
  const parts: Buffer[] = [header];
  for (let i = 0; i < fps * seconds; i++) {
    const rgb = frameRgb(i);
    const yuv = Buffer.alloc(ySize + 2 * cSize);
    for (let p = 0; p < ySize; p++) {
      const [r, g, b] = [rgb[3 * p] ?? 0, rgb[3 * p + 1] ?? 0, rgb[3 * p + 2] ?? 0].map(
        (v) => v / 255,
      );
      yuv[p] = Math.round(16 + 65.481 * (r ?? 0) + 128.553 * (g ?? 0) + 24.966 * (b ?? 0));
    }
    for (let cy = 0; cy < h / 2; cy++) {
      for (let cx = 0; cx < w / 2; cx++) {
        const acc = [0, 0, 0];
        for (const [dx, dy] of [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1],
        ] as const) {
          const k = ((2 * cy + dy) * w + 2 * cx + dx) * 3;
          for (let c = 0; c < 3; c++) acc[c] = (acc[c] ?? 0) + (rgb[k + c] ?? 0) / 255 / 4;
        }
        const [r, g, b] = [acc[0] ?? 0, acc[1] ?? 0, acc[2] ?? 0];
        const q = cy * (w / 2) + cx;
        yuv[ySize + q] = Math.round(128 - 37.797 * r - 74.203 * g + 112 * b);
        yuv[ySize + cSize + q] = Math.round(128 + 112 * r - 93.786 * g - 18.214 * b);
      }
    }
    parts.push(frameHeader, yuv);
  }
  writeFileSync(path, Buffer.concat(parts));
}

/** Write one frame as a PNG (for the dropped-image test). */
export function writePng(path: string, i: number): void {
  const { width: w, height: h } = CLIP;
  const png = new PNG({ width: w, height: h });
  const rgb = frameRgb(i);
  for (let p = 0; p < w * h; p++) {
    png.data[4 * p] = rgb[3 * p] ?? 0;
    png.data[4 * p + 1] = rgb[3 * p + 1] ?? 0;
    png.data[4 * p + 2] = rgb[3 * p + 2] ?? 0;
    png.data[4 * p + 3] = 255;
  }
  writeFileSync(path, PNG.sync.write(png));
}
