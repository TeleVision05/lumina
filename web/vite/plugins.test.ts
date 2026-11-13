import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { listSamples } from './plugins';

describe('listSamples', () => {
  it('returns an empty list when public/samples does not exist', () => {
    expect(listSamples(join(tmpdir(), 'lumina-no-such-dir-20261113'))).toEqual([]);
  });

  it('lists mp4 and webm clips, sorted, ignoring other files', () => {
    const dir = mkdtempSync(join(tmpdir(), 'lumina-samples-'));
    mkdirSync(join(dir, 'sub'));
    for (const f of ['b.mp4', 'a.webm', 'notes.txt', '.hidden.mp4', 'C.MP4']) {
      writeFileSync(join(dir, f), '');
    }
    expect(listSamples(dir)).toEqual(['C.MP4', 'a.webm', 'b.mp4']);
  });
});
