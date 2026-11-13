import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../app/settings';
import {
  describeDirection,
  describeQuality,
  formatKelvin,
  formatMs,
  kelvinPosition,
} from './describe';
import { pitchFromOrientation } from './tilt';

describe('describeDirection (display frame)', () => {
  it.each([
    [[-0.5, 0.7, 0.5], 'Light from the upper left, camera side'],
    [[0.6, 0.6, -0.5], 'Light from the upper right, behind the object'],
    [[0, 1, 0], 'Light from above'],
    [[0.9, 0, 0.1], 'Light from the right'],
    [[0, 0, 1], 'Light from the camera'],
    [[0, 0.1, -1], 'Light from behind the object'],
    [[-0.5, -0.6, 0.2], 'Light from the lower left'],
  ] as const)('%j -> %s', (d, want) => {
    expect(describeDirection(d)).toBe(want);
  });

  it('says so when there is no main direction', () => {
    expect(describeDirection(null)).toBe('Even light, no main direction');
  });
});

describe('describeQuality', () => {
  it('names flat, soft and directional light with the percentage', () => {
    expect(describeQuality(0.05)).toBe('Flat, even light (5% directional)');
    expect(describeQuality(0.25)).toBe('Soft light (25% directional)');
    expect(describeQuality(0.8)).toBe('Directional light (80% directional)');
  });
});

describe('Kelvin scale', () => {
  it('maps 2,000 K to the warm end, 10,000 K to the cool end, log in between', () => {
    expect(kelvinPosition(2000)).toBe(0);
    expect(kelvinPosition(10000)).toBe(1);
    expect(kelvinPosition(Math.sqrt(2000 * 10000))).toBeCloseTo(0.5, 9);
    expect(kelvinPosition(1000)).toBe(0);
  });

  it('formats to the nearest 100 K', () => {
    expect(formatKelvin(4321)).toBe('4,300 K');
    expect(formatKelvin(null)).toBe('–');
    expect(formatMs(3.14159)).toBe('3.1 ms');
    expect(formatMs(42.6)).toBe('43 ms');
  });
});

describe('sanitizeSettings', () => {
  it('returns the defaults for garbage', () => {
    expect(sanitizeSettings('nope')).toEqual(DEFAULT_SETTINGS);
    expect(sanitizeSettings(null)).toEqual(DEFAULT_SETTINGS);
  });

  it('keeps valid values, clamps numbers and rejects unknown choices', () => {
    const s = sanitizeSettings({
      view: 'lumina',
      object: 'teapot',
      grain: 7,
      tiltDeg: -40,
      deviceTilt: true,
    });
    expect(s.view).toBe('lumina');
    expect(s.object).toBe(DEFAULT_SETTINGS.object);
    expect(s.grain).toBe(1);
    expect(s.tiltDeg).toBe(-10);
    expect(s.deviceTilt).toBe(false); // needs a fresh permission every visit
  });
});

describe('pitchFromOrientation', () => {
  it('reads an upright portrait phone as level and a phone tipped toward the table as looking down', () => {
    expect(pitchFromOrientation(90, 0, 0)).toBe(0);
    expect(pitchFromOrientation(60, 0, 0)).toBe(30);
    expect(pitchFromOrientation(0, 0, 0)).toBe(60); // flat on its back: clamped
  });

  it('uses gamma in landscape', () => {
    expect(pitchFromOrientation(0, -70, 90)).toBe(20);
  });
});
