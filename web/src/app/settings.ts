/**
 * Visitor settings. Remembered per browser in localStorage as a convenience only: storage can be
 * missing or blocked (private windows, previews), so every read and write is guarded and the app
 * works with the defaults.
 */
import type { ObjectId } from '../render/objects';
import { OBJECT_IDS } from '../render/objects';
import type { View } from '../render/stage';

export type LuminaEstimator = 'neural' | 'stats';

export interface Settings {
  view: View;
  object: ObjectId;
  estimator: LuminaEstimator;
  /** 0 steady .. 1 quick (temporal filter). */
  responsiveness: number;
  /** 0..1 scale on shadow darkness. */
  shadow: number;
  /** Camera pitch below horizontal, degrees. */
  tiltDeg: number;
  deviceTilt: boolean;
  /** 0..1 film grain amount. */
  grain: number;
  /** 0..1 softness (blur) amount. */
  blur: number;
  envFromCamera: boolean;
  /** Split position, 0..1 of the width. */
  split: number;
}

export const DEFAULT_SETTINGS: Settings = {
  view: 'split',
  object: 'matte',
  estimator: 'neural',
  responsiveness: 0.5,
  shadow: 0.85,
  tiltDeg: 15,
  deviceTilt: false,
  grain: 0.35,
  blur: 0.4,
  envFromCamera: true,
  split: 0.5,
};

const KEY = 'lumina.settings.v1';

const num = (v: unknown, lo: number, hi: number, d: number): number =>
  typeof v === 'number' && Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : d;
const pick = <T extends string>(v: unknown, allowed: readonly T[], d: T): T =>
  allowed.includes(v as T) ? (v as T) : d;
const bool = (v: unknown, d: boolean): boolean => (typeof v === 'boolean' ? v : d);

/** Accept only well-formed values; anything else falls back to the default. */
export function sanitizeSettings(raw: unknown): Settings {
  const r = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;
  const d = DEFAULT_SETTINGS;
  return {
    view: pick(r.view, ['fixed', 'lumina', 'split'] as const, d.view),
    object: pick(r.object, OBJECT_IDS, d.object),
    estimator: pick(r.estimator, ['neural', 'stats'] as const, d.estimator),
    responsiveness: num(r.responsiveness, 0, 1, d.responsiveness),
    shadow: num(r.shadow, 0, 1, d.shadow),
    tiltDeg: num(r.tiltDeg, -10, 60, d.tiltDeg),
    deviceTilt: false, // needs a fresh permission each visit
    grain: num(r.grain, 0, 1, d.grain),
    blur: num(r.blur, 0, 1, d.blur),
    envFromCamera: bool(r.envFromCamera, d.envFromCamera),
    split: num(r.split, 0, 1, d.split),
  };
}

export function loadSettings(): Settings {
  try {
    const text = localStorage.getItem(KEY);
    return sanitizeSettings(text === null ? null : JSON.parse(text));
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // Storage unavailable: settings last for this visit only.
  }
}
