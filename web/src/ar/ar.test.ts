import { afterEach, describe, expect, it, vi } from 'vitest';
import { Quaternion, Ray, Vector3 } from 'three';
import { dominantDirection, shBasis } from '../lighting/sh';
import { rotateSH9 } from '../lighting/shRotate';
import { Ground } from '../render/ground';
import { detectArSupport, isQuickLookSupported, isWebXRArSupported } from './capabilities';
import { rayHitsDisc, rayHitsSphere, RETICLE_TAP_RADIUS } from './placement';

describe('AR capability detection', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns false for WebXR when navigator.xr is absent', async () => {
    vi.stubGlobal('navigator', {});
    expect(await isWebXRArSupported()).toBe(false);
  });

  it('returns true for WebXR when immersive-ar is supported', async () => {
    vi.stubGlobal('navigator', {
      xr: { isSessionSupported: vi.fn().mockResolvedValue(true) },
    });
    expect(await isWebXRArSupported()).toBe(true);
    expect(await detectArSupport()).toBe('webxr');
  });

  it('handles navigator.xr.isSessionSupported rejecting', async () => {
    vi.stubGlobal('navigator', {
      xr: { isSessionSupported: vi.fn().mockRejectedValue(new Error('blocked')) },
    });
    expect(await isWebXRArSupported()).toBe(false);
  });

  it('falls back to quicklook when WebXR is unsupported and <a rel="ar"> is supported', async () => {
    vi.stubGlobal('navigator', {});
    vi.stubGlobal('document', {
      createElement: () => ({
        relList: { supports: (token: string) => token === 'ar' },
      }),
    });
    expect(isQuickLookSupported()).toBe(true);
    expect(await detectArSupport()).toBe('quicklook');
  });

  it('returns none when neither WebXR nor Quick Look is available', async () => {
    vi.stubGlobal('navigator', {});
    vi.stubGlobal('document', {
      createElement: () => ({
        relList: { supports: () => false },
      }),
    });
    expect(await detectArSupport()).toBe('none');
  });
});

describe('Ground.setPose for WebXR floor placement', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sets horizontal normal [0, 1, 0] and world position', () => {
    vi.stubGlobal('document', {
      createElement: () => ({ width: 0, height: 0, getContext: () => null }),
    });
    const g = new Ground();
    g.setTilt(0.5);
    expect(g.tiltRad).toBeCloseTo(0.5);

    g.setPose(new Vector3(0.4, -0.8, -1.5));
    expect(g.tiltRad).toBe(0);
    expect(g.normal).toEqual([0, 1, 0]);
    expect(g.group.position.toArray()).toEqual([0.4, -0.8, -1.5]);
    expect(g.group.rotation.x).toBe(0);
    expect(g.group.rotation.y).toBe(0);
    expect(g.group.rotation.z).toBe(0);
  });
});

describe('AR tap hit tests', () => {
  it('hits the reticle disc only when aimed near it', () => {
    const center = new Vector3(0, -1, -2);
    const hitRay = new Ray(new Vector3(0, 0, 0), new Vector3(0.05, -1, -2).normalize());
    const missRay = new Ray(new Vector3(0, 0, 0), new Vector3(1, -1, -2).normalize());
    expect(rayHitsDisc(hitRay, center, RETICLE_TAP_RADIUS)).toBe(true);
    expect(rayHitsDisc(missRay, center, RETICLE_TAP_RADIUS)).toBe(false);
  });

  it('hits the placed object sphere only when aimed at it', () => {
    const center = new Vector3(0, -0.85, -2);
    const hitRay = new Ray(new Vector3(0, 0, 0), new Vector3(0, -0.85, -2).normalize());
    const missRay = new Ray(new Vector3(0, 0, 0), new Vector3(1, 0, -2).normalize());
    expect(rayHitsSphere(hitRay, center, 0.21)).toBe(true);
    expect(rayHitsSphere(missRay, center, 0.21)).toBe(false);
  });
});

describe('rotateSH9', () => {
  it('rotates a directional lobe by 90 degrees around Y', () => {
    const basis = shBasis([1, 0, 0]);
    const sh = basis.map((b) => [b, b, b] as const);
    const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 2);
    const rotated = rotateSH9(sh, q);
    const { direction } = dominantDirection(rotated);
    expect(direction).not.toBeNull();
    expect(direction?.[0]).toBeCloseTo(0, 4);
    expect(direction?.[1]).toBeCloseTo(0, 4);
    expect(direction?.[2]).toBeCloseTo(-1, 4);
  });
});
