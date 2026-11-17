import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vector3 } from 'three';
import { Ground } from '../render/ground';
import { detectArSupport, isQuickLookSupported, isWebXRArSupported } from './capabilities';

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
