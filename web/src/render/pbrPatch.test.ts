import { ShaderChunk } from 'three';
import { describe, expect, it } from 'vitest';
import { patchedLightsFragmentEnd } from './pbrPatch';

describe('probe-diffuse shader patch', () => {
  it('removes the separate probe diffuse call and feeds the probe irradiance to the IBL path', () => {
    const out = patchedLightsFragmentEnd();
    expect(out).not.toMatch(/RE_IndirectDiffuse\(/);
    expect(out).toContain('RE_IndirectSpecular( radiance, irradiance,');
    expect(out).not.toContain('iblIrradiance');
  });

  it('applies to the chunk shipped with the installed three.js', () => {
    expect(ShaderChunk.lights_fragment_end).toContain(
      'RE_IndirectSpecular( radiance, iblIrradiance,',
    );
  });

  it('refuses a chunk it does not recognise', () => {
    expect(() => patchedLightsFragmentEnd('void main() {}')).toThrow(/changed shape/);
  });
});
