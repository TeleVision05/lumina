/**
 * Make an env-mapped MeshStandardMaterial take its DIFFUSE light from the LightProbe only and its
 * SPECULAR reflections from the env map.
 *
 * Why (three r187, read from the source): with an env map, lights_fragment_maps adds
 * getIBLIrradiance() to `iblIrradiance`, and RE_IndirectSpecular_Physical turns that into diffuse
 * light; RE_IndirectDiffuse_Physical separately adds the LightProbe's `irradiance`. Lumina's env map
 * is synthesised from the same SH as the probe, so the diffuse light would be counted twice.
 *
 * The patch routes the probe irradiance through three.js's own IBL path instead: it drops the
 * RE_IndirectDiffuse call and passes `irradiance` (probe) where `iblIrradiance` (env) was. The
 * diffuse term, its energy split with the specular lobe and multiple-scattering compensation are
 * then exactly three.js's, with the probe as the irradiance source. The patch throws if the chunk
 * changes shape, so a three.js upgrade cannot silently undo it (pbrPatch.test.ts).
 */
import { ShaderChunk, type MeshStandardMaterial } from 'three';

const DIFFUSE_BLOCK =
  /#if defined\( RE_IndirectDiffuse \)\s*#if defined\( LAMBERT \) \|\| defined\( PHONG \)\s*irradiance \+= iblIrradiance;\s*#endif\s*RE_IndirectDiffuse\([^;]*\);\s*#endif/;
const SPECULAR_CALL = 'RE_IndirectSpecular( radiance, iblIrradiance,';

export function patchedLightsFragmentEnd(source: string = ShaderChunk.lights_fragment_end): string {
  if (!DIFFUSE_BLOCK.test(source) || !source.includes(SPECULAR_CALL)) {
    throw new Error('three.js lights_fragment_end changed shape; re-check render/pbrPatch.ts');
  }
  return source
    .replace(
      DIFFUSE_BLOCK,
      '// Lumina: diffuse comes from the probe via RE_IndirectSpecular below.',
    )
    .replace(SPECULAR_CALL, 'RE_IndirectSpecular( radiance, irradiance,');
}

let patched: string | null = null;

/** Apply the probe-diffuse patch to a standard or physical material. */
export function useProbeDiffuse(material: MeshStandardMaterial): void {
  patched ??= patchedLightsFragmentEnd();
  const chunk = patched;
  material.onBeforeCompile = (shader) => {
    const before = shader.fragmentShader;
    shader.fragmentShader = before.replace('#include <lights_fragment_end>', chunk);
    if (shader.fragmentShader === before) throw new Error('lights_fragment_end include not found');
  };
  material.customProgramCacheKey = () => 'lumina-probe-diffuse';
}
