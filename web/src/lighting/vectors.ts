/** Typed reader for shared/sh_vectors.json (written by `uv run python -m lumina_train.vectors`). */
import type { RGB, SH9, Vec3 } from './sh';

export interface EnvironmentRecord {
  name: string;
  description: string;
  sh: SH9;
  irradiance: RGB[];
  dominant_direction: Vec3 | null;
  directionality: number;
  sh_mirrored_x: SH9;
}

export interface SphereReference {
  file: string;
  environment: string;
  size: number;
  albedo: number;
  compare_mask_radius: number;
}

export interface ShVectors {
  schema_version: number;
  normals: Vec3[];
  basis_at_normals: number[][];
  environments: EnvironmentRecord[];
  sphere_reference: SphereReference;
}

export const SUPPORTED_SCHEMA = 1;

/** Parse and minimally validate the vectors file; throws with a clear message when malformed. */
export function parseShVectors(text: string): ShVectors {
  const data = JSON.parse(text) as ShVectors;
  if (data.schema_version !== SUPPORTED_SCHEMA) {
    throw new Error(`sh_vectors.json schema ${data.schema_version}, expected ${SUPPORTED_SCHEMA}`);
  }
  if (data.normals.length === 0 || data.environments.length === 0) {
    throw new Error('sh_vectors.json has no normals or environments');
  }
  return data;
}

export function environmentByName(vectors: ShVectors, name: string): EnvironmentRecord {
  const env = vectors.environments.find((e) => e.name === name);
  if (env === undefined) throw new Error(`environment ${name} not in sh_vectors.json`);
  return env;
}
