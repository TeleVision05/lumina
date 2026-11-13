"""Write shared/sh_vectors.json and shared/sphere_reference.png.

Run: uv run python -m lumina_train.vectors [--out shared]
Both the pytest and vitest suites check their SH implementations against these files, and the
Playwright parity test compares a three.js render against the PNG. Output is deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lumina_train import sh as shm
from lumina_train.environments import (
    SPHERE_REFERENCE_ENV,
    Environment,
    analytic_environments,
    environment_by_name,
)

SCHEMA_VERSION = 1
SPHERE_SIZE = 128
SPHERE_ALBEDO = 1.0
SPHERE_MASK_RADIUS = 0.95
NORMALS_SEED = 20261113
REPO_ROOT = Path(__file__).resolve().parents[2]


def vector_normals() -> np.ndarray:
    """6 axes, 8 cube diagonals and 6 seeded random unit vectors -> [20, 3]."""
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
    diagonals = signs / math.sqrt(3.0)
    rng = np.random.default_rng(NORMALS_SEED)
    rand = rng.normal(size=(6, 3))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    return np.concatenate([axes, diagonals, rand], axis=0)


def _env_record(env: Environment, normals: np.ndarray) -> dict[str, Any]:
    coeffs = env.sh()
    direction, ratio = shm.dominant_direction(coeffs)
    return {
        "name": env.name,
        "description": env.description,
        "sh": coeffs.tolist(),
        "irradiance": shm.irradiance(coeffs, normals).tolist(),
        "dominant_direction": None if direction is None else direction.tolist(),
        "directionality": ratio,
        "sh_mirrored_x": shm.mirror_x(coeffs).tolist(),
    }


def build_vectors() -> dict[str, Any]:
    normals = vector_normals()
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "uv run python -m lumina_train.vectors",
        "conventions": {
            "frame": "camera space, right-handed, +X right, +Y up, -Z forward",
            "basis": [
                "0.282095",
                "0.488603*y",
                "0.488603*z",
                "0.488603*x",
                "1.092548*x*y",
                "1.092548*y*z",
                "0.315392*(3*z*z-1)",
                "1.092548*x*z",
                "0.546274*(x*x-y*y)",
            ],
            "sh_layout": "sh[coefficient][rgb], linear RGB incident radiance",
            "irradiance": "E(n) = sum_i A_band(i) * sh[i] * Y_i(n), A = [pi, 2pi/3, pi/4]",
            "band_factors": list(shm.BAND_FACTORS),
            "luminance_weights": shm.LUMA.tolist(),
            "directionality": "|L1 of luminance| / (sqrt(3) * L0 of luminance), in [0, 1]",
        },
        "normals": normals.tolist(),
        "basis_at_normals": shm.sh_basis(normals).tolist(),
        "environments": [_env_record(env, normals) for env in analytic_environments()],
        "sphere_reference": {
            "file": "sphere_reference.png",
            "environment": SPHERE_REFERENCE_ENV,
            "size": SPHERE_SIZE,
            "albedo": SPHERE_ALBEDO,
            "camera": "orthographic, looking down -Z, image spans x and y in [-1, 1], row 0 = +Y",
            "shading": "albedo / pi * E(n), Lambertian, black background",
            "encoding": "8-bit sRGB (IEC 61966-2-1), round half up, RGB",
            "compare_mask_radius": SPHERE_MASK_RADIUS,
        },
    }


def render_sphere_reference() -> np.ndarray:
    """8-bit sRGB RGB image [S, S, 3] of the reference sphere."""
    coeffs = environment_by_name(SPHERE_REFERENCE_ENV).sh()
    linear = shm.render_diffuse_sphere(coeffs, SPHERE_SIZE, SPHERE_ALBEDO)
    return shm.to_srgb8(linear)


def write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """OpenCV stores BGR; convert explicitly so the file is RGB on disk."""
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"could not write {path}")


def read_png_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "shared")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors()
    (args.out / "sh_vectors.json").write_text(json.dumps(vectors, indent=1) + "\n")
    write_png_rgb(args.out / "sphere_reference.png", render_sphere_reference())
    print(f"wrote {args.out / 'sh_vectors.json'} and {args.out / 'sphere_reference.png'}")


if __name__ == "__main__":
    main()
