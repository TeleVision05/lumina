"""Random parameters for one synthetic camera frame: pose, lens and image pipeline.

A batch of parameters is a dict of equal-length numpy arrays, so the same values can be saved to
JSON (the fixed evaluation crops) and replayed exactly. Distributions are documented in
docs/DATA.md (Synthetic camera model).
"""

from __future__ import annotations

import math

import numpy as np

from lumina_train import sh as shm

Params = dict[str, np.ndarray]

DEG = math.pi / 180.0
#: name -> dtype, in the order they are sampled and serialised.
FIELDS: dict[str, type] = {
    "yaw": float,
    "pitch": float,
    "roll": float,
    "vfov": float,
    "flip": bool,
    "ae_percentile": float,
    "ev": float,
    "kelvin": float,
    "tint": float,
    "vignette": float,
    "blur": float,
    "noise_a": float,
    "noise_b": float,
    "tone": int,
    "gamma": float,
    "contrast": float,
    "saturation": float,
    "jpeg": int,
    "seed": int,
}

TONE_SRGB, TONE_GAMMA, TONE_FILMIC = 0, 1, 2
AE_TARGET = 0.25
VFOV_RANGE_DEG = (40.0, 75.0)
KELVIN_RANGE = (3000.0, 9000.0)
REFERENCE_KELVIN = 6500.0


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), n))


def sample_pose(rng: np.random.Generator, n: int) -> Params:
    """Webcam-like poses: any yaw, pitch N(0, 15 deg) in +/-40, roll N(0, 3 deg) in +/-10."""
    return {
        "yaw": rng.uniform(-math.pi, math.pi, n),
        "pitch": np.clip(rng.normal(0.0, 15.0, n), -40.0, 40.0) * DEG,
        "roll": np.clip(rng.normal(0.0, 3.0, n), -10.0, 10.0) * DEG,
        "vfov": rng.uniform(*VFOV_RANGE_DEG, n) * DEG,
        "flip": rng.random(n) < 0.5,
    }


def sample_pipeline(rng: np.random.Generator, n: int) -> Params:
    """Exposure, white balance and image-pipeline settings."""
    mired = rng.uniform(1e6 / KELVIN_RANGE[1], 1e6 / KELVIN_RANGE[0], n)  # uniform in mired
    jpeg = rng.integers(50, 96, n)
    jpeg[rng.random(n) < 0.25] = 0  # 0 = no JPEG
    return {
        "ae_percentile": rng.uniform(0.5, 0.85, n),
        "ev": rng.uniform(-1.0, 1.0, n),
        "kelvin": 1e6 / mired,
        "tint": rng.uniform(-0.08, 0.08, n),
        "vignette": rng.uniform(0.0, 0.35, n),
        "blur": rng.uniform(0.0, 1.2, n),
        "noise_a": _log_uniform(rng, 1e-4, 5e-3, n),
        "noise_b": _log_uniform(rng, 1e-6, 2e-4, n),
        "tone": rng.integers(0, 3, n),
        "gamma": rng.uniform(1.8, 2.6, n),
        "contrast": rng.uniform(0.8, 1.5, n),
        "saturation": rng.uniform(0.85, 1.3, n),
        "jpeg": jpeg,
        "seed": rng.integers(0, 2**31 - 1, n),
    }


def sample_params(rng: np.random.Generator, n: int) -> Params:
    out = sample_pose(rng, n) | sample_pipeline(rng, n)
    return {k: np.asarray(out[k], dtype=FIELDS[k]) for k in FIELDS}


def neutral_pipeline(n: int) -> Params:
    """A clean camera: exact sRGB, no noise, blur, vignette, JPEG; 6500 K; median AE, 0 EV."""
    return {
        "ae_percentile": np.full(n, 0.5),
        "ev": np.zeros(n),
        "kelvin": np.full(n, REFERENCE_KELVIN),
        "tint": np.zeros(n),
        "vignette": np.zeros(n),
        "blur": np.zeros(n),
        "noise_a": np.zeros(n),
        "noise_b": np.zeros(n),
        "tone": np.full(n, TONE_SRGB),
        "gamma": np.full(n, 2.2),
        "contrast": np.ones(n),
        "saturation": np.ones(n),
        "jpeg": np.zeros(n, dtype=int),
        "seed": np.zeros(n, dtype=int),
    }


def select(params: Params, idx: np.ndarray | slice) -> Params:
    return {k: v[idx] for k, v in params.items()}


def concat(parts: list[Params]) -> Params:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def to_json(params: Params) -> dict[str, list[float | int | bool]]:
    return {k: params[k].tolist() for k in FIELDS}


def from_json(data: dict[str, list[float | int | bool]]) -> Params:
    return {k: np.asarray(data[k], dtype=FIELDS[k]) for k in FIELDS}


def _planck_xy(kelvin: float) -> tuple[float, float]:
    """CIE 1931 xy of the Planckian locus (Kim et al. 2002 cubic spline, 1667-25000 K)."""
    t = kelvin
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


_XYZ_TO_SRGB = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)


def kelvin_rgb(kelvin: float) -> np.ndarray:
    """Linear sRGB of a blackbody at kelvin, luminance 1."""
    x, y = _planck_xy(kelvin)
    rgb = _XYZ_TO_SRGB @ np.array([x / y, 1.0, (1.0 - x - y) / y])
    return rgb / float(rgb @ shm.LUMA)


def white_balance_gains(kelvin: np.ndarray, tint: np.ndarray) -> np.ndarray:
    """[B] -> [B, 3] luminance-preserving gains: a light of `kelvin` seen by a camera balanced
    for 6500 K (warm below, cool above), with a green-magenta tint of 2**tint on green."""
    ref = kelvin_rgb(REFERENCE_KELVIN)
    gains = np.stack([kelvin_rgb(float(k)) / ref for k in kelvin])
    gains[:, 1] *= np.exp2(tint)
    return gains / (gains @ shm.LUMA)[:, None]
