"""Baseline estimators, scored with exactly the same code as the network (evaluate.py).

- mean:      the training-set mean label (SH9), whatever the picture shows.
- image_mean: the picture's mean colour (sRGB decoded to linear) as a uniform environment: the
             grey-world idea, L0 only.
- image_dir: image_mean plus a band-1 term toward the brightest image region (the luminance-
             weighted centroid of the brightest 2 % of pixels, through a pinhole of an assumed
             60 degree vertical FOV), with the training-set mean directionality.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lumina_train import sh as shm
from lumina_train import sh_torch
from lumina_train.camera import pixel_direction
from lumina_train.fixedsets import render_set
from lumina_train.hdri import load_split_cache
from lumina_train.params import sample_params
from lumina_train.paths import CACHE_DIR

ASSUMED_VFOV = math.radians(60.0)
TOP_FRACTION = 0.02
TRAIN_STATS_SEED = 3003
TRAIN_STATS_CROPS = 8
_UNIFORM_L0 = 4.0 * math.pi * shm.C0  # L0 of a uniform environment of radiance 1


def train_statistics() -> dict[str, np.ndarray | float]:
    """Mean label and mean directionality over 8 seeded crops of every training HDRI (cached)."""
    path = CACHE_DIR / "train_stats.npz"
    if path.exists():
        with np.load(path) as data:
            return {"mean_sh": data["mean_sh"], "directionality": float(data["directionality"])}
    cache = load_split_cache("train")
    n = len(cache["ids"]) * TRAIN_STATS_CROPS
    params = sample_params(np.random.default_rng(TRAIN_STATS_SEED), n)
    labels = torch.from_numpy(render_set(cache, params, TRAIN_STATS_CROPS)["labels"])
    out = {
        "mean_sh": labels.double().mean(0).numpy(),
        "directionality": float(sh_torch.directionality(labels).double().mean()),
    }
    np.savez(path, **out)
    return out


def mean_baseline(n: int, mean_sh: np.ndarray) -> np.ndarray:
    return np.broadcast_to(mean_sh, (n, 9, 3)).copy()


def _linear(images_u8: np.ndarray) -> np.ndarray:
    return shm.srgb_to_linear(images_u8.astype(np.float64) / 255.0)


def image_mean_baseline(images_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB images [N, H, W, 3] -> SH [N, 9, 3]: a uniform environment of the mean colour."""
    mean = _linear(images_u8).mean(axis=(1, 2))
    out = np.zeros((len(images_u8), 9, 3))
    out[:, 0] = _UNIFORM_L0 * mean
    return out


def brightest_direction(image_u8: np.ndarray) -> np.ndarray:
    """Camera-frame unit ray through the luminance-weighted centroid of the brightest pixels."""
    lum = _linear(image_u8) @ shm.LUMA
    h, w = lum.shape
    k = max(1, int(TOP_FRACTION * lum.size))
    flat = np.argsort(lum, axis=None)[-k:]
    rows, cols = np.unravel_index(flat, lum.shape)
    # Weight by how far above the cut-off each pixel is, so ties at the cut-off do not pull.
    weights = lum[rows, cols] - lum[rows, cols].min() + 1e-9
    row = float(np.average(rows, weights=weights))
    col = float(np.average(cols, weights=weights))
    return pixel_direction(row, col, ASSUMED_VFOV, h, w)


def image_dir_baseline(images_u8: np.ndarray, directionality: float) -> np.ndarray:
    out = image_mean_baseline(images_u8)
    for i, img in enumerate(images_u8):
        d = brightest_direction(img)
        band1 = d * directionality * shm.MAX_L1_OVER_L0  # times L0, per channel
        for axis, coeff in ((0, 3), (1, 1), (2, 2)):  # x -> Y_3, y -> Y_1, z -> Y_2
            out[i, coeff] = band1[axis] * out[i, 0]
    return out
