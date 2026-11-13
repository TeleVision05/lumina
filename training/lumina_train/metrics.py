"""Evaluation metrics on a rendered diffuse sphere (docs/EVAL.md defines each one).

Every metric compares a predicted and a true SH9 [B, 9, 3] through what a viewer would see: a
Lambertian sphere of albedo 0.5, seen by an orthographic camera down -Z (the side facing the
camera), SPHERE_SIZE x SPHERE_SIZE pixels, linear radiance albedo / pi * E(n).

- rmse:        RMSE over sphere pixels and RGB, linear units (the label's exposure scale).
- si_rmse:     RMSE after the per-sample scale s* = <p, t> / <p, p> (shape and colour only).
- srgb_rmse:   RMSE in 8-bit sRGB levels after clipping to [0, 1] (what the screen shows).
- colour_deg:  angle between the mean RGB of the predicted and true spheres (colour cast only).
- dir_deg:     angle between the dominant-light directions (luminance band 1) of both SH.
- directionality: of the TRUE SH, |L1| / (sqrt(3) L0), to split directional and diffuse scenes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from lumina_train import sh as shm
from lumina_train import sh_torch

SPHERE_SIZE = 64
ALBEDO = 0.5
METRICS = ("rmse", "si_rmse", "srgb_rmse", "colour_deg", "dir_deg")
#: Scenes whose true directionality is at least this are "directional"; below, "diffuse".
DIRECTIONAL_THRESHOLD = 0.3

_normals, _mask = shm.sphere_normals(SPHERE_SIZE)
SPHERE_NORMALS = torch.from_numpy(_normals[_mask]).float()  # [M, 3]


def sphere_radiance(sh: Tensor) -> Tensor:
    """[B, 9, 3] -> linear radiance at the visible sphere pixels [B, M, 3]."""
    return ALBEDO / math.pi * sh_torch.irradiance(sh, SPHERE_NORMALS.to(sh.device))


def _srgb(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def _angle_deg(a: Tensor, b: Tensor) -> Tensor:
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def compute(pred: Tensor, target: Tensor) -> dict[str, np.ndarray]:
    """Per-sample metrics, float64 numpy arrays of shape [B]."""
    pred, target = pred.detach().cpu().double(), target.detach().cpu().double()
    p, t = sphere_radiance(pred), sphere_radiance(target)
    rmse = (p - t).pow(2).mean((1, 2)).sqrt()
    s = (p * t).sum((1, 2)) / p.pow(2).sum((1, 2)).clamp_min(1e-12)
    si = (s[:, None, None] * p - t).pow(2).mean((1, 2)).sqrt()
    srgb = (255.0 * (_srgb(p) - _srgb(t))).pow(2).mean((1, 2)).sqrt()
    colour = _angle_deg(p.mean(1), t.mean(1))
    direction = _angle_deg(sh_torch.band1_vector(pred), sh_torch.band1_vector(target))
    return {
        "rmse": rmse.numpy(),
        "si_rmse": si.numpy(),
        "srgb_rmse": srgb.numpy(),
        "colour_deg": colour.numpy(),
        "dir_deg": direction.numpy(),
        "directionality": sh_torch.directionality(target).numpy(),
    }


def per_group_mean(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of values per group id -> (group ids sorted, means)."""
    ids, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return ids, sums / counts


def bootstrap_ci(
    per_unit: np.ndarray, seed: int, resamples: int = 2000, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over units (HDRIs): (mean, low, high)."""
    rng = np.random.default_rng(seed)
    n = len(per_unit)
    if n == 0:
        return math.nan, math.nan, math.nan
    draws = rng.integers(0, n, size=(resamples, n))
    means = per_unit[draws].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(per_unit.mean()), float(lo), float(hi)
