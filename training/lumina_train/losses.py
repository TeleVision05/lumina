"""Training loss: rendered irradiance + scale-invariant + coefficient + brightness + direction.

All terms are per-sample relative (divided by the target's own energy), so a sunlit crop with a
label 100 times brighter than a dim room does not dominate the batch.

- irr:   ||E_p - E_t||^2 / ||E_t||^2 over 64 fixed normals covering the whole sphere and RGB.
         E is the irradiance of a diffuse sphere, sum_i A_band L_i Y_i(n).
- si:    the same after the best per-sample scale s* = <E_p, E_t> / <E_p, E_p>: shape and colour
         independent of overall brightness.
- coef:  ||L_p - L_t||^2 / ||L_t||^2 over the 27 coefficients.
- logl0: (log L0_p - log L0_t)^2 per channel: symmetric brightness error.
- dir:   directionality_t * (1 - cos(v_p, v_t)) for the luminance band-1 vector, so the dominant
         direction is learnt where it exists and ignored in flat light.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lumina_train import sh_torch

WEIGHTS = {"irr": 1.0, "si": 0.5, "coef": 0.25, "logl0": 0.1, "dir": 0.1}
_EPS = 1e-8


@dataclass
class LossParts:
    total: Tensor
    parts: dict[str, Tensor]


class LightingLoss:
    def __init__(self, device: torch.device, weights: dict[str, float] | None = None) -> None:
        self.normals = sh_torch.fibonacci_normals(64).to(device)
        self.weights = weights or WEIGHTS

    def __call__(self, pred: Tensor, target: Tensor) -> LossParts:
        e_p = sh_torch.irradiance(pred, self.normals).flatten(1)
        e_t = sh_torch.irradiance(target, self.normals).flatten(1)
        energy = e_t.pow(2).sum(1) + _EPS
        irr = (e_p - e_t).pow(2).sum(1) / energy
        s = (e_p * e_t).sum(1) / (e_p.pow(2).sum(1) + _EPS)
        si = (s[:, None] * e_p - e_t).pow(2).sum(1) / energy
        coef = (pred - target).flatten(1).pow(2).sum(1) / (target.flatten(1).pow(2).sum(1) + _EPS)
        logl0 = (
            (torch.log(pred[:, 0].clamp_min(_EPS)) - torch.log(target[:, 0].clamp_min(_EPS)))
            .pow(2)
            .mean(1)
        )
        v_p, v_t = sh_torch.band1_vector(pred), sh_torch.band1_vector(target)
        cos = (v_p * v_t).sum(1) / (v_p.norm(dim=1) * v_t.norm(dim=1) + _EPS)
        direction = sh_torch.directionality(target) * (1.0 - cos)
        parts = {
            "irr": irr.mean(),
            "si": si.mean(),
            "coef": coef.mean(),
            "logl0": logl0.mean(),
            "dir": direction.mean(),
        }
        total = sum(self.weights[k] * v for k, v in parts.items())
        assert isinstance(total, Tensor)
        return LossParts(total=total, parts=parts)
