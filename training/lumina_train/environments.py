"""Analytic test environments with exact SH9 coefficients.

Every environment is a sum of zonal terms: ambient colour plus colour * f(w . axis). By the
Funk-Hecke theorem the SH coefficients of a zonal term are L_lm = lambda_l * Y_lm(axis) with
lambda_l = 2 pi * integral_{-1}^{1} f(t) P_l(t) dt, computed here with Gauss-Legendre quadrature
split at discontinuities, so the coefficients are exact to about 1e-12.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lumina_train.sh import BAND_OF_COEFF, FloatArray, sh_basis

Profile = Callable[[FloatArray], FloatArray]


def _legendre(t: FloatArray) -> FloatArray:
    """P_0, P_1, P_2 at t -> [..., 3]."""
    return np.stack([np.ones_like(t), t, 0.5 * (3.0 * t * t - 1.0)], axis=-1)


def zonal_band_weights(profile: Profile, lo: float = -1.0, hi: float = 1.0) -> FloatArray:
    """lambda_0..2 for a zonal profile f(t) supported on [lo, hi] (zero elsewhere)."""
    nodes, weights = np.polynomial.legendre.leggauss(256)
    t = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * weights
    return 2.0 * math.pi * (w * profile(t)) @ _legendre(t)


@dataclass(frozen=True)
class ZonalTerm:
    """colour * f(w . axis), with f supported on t in [lo, hi]."""

    axis: tuple[float, float, float]
    colour: tuple[float, float, float]
    profile: Profile
    lo: float = -1.0
    hi: float = 1.0

    def unit_axis(self) -> FloatArray:
        a = np.asarray(self.axis, dtype=np.float64)
        return a / np.linalg.norm(a)

    def sh(self) -> FloatArray:
        lam = zonal_band_weights(self.profile, self.lo, self.hi)
        per_coeff = lam[list(BAND_OF_COEFF)] * sh_basis(self.unit_axis())
        return per_coeff[:, None] * np.asarray(self.colour, dtype=np.float64)[None, :]

    def radiance(self, dirs: FloatArray) -> FloatArray:
        t = np.asarray(dirs, dtype=np.float64) @ self.unit_axis()
        inside = (t >= self.lo) & (t <= self.hi)
        values = np.where(inside, self.profile(t), 0.0)
        return values[..., None] * np.asarray(self.colour, dtype=np.float64)


@dataclass(frozen=True)
class Environment:
    """Ambient colour plus zonal terms; radiance in linear RGB, directions in the camera frame."""

    name: str
    description: str
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    terms: tuple[ZonalTerm, ...] = field(default_factory=tuple)
    dominant: tuple[float, float, float] | None = None

    def sh(self) -> FloatArray:
        out = np.zeros((9, 3), dtype=np.float64)
        out[0] = 4.0 * math.pi * sh_basis(np.array([0.0, 1.0, 0.0]))[0] * np.asarray(self.ambient)
        for term in self.terms:
            out += term.sh()
        return out

    def radiance(self, dirs: FloatArray) -> FloatArray:
        d = np.asarray(dirs, dtype=np.float64)
        out = np.broadcast_to(np.asarray(self.ambient, dtype=np.float64), (*d.shape[:-1], 3))
        out = out.copy()
        for term in self.terms:
            out += term.radiance(d)
        return out


LOBE_SHARPNESS = 8.0


def _lobe(t: FloatArray) -> FloatArray:
    """Soft lobe exp(k (t - 1)), peak 1 at the axis, about 1/e at 29 degrees."""
    return np.exp(LOBE_SHARPNESS * (t - 1.0))


def _one(t: FloatArray) -> FloatArray:
    return np.ones_like(t)


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return (float(a[0]), float(a[1]), float(a[2]))


# Upper right and slightly behind the camera, so it lights the side of an object the camera sees.
OBLIQUE = _unit((2.0, 2.0, 1.0))


def _lobe_env(name: str, axis: tuple[float, float, float], label: str) -> Environment:
    return Environment(
        name=name,
        description=f"White soft lobe exp(8(w.d - 1)) toward {label}, no ambient.",
        terms=(ZonalTerm(axis=axis, colour=(1.0, 1.0, 1.0), profile=_lobe),),
        dominant=_unit(axis),
    )


def analytic_environments() -> list[Environment]:
    """The environments written to shared/sh_vectors.json, in a fixed order."""
    return [
        Environment(
            name="constant_white",
            description="Uniform white radiance 1 in every direction.",
            ambient=(1.0, 1.0, 1.0),
        ),
        _lobe_env("lobe_pos_x", (1.0, 0.0, 0.0), "+X (camera right)"),
        _lobe_env("lobe_pos_y", (0.0, 1.0, 0.0), "+Y (up)"),
        _lobe_env("lobe_pos_z", (0.0, 0.0, 1.0), "+Z (behind the camera)"),
        _lobe_env("lobe_neg_z", (0.0, 0.0, -1.0), "-Z (in front of the camera)"),
        Environment(
            name="lobe_oblique_warm",
            description=(
                "Warm soft lobe colour (3.0, 2.4, 1.65) toward (2, 2, 1)/3 (upper right, toward"
                " the viewer) over a cool ambient (0.05, 0.06, 0.08)."
                " Used for sphere_reference.png."
            ),
            ambient=(0.05, 0.06, 0.08),
            terms=(ZonalTerm(axis=OBLIQUE, colour=(3.0, 2.4, 1.65), profile=_lobe),),
            dominant=OBLIQUE,
        ),
        Environment(
            name="sky_ground",
            description=(
                "Two-colour sky and ground: (0.6, 0.75, 1.0) where y > 0, (0.35, 0.25, 0.15) where"
                " y < 0."
            ),
            terms=(
                ZonalTerm(axis=(0.0, 1.0, 0.0), colour=(0.6, 0.75, 1.0), profile=_one, lo=0.0),
                ZonalTerm(axis=(0.0, 1.0, 0.0), colour=(0.35, 0.25, 0.15), profile=_one, hi=0.0),
            ),
            dominant=(0.0, 1.0, 0.0),
        ),
    ]


SPHERE_REFERENCE_ENV = "lobe_oblique_warm"


def environment_by_name(name: str) -> Environment:
    for env in analytic_environments():
        if env.name == name:
            return env
    raise KeyError(name)
