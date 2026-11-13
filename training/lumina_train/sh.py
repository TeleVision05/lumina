"""Real spherical harmonics up to band 2 (SH9), matching three.js SphericalHarmonics3.

Conventions (see docs/CONVENTIONS.md, which is the contract):
- Directions are unit vectors in a right-handed, +Y-up frame. Camera frame: +X right, +Y up,
  -Z forward (three.js camera default).
- Basis and ordering are exactly three.js SphericalHarmonics3.getBasisAt.
- Coefficients describe incident RADIANCE in linear RGB. Arrays of coefficients have shape [9, 3]
  (coefficient, RGB). Irradiance is E(n) = sum_i A_band(i) * L_i * Y_i(n), A = [pi, 2pi/3, pi/4].
- Equirectangular images use the three.js equirect mapping: u = atan2(z, x) / (2 pi) + 0.5 from
  the left edge, row 0 at the zenith (+Y), so the image centre column looks along +X and the
  quarter column u = 0.25 looks along -Z.
All maths is float64.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Basis constants, three.js SphericalHarmonics3.getBasisAt (r187).
C0 = 0.282095
C1 = 0.488603
C2 = 1.092548
C3 = 0.315392
C4 = 0.546274

#: Band of each of the nine coefficients.
BAND_OF_COEFF: tuple[int, ...] = (0, 1, 1, 1, 2, 2, 2, 2, 2)
#: Cosine-lobe convolution factor per band (Ramamoorthi and Hanrahan 2001).
BAND_FACTORS: tuple[float, float, float] = (math.pi, 2.0 * math.pi / 3.0, math.pi / 4.0)
#: Per-coefficient convolution factor A_band(i).
COEFF_FACTORS: FloatArray = np.array([BAND_FACTORS[b] for b in BAND_OF_COEFF], dtype=np.float64)
#: Rec. 709 / sRGB linear luminance weights.
LUMA: FloatArray = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
#: Coefficients that change sign when the X axis is mirrored (x -> -x): Y_3 ~ x, Y_4 ~ xy, Y_7 ~ xz.
MIRROR_X_SIGNS: FloatArray = np.array([1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float64)
#: For a non-negative radiance function |L1| / L0 <= sqrt(3); equality for a single point light.
MAX_L1_OVER_L0 = C1 / C0


def sh_basis(dirs: FloatArray) -> FloatArray:
    """Evaluate the nine basis functions at unit directions [..., 3] -> [..., 9]."""
    d = np.asarray(dirs, dtype=np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack(
        [
            np.full_like(x, C0),
            C1 * y,
            C1 * z,
            C1 * x,
            C2 * x * y,
            C2 * y * z,
            C3 * (3.0 * z * z - 1.0),
            C2 * x * z,
            C4 * (x * x - y * y),
        ],
        axis=-1,
    )


def eval_sh(sh: FloatArray, dirs: FloatArray) -> FloatArray:
    """Radiance reconstructed from SH9 at directions [..., 3] -> [..., 3]."""
    return sh_basis(dirs) @ np.asarray(sh, dtype=np.float64)


def equirect_directions(height: int, width: int) -> FloatArray:
    """Unit direction at each pixel centre of an equirect image, shape [H, W, 3]."""
    theta = (np.arange(height, dtype=np.float64) + 0.5) / height * math.pi  # polar angle from +Y
    phi = ((np.arange(width, dtype=np.float64) + 0.5) / width - 0.5) * 2.0 * math.pi
    sin_t, cos_t = np.sin(theta)[:, None], np.cos(theta)[:, None]
    x = np.cos(phi)[None, :] * sin_t
    z = np.sin(phi)[None, :] * sin_t
    y = np.broadcast_to(cos_t, x.shape)
    return np.stack([x, y, z], axis=-1)


def equirect_solid_angles(height: int, width: int) -> FloatArray:
    """Exact solid angle of each pixel of an equirect image, shape [H, W]; sums to 4 pi."""
    edges = np.arange(height + 1, dtype=np.float64) / height * math.pi
    row = (2.0 * math.pi / width) * (np.cos(edges[:-1]) - np.cos(edges[1:]))
    return np.broadcast_to(row[:, None], (height, width)).copy()


def camera_rotation(yaw: float, pitch: float, roll: float) -> FloatArray:
    """Camera-to-world rotation R = Ry(yaw) @ Rx(pitch) @ Rz(roll), angles in radians.

    Columns of R are the camera's +X, +Y, +Z axes in world coordinates. Positive yaw turns the
    camera left (forward -Z swings toward -X), positive pitch looks up (-Z toward +Y), positive
    roll turns the camera body counter-clockwise as seen from behind it (+X toward +Y), so the
    scene appears rotated clockwise in the image. A camera-frame direction d_c corresponds to the
    world direction R @ d_c.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def world_to_camera(dirs: FloatArray, cam_to_world: FloatArray) -> FloatArray:
    """Express world-frame directions [..., 3] in the camera frame (d_c = R^T d_w)."""
    return np.asarray(dirs, dtype=np.float64) @ np.asarray(cam_to_world, dtype=np.float64)


def project_equirect(radiance: FloatArray, cam_to_world: FloatArray | None = None) -> FloatArray:
    """Project a linear-RGB equirect environment [H, W, 3] onto SH9 -> [9, 3].

    The environment is in the world frame. With cam_to_world, the result describes the same
    environment in that camera's frame, which is what the network label must be.
    """
    img = np.asarray(radiance, dtype=np.float64)
    height, width = img.shape[:2]
    dirs = equirect_directions(height, width)
    if cam_to_world is not None:
        dirs = world_to_camera(dirs, cam_to_world)
    weights = equirect_solid_angles(height, width)
    basis = sh_basis(dirs) * weights[..., None]
    return np.einsum("hwk,hwc->kc", basis, img)


def fibonacci_sphere(count: int) -> FloatArray:
    """Deterministic, near-uniform unit directions [count, 3]."""
    i = np.arange(count, dtype=np.float64) + 0.5
    y = 1.0 - 2.0 * i / count
    r = np.sqrt(1.0 - y * y)
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), y, r * np.sin(phi)], axis=-1)


_FIT_DIRS = fibonacci_sphere(64)
_FIT_PINV = np.linalg.pinv(sh_basis(_FIT_DIRS))


def rotate_sh(sh: FloatArray, cam_to_world: FloatArray) -> FloatArray:
    """Re-express world-frame SH9 in a camera frame; equals project_equirect(env, cam_to_world).

    Rotation never mixes bands, so fitting the rotated function on 64 directions is exact up to
    float rounding. Use it to label many crops of one HDRI from a single projection.
    """
    world_dirs = _FIT_DIRS @ np.asarray(cam_to_world, dtype=np.float64).T  # R @ d_c
    values = eval_sh(sh, world_dirs)
    return _FIT_PINV @ values


def irradiance(sh: FloatArray, normals: FloatArray) -> FloatArray:
    """Irradiance E(n) = sum_i A_band(i) L_i Y_i(n) for unit normals [..., 3] -> [..., 3]."""
    return sh_basis(normals) @ (COEFF_FACTORS[:, None] * np.asarray(sh, dtype=np.float64))


def mirror_x(sh: FloatArray) -> FloatArray:
    """SH of the environment mirrored left-right (x -> -x), for a mirrored front-camera view."""
    return np.asarray(sh, dtype=np.float64) * MIRROR_X_SIGNS[:, None]


def dominant_direction(sh: FloatArray, min_ratio: float = 1e-6) -> tuple[FloatArray | None, float]:
    """Dominant light direction from the luminance of the band-1 coefficients.

    Returns (unit direction or None, directionality). directionality = |L1| / (sqrt(3) L0) of the
    luminance SH, in [0, 1]: 0 for uniform light, 1 for a single point light. The direction is None
    when the luminance L0 is not positive or directionality is below min_ratio.
    """
    lum = np.asarray(sh, dtype=np.float64) @ LUMA
    l0 = float(lum[0])
    v = np.array([lum[3], lum[1], lum[2]])  # (x, y, z) from (Y_3 ~ x, Y_1 ~ y, Y_2 ~ z)
    norm = float(np.linalg.norm(v))
    if l0 <= 0.0:
        return None, 0.0
    ratio = min(norm / (MAX_L1_OVER_L0 * l0), 1.0)
    if ratio < min_ratio:
        return None, ratio
    return v / norm, ratio


def sphere_normals(size: int) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Normals of a unit sphere seen by an orthographic camera looking down -Z.

    The image spans x, y in [-1, 1]; row 0 is the top (+Y). Returns (normals [S, S, 3], mask).
    Pixels outside the disc have a zero normal and mask False.
    """
    centres = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    x = np.broadcast_to(centres[None, :], (size, size))
    y = np.broadcast_to(-centres[:, None], (size, size))
    r2 = x * x + y * y
    mask = r2 < 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    normals = np.stack([x, y, z], axis=-1) * mask[..., None]
    return normals, mask


def render_diffuse_sphere(sh: FloatArray, size: int, albedo: float = 1.0) -> FloatArray:
    """Linear outgoing radiance of a Lambertian sphere: albedo / pi * E(n); black background."""
    normals, mask = sphere_normals(size)
    img = albedo / math.pi * irradiance(sh, normals)
    return img * mask[..., None]


def linear_to_srgb(linear: FloatArray) -> FloatArray:
    """sRGB transfer function (IEC 61966-2-1), the same piecewise curve three.js uses."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear(encoded: FloatArray) -> FloatArray:
    """Inverse sRGB transfer function."""
    s = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0)
    return np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4))


def to_srgb8(linear: FloatArray) -> NDArray[np.uint8]:
    """Linear RGB in [0, 1] -> 8-bit sRGB with round-half-up quantisation."""
    return np.floor(linear_to_srgb(linear) * 255.0 + 0.5).astype(np.uint8)
