"""Batched torch versions of the SH9 maths in sh.py (same basis, ordering and constants).

Used on MPS during training and on CPU in evaluation. tests/test_sh_torch.py checks every function
against its float64 numpy twin in sh.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lumina_train import sh as shm

_FIT_DIRS = torch.from_numpy(shm.fibonacci_sphere(64)).float()
_FIT_PINV = torch.from_numpy(shm._FIT_PINV).float()
_COEFF_FACTORS = torch.from_numpy(shm.COEFF_FACTORS).float()
_LUMA = torch.from_numpy(shm.LUMA).float()
MIRROR_X_SIGNS = torch.from_numpy(shm.MIRROR_X_SIGNS).float()


def sh_basis(dirs: Tensor) -> Tensor:
    """[..., 3] unit directions -> [..., 9] basis values."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    return torch.stack(
        [
            torch.full_like(x, shm.C0),
            shm.C1 * y,
            shm.C1 * z,
            shm.C1 * x,
            shm.C2 * x * y,
            shm.C2 * y * z,
            shm.C3 * (3.0 * z * z - 1.0),
            shm.C2 * x * z,
            shm.C4 * (x * x - y * y),
        ],
        dim=-1,
    )


def rotate_sh(sh: Tensor, cam_to_world: Tensor) -> Tensor:
    """World-frame SH [B, 9, 3] -> camera-frame SH for camera-to-world rotations [B, 3, 3]."""
    dirs = _FIT_DIRS.to(sh.device, sh.dtype)
    world_dirs = torch.einsum("bij,nj->bni", cam_to_world.to(sh.dtype), dirs)  # R @ d_c
    values = sh_basis(world_dirs) @ sh  # [B, 64, 3]
    return _FIT_PINV.to(sh.device, sh.dtype) @ values


def irradiance(sh: Tensor, normals: Tensor) -> Tensor:
    """SH [B, 9, 3] and unit normals [N, 3] -> irradiance [B, N, 3]."""
    factors = _COEFF_FACTORS.to(sh.device, sh.dtype)[None, :, None]
    return sh_basis(normals.to(sh.dtype)) @ (factors * sh)


def luminance_sh(sh: Tensor) -> Tensor:
    """[B, 9, 3] -> [B, 9] Rec. 709 luminance coefficients."""
    return sh @ _LUMA.to(sh.device, sh.dtype)


def band1_vector(sh: Tensor) -> Tensor:
    """Luminance band-1 vector (x, y, z) = (l_3, l_1, l_2), unnormalised, [B, 3]."""
    lum = luminance_sh(sh)
    return torch.stack([lum[:, 3], lum[:, 1], lum[:, 2]], dim=-1)


def directionality(sh: Tensor) -> Tensor:
    """|L1| / (sqrt(3) L0) of the luminance SH, clamped to [0, 1], [B]."""
    lum = luminance_sh(sh)
    norm = band1_vector(sh).norm(dim=-1)
    return (norm / (shm.MAX_L1_OVER_L0 * lum[:, 0].clamp_min(1e-12))).clamp(0.0, 1.0)


def mirror_x(sh: Tensor) -> Tensor:
    return sh * MIRROR_X_SIGNS.to(sh.device, sh.dtype)[:, None]


def fibonacci_normals(count: int = 64) -> Tensor:
    return torch.from_numpy(shm.fibonacci_sphere(count)).float()
