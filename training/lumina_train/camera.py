"""Pinhole camera: poses, rays, and perspective crops of equirect panoramas (batched torch).

Frame conventions are docs/CONVENTIONS.md sections 1 and 4: camera +X right, +Y up, -Z forward;
camera_rotation(yaw, pitch, roll) = Ry Rx Rz is camera-to-world; the equirect column coordinate is
u = atan2(z, x) / (2 pi) + 0.5 and row 0 is the zenith. Image row 0 is the top of the picture.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as nnf

#: Panoramas are padded with this many wrapped columns on each side so bilinear lookups across
#: the u = 0 / u = 1 seam blend the right neighbours.
WRAP = 1


def rotations(yaw: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
    """Batched camera_rotation: [B] angles in radians -> camera-to-world [B, 3, 3]."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    one, zero = torch.ones_like(yaw), torch.zeros_like(yaw)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(-1, 3, 3)
    rx = torch.stack([one, zero, zero, zero, cp, -sp, zero, sp, cp], -1).reshape(-1, 3, 3)
    rz = torch.stack([cr, -sr, zero, sr, cr, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    return ry @ rx @ rz


def camera_rays(vfov: Tensor, height: int, width: int) -> Tensor:
    """Unit camera-frame rays through pixel centres: [B] vertical FOV (radians) -> [B, H, W, 3]."""
    device, dtype = vfov.device, vfov.dtype
    tan_half = torch.tan(0.5 * vfov)[:, None, None]
    aspect = width / height
    cols = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width * 2.0 - 1.0
    rows = 1.0 - (torch.arange(height, device=device, dtype=dtype) + 0.5) / height * 2.0
    x = cols[None, None, :] * tan_half * aspect
    y = rows[None, :, None] * tan_half
    x, y = torch.broadcast_tensors(x, y)
    rays = torch.stack([x, y, -torch.ones_like(x)], dim=-1)
    return rays / rays.norm(dim=-1, keepdim=True)


def pad_wrap(panos: Tensor) -> Tensor:
    """[N, C, H, W] -> [N, C, H, W + 2 WRAP] with wrapped columns at both edges."""
    return torch.cat([panos[..., -WRAP:], panos, panos[..., :WRAP]], dim=-1)


def equirect_grid(world_dirs: Tensor, pano_width: int) -> Tensor:
    """World directions [..., 3] -> grid_sample coordinates [..., 2] for a WRAP-padded panorama.

    pano_width is the unpadded width. Uses align_corners=False pixel-centre conventions.
    """
    x, y, z = world_dirs[..., 0], world_dirs[..., 1], world_dirs[..., 2]
    u = torch.atan2(z, x) / (2.0 * math.pi) + 0.5  # [0, 1)
    theta = torch.acos(y.clamp(-1.0, 1.0))  # polar angle from +Y
    padded = pano_width + 2 * WRAP
    px = WRAP + u * pano_width  # continuous pixel coordinate (edge-based) in the padded image
    gx = 2.0 * px / padded - 1.0
    gy = 2.0 * theta / math.pi - 1.0
    return torch.stack([gx, gy], dim=-1)


def render_crops(
    padded_panos: Tensor, cam_to_world: Tensor, vfov: Tensor, height: int, width: int, ss: int = 2
) -> Tensor:
    """Perspective crops, linear radiance, [G, C, H', W'] panos and [G*K] poses -> [G*K, C, H, W].

    padded_panos holds G panoramas (WRAP-padded); the G*K poses are grouped K per panorama in
    order, so crop i comes from panorama i // K. Renders at ss x the resolution and box-filters
    down, which anti-aliases the panorama lookup.
    """
    g = padded_panos.shape[0]
    total = cam_to_world.shape[0]
    k = total // g
    rays = camera_rays(vfov, height * ss, width * ss)  # [GK, h, w, 3]
    world = torch.einsum("bij,bhwj->bhwi", cam_to_world, rays)
    grid = equirect_grid(world, padded_panos.shape[-1] - 2 * WRAP)  # [GK, h, w, 2]
    hs, ws = height * ss, width * ss
    grid = grid.reshape(g, k, hs, ws, 2).permute(0, 2, 1, 3, 4).reshape(g, hs, k * ws, 2)
    out = nnf.grid_sample(
        padded_panos,
        grid.to(padded_panos.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    out = out.reshape(g, -1, hs, k, ws).permute(0, 3, 1, 2, 4).reshape(total, -1, hs, ws)
    if ss > 1:
        out = nnf.avg_pool2d(out, ss)
    return out


def pixel_direction(row: float, col: float, vfov: float, height: int, width: int) -> np.ndarray:
    """Camera-frame unit ray through a (fractional) pixel centre, numpy, for tests and baselines."""
    tan_half = math.tan(0.5 * vfov)
    x = ((col + 0.5) / width * 2.0 - 1.0) * tan_half * width / height
    y = (1.0 - (row + 0.5) / height * 2.0) * tan_half
    d = np.array([x, y, -1.0])
    return d / np.linalg.norm(d)
