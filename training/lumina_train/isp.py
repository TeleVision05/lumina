"""Synthetic webcam image pipeline (batched torch), applied to linear radiance crops.

Order: lens blur -> vignetting -> auto-exposure scale k -> white-balance gains g -> sensor noise
-> clip at 1 -> tone curve -> saturation -> 8-bit quantisation (-> JPEG, on CPU, in jpeg()).
The label scale returned for each sample is k * g (per channel): the same factors the image got.
Blur, vignetting, noise, clipping, the tone curve and JPEG are things the camera does to the
picture, not to the light, so they are not in the label.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as nnf

from lumina_train import sh as shm
from lumina_train.params import AE_TARGET, TONE_FILMIC, TONE_GAMMA, Params, white_balance_gains

BLUR_RADIUS = 4


def _t(params: Params, key: str, device: torch.device) -> Tensor:
    return torch.as_tensor(np.asarray(params[key], dtype=np.float32), device=device)


def gaussian_blur(img: Tensor, sigma: Tensor) -> Tensor:
    """Separable Gaussian blur with a per-sample sigma (pixels) [B]; sigma ~ 0 is identity."""
    b, c, h, w = img.shape
    offsets = torch.arange(-BLUR_RADIUS, BLUR_RADIUS + 1, device=img.device, dtype=img.dtype)
    s = sigma.clamp_min(0.05)[:, None]
    kernel = torch.exp(-0.5 * (offsets[None, :] / s) ** 2)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)  # [B, K]
    kernel = kernel.repeat_interleave(c, dim=0)  # [B*C, K]
    x = img.reshape(1, b * c, h, w)
    pad = BLUR_RADIUS
    x = nnf.conv2d(
        nnf.pad(x, (pad, pad, 0, 0), mode="replicate"), kernel[:, None, None, :], groups=b * c
    )
    x = nnf.conv2d(
        nnf.pad(x, (0, 0, pad, pad), mode="replicate"), kernel[:, None, :, None], groups=b * c
    )
    return x.reshape(b, c, h, w)


def vignette(img: Tensor, strength: Tensor) -> Tensor:
    """Multiply by 1 - strength * r^2, r = 1 at the image corners."""
    _, _, h, w = img.shape
    ys = torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype)
    r2 = (ys[:, None] ** 2 + xs[None, :] ** 2) / 2.0
    return img * (1.0 - strength[:, None, None, None] * r2[None, None])


def luminance(img: Tensor) -> Tensor:
    weights = torch.as_tensor(shm.LUMA, dtype=img.dtype, device=img.device)
    return torch.einsum("bchw,c->bhw", img, weights)


def auto_exposure(img: Tensor, percentile: Tensor, ev: Tensor) -> Tensor:
    """Scale k [B] that puts the given luminance percentile at AE_TARGET, times 2**ev."""
    lum = luminance(img).flatten(1)
    ordered = lum.sort(dim=1).values
    idx = (percentile * (ordered.shape[1] - 1)).round().long().clamp(0, ordered.shape[1] - 1)
    level = ordered.gather(1, idx[:, None])[:, 0].clamp_min(1e-6)
    return AE_TARGET / level * torch.exp2(ev)


def tone_curve(x: Tensor, mode: Tensor, gamma: Tensor, contrast: Tensor) -> Tensor:
    """Linear [0, 1] -> encoded [0, 1]: sRGB (0), pure gamma (1) or sRGB plus an S-curve (2)."""
    srgb = torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
    g = gamma[:, None, None, None]
    gam = x.clamp_min(1e-8) ** (1.0 / g)
    c = contrast[:, None, None, None]
    s = srgb.clamp(1e-6, 1 - 1e-6)
    film = s**c / (s**c + (1.0 - s) ** c)
    m = mode[:, None, None, None]
    return torch.where(m == TONE_GAMMA, gam, torch.where(m == TONE_FILMIC, film, srgb))


def saturate(y: Tensor, amount: Tensor) -> Tensor:
    grey = luminance(y)[:, None]
    return (grey + amount[:, None, None, None] * (y - grey)).clamp(0.0, 1.0)


def sensor_noise(x: Tensor, a: Tensor, b: Tensor, noise: Tensor) -> Tensor:
    """Poisson-Gaussian approximation: x + sqrt(a x + b) * n, with standard normal n."""
    var = a[:, None, None, None] * x.clamp_min(0.0) + b[:, None, None, None]
    return x + var.sqrt() * noise


def standard_noise(params: Params, shape: tuple[int, ...], device: torch.device) -> Tensor:
    """Per-sample deterministic N(0, 1) noise from each sample's seed (CPU generator)."""
    out = torch.empty(shape)
    for i, seed in enumerate(params["seed"]):
        gen = torch.Generator().manual_seed(int(seed))
        out[i] = torch.randn(shape[1:], generator=gen)
    return out.to(device)


def run_isp(
    linear: Tensor, params: Params, noise: Tensor | None = None, exposure: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Linear crops [B, 3, H, W] -> (8-bit-quantised encoded image in [0, 1], label scale [B, 3]).

    noise: standard normal [B, 3, H, W]; None draws it from each sample's seed (deterministic).
    exposure: optional precomputed k [B] (for temporally smoothed auto-exposure in clips).
    """
    dev = linear.device
    x = gaussian_blur(linear, _t(params, "blur", dev))
    x = vignette(x, _t(params, "vignette", dev))
    if exposure is None:
        exposure = auto_exposure(x, _t(params, "ae_percentile", dev), _t(params, "ev", dev))
    k = exposure
    gains = torch.as_tensor(
        white_balance_gains(params["kelvin"], params["tint"]), dtype=x.dtype, device=dev
    )
    scale = k[:, None] * gains  # [B, 3]
    x = x * scale[:, :, None, None]
    if noise is None:
        noise = standard_noise(params, tuple(x.shape), dev)
    x = sensor_noise(x, _t(params, "noise_a", dev), _t(params, "noise_b", dev), noise)
    x = x.clamp(0.0, 1.0)
    y = tone_curve(
        x, _t(params, "tone", dev), _t(params, "gamma", dev), _t(params, "contrast", dev)
    )
    y = saturate(y, _t(params, "saturation", dev))
    return torch.round(y * 255.0) / 255.0, scale


def jpeg(images: Tensor, quality: np.ndarray) -> Tensor:
    """JPEG round trip (OpenCV, 4:2:0) for samples with quality > 0; others pass through."""
    if not np.any(quality > 0):
        return images
    u8 = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    for i in np.nonzero(quality > 0)[0]:
        bgr = np.ascontiguousarray(u8[i, :, :, ::-1])
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality[i])])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        u8[i] = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    out = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0
    return out.to(images.device)
