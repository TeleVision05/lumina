"""(image, label) pairs from cached panoramas: crop -> camera pipeline -> label in camera frame.

The label of a crop is rotate_sh(sh_world, R) * (k * g): the whole panorama's SH9 expressed in the
camera frame and scaled by the same exposure k and white-balance gains g as the image. A
horizontally flipped sample is the mirror world: image flipped left-right and label mirror_x'd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from lumina_train import sh_torch
from lumina_train.camera import pad_wrap, render_crops, rotations
from lumina_train.isp import jpeg, run_isp
from lumina_train.model_contract import INPUT_SHAPE
from lumina_train.params import Params

OUT_H, OUT_W = INPUT_SHAPE[2], INPUT_SHAPE[3]


@dataclass
class Batch:
    images: Tensor  # [B, 3, 96, 128] sRGB-encoded 0..1 (the model input)
    labels: Tensor  # [B, 9, 3] linear radiance SH9, camera frame, exposure and WB applied
    pano_index: np.ndarray  # [B] index into the bank
    linear: Tensor | None = None  # [B, 3, 96, 128] linear crop before the pipeline (optional)


def finish_images(images: Tensor, params: Params) -> Tensor:
    """JPEG round trip and the horizontal flip; CPU work, safe to run on a worker thread."""
    out = jpeg(images, params["jpeg"])
    flip = torch.as_tensor(params["flip"], device=out.device)
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


class PanoBank:
    """Panoramas [N, 3, H, W] (float16) and world SH [N, 9, 3] resident on one device."""

    def __init__(self, panos: np.ndarray, sh: np.ndarray, device: torch.device) -> None:
        self.device = device
        self.panos = torch.from_numpy(panos).to(device).permute(0, 3, 1, 2).contiguous()
        self.sh = torch.from_numpy(np.asarray(sh, dtype=np.float32)).to(device)

    def __len__(self) -> int:
        return int(self.panos.shape[0])

    def render_device(
        self,
        pano_index: np.ndarray,
        params: Params,
        crops_per_pano: int,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Device part: (images before JPEG and flip, labels with the flip applied, linear)."""
        dev = self.device
        idx = torch.as_tensor(pano_index, device=dev, dtype=torch.long)
        angles = (torch.as_tensor(params[k], dtype=torch.float32) for k in ("yaw", "pitch", "roll"))
        rot = rotations(*angles).to(dev)
        vfov = torch.as_tensor(params["vfov"], dtype=torch.float32, device=dev)
        linear = render_crops(pad_wrap(self.panos[idx].float()), rot, vfov, OUT_H, OUT_W, ss=2)
        images, scale = run_isp(linear, params, noise=noise)
        labels = sh_torch.rotate_sh(self.sh[idx].repeat_interleave(crops_per_pano, 0), rot)
        labels = labels * scale[:, None, :]
        flip = torch.as_tensor(params["flip"], device=dev)
        labels = torch.where(flip[:, None, None], sh_torch.mirror_x(labels), labels)
        return images, labels, linear

    def render(
        self,
        pano_index: np.ndarray,
        params: Params,
        crops_per_pano: int,
        noise: Tensor | None = None,
        keep_linear: bool = False,
    ) -> Batch:
        """Render len(pano_index) * crops_per_pano samples; params are grouped per panorama."""
        images, labels, linear = self.render_device(pano_index, params, crops_per_pano, noise)
        images = finish_images(images, params)
        if keep_linear:
            flip = torch.as_tensor(params["flip"], device=self.device)
            linear = torch.where(flip[:, None, None, None], linear.flip(-1), linear)
        return Batch(
            images=images,
            labels=labels,
            pano_index=np.repeat(np.asarray(pano_index), crops_per_pano),
            linear=linear if keep_linear else None,
        )
