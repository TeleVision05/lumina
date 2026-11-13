"""LuminaNet: a compact residual CNN (about 1.5 M parameters) -> SH9 radiance in the camera frame.

Input: [B, 3, 96, 128] sRGB-encoded 0..1 (the model contract); the normalisation is the first op
of the graph. Five stride-2 stages (24, 32, 64, 128, 192 channels) with one residual block each
bring the picture to a 3 x 4 map. A 1 x 1 conv squeezes it to 64 channels and the map is
FLATTENED rather than pooled, so the head knows where in the picture each feature is (where the
bright window is matters for the light direction).

Why dense 3 x 3 convolutions and not MobileNetV2: on the M2 Max's MPS backend a training step at
batch 256 took 0.221 s for MobileNetV2 (width 0.75, 1.26 M parameters) and 0.049 s for a plain
dense CNN of similar size and multiply count (measured 2026-11-13 with a throwaway benchmark), so
the same wall-clock budget sees about four times as many samples. Dense 3 x 3 convs are also the
best-optimised kernels in onnxruntime-web's wasm backend.

Output parametrisation: per colour channel c, L0_c = exp(a_c) and L_i,c = L0_c * r_i,c for
i = 1..8, so the network predicts a log brightness and a shape relative to it; this keeps the huge
brightness range of HDR scenes well conditioned. All of it is inside the exported graph: the ONNX
output is plain radiance SH [1, 9, 3].

Weights are randomly initialised: no ImageNet weights, whose terms are research-only.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn

STAGES = (24, 32, 64, 128, 192)
SQUEEZE = 64
HIDDEN = 256
LOG_L0_CLAMP = 9.0
#: Initial log L0, near the median label L0 (about 2.9 on a validation batch), so training
#: starts near the right brightness.
INIT_LOG_L0 = 1.0


def conv_bn(c_in: int, c_out: int, stride: int = 1, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
        nn.BatchNorm2d(c_out),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(channels, channels), conv_bn(channels, channels, act=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.body(x))


class LuminaNet(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = [conv_bn(3, STAGES[0], stride=2)]
        for c_in, c_out in itertools.pairwise(STAGES):
            layers += [conv_bn(c_in, c_out, stride=2), ResBlock(c_out)]
        self.backbone = nn.Sequential(*layers)
        self.squeeze = nn.Sequential(
            nn.Conv2d(STAGES[-1], SQUEEZE, 1, bias=False),
            nn.BatchNorm2d(SQUEEZE),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(SQUEEZE * 3 * 4, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, 27),
        )
        last = self.head[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.mul_(0.1)
            bias = torch.zeros(9, 3)
            bias[0] = INIT_LOG_L0
            last.bias.copy_(bias.reshape(-1))

    def raw(self, image: Tensor) -> Tensor:
        x = (image - 0.5) / 0.25
        return self.head(self.squeeze(self.backbone(x))).reshape(-1, 9, 3)

    def forward(self, image: Tensor) -> Tensor:
        return raw_to_sh(self.raw(image))


def raw_to_sh(raw: Tensor) -> Tensor:
    """[B, 9, 3] raw outputs -> radiance SH: L0 = exp(a), L_i = L0 * r_i."""
    l0 = torch.exp(raw[:, :1, :].clamp(-LOG_L0_CLAMP, LOG_L0_CLAMP))
    return torch.cat([l0, l0 * raw[:, 1:, :]], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
