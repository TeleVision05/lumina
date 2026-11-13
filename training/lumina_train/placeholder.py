"""Placeholder model: random weights, the exact input/output contract of the real one.

Run: uv run python -m lumina_train.placeholder [--out web/public/models/lumina-sh.onnx]
The training engineer replaces the output file with the trained network. The output is biased
toward a dim neutral light from above so a web app wired to it renders something plausible; it
carries no information about the scene.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
import torch
from torch import nn

from lumina_train.model_contract import (
    CONTRACT_VERSION,
    INPUT_NAME,
    INPUT_SHAPE,
    MIN_OPSET,
    MODEL_PATH,
    OUTPUT_NAME,
)

SEED = 20261113
#: Bias SH: uniform grey radiance 0.5 plus a band-1 term toward +Y (light from above).
NEUTRAL_L00 = 0.5 * 4.0 * math.pi * 0.282095
NEUTRAL_L1Y = 0.5


class PlaceholderNet(nn.Module):
    """Three strided convs, global pool, linear to 27. Normalisation lives inside the graph."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(32, 27)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - 0.5) / 0.25
        return self.head(self.features(x)).reshape(-1, 9, 3)


def build_placeholder() -> PlaceholderNet:
    torch.manual_seed(SEED)
    model = PlaceholderNet()
    with torch.no_grad():
        model.head.weight.mul_(0.01)
        bias = torch.zeros(9, 3)
        bias[0, :] = NEUTRAL_L00
        bias[1, :] = NEUTRAL_L1Y
        model.head.bias.copy_(bias.reshape(-1))
    return model.eval()


def export_onnx(model: nn.Module, path: Path, placeholder: bool) -> None:
    """Export with the contract's names, static shapes and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.rand(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=max(MIN_OPSET, 18),
        dynamo=True,
        external_data=False,
    )
    proto = onnx.load(str(path))
    meta = {
        "lumina.contract": CONTRACT_VERSION,
        "lumina.placeholder": "true" if placeholder else "false",
        "lumina.input": "image float32 [1,3,96,128] NCHW RGB sRGB-encoded 0..1",
        "lumina.output": "sh float32 [1,9,3] linear radiance SH9, camera frame",
    }
    onnx.helper.set_model_props(proto, meta)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)
    export_onnx(build_placeholder(), args.out, placeholder=True)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
