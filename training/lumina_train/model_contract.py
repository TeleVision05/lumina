"""The model file contract shared with the web app (docs/CONVENTIONS.md, section Model file)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "web" / "public" / "models" / "lumina-sh.onnx"

INPUT_NAME = "image"
#: NCHW, RGB, sRGB-encoded values in [0, 1]; centre-cropped to 4:3 then resized.
INPUT_SHAPE = (1, 3, 96, 128)
OUTPUT_NAME = "sh"
#: (batch, coefficient, RGB): linear incident-radiance SH9 in the camera frame.
OUTPUT_SHAPE = (1, 9, 3)
MIN_OPSET = 17
#: Bumped when the meaning of the input or output changes; stored in the ONNX metadata.
CONTRACT_VERSION = "1"
