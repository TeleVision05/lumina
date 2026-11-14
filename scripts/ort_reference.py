"""Run the shipped ONNX model with Python onnxruntime on frames captured in the browser.

Usage: python scripts/ort_reference.py IN.json OUT.json
IN.json:  {"model": "<path to .onnx>", "frames": [[R,G,B,A, ...] x (96*128*4 bytes)], ...}
OUT.json: {"sh": [[[9 x 3]], ...]}  one SH9 RGB array per frame

Preprocessing is the model contract in docs/CONVENTIONS.md: RGB, byte / 255, NCHW float32
[1, 3, 96, 128]. Normalisation is inside the graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

H, W = 96, 128


def run(model: Path, frames: list[list[int]]) -> list[list[list[float]]]:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out: list[list[list[float]]] = []
    for rgba in frames:
        px = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        x = (px.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        (sh,) = session.run(["sh"], {"image": np.ascontiguousarray(x)})
        out.append(sh[0].tolist())
    return out


def main() -> None:
    src = json.loads(Path(sys.argv[1]).read_text())
    sh = run(Path(src["model"]), src["frames"])
    Path(sys.argv[2]).write_text(json.dumps({"sh": sh}))


if __name__ == "__main__":
    main()
