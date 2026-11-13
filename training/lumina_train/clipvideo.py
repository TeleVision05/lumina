"""H.264 encode / decode for the sample clips (ffmpeg CLI) and the model-input resize.

Frames are 8-bit sRGB RGB. Encoding converts to YUV 4:2:0 with the BT.709 matrix in limited range
and tags the stream BT.709, so a browser decodes the colours that were encoded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from lumina_train.model_contract import INPUT_SHAPE

MAX_BYTES = 1_500_000


def encode_mp4(frames: np.ndarray, fps: int, path: Path, crf: int = 23) -> int:
    """frames uint8 [N, H, W, 3] RGB -> MP4; raises the CRF until the file is under MAX_BYTES."""
    _, h, w, _ = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "main",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-color_range", "tv", "-movflags", "+faststart", "-an", str(path),
        ]  # fmt: skip
        subprocess.run(cmd, input=frames.tobytes(), check=True)
        size = path.stat().st_size
        if size <= MAX_BYTES or crf >= 40:
            return crf
        crf += 2


def decode_mp4(path: Path, width: int, height: int) -> np.ndarray:
    """MP4 -> uint8 [N, H, W, 3] RGB, converting with the BT.709 limited-range matrix."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def model_inputs(frames: np.ndarray) -> np.ndarray:
    """Centre-crop to 4:3 and area-resize to the model's 128 x 96: uint8 [N, 96, 128, 3]."""
    out_h, out_w = INPUT_SHAPE[2], INPUT_SHAPE[3]
    _, h, w, _ = frames.shape
    if w * 3 > h * 4:
        cw = h * 4 // 3
        frames = frames[:, :, (w - cw) // 2 : (w - cw) // 2 + cw]
    elif w * 3 < h * 4:
        ch = w * 3 // 4
        frames = frames[:, (h - ch) // 2 : (h - ch) // 2 + ch]
    return np.stack([cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA) for f in frames])
