"""Load HDRI panoramas and build the per-split cache the sampler trains from.

    uv run python -m lumina_train.hdri          # builds data/cache/<split>.npz for every split

Each cached panorama is the 1k equirect (512 x 1024) in linear RGB, divided by its own median
luminance (so the median is 1) and stored as float16, clamped at FP16_CLAMP. The world-frame SH9
of the same normalised panorama is computed from the float32 original BEFORE the clamp, so a sun
brighter than float16 can hold still contributes its full energy to the label. Exposure is
re-chosen per crop by auto-exposure anyway, so the per-panorama normalisation never reaches a
label without the matching image scale.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from lumina_train import sh as shm
from lumina_train.paths import AMBIENTCG_DIR, CACHE_DIR, POLYHAVEN_DIR, SPLIT_PATH

PANO_HEIGHT = 512
PANO_WIDTH = 1024
FP16_CLAMP = 60000.0


def load_hdri(path: Path) -> np.ndarray:
    """Read a .hdr or .exr file -> float32 linear RGB [H, W, 3], non-finite and negative -> 0."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"could not read HDRI {path}")
    rgb = np.ascontiguousarray(img[..., 2::-1], dtype=np.float32)  # BGR(A) -> RGB
    rgb[~np.isfinite(rgb)] = 0.0
    return np.maximum(rgb, 0.0)


def resize_equirect(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Area-resample to [height, width]; energy-preserving per solid angle row for downscales."""
    if img.shape[:2] == (height, width):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def normalise_panorama(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Divide by the median pixel luminance (robust to the sun). Returns (image, scale)."""
    lum = img @ shm.LUMA.astype(np.float32)
    median = float(np.median(lum))
    scale = median if median > 1e-8 else max(float(lum.mean()), 1e-8)
    return img / scale, scale


def prepare(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(float16 panorama [512, 1024, 3], world SH [9, 3] float64) for one HDRI file."""
    img = resize_equirect(load_hdri(path), PANO_HEIGHT, PANO_WIDTH)
    img, _ = normalise_panorama(img)
    sh_world = shm.project_equirect(img)
    pano = np.minimum(img, FP16_CLAMP).astype(np.float16)
    return pano, sh_world


def hdri_path(source: str, asset_id: str) -> Path:
    if source == "polyhaven":
        return POLYHAVEN_DIR / "1k" / f"{asset_id}_1k.hdr"
    return AMBIENTCG_DIR / "1k" / f"{asset_id}_1k.exr"


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}.npz"


def build_split_cache(split: str, entries: list[dict[str, str]], workers: int = 4) -> Path:
    """Write data/cache/<split>.npz with panos [N, H, W, 3] f16, sh [N, 9, 3] f64, ids, env."""
    paths = [hdri_path(e["source"], e["id"]) for e in entries]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} HDRIs missing, e.g. {missing[0]}; download first")
    panos = np.empty((len(paths), PANO_HEIGHT, PANO_WIDTH, 3), dtype=np.float16)
    shs = np.empty((len(paths), 9, 3), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        for i, (pano, sh_world) in enumerate(pool.map(prepare, paths)):
            panos[i], shs[i] = pano, sh_world
    out = cache_path(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        panos=panos,
        sh=shs,
        ids=np.array([e["id"] for e in entries]),
        environment=np.array([e["environment"] for e in entries]),
    )
    print(f"wrote {out}: {len(entries)} panoramas", flush=True)
    return out


def load_split_cache(split: str) -> dict[str, np.ndarray]:
    with np.load(cache_path(split)) as data:
        return {k: data[k] for k in data.files}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args(argv)
    split = json.loads(SPLIT_PATH.read_text())
    for name, entries in split["splits"].items():
        if args.splits and name not in args.splits:
            continue
        build_split_cache(name, entries)


if __name__ == "__main__":
    main()
