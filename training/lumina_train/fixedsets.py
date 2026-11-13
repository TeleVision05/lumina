"""Fixed, pre-generated sample sets: the validation set and the evaluation test crops.

    uv run python -m lumina_train.fixedsets          # writes eval/test_crops.json

Both are rendered on the CPU with per-sample noise seeds, so they are bit-for-bit repeatable on
one machine. The validation set (8 fully random crops per validation HDRI) is cached under
data/cache/; the test crop list is committed as eval/test_crops.json: for every HDRI of a test
split, 5 camera poses x 3 exposure / white-balance settings, every parameter written out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lumina_train import params as pm
from lumina_train.hdri import load_split_cache
from lumina_train.paths import CACHE_DIR, EVAL_DIR
from lumina_train.synth import PanoBank

VAL_SEED = 1001
VAL_CROPS = 8
TEST_SEED = 2002
TEST_POSES = 5
#: (name, EV offset, kelvin): neutral, warm and darker, cool and brighter.
TEST_SETTINGS = (
    ("neutral", 0.0, 6500.0),
    ("warm_dark", -1.0, 3200.0),
    ("cool_bright", 1.0, 8500.0),
)
TEST_CROPS_PATH = EVAL_DIR / "test_crops.json"
CPU = torch.device("cpu")


def val_params(n_panos: int) -> pm.Params:
    return pm.sample_params(np.random.default_rng(VAL_SEED), n_panos * VAL_CROPS)


def test_params(n_panos: int, seed: int) -> pm.Params:
    """Per panorama: TEST_POSES poses, each under every TEST_SETTINGS entry (un-flipped)."""
    rng = np.random.default_rng(seed)
    per = TEST_POSES * len(TEST_SETTINGS)
    out = pm.sample_params(rng, n_panos * per)
    out["flip"][:] = False
    poses = pm.sample_pose(rng, n_panos * TEST_POSES)
    for key in ("yaw", "pitch", "roll", "vfov"):
        out[key] = np.repeat(poses[key], len(TEST_SETTINGS))
    settings = np.tile(np.arange(len(TEST_SETTINGS)), n_panos * TEST_POSES)
    out["ev"] = np.array([TEST_SETTINGS[s][1] for s in settings])
    out["kelvin"] = np.array([TEST_SETTINGS[s][2] for s in settings])
    out["tint"][:] = 0.0
    out["ae_percentile"][:] = 0.6
    return out


def render_set(
    cache: dict[str, np.ndarray], params: pm.Params, crops_per_pano: int, chunk: int = 8
) -> dict[str, np.ndarray]:
    """Render each panorama's crops on the CPU: images uint8 [N, 96, 128, 3], labels [N, 9, 3]."""
    bank = PanoBank(cache["panos"], cache["sh"], CPU)
    images, labels, index = [], [], []
    for start in range(0, len(bank), chunk):
        pano_idx = np.arange(start, min(start + chunk, len(bank)))
        sl = slice(start * crops_per_pano, (start + len(pano_idx)) * crops_per_pano)
        with torch.no_grad():
            batch = bank.render(pano_idx, pm.select(params, sl), crops_per_pano)
        u8 = (batch.images.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
        images.append(u8)
        labels.append(batch.labels.numpy())
        index.append(batch.pano_index)
    return {
        "images": np.concatenate(images),
        "labels": np.concatenate(labels).astype(np.float32),
        "pano_index": np.concatenate(index),
    }


def val_set_path() -> Path:
    return CACHE_DIR / "val_samples.npz"


def load_or_build_val_set() -> dict[str, np.ndarray]:
    path = val_set_path()
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    cache = load_split_cache("val")
    out = render_set(cache, val_params(len(cache["ids"])), VAL_CROPS)
    out["ids"] = cache["ids"][out["pano_index"]]
    np.savez(path, **out)
    return out


def write_test_crops() -> Path:
    """Write the committed crop list for the 'test' and 'ood' splits."""
    doc: dict[str, object] = {
        "schema": 1,
        "description": (
            f"{TEST_POSES} poses x {len(TEST_SETTINGS)} settings per HDRI; angles in radians;"
            " settings cycle fastest; replay with lumina_train.evaluate"
        ),
        "settings": [{"name": n, "ev": e, "kelvin": k} for n, e, k in TEST_SETTINGS],
        "crops_per_hdri": TEST_POSES * len(TEST_SETTINGS),
        "splits": {},
    }
    splits: dict[str, object] = {}
    for offset, split in enumerate(("test", "ood")):
        ids = load_split_cache(split)["ids"].tolist()
        params = test_params(len(ids), TEST_SEED + offset)
        splits[split] = {"seed": TEST_SEED + offset, "hdri_ids": ids, "params": _rounded(params)}
    doc["splits"] = splits
    TEST_CROPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_CROPS_PATH.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return TEST_CROPS_PATH


def _rounded(params: pm.Params) -> dict[str, list[float | int | bool]]:
    out = pm.to_json(params)
    for key, values in out.items():
        if pm.FIELDS[key] is float:
            out[key] = [round(float(v), 7) for v in values]
    return out


def load_test_crops(split: str) -> tuple[list[str], pm.Params, int]:
    doc = json.loads(TEST_CROPS_PATH.read_text())
    entry = doc["splits"][split]
    return entry["hdri_ids"], pm.from_json(entry["params"]), int(doc["crops_per_hdri"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(f"wrote {write_test_crops()}")
    val = load_or_build_val_set()
    print(f"validation set: {len(val['images'])} samples at {val_set_path()}")


if __name__ == "__main__":
    main()
