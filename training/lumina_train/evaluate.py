"""Evaluate the shipped ONNX model and the baselines on the fixed test crops.

    uv run python -m lumina_train.evaluate

Replays eval/test_crops.json (5 poses x 3 exposure/white-balance settings for every HDRI of the
held-out Poly Haven test split and of the ambientCG out-of-distribution split), runs
web/public/models/lumina-sh.onnx through onnxruntime (CPU) and the three baselines, and scores all
of them with lumina_train.metrics. Means and 95 % bootstrap CIs are over HDRIs (each HDRI's crops
are averaged first), with a fixed bootstrap seed. Writes eval/results.json, docs/EVAL.md and the
figures in docs/media/. Needs the downloads and caches (see docs/DATA.md).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from lumina_train import baselines, metrics, report
from lumina_train.fixedsets import TEST_SETTINGS, load_test_crops, render_set
from lumina_train.hdri import load_split_cache
from lumina_train.model_contract import INPUT_NAME, MODEL_PATH, OUTPUT_NAME
from lumina_train.paths import EVAL_DIR

BOOTSTRAP_SEED = 4004
METHODS = ("cnn", "mean", "image_mean", "image_dir")
SPLITS = ("test", "ood")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_onnx(images_u8: np.ndarray, model_path: Path) -> np.ndarray:
    """uint8 [N, 96, 128, 3] -> SH [N, 9, 3], one frame at a time as the web app runs it."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    x = images_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    out = [session.run([OUTPUT_NAME], {INPUT_NAME: x[i : i + 1]})[0][0] for i in range(len(x))]
    return np.stack(out).astype(np.float64)


def render_split(split: str) -> dict[str, Any]:
    ids, params, per = load_test_crops(split)
    cache = load_split_cache(split)
    if cache["ids"].tolist() != ids:
        raise ValueError(f"cache order for {split} differs from eval/test_crops.json")
    rendered = render_set(cache, params, per)
    idx = rendered["pano_index"]
    rendered["hdri"] = cache["ids"][idx]
    rendered["environment"] = cache["environment"][idx]
    rendered["setting"] = np.tile(np.arange(len(TEST_SETTINGS)), len(idx) // len(TEST_SETTINGS))
    return rendered


def predictions(
    rendered: dict[str, Any], model_path: Path, stats: dict[str, Any]
) -> dict[str, np.ndarray]:
    images = rendered["images"]
    return {
        "cnn": predict_onnx(images, model_path),
        "mean": baselines.mean_baseline(len(images), stats["mean_sh"]),
        "image_mean": baselines.image_mean_baseline(images),
        "image_dir": baselines.image_dir_baseline(images, stats["directionality"]),
    }


def summarise(per: dict[str, np.ndarray], hdri: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Mean and bootstrap CI over HDRIs of every metric, for the crops in mask."""
    out: dict[str, Any] = {}
    strong = per["directionality"] >= metrics.DIRECTIONAL_THRESHOLD
    columns = {key: (per[key], mask) for key in metrics.METRICS if key != "dir_deg"}
    columns["dir_deg_directional"] = (per["dir_deg"], mask & strong)
    columns["dir_deg_diffuse"] = (per["dir_deg"], mask & ~strong)
    for key, (values, m) in columns.items():
        if not m.any():
            out[key] = {"mean": None, "ci": [None, None], "n_hdri": 0}
            continue
        _, per_hdri = metrics.per_group_mean(values[m], hdri[m])
        mean, lo, hi = metrics.bootstrap_ci(per_hdri, BOOTSTRAP_SEED)
        out[key] = {"mean": mean, "ci": [lo, hi], "n_hdri": len(per_hdri)}
    return out


def win_rates(scores: dict[str, dict[str, np.ndarray]], hdri: np.ndarray) -> dict[str, Any]:
    """Share of HDRIs on which the CNN's mean error is lower than each baseline's."""
    out: dict[str, Any] = {}
    for base in METHODS[1:]:
        out[base] = {}
        for key in ("rmse", "si_rmse", "srgb_rmse", "colour_deg"):
            _, cnn = metrics.per_group_mean(scores["cnn"][key], hdri)
            _, other = metrics.per_group_mean(scores[base][key], hdri)
            out[base][key] = float(np.mean(cnn < other))
    return out


def evaluate_split(
    split: str, model_path: Path, stats: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rendered = render_split(split)
    preds = predictions(rendered, model_path, stats)
    labels = torch.from_numpy(rendered["labels"]).double()
    scores = {m: metrics.compute(torch.from_numpy(p), labels) for m, p in preds.items()}
    hdri, env = rendered["hdri"], rendered["environment"]
    subsets = {
        "all": np.ones(len(hdri), bool),
        "indoor": env == "indoor",
        "outdoor": env == "outdoor",
    }
    for i, (name, _, _) in enumerate(TEST_SETTINGS):
        subsets[f"setting_{name}"] = rendered["setting"] == i
    result = {
        "n_hdri": len(np.unique(hdri)),
        "n_crops": len(hdri),
        "n_hdri_by_environment": {e: len(np.unique(hdri[env == e])) for e in ("indoor", "outdoor")},
        "methods": {
            m: {s: summarise(scores[m], hdri, mask) for s, mask in subsets.items()} for m in METHODS
        },
        "win_rate_cnn_vs": win_rates(scores, hdri),
    }
    detail = {"rendered": rendered, "preds": preds, "scores": scores}
    return result, detail


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    stats = baselines.train_statistics()
    results: dict[str, Any] = {
        "schema": 1,
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "command": "uv run python -m lumina_train.evaluate",
        "model": {
            "path": "web/public/models/lumina-sh.onnx",
            "sha256": sha256_of(args.model),
            "bytes": args.model.stat().st_size,
        },
        "protocol": report.protocol(stats, BOOTSTRAP_SEED),
        "splits": {},
    }
    details = {}
    for split in SPLITS:
        results["splits"][split], details[split] = evaluate_split(split, args.model, stats)
        print(f"{split}: {results['splits'][split]['n_crops']} crops scored", flush=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results["worst_test_crops"] = report.worst_cases_figure(details["test"])
    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    report.write_eval_md(results)
    print(f"wrote {EVAL_DIR / 'results.json'} and docs/EVAL.md", flush=True)


if __name__ == "__main__":
    main()
