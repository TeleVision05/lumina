"""Export the trained network to the ONNX model contract and write its metadata file.

    uv run python -m lumina_train.export --run main

Loads the EMA weights of data/runs/<run>/best.pt, exports web/public/models/lumina-sh.onnx with
the contract's names, shapes and metadata (normalisation and the output parametrisation are
inside the graph), checks PyTorch against onnxruntime on fixed inputs, writes the parity fixture
training/tests/fixtures/model_parity.npz (so pytest can re-check the committed file) and
web/public/models/lumina-sh.json (version, training date, validation metrics, SHA-256), and copies
the run's log to eval/train_log.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lumina_train.fixedsets import load_or_build_val_set
from lumina_train.model import LuminaNet, count_parameters
from lumina_train.model_contract import CONTRACT_VERSION, INPUT_NAME, MODEL_PATH, OUTPUT_NAME
from lumina_train.paths import EVAL_DIR, MODEL_META_PATH, REPO_ROOT, RUNS_DIR
from lumina_train.placeholder import export_onnx
from lumina_train.train import CROPS_PER_GROUP, GROUPS

MODEL_VERSION = "1.0.0"
FIXTURE_PATH = REPO_ROOT / "training" / "tests" / "fixtures" / "model_parity.npz"
#: Maximum size the brief aims for.
SIZE_BUDGET = 8 * 1024 * 1024
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4


def load_ema(run_dir: Path, name: str = "best.pt") -> tuple[LuminaNet, dict[str, Any]]:
    ckpt = torch.load(run_dir / name, map_location="cpu")
    model = LuminaNet()
    model.load_state_dict(ckpt["ema"])
    return model.eval(), ckpt


def parity_inputs() -> np.ndarray:
    """8 fixed validation frames plus 2 seeded noise images, uint8 [10, 96, 128, 3]."""
    val = load_or_build_val_set()["images"]
    picks = val[np.linspace(0, len(val) - 1, 8).astype(int)]
    noise = np.random.default_rng(5005).integers(0, 256, (2, 96, 128, 3), dtype=np.uint8)
    return np.concatenate([picks, noise])


def check_parity(model: LuminaNet, path: Path, images_u8: np.ndarray) -> dict[str, float]:
    x = images_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = np.stack(
        [session.run([OUTPUT_NAME], {INPUT_NAME: x[i : i + 1]})[0][0] for i in range(len(x))]
    )
    np.testing.assert_allclose(got, expected, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    abs_err = np.abs(got - expected)
    rel = abs_err / np.maximum(np.abs(expected), 1e-6)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_PATH, images=images_u8, expected=expected)
    return {"max_abs": float(abs_err.max()), "max_rel": float(rel.max()), "n_inputs": len(x)}


def add_metadata(path: Path, extra: dict[str, str]) -> None:
    proto = onnx.load(str(path))
    props = {p.key: p.value for p in proto.metadata_props} | extra
    del proto.metadata_props[:]
    onnx.helper.set_model_props(proto, props)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def _log_rows(run_dir: Path, step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(the log row of the exported step, all rows)."""
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    return next(r for r in rows if r["step"] == step), rows


def write_meta(
    run_dir: Path, ckpt: dict[str, Any], parity: dict[str, float], model: LuminaNet
) -> dict[str, Any]:
    best, rows = _log_rows(run_dir, int(ckpt["step"]))
    cfg = ckpt["config"]
    val_keys = (
        "val_loss",
        "val_rmse",
        "val_si_rmse",
        "val_srgb_rmse",
        "val_colour_deg",
        "val_dir_deg",
        "val_dir_deg_directional",
    )
    trained = dt.datetime.fromtimestamp((run_dir / "best.pt").stat().st_mtime, dt.UTC)
    val = load_or_build_val_set()
    meta = {
        "name": "lumina-sh",
        "version": MODEL_VERSION,
        "contract": CONTRACT_VERSION,
        "placeholder": False,
        "file": "lumina-sh.onnx",
        "bytes": MODEL_PATH.stat().st_size,
        "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "opset": max(
            o.version
            for o in onnx.load(str(MODEL_PATH)).opset_import
            if o.domain in ("", "ai.onnx")
        ),
        "input": {
            "name": INPUT_NAME,
            "shape": [1, 3, 96, 128],
            "dtype": "float32",
            "encoding": "sRGB-encoded RGB in 0..1, NCHW, centre-cropped to 4:3",
        },
        "output": {
            "name": OUTPUT_NAME,
            "shape": [1, 9, 3],
            "dtype": "float32",
            "meaning": "linear radiance SH9 (three.js basis), camera frame, exposure-relative",
        },
        "architecture": f"compact residual CNN, {count_parameters(model)} parameters",
        "source": "training/lumina_train/model.py",
        "trained": trained.strftime("%Y-%m-%d"),
        "training": {
            "command": (
                f"uv run python -m lumina_train.train --run {cfg['run']} --steps {cfg['steps']}"
            ),
            "seed": cfg["seed"],
            "steps": rows[-1]["step"],
            "exported_step": int(ckpt["step"]),
            "batch": GROUPS * CROPS_PER_GROUP,
            "samples_seen": rows[-1]["step"] * GROUPS * CROPS_PER_GROUP,
            "device": "Apple M2 Max (MPS)",
            "wall_clock_hours": rows[-1]["elapsed_s"] / 3600.0,
            "val_samples": len(val["images"]),
            "val_hdris": len(np.unique(val["ids"])),
        },
        "validation": {k.removeprefix("val_"): best[k] for k in val_keys},
        "parity_torch_vs_onnxruntime": parity,
        "data": "Poly Haven HDRIs (CC0), see docs/DATA.md",
        "license": "MIT (weights trained only on CC0 data)",
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    shutil.copyfile(run_dir / "log.jsonl", EVAL_DIR / "train_log.jsonl")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="main")
    args = parser.parse_args(argv)
    run_dir = RUNS_DIR / args.run
    model, ckpt = load_ema(run_dir)
    export_onnx(model, MODEL_PATH, placeholder=False)
    add_metadata(
        MODEL_PATH, {"lumina.version": MODEL_VERSION, "lumina.exported_step": str(ckpt["step"])}
    )
    size = MODEL_PATH.stat().st_size
    if size > SIZE_BUDGET:
        raise SystemExit(f"model is {size} bytes, over the {SIZE_BUDGET} byte budget")
    parity = check_parity(model, MODEL_PATH, parity_inputs())
    meta = write_meta(run_dir, ckpt, parity, model)
    print(f"wrote {MODEL_PATH} ({size} bytes, sha256 {meta['sha256']}); parity {parity}")


if __name__ == "__main__":
    main()
