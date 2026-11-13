"""Synthetic camera clips with per-frame ground truth, for the app's no-camera mode and e2e test.

    uv run python -m lumina_train.clips

From held-out test-split HDRIs (downloaded again at 4k for sharp 640 x 480 frames), renders
through the same camera model as training: three slow pans and one clip where the camera is still
and the environment turns a full circle about the vertical axis. Auto-exposure is re-metered every
frame and smoothed like a webcam's; white balance is fixed per clip. Writes
web/public/samples/<name>.mp4 (H.264, BT.709, under 1.5 MB), <name>.json (true SH per frame, plus
what the shipped ONNX model predicts on the decoded video, so an end-to-end test can set its
tolerance), and index.json.

HDRIs are chosen by rule from the test split's world SH: the most directional outdoor scene, the
most directional indoor scene, the least directional outdoor scene, and the second most
directional outdoor scene for the rotating environment.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from lumina_train import metrics, sh_torch
from lumina_train import sh as shm
from lumina_train.camera import pad_wrap, render_crops, rotations
from lumina_train.clipvideo import decode_mp4, encode_mp4, model_inputs
from lumina_train.download import PH_FILES, PH_LICENSE
from lumina_train.evaluate import predict_onnx, sha256_of
from lumina_train.fetch import download_file, get_json
from lumina_train.hdri import load_hdri, load_split_cache, normalise_panorama, resize_equirect
from lumina_train.isp import auto_exposure, run_isp
from lumina_train.model_contract import MODEL_PATH, REPO_ROOT
from lumina_train.params import Params, neutral_pipeline
from lumina_train.paths import POLYHAVEN_DIR, SAMPLES_DIR

WIDTH, HEIGHT, FPS = 640, 480, 24
AE_SMOOTHING = 0.15  # per-frame EMA weight on log exposure: a webcam's AE settles in about 1 s
AE_PERCENTILE = 0.6
SEED = 6006
CHUNK = 16


@dataclass(frozen=True)
class ClipSpec:
    name: str
    hdri: str
    environment: str
    seconds: float
    kind: str  # "pan" or "rotate_env"
    yaw0_deg: float
    sweep_deg: float
    pitch_deg: float
    vfov_deg: float
    kelvin: float
    description: str


def choose_clips() -> list[ClipSpec]:
    cache = load_split_cache("test")
    ratio = np.array([shm.dominant_direction(s)[1] for s in cache["sh"]])
    ids, env = cache["ids"], cache["environment"]
    outdoor = [i for i in np.argsort(-ratio) if env[i] == "outdoor"]
    indoor = [i for i in np.argsort(-ratio) if env[i] == "indoor"]
    sunny, second, flat, room = outdoor[0], outdoor[1], outdoor[-1], indoor[0]
    return [
        ClipSpec("pan-outdoor-sun", str(ids[sunny]), "outdoor", 8.0, "pan", 0.0, 70.0, 4.0, 60.0,
                 5600.0, "slow 70 degree pan, the test split's most directional outdoor scene"),
        ClipSpec("pan-indoor", str(ids[room]), "indoor", 8.0, "pan", 30.0, 60.0, 2.0, 62.0,
                 4200.0, "slow 60 degree pan, the test split's most directional indoor scene"),
        ClipSpec("pan-outdoor-diffuse", str(ids[flat]), "outdoor", 8.0, "pan", -20.0, 60.0, 6.0,
                 58.0, 6500.0, "slow 60 degree pan, the test split's least directional scene"),
        ClipSpec("rotate-environment", str(ids[second]), "outdoor", 10.0, "rotate_env", 0.0,
                 360.0, 3.0, 60.0, 5600.0,
                 "camera still; the environment turns 360 degrees about +Y (up)"),
    ]  # fmt: skip


def poses(spec: ClipSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (yaw, pitch, roll), radians. Pans ease in and out across the sweep."""
    n = round(spec.seconds * FPS)
    t = np.arange(n) / max(1, n - 1)
    if spec.kind == "pan":
        offset = spec.sweep_deg * (0.5 - 0.5 * np.cos(math.pi * t)) - 0.5 * spec.sweep_deg
        yaw = np.radians(spec.yaw0_deg - offset)  # yaw decreases: the camera turns right
    else:
        # Turning the environment by +a about +Y is the same picture as yawing the camera by -a.
        yaw = np.radians(spec.yaw0_deg - spec.sweep_deg * np.arange(n) / n)
    pitch = np.radians(spec.pitch_deg + 0.6 * np.sin(2 * math.pi * t))
    roll = np.radians(0.4 * np.sin(3 * math.pi * t))
    return yaw, pitch, roll


def load_4k(asset_id: str) -> tuple[torch.Tensor, np.ndarray]:
    """(WRAP-padded 4k panorama [1, 3, H, W + 2] float32, world SH [9, 3]) normalised as in hdri."""
    info = get_json(PH_FILES.format(id=asset_id))["hdri"]["4k"]["hdr"]
    path = POLYHAVEN_DIR / "4k" / f"{asset_id}_4k.hdr"
    download_file(info["url"], path, md5=info["md5"], size=info["size"])
    img, _ = normalise_panorama(load_hdri(path))
    sh_world = shm.project_equirect(resize_equirect(img, 512, 1024))
    pano = torch.from_numpy(img).permute(2, 0, 1)[None].contiguous()
    return pad_wrap(pano), sh_world


def clip_params(spec: ClipSpec, n: int) -> Params:
    p = neutral_pipeline(n)
    p["kelvin"] = np.full(n, spec.kelvin)
    p["vignette"] = np.full(n, 0.2)
    p["blur"] = np.full(n, 0.7)
    p["noise_a"] = np.full(n, 6e-4)
    p["noise_b"] = np.full(n, 2e-5)
    p["saturation"] = np.full(n, 1.05)
    p["seed"] = SEED + np.arange(n)
    p["flip"] = np.zeros(n, bool)
    return p


def render_clip(spec: ClipSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (frames uint8 [N, H, W, 3], true SH [N, 9, 3], smoothed exposure [N])."""
    pano, sh_world = load_4k(spec.hdri)
    yaw, pitch, roll = poses(spec)
    n = len(yaw)
    rot = rotations(*(torch.from_numpy(a).float() for a in (yaw, pitch, roll)))
    vfov = torch.full((n,), math.radians(spec.vfov_deg))
    params = clip_params(spec, n)
    linear = torch.cat(
        [render_crops(pano, rot[i : i + CHUNK], vfov[i : i + CHUNK], HEIGHT, WIDTH, ss=1)
         for i in range(0, n, CHUNK)]
    )  # fmt: skip
    raw = auto_exposure(linear, torch.full((n,), AE_PERCENTILE), torch.zeros(n)).double().numpy()
    smooth = np.empty(n)
    log_k = math.log(raw[0])
    for i in range(n):
        log_k += AE_SMOOTHING * (math.log(raw[i]) - log_k)
        smooth[i] = math.exp(log_k)
    frames, scales = [], []
    for i in range(0, n, CHUNK):
        sl = slice(i, i + CHUNK)
        sub = {k: v[sl] for k, v in params.items()}
        img, scale = run_isp(linear[sl], sub, exposure=torch.from_numpy(smooth[sl]).float())
        frames.append((img.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8))
        scales.append(scale)
    labels = sh_torch.rotate_sh(torch.from_numpy(sh_world).float()[None].expand(n, 9, 3), rot)
    labels = labels * torch.cat(scales)[:, None, :]
    return np.concatenate(frames), labels.double().numpy(), smooth


def _round(a: np.ndarray, digits: int = 6) -> list[Any]:
    return np.vectorize(lambda v: float(f"{v:.{digits}g}"))(a).tolist()


def frame_records(
    truth: np.ndarray, pred: np.ndarray, exposure: np.ndarray
) -> list[dict[str, Any]]:
    labels = torch.from_numpy(truth)
    scores = metrics.compute(torch.from_numpy(pred), labels)
    out = []
    for i in range(len(truth)):
        direction, ratio = shm.dominant_direction(truth[i])
        out.append(
            {
                "t": round(i / FPS, 4),
                "sh": _round(truth[i]),
                "dominant_direction": None if direction is None else _round(direction, 5),
                "directionality": round(ratio, 4),
                "exposure_scale": float(f"{exposure[i]:.6g}"),
                "model_sh": _round(pred[i]),
                "model_srgb_rmse": round(float(scores["srgb_rmse"][i]), 2),
                "model_dir_deg": round(float(scores["dir_deg"][i]), 2),
            }
        )
    return out


def make_clip(spec: ClipSpec) -> dict[str, Any]:
    frames, truth, exposure = render_clip(spec)
    video = SAMPLES_DIR / f"{spec.name}.mp4"
    crf = encode_mp4(frames, FPS, video)
    decoded = decode_mp4(video, WIDTH, HEIGHT)
    pred = predict_onnx(model_inputs(decoded), MODEL_PATH)
    records = frame_records(truth, pred, exposure)
    hdri = {
        "id": spec.hdri,
        "source": "Poly Haven",
        "split": "test",
        "page": f"https://polyhaven.com/a/{spec.hdri}",
        "license": PH_LICENSE,
    }
    model = {
        "file": "models/lumina-sh.onnx",
        "sha256": sha256_of(MODEL_PATH),
        "input": "decoded frame, centre-cropped to 4:3, area-resized to 128 x 96",
    }
    errors = np.array([[r["model_srgb_rmse"], r["model_dir_deg"]] for r in records])
    doc = {
        "schema": 1,
        "video": f"{spec.name}.mp4",
        "bytes": video.stat().st_size,
        "h264_crf": crf,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frames": len(records),
        "vfov_deg": spec.vfov_deg,
        "mirrored": False,
        "frame_convention": "frame i is displayed at t = i / fps; SH in docs/CONVENTIONS.md",
        "spec": asdict(spec),
        "hdri": hdri,
        "model": model,
        "summary": {
            "model_srgb_rmse_mean": round(float(errors[:, 0].mean()), 2),
            "model_dir_deg_median": round(float(np.median(errors[:, 1])), 2),
        },
        "per_frame": records,
    }
    out = SAMPLES_DIR / f"{spec.name}.json"
    out.write_text(json.dumps(doc) + "\n")
    print(f"{video.name}: {doc['bytes']} bytes, crf {crf}, {doc['summary']}", flush=True)
    entry = {k: doc[k] for k in ("video", "bytes", "fps", "frames", "vfov_deg")}
    entry |= {"truth": out.name, "description": spec.description, "hdri": spec.hdri}
    entry["environment"] = spec.environment
    return entry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    torch.set_num_threads(4)
    entries = [make_clip(spec) for spec in choose_clips()]
    index = {
        "schema": 1,
        "command": "uv run python -m lumina_train.clips",
        "license": "rendered from CC0 Poly Haven HDRIs; see docs/DATA.md",
        "clips": entries,
    }
    (SAMPLES_DIR / "index.json").write_text(json.dumps(index) + "\n")
    prettier = REPO_ROOT / "web" / "node_modules" / ".bin" / "prettier"
    files = [str(p) for p in sorted(SAMPLES_DIR.glob("*.json"))]
    subprocess.run([str(prettier), "--write", "--log-level", "warn", *files], check=True)


if __name__ == "__main__":
    main()
