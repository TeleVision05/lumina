"""docs/EVAL.md and the evaluation figures, generated from eval/results.json data.

Nothing in EVAL.md is typed by hand except the fixed explanatory text below: every number comes
from the results dict, and every better/worse statement is derived from the confidence intervals.
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from lumina_train import metrics, plots
from lumina_train import sh as shm
from lumina_train.fixedsets import TEST_POSES, TEST_SETTINGS
from lumina_train.paths import DOCS_DIR, EVAL_DIR, MEDIA_DIR

METHOD_NAMES = {
    "cnn": "LuminaNet (CNN)",
    "mean": "Training-set mean lighting",
    "image_mean": "Image-mean colour, ambient only",
    "image_dir": "Image mean + brightest-region direction",
}
COLUMNS = {
    "rmse": "RMSE (linear)",
    "si_rmse": "si-RMSE (linear)",
    "srgb_rmse": "sRGB RMSE (8-bit levels)",
    "colour_deg": "Colour error (deg)",
    "dir_deg_directional": "Direction error, directional scenes (deg)",
    "dir_deg_diffuse": "Direction error, diffuse scenes (deg)",
}
FIGURE_WORST = "eval-worst-cases.png"
FIGURE_TYPICAL = "eval-typical-cases.png"
FIGURE_CURVES = "train-curves.png"


def protocol(stats: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "crops_per_hdri": f"{TEST_POSES} poses x {len(TEST_SETTINGS)} settings",
        "settings": [{"name": n, "ev": e, "kelvin": k} for n, e, k in TEST_SETTINGS],
        "sphere": (
            f"Lambertian, albedo {metrics.ALBEDO}, {metrics.SPHERE_SIZE} px square,"
            " orthographic view down -Z, visible hemisphere"
        ),
        "directional_threshold": metrics.DIRECTIONAL_THRESHOLD,
        "bootstrap": {"resamples": 2000, "level": 0.95, "unit": "HDRI", "seed": seed},
        "train_mean_directionality": stats["directionality"],
    }


def _cell(entry: dict[str, Any]) -> str:
    if entry["mean"] is None:
        return "n/a"
    lo, hi = entry["ci"]
    digits = 3 if entry["mean"] < 10 else 1
    return f"{entry['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(split: dict[str, Any], subset: str, columns: dict[str, str] = COLUMNS) -> str:
    head = "| Method | " + " | ".join(columns.values()) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    rows = [
        f"| {METHOD_NAMES[m]} | "
        + " | ".join(_cell(split["methods"][m][subset][c]) for c in columns)
        + " |"
        for m in split["methods"]
    ]
    return "\n".join([head, sep, *rows])


BASELINES = ("mean", "image_mean", "image_dir")


def verdict(diff: dict[str, Any]) -> str:
    """From the paired difference CNN - baseline (lower is better)."""
    if diff["mean"] is None:
        return "n/a"
    if diff["ci"][1] < 0:
        return "better"
    if diff["ci"][0] > 0:
        return "**WORSE**"
    return "no clear difference"


def paired_table(split: dict[str, Any], subset: str) -> str:
    """CNN minus each baseline, paired per HDRI, with the 95 % CI and a verdict per cell."""
    cols = {k: v for k, v in COLUMNS.items() if k != "dir_deg_diffuse"}
    head = "| CNN minus | " + " | ".join(cols.values()) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    rows = []
    for base in BASELINES:
        diffs = split["paired_cnn_minus"][base][subset]
        cells = []
        for c in cols:
            d = diffs[c]
            if d["mean"] is None:
                cells.append("n/a")
                continue
            digits = 3 if abs(d["mean"]) < 10 else 1
            lo, hi = d["ci"]
            cells.append(
                f"{d['mean']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] {verdict(d)}"
            )
        rows.append(f"| {METHOD_NAMES[base]} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def _sphere_srgb(sh: np.ndarray, size: int) -> np.ndarray:
    img = shm.to_srgb8(shm.render_diffuse_sphere(sh, size, albedo=metrics.ALBEDO))
    _, mask = shm.sphere_normals(size)
    img[~mask] = 32
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _row(detail: dict[str, Any], i: int) -> np.ndarray:
    r, p, s = detail["rendered"], detail["preds"], detail["scores"]
    crop = cv2.resize(r["images"][i], (256, 192), interpolation=cv2.INTER_NEAREST)
    cells = [_label(crop, f"{r['hdri'][i]} ({r['environment'][i]})"[:40])]
    cells.append(_label(_sphere_srgb(r["labels"][i].astype(np.float64), 192), "true"))
    err = s["cnn"]["srgb_rmse"][i]
    cells.append(_label(_sphere_srgb(p["cnn"][i], 192), f"CNN  sRGB RMSE {err:.0f}"))
    err_b = s["image_dir"]["srgb_rmse"][i]
    cells.append(_label(_sphere_srgb(p["image_dir"][i], 192), f"img+dir  {err_b:.0f}"))
    gap = np.full((214, 6, 3), 255, np.uint8)
    parts = []
    for c in cells:
        parts += [c, gap]
    return np.concatenate(parts[:-1], axis=1)


def _figure(detail: dict[str, Any], indices: list[int], name: str) -> None:
    rows = [_row(detail, i) for i in indices]
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    parts = []
    for r in rows:
        parts += [r, gap]
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MEDIA_DIR / name), np.concatenate(parts[:-1])[:, :, ::-1])


def worst_cases_figure(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Five worst test crops by CNN sRGB RMSE, and five typical ones (evenly spaced ranks)."""
    err = detail["scores"]["cnn"]["srgb_rmse"]
    order = np.argsort(err)
    worst = [int(i) for i in order[::-1][:5]]
    typical = [int(order[int(q * (len(order) - 1))]) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    _figure(detail, worst, FIGURE_WORST)
    _figure(detail, typical, FIGURE_TYPICAL)
    r = detail["rendered"]
    return [
        {
            "crop": i,
            "hdri": str(r["hdri"][i]),
            "environment": str(r["environment"][i]),
            "setting": TEST_SETTINGS[int(r["setting"][i])][0],
            "cnn_srgb_rmse": float(err[i]),
            "directionality": float(detail["scores"]["cnn"]["directionality"][i]),
        }
        for i in worst
    ]


def _training_section() -> str:
    log_path = EVAL_DIR / "train_log.jsonl"
    meta_path = DOCS_DIR.parent / "web" / "public" / "models" / "lumina-sh.json"
    if not log_path.exists() or not meta_path.exists():
        return "Training log not committed yet.\n"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    meta = json.loads(meta_path.read_text())
    train = meta["training"]
    _curve_figure(rows)
    keep = [r for r in rows if r["step"] % 5000 == 0 or r is rows[-1] or r["step"] <= 2000]
    lines = [
        "| Step | Minutes | Val loss | Val si-RMSE | Val sRGB RMSE | Val colour (deg)"
        " | Val direction, directional (deg) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in keep:
        lines.append(
            f"| {r['step']} | {r['elapsed_s'] / 60:.0f} | {r['val_loss']:.3f} | "
            f"{r['val_si_rmse']:.3f} | {r['val_srgb_rmse']:.1f} | {r['val_colour_deg']:.2f} | "
            f"{r['val_dir_deg_directional']:.1f} |"
        )
    return (
        f"Command: `{train['command']}` (seed {train['seed']}), {train['steps']} steps of "
        f"{train['batch']} samples on {train['device']}, "
        f"{train['wall_clock_hours']:.2f} h wall clock "
        f"({train['samples_seen'] / 1e6:.1f} M synthetic crops). Exported weights: the EMA "
        f"weights at step {train['exported_step']} (lowest validation loss). Validation: "
        f"{train['val_samples']} fixed crops of the {train['val_hdris']} validation HDRIs, scored "
        "every 1000 steps; the full log is `eval/train_log.jsonl`.\n\n"
        f"![Training curves](media/{FIGURE_CURVES})\n\n" + "\n".join(lines) + "\n"
    )


def _curve_figure(rows: list[dict[str, Any]]) -> None:
    """Validation curves (EMA weights) and the running training loss terms."""
    rows = [r for r in rows if r["step"] >= 3000]  # the first EMA evaluations are off-scale
    if len(rows) < 2:
        return
    steps = np.array([r["step"] for r in rows], float)

    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], float)

    panels = [
        plots.line_panel(steps, {"val si-RMSE": col("val_si_rmse"), "val RMSE": col("val_rmse")},
                         "Validation sphere error (linear)"),
        plots.line_panel(steps, {"val sRGB RMSE": col("val_srgb_rmse")},
                         "Validation sphere error (8-bit sRGB levels)"),
        plots.line_panel(steps, {"train irr": col("train_irr"), "val irr": col("val_irr")},
                         "Irradiance loss term"),
    ]  # fmt: skip
    plots.save_panels(panels, MEDIA_DIR / FIGURE_CURVES)


DEFINITIONS = """\
## What is measured

Each test crop is a synthetic camera frame (same camera model as training, docs/DATA.md) of a
held-out HDRI. Its true label is the HDRI's SH9 in that camera's frame at that frame's exposure
and white balance. Every method predicts SH9 from the 128 x 96 frame alone, and both SH are
compared through a rendered sphere: Lambertian, albedo 0.5, 64 x 64, orthographic view down -Z
(the side the camera sees), linear radiance albedo / pi * E(n) (docs/CONVENTIONS.md section 3).

| Metric | Definition |
|---|---|
| RMSE | root mean square difference over sphere pixels and RGB, linear, frame exposure |
| si-RMSE | the same after scaling the prediction by the best scalar (shape and colour only) |
| sRGB RMSE | after clipping to [0, 1] and the sRGB curve, in 8-bit levels: what a screen shows |
| Colour error | angle between the mean RGB of the predicted and true spheres (colour cast) |
| Direction error | angle between dominant-light directions (luminance band 1, CONVENTIONS s. 6) |

Direction error is reported separately for **directional** crops (true directionality at least
{threshold}) and **diffuse** crops (below it), where a dominant direction hardly exists and the
number is close to meaningless; it is listed only to be complete.

Baselines, scored by the same code:

- **Training-set mean lighting**: the mean label over 8 seeded crops of every training HDRI.
- **Image-mean colour, ambient only**: the frame's mean colour (sRGB decoded) as a uniform
  environment (grey-world style); it has no direction.
- **Image mean + brightest-region direction**: the same plus a band-1 term toward the
  luminance-weighted centroid of the brightest 2 % of pixels (pinhole with an assumed 60 degree
  vertical FOV), with the training-set mean directionality ({directionality:.3f}).

Each cell is the mean over HDRIs (each HDRI's crops averaged first) with a 95 % percentile
bootstrap CI over HDRIs (2000 resamples, seed {seed}). Lower is better everywhere.
"""

LIMITS = """\
## Limits of this evaluation

- The test crops come from the same synthetic camera model as training. They measure
  generalisation to **unseen scenes**, not to real webcams. Real-camera accuracy is UNVERIFIED:
  there is no real-photo test set with ground-truth lighting here.
- Poly Haven skews outdoor and professional; the ambientCG set is a small second source (48
  HDRIs), not a different camera.
- Labels are relative to the frame's exposure: absolute scene brightness is not recoverable and is
  never claimed.
- SH9 is low frequency: hard shadows and sharp highlights are outside what any method here can
  represent, so the true sphere is itself a soft approximation of the scene.
- Numbers from papers (DeepLight, Garon et al.) use other test sets and definitions; they are not
  comparable with these.
- Browser speed is not measured here (the web app's job).
"""


def _split_section(title: str, split: dict[str, Any]) -> str:
    env = split["n_hdri_by_environment"]
    parts = [
        f"## {title}\n",
        f"{split['n_hdri']} HDRIs ({env['indoor']} indoor, {env['outdoor']} outdoor), "
        f"{split['n_crops']} crops.\n",
        "### All scenes\n",
        table(split, "all"),
        "\nPaired comparison, CNN minus baseline (negative = the CNN is better), per-HDRI "
        "difference with its 95 % bootstrap CI; the verdict is generated from the CI:\n",
        paired_table(split, "all"),
        "\n### Indoor\n",
        table(split, "indoor"),
        "",
        paired_table(split, "indoor"),
        "\n### Outdoor\n",
        table(split, "outdoor"),
        "",
        paired_table(split, "outdoor"),
        "\n### Win rate: share of HDRIs where the CNN's error is lower\n",
        _win_table(split["win_rate_cnn_vs"]),
        "\n### CNN by exposure / white-balance setting\n",
        _setting_table(split),
    ]
    return "\n".join(parts) + "\n"


def _win_table(win: dict[str, dict[str, float]]) -> str:
    cols = ("rmse", "si_rmse", "srgb_rmse", "colour_deg")
    lines = ["| CNN vs | " + " | ".join(COLUMNS[c] for c in cols) + " |", "|---" * 5 + "|"]
    for base, row in win.items():
        lines.append(
            f"| {METHOD_NAMES[base]} | " + " | ".join(f"{row[c]:.0%}" for c in cols) + " |"
        )
    return "\n".join(lines)


def _setting_table(split: dict[str, Any]) -> str:
    cols = ("rmse", "si_rmse", "srgb_rmse", "colour_deg", "dir_deg_directional")
    lines = ["| Setting | " + " | ".join(COLUMNS[c] for c in cols) + " |", "|---" * 6 + "|"]
    for name, ev, kelvin in TEST_SETTINGS:
        cells = split["methods"]["cnn"][f"setting_{name}"]
        lines.append(
            f"| {name} ({ev:+.0f} EV, {kelvin:.0f} K) | "
            + " | ".join(_cell(cells[c]) for c in cols)
            + " |"
        )
    return "\n".join(lines)


def _worst_list(worst: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{i + 1}. `{w['hdri']}` ({w['environment']}, {w['setting']}): sRGB RMSE "
        f"{w['cnn_srgb_rmse']:.1f}, true directionality {w['directionality']:.2f}"
        for i, w in enumerate(worst)
    )


def summary(results: dict[str, Any]) -> str:
    """Where the CNN beats every baseline (paired CI below zero), and where it does not."""
    lines = [
        "## Summary\n",
        "Generated from the paired per-HDRI differences (CNN minus baseline, 95 % bootstrap CI), "
        "all scenes. 'Better' means the whole CI is below zero.\n",
    ]
    for key, title in (("test", "Held-out Poly Haven test"), ("ood", "ambientCG (OOD)")):
        split = results["splits"][key]
        wins, losses = [], []
        for col, label in COLUMNS.items():
            if col == "dir_deg_diffuse":
                continue
            words = {b: verdict(split["paired_cnn_minus"][b]["all"][col]) for b in BASELINES}
            if all(w == "better" for w in words.values()):
                wins.append(label)
                continue
            detail = "; ".join(
                f"{w.strip('*').lower()} vs {METHOD_NAMES[b].lower()}"
                for b, w in words.items()
                if w != "better"
            )
            losses.append(f"{label} ({detail})")
        lines.append(
            f"- **{title}**: better than every baseline on {', '.join(wins) or 'no metric'}. "
            f"Not better than every baseline on: {', '.join(losses) or 'none'}."
        )
    return "\n".join(lines) + "\n"


def write_eval_md(results: dict[str, Any]) -> None:
    proto = results["protocol"]
    model = results["model"]
    head = (
        "# Evaluation\n\n"
        f"Generated by `{results['command']}` on {results['generated']} for "
        f"`{model['path']}` ({model['bytes']:,} bytes, sha256 `{model['sha256']}`). Every number "
        "below is in `eval/results.json`; re-running the command reproduces them from the fixed "
        "crop list `eval/test_crops.json` and the fixed bootstrap seed. Setup: docs/DATA.md.\n\n"
    )
    body = [
        head,
        summary(results),
        DEFINITIONS.format(
            threshold=proto["directional_threshold"],
            directionality=proto["train_mean_directionality"],
            seed=proto["bootstrap"]["seed"],
        ),
        _split_section("Held-out Poly Haven test split", results["splits"]["test"]),
        _split_section("Out-of-distribution: ambientCG", results["splits"]["ood"]),
        "## Failure cases\n",
        "The five test crops with the largest CNN sRGB RMSE. Columns: the input frame, the true "
        "sphere, the CNN's sphere, and the image-mean + direction baseline's sphere.\n",
        f"![Five worst test crops](media/{FIGURE_WORST})\n",
        _worst_list(results["worst_test_crops"]) + "\n",
        "For contrast, five crops at the 10th, 30th, 50th, 70th and 90th percentile of CNN "
        "error:\n",
        f"![Typical test crops](media/{FIGURE_TYPICAL})\n",
        "## Training\n",
        _training_section(),
        LIMITS,
    ]
    (DOCS_DIR / "EVAL.md").write_text("\n".join(body))
