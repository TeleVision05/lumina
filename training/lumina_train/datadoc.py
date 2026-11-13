"""Write docs/DATA.md from the download manifests and the committed split.

    uv run python -m lumina_train.datadoc

Needs data/hdri/*/manifest.json (written by lumina_train.download) and training/splits/split.json.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from typing import Any

from lumina_train.paths import AMBIENTCG_DIR, DOCS_DIR, POLYHAVEN_DIR, SPLIT_PATH

CAMERA_MODEL = """\
## Synthetic camera model

Each training sample is made on the fly (`lumina_train.params`, `camera`, `isp`, `synth`):

| Step | Distribution |
|---|---|
| Yaw | uniform over 360 degrees |
| Pitch | normal, mean 0, sd 15 degrees, clipped to +/-40 |
| Roll | normal, mean 0, sd 3 degrees, clipped to +/-10 |
| Vertical FOV | uniform 40 to 75 degrees; 128 x 96 (4:3), rendered at 2x, box-filtered |
| Mirror | 50 %: image flipped left-right and label `mirror_x`'d (a mirror-image world) |
| Lens blur | Gaussian, sigma uniform 0 to 1.2 output pixels |
| Vignetting | 1 - v r^2, v uniform 0 to 0.35, r = 1 at the corners |
| Auto-exposure | p-th luminance percentile to 0.25 (p in 0.5..0.85), then 2^ev, ev in -1..1 |
| White balance | blackbody T seen by a 6500 K-balanced camera; T 3000-9000 K (uniform mired) |
| Tint | green gain 2^t, t uniform +/-0.08; all gains keep luminance |
| Sensor noise | x + sqrt(a x + b) n, a log-uniform 1e-4 to 5e-3, b log-uniform 1e-6 to 2e-4 |
| Clipping | at 1.0 |
| Tone curve | sRGB, or gamma 1.8 to 2.6, or sRGB plus an S-curve of contrast 0.8 to 1.5 |
| Saturation | 0.85 to 1.3 around Rec. 709 luminance |
| Quantisation | 8 bit |
| JPEG | 75 %: OpenCV quality 50 to 95; 25 %: none |

The label is the SH9 of the **whole** panorama rotated into the camera frame, times the same
exposure scale and white-balance gains as the image (docs/CONVENTIONS.md section 5). Blur,
vignetting, noise, clipping, the tone curve, saturation and JPEG change the picture, not the light,
so they are not in the label. `training/tests/test_synth.py` checks, on analytic environments,
that a bright lobe lands on the expected pixel with the expected label direction, that rotating the
camera rotates the label consistently, that exposure and white balance scale image and label
together, and that a white sphere lit by the label matches the photo of a uniform environment.

Panoramas are cached at 1024 x 512, divided by their median luminance and stored as float16
(clamped at 60000); the world SH is projected from the float32 image before the clamp, so a sun too
bright for float16 still contributes its full energy to the label.
"""


def _wrap_ids(ids: list[str]) -> str:
    return "\n".join(textwrap.wrap(", ".join(ids), width=100, break_on_hyphens=False))


def _source_section(manifest: dict[str, Any], title: str) -> str:
    assets = manifest["assets"]
    env = Counter(a["environment"] for a in assets)
    return (
        f"### {title}\n\n"
        f"- API: `{manifest['api']}`\n"
        f"- Licence: {manifest['license']}\n"
        f"- Downloaded: {manifest['downloaded']} (UTC), resolution {manifest['resolution']}, "
        f"format .{manifest['format']}, User-Agent `{manifest['user_agent']}`\n"
        f"- The API listed {manifest['listed']} HDRIs; {len(assets)} were downloaded "
        f"({env.get('indoor', 0)} indoor, {env.get('outdoor', 0)} outdoor).\n"
    )


def build_markdown(ph: dict[str, Any], acg: dict[str, Any], split: dict[str, Any]) -> str:
    counts = split["counts"]
    rows = "\n".join(
        f"| {name} | {c['total']} | {c['indoor']} | {c['outdoor']} |" for name, c in counts.items()
    )
    lists = "\n\n".join(
        f"### {name} ({len(entries)})\n\n```\n{_wrap_ids([e['id'] for e in entries])}\n```"
        for name, entries in split["splits"].items()
    )
    acg_note = acg.get("selection", "")
    return f"""# Data

Lumina is trained only on CC0 panoramas. Nothing research-only (Laval, SUN360) and no ImageNet
weights are used. Every HDRI below is listed by id with its source; the per-asset URL, page,
authors and md5 are in `data/hdri/<source>/manifest.json` after a download.

Regenerate this file: `uv run python -m lumina_train.datadoc` (after
`uv run python -m lumina_train.download polyhaven` and `... download ambientcg --count 48`, then
`uv run python -m lumina_train.split`).

## Sources

{_source_section(ph, "Poly Haven (training, validation, test)")}
- Licence page: https://polyhaven.com/license (CC0: "You can use our assets for any purpose").
  The API terms (https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) ask for a unique
  User-Agent naming the software; the downloader sends `{ph["user_agent"]}` and nothing personal,
  runs at most 4 downloads at once, retries with backoff and skips files already present.
- Asset pages: `https://polyhaven.com/a/<id>`.

{_source_section(acg, "ambientCG (out-of-distribution test only)")}
- Licence page: https://docs.ambientcg.com/license/ ("All ambientCG assets are provided under the
  Creative Commons CC0 1.0 Universal License").
- Selection: {acg_note}. The 1K zip is downloaded and only its `_HDR.exr` is kept.
- Asset pages: `https://ambientcg.com/a/<id>`.

## Split

`training/splits/split.json` (seed {split["seed"]}). Split by **scene**, never by crop: HDRIs
sharing GPS coordinates to 0.01 degrees or an id stem (time-of-day words and trailing numbers
removed) form one group ({split["groups"]} groups), and whole groups are assigned so indoor and
outdoor are each split about 80/10/10. "indoor"/"outdoor" is Poly Haven's `environment` attribute
(ambientCG: the "Indoor Environment HDRI" category).

| Split | Total | Indoor | Outdoor |
|---|---|---|---|
{rows}

{CAMERA_MODEL}
## HDRI ids by split

{lists}
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    ph = json.loads((POLYHAVEN_DIR / "manifest.json").read_text())
    acg = json.loads((AMBIENTCG_DIR / "manifest.json").read_text())
    split = json.loads(SPLIT_PATH.read_text())
    out = DOCS_DIR / "DATA.md"
    out.write_text(build_markdown(ph, acg, split))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
