"""Download CC0 HDRI panoramas through the sources' public APIs, politely and resumably.

    uv run python -m lumina_train.download polyhaven            # every Poly Haven HDRI at 1k
    uv run python -m lumina_train.download ambientcg --count 48 # out-of-distribution test set

Poly Haven: GET api.polyhaven.com/assets?t=hdris, then /files/<id> for the 1k .hdr URL and md5.
ambientCG: GET ambientcg.com/api/v2/full_json?type=HDRI, then the 1K zip; the _HDR.exr inside it
is kept and the zip deleted. At most --workers (default 4) downloads run at once. Re-running skips
every file already present with the right checksum, so an interrupted run resumes.
Each source gets data/hdri/<source>/manifest.json with the metadata docs/DATA.md is built from.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lumina_train.fetch import USER_AGENT, download_file, get_json
from lumina_train.paths import AMBIENTCG_DIR, POLYHAVEN_DIR

PH_ASSETS = "https://api.polyhaven.com/assets?t=hdris"
PH_FILES = "https://api.polyhaven.com/files/{id}"
PH_LICENSE = "CC0 1.0 (https://polyhaven.com/license)"
ACG_LIST = "https://ambientcg.com/api/v2/full_json?type=HDRI&limit=1000&include=downloadData"
ACG_LICENSE = "CC0 1.0 (https://docs.ambientcg.com/license/)"
#: ambientCG categories with a photographed ground (sky-only sets are skipped).
ACG_CATEGORIES = (
    "Indoor Environment HDRI",
    "Day Environment HDRI",
    "Evening Environment HDRI",
    "Night Environment HDRI",
)
ACG_SEED = 20261113


def _today() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")


def _run_pool(jobs: list[Callable[[], dict[str, Any]]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        futures = [pool.submit(job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                results.append(fut.result())
            except Exception as err:
                failures.append(str(err))
            if i % 50 == 0 or i == len(futures):
                print(f"[{i}/{len(futures)}] done, {len(failures)} failed", flush=True)
    for msg in failures:
        print(f"FAILED: {msg}", flush=True)
    return results


def _polyhaven_job(asset_id: str, meta: dict[str, Any], res: str) -> dict[str, Any]:
    info = get_json(PH_FILES.format(id=asset_id))["hdri"][res]["hdr"]
    dest = POLYHAVEN_DIR / res / f"{asset_id}_{res}.hdr"
    fetched = download_file(info["url"], dest, md5=info["md5"], size=info["size"])
    attrs = meta.get("attributes", {})
    return {
        "id": asset_id,
        "name": meta.get("name", asset_id),
        "environment": attrs.get("environment"),
        "contrast": attrs.get("contrast"),
        "light_type": attrs.get("light_type"),
        "categories": meta.get("categories", []),
        "authors": sorted(meta.get("authors", {}).keys()),
        "whitebalance": meta.get("whitebalance"),
        "evs_cap": meta.get("evs_cap"),
        "coords": meta.get("coords"),
        "url": info["url"],
        "page": f"https://polyhaven.com/a/{asset_id}",
        "md5": info["md5"],
        "file": str(dest.relative_to(POLYHAVEN_DIR)),
        "fetched_now": fetched,
    }


def download_polyhaven(res: str, limit: int | None, workers: int) -> Path:
    assets: dict[str, Any] = get_json(PH_ASSETS)
    ids = sorted(assets)[:limit] if limit else sorted(assets)
    print(f"Poly Haven lists {len(assets)} HDRIs; fetching {len(ids)} at {res}", flush=True)
    jobs = [(lambda a=a: _polyhaven_job(a, assets[a], res)) for a in ids]
    rows = sorted(_run_pool(jobs, workers), key=lambda r: r["id"])
    manifest = {
        "source": "Poly Haven",
        "api": PH_ASSETS,
        "license": PH_LICENSE,
        "resolution": res,
        "format": "hdr",
        "downloaded": _today(),
        "user_agent": USER_AGENT,
        "listed": len(assets),
        "assets": rows,
    }
    out = POLYHAVEN_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out}: {len(rows)} of {len(ids)} HDRIs present", flush=True)
    return out


def _acg_zip(asset: dict[str, Any]) -> dict[str, Any]:
    for folder in asset["downloadFolders"].values():
        for cat in folder["downloadFiletypeCategories"].values():
            for item in cat["downloads"]:
                if item["attribute"] == "1K" and item["filetype"] == "zip":
                    return item
    raise KeyError(f"no 1K zip for {asset['assetId']}")


def _ambientcg_job(asset: dict[str, Any]) -> dict[str, Any]:
    asset_id = asset["assetId"]
    exr = AMBIENTCG_DIR / "1k" / f"{asset_id}_1k.exr"
    item = _acg_zip(asset)
    fetched = False
    if not exr.exists():
        archive = AMBIENTCG_DIR / "zips" / item["fileName"]
        download_file(item["downloadLink"], archive, size=item["size"])
        with zipfile.ZipFile(archive) as zf:
            member = next(n for n in zf.namelist() if n.endswith("_HDR.exr"))
            exr.parent.mkdir(parents=True, exist_ok=True)
            part = exr.with_name(exr.name + ".part")
            part.write_bytes(zf.read(member))
            part.replace(exr)
        archive.unlink()
        fetched = True
    indoor = asset["displayCategory"].startswith("Indoor")
    return {
        "id": asset_id,
        "name": asset["displayName"],
        "environment": "indoor" if indoor else "outdoor",
        "category": asset["displayCategory"],
        "url": item["downloadLink"],
        "page": asset["shortLink"],
        "file": str(exr.relative_to(AMBIENTCG_DIR)),
        "fetched_now": fetched,
    }


def select_ambientcg(assets: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """All indoor environments, then a seeded sample of the outdoor ones, up to count."""
    usable = [a for a in assets if a["displayCategory"] in ACG_CATEGORIES]
    usable.sort(key=lambda a: a["assetId"])
    indoor = [a for a in usable if a["displayCategory"].startswith("Indoor")]
    outdoor = [a for a in usable if not a["displayCategory"].startswith("Indoor")]
    chosen = indoor[:count]
    rng = random.Random(ACG_SEED)
    chosen += rng.sample(outdoor, max(0, min(len(outdoor), count - len(chosen))))
    return sorted(chosen, key=lambda a: a["assetId"])


def download_ambientcg(count: int, workers: int) -> Path:
    listing = get_json(ACG_LIST)
    chosen = select_ambientcg(listing["foundAssets"], count)
    print(
        f"ambientCG lists {len(listing['foundAssets'])} HDRIs; fetching {len(chosen)}", flush=True
    )
    jobs = [(lambda a=a: _ambientcg_job(a)) for a in chosen]
    rows = sorted(_run_pool(jobs, workers), key=lambda r: r["id"])
    manifest = {
        "source": "ambientCG",
        "api": ACG_LIST,
        "license": ACG_LICENSE,
        "resolution": "1k",
        "format": "exr",
        "downloaded": _today(),
        "user_agent": USER_AGENT,
        "listed": len(listing["foundAssets"]),
        "selection": f"all of {ACG_CATEGORIES[0]}, then seed {ACG_SEED} sample of the others",
        "assets": rows,
    }
    out = AMBIENTCG_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out}: {len(rows)} of {len(chosen)} HDRIs present", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    ph = sub.add_parser("polyhaven")
    ph.add_argument("--res", default="1k")
    ph.add_argument("--limit", type=int, default=None)
    ph.add_argument("--workers", type=int, default=4)
    acg = sub.add_parser("ambientcg")
    acg.add_argument("--count", type=int, default=48)
    acg.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.source == "polyhaven":
        download_polyhaven(args.res, args.limit, args.workers)
    else:
        download_ambientcg(args.count, args.workers)


if __name__ == "__main__":
    main()
