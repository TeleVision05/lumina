"""Split the HDRIs into train / validation / test by scene, never by crop.

    uv run python -m lumina_train.split      # writes training/splits/split.json

Poly Haven often has several HDRIs of one place (different times of day, positions a few metres
apart). Putting siblings on both sides of a split would leak the test scenes into training, so
HDRIs are first grouped: two HDRIs share a group if their GPS coordinates agree to 0.01 degrees
(about 1 km) or their ids share a stem once time-of-day words and trailing numbers are removed.
Groups are shuffled with a fixed seed and assigned greedily so each environment type (indoor,
outdoor) is split about 80/10/10. The ambientCG HDRIs form a separate out-of-distribution test
split ("ood") that training never sees.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from typing import Any

from lumina_train.paths import AMBIENTCG_DIR, POLYHAVEN_DIR, SPLIT_PATH

SEED = 20261113
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
_TIME_WORDS = re.compile(
    r"_(\d+k|puresky|sunset|sunrise|dawn|dusk|night|morning|evening|afternoon|noon|"
    r"partly_cloudy|clear|cloudy|overcast)\b.*"
)
_TRAILING_NUM = re.compile(r"(_\d+[a-z]?)+$")


def stem(asset_id: str) -> str:
    return _TRAILING_NUM.sub("", _TIME_WORDS.sub("", asset_id))


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_assets(assets: list[dict[str, Any]], coords: dict[str, list[float]]) -> list[list[str]]:
    """Connected components of 'same place' (coords to 0.01 deg) or 'same stem'."""
    ids = [a["id"] for a in assets]
    uf = _UnionFind(ids)
    by_key: dict[str, str] = {}
    for asset_id in ids:
        keys = [f"stem:{stem(asset_id)}"]
        if asset_id in coords:
            lat, lon = coords[asset_id]
            keys.append(f"gps:{round(lat, 2)},{round(lon, 2)}")
        for key in keys:
            if key in by_key:
                uf.union(asset_id, by_key[key])
            else:
                by_key[key] = asset_id
    groups: dict[str, list[str]] = defaultdict(list)
    for asset_id in ids:
        groups[uf.find(asset_id)].append(asset_id)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def assign(groups: list[list[str]], env: dict[str, str]) -> dict[str, list[str]]:
    """Seeded shuffle, then give each group to the split furthest below its target share."""
    rng = random.Random(SEED)
    order = groups[:]
    rng.shuffle(order)
    totals: dict[str, int] = defaultdict(int)
    for g in groups:
        totals[_majority(g, env)] += len(g)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    out: dict[str, list[str]] = {name: [] for name in FRACTIONS}
    for g in order:
        kind = _majority(g, env)
        deficit = {s: FRACTIONS[s] * totals[kind] - counts[(s, kind)] for s in FRACTIONS}
        best = max(FRACTIONS, key=lambda s: (deficit[s] / FRACTIONS[s], -list(FRACTIONS).index(s)))
        out[best].extend(g)
        counts[(best, kind)] += len(g)
    return {s: sorted(v) for s, v in out.items()}


def _majority(group: list[str], env: dict[str, str]) -> str:
    indoor = sum(1 for a in group if env[a] == "indoor")
    return "indoor" if indoor * 2 > len(group) else "outdoor"


def build_split(ph_manifest: dict[str, Any], acg_manifest: dict[str, Any]) -> dict[str, Any]:
    assets = ph_manifest["assets"]
    env = {a["id"]: a["environment"] or "outdoor" for a in assets}
    coords = {a["id"]: a["coords"] for a in assets if a.get("coords")}
    groups = group_assets(assets, coords)
    parts = assign(groups, env)
    splits: dict[str, list[dict[str, str]]] = {}
    for name, ids in parts.items():
        splits[name] = [{"id": i, "source": "polyhaven", "environment": env[i]} for i in ids]
    splits["ood"] = [
        {"id": a["id"], "source": "ambientcg", "environment": a["environment"]}
        for a in acg_manifest["assets"]
    ]
    return {
        "schema": 1,
        "seed": SEED,
        "method": "group by GPS (0.01 deg) or id stem, seeded shuffle, greedy 80/10/10 per env",
        "groups": len(groups),
        "counts": {s: _counts(v) for s, v in splits.items()},
        "splits": splits,
    }


def _counts(entries: list[dict[str, str]]) -> dict[str, int]:
    out = {"total": len(entries), "indoor": 0, "outdoor": 0}
    for e in entries:
        out[e["environment"]] += 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    ph = json.loads((POLYHAVEN_DIR / "manifest.json").read_text())
    acg = json.loads((AMBIENTCG_DIR / "manifest.json").read_text())
    split = build_split(ph, acg)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=1) + "\n")
    print(f"wrote {SPLIT_PATH}: {json.dumps(split['counts'])}, {split['groups']} groups")


if __name__ == "__main__":
    main()
