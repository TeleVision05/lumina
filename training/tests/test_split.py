"""The committed split keeps scenes apart and has the promised shape."""

from __future__ import annotations

import json

from lumina_train.paths import SPLIT_PATH
from lumina_train.split import group_assets, stem

SPLIT = json.loads(SPLIT_PATH.read_text())


def test_no_hdri_in_two_splits():
    seen: dict[str, str] = {}
    for name, entries in SPLIT["splits"].items():
        for e in entries:
            assert e["id"] not in seen, f"{e['id']} in {seen.get(e['id'])} and {name}"
            seen[e["id"]] = name


def test_no_id_stem_shared_between_polyhaven_splits():
    stems = {
        name: {stem(e["id"]) for e in SPLIT["splits"][name]} for name in ("train", "val", "test")
    }
    assert not stems["train"] & stems["val"]
    assert not stems["train"] & stems["test"]
    assert not stems["val"] & stems["test"]


def test_roughly_80_10_10_per_environment():
    counts = SPLIT["counts"]
    for env in ("indoor", "outdoor"):
        total = sum(counts[s][env] for s in ("train", "val", "test"))
        assert abs(counts["train"][env] / total - 0.8) < 0.03
        for s in ("val", "test"):
            assert abs(counts[s][env] / total - 0.1) < 0.03


def test_grouping_joins_same_place_and_same_stem():
    assets = [{"id": i} for i in ("a_1", "a_2", "b_sunset", "b_noon", "c", "d")]
    coords = {"c": [10.001, 20.0], "d": [10.002, 20.001]}
    groups = group_assets(assets, coords)
    assert sorted(map(sorted, groups)) == [["a_1", "a_2"], ["b_noon", "b_sunset"], ["c", "d"]]
