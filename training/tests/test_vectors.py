"""shared/ test vectors: committed files are current and agree with the Python implementation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lumina_train import sh as shm
from lumina_train.vectors import REPO_ROOT, build_vectors, read_png_rgb, render_sphere_reference

SHARED = REPO_ROOT / "shared"
VECTORS = json.loads((SHARED / "sh_vectors.json").read_text())


def _assert_tree_close(a, b, path="$"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _assert_tree_close(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_tree_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, abs=1e-9), path
    else:
        assert a == b, path


def test_committed_vectors_are_current():
    """Fails when the maths changed but shared/ was not regenerated."""
    _assert_tree_close(VECTORS, json.loads(json.dumps(build_vectors())))


def test_committed_sphere_png_is_current():
    committed = read_png_rgb(SHARED / "sphere_reference.png").astype(int)
    assert committed.shape == (128, 128, 3)
    # Allow one quantisation level for platform float differences at .5 boundaries.
    assert np.abs(committed - render_sphere_reference().astype(int)).max() <= 1


def test_png_channel_order_is_rgb():
    """The warm light makes red > blue on the lit side; a BGR mix-up would invert it."""
    img = read_png_rgb(SHARED / "sphere_reference.png").astype(int)
    lit = img[30, 98]
    assert lit[0] > lit[2] + 20


NORMALS = np.array(VECTORS["normals"])


def test_basis_matches_vectors():
    np.testing.assert_allclose(shm.sh_basis(NORMALS), VECTORS["basis_at_normals"], atol=1e-12)


@pytest.mark.parametrize("record", VECTORS["environments"], ids=lambda r: r["name"])
def test_python_matches_vectors(record):
    sh = np.array(record["sh"])
    np.testing.assert_allclose(shm.irradiance(sh, NORMALS), record["irradiance"], atol=1e-9)
    np.testing.assert_allclose(shm.mirror_x(sh), record["sh_mirrored_x"], atol=1e-12)
    direction, ratio = shm.dominant_direction(sh)
    assert ratio == pytest.approx(record["directionality"], abs=1e-9)
    if record["dominant_direction"] is None:
        assert direction is None
    else:
        np.testing.assert_allclose(direction, record["dominant_direction"], atol=1e-9)
