"""The committed test-crop list and the clip helpers behave as documented."""

from __future__ import annotations

import json

import numpy as np

from lumina_train.clips import FPS, ClipSpec, poses
from lumina_train.clipvideo import model_inputs
from lumina_train.fixedsets import TEST_CROPS_PATH, TEST_POSES, TEST_SETTINGS, load_test_crops
from lumina_train.paths import SPLIT_PATH


def test_test_crops_cover_every_test_and_ood_hdri_with_poses_times_settings():
    split = json.loads(SPLIT_PATH.read_text())["splits"]
    for name in ("test", "ood"):
        ids, params, per = load_test_crops(name)
        assert ids == [e["id"] for e in split[name]]
        assert per == TEST_POSES * len(TEST_SETTINGS)
        assert all(len(v) == len(ids) * per for v in params.values())
        assert not params["flip"].any()
        kelvin = params["kelvin"].reshape(-1, len(TEST_SETTINGS))
        assert np.all(kelvin == [k for _, _, k in TEST_SETTINGS])
        yaw = params["yaw"].reshape(-1, len(TEST_SETTINGS))
        assert np.all(yaw == yaw[:, :1])  # each pose is shown under every setting


def test_test_crops_file_is_compact_json():
    doc = json.loads(TEST_CROPS_PATH.read_text())
    assert doc["crops_per_hdri"] == TEST_POSES * len(TEST_SETTINGS)


def test_model_inputs_centre_crops_wide_frames_to_4_by_3():
    frame = np.zeros((1, 90, 160, 3), np.uint8)
    frame[:, :, :20] = 255  # a bright band on the far left, outside the 4:3 centre (120 wide)
    out = model_inputs(frame)
    assert out.shape == (1, 96, 128, 3)
    assert out.max() == 0


def test_rotating_environment_clip_turns_a_full_circle_at_constant_speed():
    spec = ClipSpec("r", "x", "outdoor", 10.0, "rotate_env", 0.0, 360.0, 0.0, 60.0, 6500.0, "")
    yaw, _, _ = poses(spec)
    assert len(yaw) == round(10.0 * FPS)
    step = np.diff(yaw)
    np.testing.assert_allclose(step, -2 * np.pi / len(yaw), rtol=1e-9)
