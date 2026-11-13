"""LuminaNet's output contract, the loss, the metrics and the baselines on known inputs."""

from __future__ import annotations

import math

import numpy as np
import torch

from lumina_train import baselines, metrics
from lumina_train import sh as shm
from lumina_train.camera import pixel_direction
from lumina_train.environments import analytic_environments, environment_by_name
from lumina_train.losses import LightingLoss
from lumina_train.model import LuminaNet, count_parameters, raw_to_sh

SH = torch.from_numpy(np.stack([e.sh() for e in analytic_environments()])).float()


def test_model_shape_size_and_positive_l0():
    model = LuminaNet().eval()
    assert count_parameters(model) < 2_000_000  # under 8 MB as float32
    with torch.no_grad():
        out = model(torch.rand(2, 3, 96, 128))
    assert out.shape == (2, 9, 3)
    assert torch.all(out[:, 0] > 0)


def test_raw_to_sh_is_l0_times_relative_shape():
    raw = torch.zeros(1, 9, 3)
    raw[0, 0] = torch.tensor([0.0, 1.0, -1.0])
    raw[0, 3] = 0.5
    sh = raw_to_sh(raw)
    np.testing.assert_allclose(sh[0, 0].numpy(), np.exp([0.0, 1.0, -1.0]), rtol=1e-6)
    np.testing.assert_allclose(sh[0, 3].numpy(), 0.5 * np.exp([0.0, 1.0, -1.0]), rtol=1e-6)


def test_loss_is_zero_for_a_perfect_prediction_and_positive_otherwise():
    loss = LightingLoss(torch.device("cpu"))
    assert float(loss(SH, SH).total) < 1e-6
    assert float(loss(SH * 1.5, SH).parts["si"]) < 1e-6  # scale-invariant term ignores scale
    assert float(loss(SH * 1.5, SH).parts["irr"]) > 0.1


def test_metrics_on_identical_scaled_and_rotated_predictions():
    same = metrics.compute(SH, SH)
    has_direction = same["directionality"] > 1e-6  # constant white has no direction at all
    assert has_direction.sum() == len(SH) - 1
    for key in metrics.METRICS:
        assert np.all(np.abs(same[key][has_direction]) < 1e-3), key
    scaled = metrics.compute(SH * 2.0, SH)
    assert np.all(scaled["si_rmse"] < 1e-6) and np.all(scaled["colour_deg"] < 1e-3)
    assert np.all(scaled["rmse"] > 0)
    lobe = environment_by_name("lobe_pos_x").sh()
    mirrored = torch.from_numpy(np.stack([shm.mirror_x(lobe)])).float()
    flipped = metrics.compute(mirrored, torch.from_numpy(lobe[None]).float())
    assert abs(flipped["dir_deg"][0] - 180.0) < 1e-3


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = np.random.default_rng(0).normal(5.0, 1.0, 100)
    mean, lo, hi = metrics.bootstrap_ci(values, seed=1)
    assert lo < mean < hi
    assert metrics.bootstrap_ci(values, seed=1) == (mean, lo, hi)
    ids, means = metrics.per_group_mean(np.array([1.0, 3.0, 10.0]), np.array(["a", "a", "b"]))
    assert ids.tolist() == ["a", "b"] and means.tolist() == [2.0, 10.0]


def test_image_mean_baseline_reproduces_a_uniform_grey_frame():
    frame = np.full((1, 96, 128, 3), 128, np.uint8)
    sh = baselines.image_mean_baseline(frame)[0]
    linear = shm.srgb_to_linear(np.array(128 / 255.0))
    # A white sphere under a uniform environment of radiance c has outgoing radiance c.
    np.testing.assert_allclose(shm.render_diffuse_sphere(sh, 8)[4, 4], linear, rtol=1e-4)


def test_brightest_region_direction_points_at_the_bright_pixel():
    frame = np.full((96, 128, 3), 20, np.uint8)
    frame[10:14, 100:104] = 255
    d = baselines.brightest_direction(frame)
    expected = pixel_direction(11.5, 101.5, baselines.ASSUMED_VFOV, 96, 128)
    assert math.degrees(math.acos(min(1.0, float(d @ expected)))) < 0.1
    sh = baselines.image_dir_baseline(frame[None], directionality=0.5)[0]
    direction, ratio = shm.dominant_direction(sh)
    assert abs(ratio - 0.5) < 1e-6
    np.testing.assert_allclose(direction, expected, atol=1e-6)
