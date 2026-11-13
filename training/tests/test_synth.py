"""Synthetic camera model on analytic environments: geometry, labels, exposure, white balance."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lumina_train import sh as shm
from lumina_train import sh_torch
from lumina_train.camera import pad_wrap, pixel_direction, render_crops, rotations
from lumina_train.environments import Environment, ZonalTerm, environment_by_name
from lumina_train.params import neutral_pipeline, white_balance_gains
from lumina_train.synth import OUT_H, OUT_W, PanoBank

H, W = 256, 512
CPU = torch.device("cpu")


def _sharp(t: np.ndarray) -> np.ndarray:
    return np.exp(300.0 * (t - 1.0))


def _panorama(env: Environment) -> np.ndarray:
    return env.radiance(shm.equirect_directions(H, W)).astype(np.float32)


def _bank(env: Environment) -> PanoBank:
    pano = _panorama(env)
    return PanoBank(pano[None].astype(np.float16), shm.project_equirect(pano)[None], CPU)


def _params(yaw: float, pitch: float, roll: float, vfov_deg: float = 60.0, flip: bool = False):
    p = neutral_pipeline(1)
    p.update(
        yaw=np.array([yaw]),
        pitch=np.array([pitch]),
        roll=np.array([roll]),
        vfov=np.array([math.radians(vfov_deg)]),
        flip=np.array([flip]),
    )
    return p


def _lobe_env(world_axis: np.ndarray, sharp: bool = True) -> Environment:
    term = ZonalTerm(axis=tuple(world_axis), colour=(4.0, 4.0, 4.0), profile=_sharp, lo=0.9)
    if not sharp:
        term = environment_by_name("lobe_pos_x").terms[0]
        term = ZonalTerm(axis=tuple(world_axis), colour=(1.0, 1.0, 1.0), profile=term.profile)
    return Environment(name="t", description="", ambient=(0.002, 0.002, 0.002), terms=(term,))


POSES = [(0.3, 0.1, 0.05), (-2.0, -0.3, -0.1), (3.0, 0.5, 0.0)]


@pytest.mark.parametrize(("yaw", "pitch", "roll"), POSES)
@pytest.mark.parametrize(("row", "col"), [(30.0, 90.0), (70.0, 20.0)])
def test_bright_lobe_lands_on_expected_pixel_and_label_direction(yaw, pitch, roll, row, col):
    vfov = math.radians(60.0)
    d_cam = pixel_direction(row, col, vfov, OUT_H, OUT_W)
    rot = shm.camera_rotation(yaw, pitch, roll)
    bank = _bank(_lobe_env(rot @ d_cam))
    batch = bank.render(np.array([0]), _params(yaw, pitch, roll), 1, keep_linear=True)
    lum = np.einsum("chw,c->hw", batch.linear[0].double().numpy(), shm.LUMA)
    peak_row, peak_col = np.unravel_index(np.argmax(lum), lum.shape)
    assert abs(peak_row - row) <= 1 and abs(peak_col - col) <= 1
    direction, ratio = shm.dominant_direction(batch.labels[0].double().numpy())
    assert ratio > 0.5
    assert math.degrees(math.acos(np.clip(direction @ d_cam, -1, 1))) < 1.0


def test_render_matches_analytic_radiance_across_the_seam():
    env = environment_by_name("lobe_oblique_warm")
    pano = torch.from_numpy(_panorama(env)).permute(2, 0, 1)[None]
    # yaw = pi/2 points the camera at world -X: the u = 0 / u = 1 seam is in the middle.
    yaw, pitch, roll = torch.tensor([math.pi / 2]), torch.tensor([0.2]), torch.tensor([0.0])
    rot = rotations(yaw, pitch, roll)
    out = render_crops(pad_wrap(pano), rot, torch.tensor([1.0]), OUT_H, OUT_W, ss=1)
    rays = [[pixel_direction(r, c, 1.0, OUT_H, OUT_W) for c in range(OUT_W)] for r in range(OUT_H)]
    expect = env.radiance(np.asarray(rays) @ rot[0].double().numpy().T)
    got = out[0].permute(1, 2, 0).double().numpy()
    assert np.max(np.abs(got - expect)) < 0.02 * np.max(expect)


@pytest.mark.parametrize(("yaw", "pitch", "roll"), POSES)
def test_label_equals_projection_in_camera_frame(yaw, pitch, roll):
    env = environment_by_name("lobe_oblique_warm")
    bank = _bank(env)
    batch = bank.render(np.array([0]), _params(yaw, pitch, roll), 1)
    rot = shm.camera_rotation(yaw, pitch, roll)
    exact = shm.rotate_sh(env.sh(), rot)
    scale = batch.labels[0, 0].double().numpy() / exact[0]
    np.testing.assert_allclose(
        batch.labels[0].double().numpy(), exact * scale, atol=2e-3 * scale[1]
    )
    projected = shm.project_equirect(_panorama(env), cam_to_world=rot)
    np.testing.assert_allclose(exact, projected, atol=2e-3)


def test_rotating_the_camera_rotates_the_label_consistently():
    lobe = environment_by_name("lobe_pos_x").terms
    bank = _bank(Environment("e", "", ambient=(0.1, 0.0, 0.0), terms=lobe))
    pose_a, pose_b = POSES[0], POSES[1]
    la = bank.render(np.array([0]), _params(*pose_a), 1).labels[0].double().numpy()
    lb = bank.render(np.array([0]), _params(*pose_b), 1).labels[0].double().numpy()
    ra, rb = shm.camera_rotation(*pose_a), shm.camera_rotation(*pose_b)
    # Express camera A's label in camera B's frame: cam_b -> cam_a is ra^T rb.
    predicted_b = shm.rotate_sh(la, ra.T @ rb)
    ratio = lb[0] / predicted_b[0]  # exposure differs between the two crops
    np.testing.assert_allclose(lb, predicted_b * ratio, atol=1e-4 * np.abs(lb).max())


def test_exposure_scales_image_and_label_together():
    bank = _bank(environment_by_name("lobe_oblique_warm"))
    base = _params(0.4, 0.1, 0.0)
    brighter = {k: v.copy() for k, v in base.items()}
    brighter["ev"] = np.array([1.0])
    a = bank.render(np.array([0]), base, 1)
    b = bank.render(np.array([0]), brighter, 1)
    np.testing.assert_allclose(b.labels.numpy(), 2.0 * a.labels.numpy(), rtol=1e-5, atol=1e-6)
    lin_a = shm.srgb_to_linear(a.images.double().numpy())
    lin_b = shm.srgb_to_linear(b.images.double().numpy())
    ok = (lin_b < 0.95) & (lin_a > 0.02)
    assert ok.sum() > 1000
    np.testing.assert_allclose(np.median(lin_b[ok] / lin_a[ok]), 2.0, rtol=0.02)


def test_white_balance_scales_image_and_label_together():
    bank = _bank(Environment("grey", "", ambient=(0.5, 0.5, 0.5)))
    params = _params(0.0, 0.0, 0.0)
    params["kelvin"] = np.array([3200.0])
    batch = bank.render(np.array([0]), params, 1)
    gains = white_balance_gains(np.array([3200.0]), np.array([0.0]))[0]
    assert gains[0] > gains[1] > gains[2]
    label = batch.labels[0].double().numpy()
    np.testing.assert_allclose(label[0] / label[0, 1], gains / gains[1], rtol=1e-5)
    pixel = shm.srgb_to_linear(batch.images[0].double().numpy()).mean(axis=(1, 2))
    np.testing.assert_allclose(pixel / pixel[1], gains / gains[1], rtol=0.02)


def test_uniform_light_sphere_matches_the_photo():
    """The contract in docs/CONVENTIONS.md section 5: a white sphere lit by the label has the
    brightness and colour the photo shows for a uniform environment."""
    bank = _bank(Environment("grey", "", ambient=(0.3, 0.5, 0.4)))
    params = _params(1.0, 0.0, 0.0)
    params["kelvin"] = np.array([5000.0])
    batch = bank.render(np.array([0]), params, 1)
    sphere = shm.render_diffuse_sphere(batch.labels[0].double().numpy(), 16)[8, 8]
    pixel = shm.srgb_to_linear(batch.images[0].double().numpy()).mean(axis=(1, 2))
    np.testing.assert_allclose(sphere, pixel, rtol=0.02)


def test_flip_mirrors_image_and_label():
    d_cam = pixel_direction(40.0, 100.0, math.radians(60.0), OUT_H, OUT_W)
    rot = shm.camera_rotation(0.2, 0.0, 0.0)
    bank = _bank(_lobe_env(rot @ d_cam))
    plain = bank.render(np.array([0]), _params(0.2, 0.0, 0.0), 1)
    flipped = bank.render(np.array([0]), _params(0.2, 0.0, 0.0, flip=True), 1)
    np.testing.assert_array_equal(flipped.images.numpy(), plain.images.flip(-1).numpy())
    np.testing.assert_allclose(flipped.labels.numpy(), sh_torch.mirror_x(plain.labels).numpy())
    direction, _ = shm.dominant_direction(flipped.labels[0].double().numpy())
    assert direction[0] < 0 < d_cam[0]


def test_white_balance_gains_neutral_at_reference():
    gains = white_balance_gains(np.array([6500.0, 3000.0, 9000.0]), np.zeros(3))
    np.testing.assert_allclose(gains[0], [1.0, 1.0, 1.0], atol=1e-9)
    assert gains[1, 0] > gains[1, 2]  # warm light: red over blue
    assert gains[2, 2] > gains[2, 0]  # cool light: blue over red
    np.testing.assert_allclose(gains @ shm.LUMA, 1.0, atol=1e-9)
