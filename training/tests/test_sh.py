"""SH maths: basis, projection, rotation, irradiance, mirroring, dominant direction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lumina_train import sh as shm
from lumina_train.environments import analytic_environments, environment_by_name

ENVS = analytic_environments()


def test_basis_matches_threejs_constants():
    d = np.array([0.6, 0.0, 0.8])  # x, y, z
    expected = [
        0.282095,
        0.0,
        0.488603 * 0.8,
        0.488603 * 0.6,
        0.0,
        0.0,
        0.315392 * (3 * 0.64 - 1),
        1.092548 * 0.48,
        0.546274 * 0.36,
    ]
    np.testing.assert_allclose(shm.sh_basis(d), expected, atol=1e-12)


def test_basis_is_orthonormal_under_solid_angle_weights():
    dirs = shm.equirect_directions(512, 1024)
    w = shm.equirect_solid_angles(512, 1024)
    b = shm.sh_basis(dirs).reshape(-1, 9)
    gram = (b * w.reshape(-1, 1)).T @ b
    # three.js constants are rounded to 6 significant digits, so the identity holds to ~1e-5.
    np.testing.assert_allclose(gram, np.eye(9), atol=2e-5)


def test_solid_angles_sum_to_four_pi():
    assert shm.equirect_solid_angles(64, 128).sum() == pytest.approx(4 * math.pi, rel=1e-12)


def test_equirect_layout_matches_threejs_mapping():
    dirs = shm.equirect_directions(180, 360)
    # Centre column looks along +X, quarter column along -Z, top row near +Y.
    np.testing.assert_allclose(dirs[90, 180], [1, 0, 0], atol=0.02)
    np.testing.assert_allclose(dirs[90, 90], [0, 0, -1], atol=0.02)
    assert dirs[0, 0, 1] > 0.9999
    # three.js: u = atan2(z, x) / (2 pi) + 0.5
    u = np.arctan2(dirs[..., 2], dirs[..., 0]) / (2 * math.pi) + 0.5
    np.testing.assert_allclose(u[45], (np.arange(360) + 0.5) / 360, atol=1e-9)


@pytest.mark.parametrize("env", ENVS, ids=lambda e: e.name)
def test_projection_matches_analytic_coefficients(env):
    dirs = shm.equirect_directions(256, 512)
    projected = shm.project_equirect(env.radiance(dirs))
    np.testing.assert_allclose(projected, env.sh(), atol=1e-4)


def test_camera_rotation_semantics():
    forward = np.array([0.0, 0.0, -1.0])
    left = shm.camera_rotation(math.pi / 2, 0, 0) @ forward
    np.testing.assert_allclose(left, [-1, 0, 0], atol=1e-12)
    up = shm.camera_rotation(0, math.pi / 2, 0) @ forward
    np.testing.assert_allclose(up, [0, 1, 0], atol=1e-12)
    right_axis = shm.camera_rotation(0, 0, math.pi / 2) @ np.array([1.0, 0, 0])
    np.testing.assert_allclose(right_axis, [0, 1, 0], atol=1e-12)
    r = shm.camera_rotation(0.3, -0.2, 0.1)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_light_ahead_of_turned_camera_lands_on_negative_z():
    # A lobe along world +X, camera yawed right by 90 deg (yaw = -pi/2) to face +X.
    env = environment_by_name("lobe_pos_x")
    r = shm.camera_rotation(-math.pi / 2, 0, 0)
    direction, _ = shm.dominant_direction(shm.rotate_sh(env.sh(), r))
    np.testing.assert_allclose(direction, [0, 0, -1], atol=1e-9)


@pytest.mark.parametrize("env", ENVS, ids=lambda e: e.name)
def test_rotate_sh_equals_projecting_rotated_environment(env):
    r = shm.camera_rotation(0.7, -0.4, 0.25)
    dirs = shm.equirect_directions(256, 512)
    direct = shm.project_equirect(env.radiance(dirs), cam_to_world=r)
    np.testing.assert_allclose(shm.rotate_sh(env.sh(), r), direct, atol=1e-4)


@pytest.mark.parametrize("env", ENVS, ids=lambda e: e.name)
def test_irradiance_matches_brute_force_cosine_integral(env):
    dirs = shm.equirect_directions(256, 512).reshape(-1, 3)
    w = shm.equirect_solid_angles(256, 512).reshape(-1)
    radiance = env.radiance(dirs)
    normals = shm.fibonacci_sphere(40)
    cos = np.clip(normals @ dirs.T, 0.0, None) * w[None, :]
    brute = cos @ radiance
    sh9 = shm.irradiance(env.sh(), normals)
    # SH9 irradiance truncation error: measured 2.2% worst case for these soft lobes.
    assert np.abs(sh9 - brute).max() <= 0.03 * brute.max()


def test_constant_white_irradiance_is_pi_and_sky_ground_is_exact_at_poles():
    white = environment_by_name("constant_white").sh()
    np.testing.assert_allclose(shm.irradiance(white, shm.fibonacci_sphere(10)), math.pi, rtol=1e-5)
    sky = environment_by_name("sky_ground").sh()
    poles = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    expected = math.pi * np.array([[0.6, 0.75, 1.0], [0.35, 0.25, 0.15]])
    np.testing.assert_allclose(shm.irradiance(sky, poles), expected, rtol=1e-5)


@pytest.mark.parametrize("env", ENVS, ids=lambda e: e.name)
def test_mirror_x_equals_evaluating_at_mirrored_directions(env):
    dirs = shm.fibonacci_sphere(50)
    flipped = dirs * np.array([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        shm.eval_sh(shm.mirror_x(env.sh()), dirs), shm.eval_sh(env.sh(), flipped), atol=1e-12
    )


@pytest.mark.parametrize("env", ENVS, ids=lambda e: e.name)
def test_dominant_direction_matches_known_light(env):
    direction, ratio = shm.dominant_direction(env.sh())
    if env.dominant is None:
        assert direction is None
        assert ratio == pytest.approx(0.0, abs=1e-9)
    else:
        assert direction is not None
        np.testing.assert_allclose(direction, env.dominant, atol=1e-9)
        assert 0.0 < ratio <= 1.0


def test_point_light_has_directionality_one():
    d = np.array([0.0, 0.6, 0.8])
    point = shm.sh_basis(d)[:, None] * np.ones((1, 3))
    direction, ratio = shm.dominant_direction(point)
    np.testing.assert_allclose(direction, d, atol=1e-12)
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_srgb_round_trip_and_known_points():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(shm.srgb_to_linear(shm.linear_to_srgb(x)), x, atol=1e-12)
    assert shm.linear_to_srgb(np.array(0.18)) == pytest.approx(0.4614, abs=1e-4)
    assert shm.to_srgb8(np.array([0.0, 0.5, 1.0])).tolist() == [0, 188, 255]


def test_white_sphere_under_white_light_is_white():
    img = shm.render_diffuse_sphere(environment_by_name("constant_white").sh(), 16)
    _, mask = shm.sphere_normals(16)
    np.testing.assert_allclose(img[mask], 1.0, rtol=1e-5)
    assert np.all(img[~mask] == 0)
