"""The torch SH9 functions agree with their float64 numpy twins in sh.py."""

from __future__ import annotations

import numpy as np
import torch

from lumina_train import sh as shm
from lumina_train import sh_torch
from lumina_train.camera import rotations
from lumina_train.environments import analytic_environments

ENVS = analytic_environments()
SH = torch.from_numpy(np.stack([e.sh() for e in ENVS])).float()


def test_basis_matches_numpy():
    dirs = shm.fibonacci_sphere(50)
    np.testing.assert_allclose(
        sh_torch.sh_basis(torch.from_numpy(dirs)).numpy(), shm.sh_basis(dirs), atol=1e-12
    )


def test_rotations_match_camera_rotation():
    angles = np.array([[0.3, -0.2, 0.1], [2.5, 0.6, -0.15], [-1.0, 0.0, 0.0]])
    got = rotations(*(torch.from_numpy(angles[:, i]) for i in range(3))).numpy()
    for i, (y, p, r) in enumerate(angles):
        np.testing.assert_allclose(got[i], shm.camera_rotation(y, p, r), atol=1e-12)


def test_rotate_sh_matches_numpy():
    rot = rotations(*(torch.full((len(ENVS),), a) for a in (0.7, -0.3, 0.2)))
    got = sh_torch.rotate_sh(SH, rot.float()).numpy()
    for i, env in enumerate(ENVS):
        np.testing.assert_allclose(got[i], shm.rotate_sh(env.sh(), rot[i].numpy()), atol=1e-5)


def test_irradiance_and_direction_match_numpy():
    normals = shm.fibonacci_sphere(20)
    got = sh_torch.irradiance(SH, torch.from_numpy(normals).float()).numpy()
    ratio = sh_torch.directionality(SH).numpy()
    for i, env in enumerate(ENVS):
        np.testing.assert_allclose(got[i], shm.irradiance(env.sh(), normals), atol=1e-5)
        direction, expected_ratio = shm.dominant_direction(env.sh())
        assert abs(ratio[i] - expected_ratio) < 1e-5
        if direction is not None:
            v = sh_torch.band1_vector(SH[i : i + 1])[0].numpy()
            np.testing.assert_allclose(v / np.linalg.norm(v), direction, atol=1e-5)


def test_mirror_x_matches_numpy():
    np.testing.assert_allclose(
        sh_torch.mirror_x(SH).numpy(), np.stack([shm.mirror_x(e.sh()) for e in ENVS]), atol=1e-6
    )
