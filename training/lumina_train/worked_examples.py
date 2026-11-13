"""Print the worked examples quoted in docs/CONVENTIONS.md section 9.

Run: uv run python -m lumina_train.worked_examples
"""

from __future__ import annotations

import math

import numpy as np

from lumina_train import sh as shm
from lumina_train.environments import environment_by_name


def _fmt(values: np.ndarray, digits: int = 6) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in np.ravel(values)) + "]"


def _dominant(coeffs: np.ndarray) -> str:
    direction, ratio = shm.dominant_direction(coeffs)
    shown = "None" if direction is None else _fmt(direction, 3)
    return f"direction {shown}, directionality {ratio:.3f}"


def main() -> None:
    white = environment_by_name("constant_white").sh()
    up = np.array([0.0, 1.0, 0.0])
    print("constant_white L0", _fmt(white[0]), "E(+Y)", _fmt(shm.irradiance(white, up)))

    lobe_y = environment_by_name("lobe_pos_y").sh()
    print("lobe_pos_y sh[:, r]", _fmt(lobe_y[:, 0]))
    for name, n in (("+Y", up), ("+X", np.array([1.0, 0, 0])), ("-Y", -up)):
        e = shm.irradiance(lobe_y, n)[0]
        print(f"  E({name}) = {e:.6f}, sRGB8 of E/pi = {shm.to_srgb8(np.array(e / math.pi))}")
    print("  ", _dominant(lobe_y))

    lobe_x = environment_by_name("lobe_pos_x").sh()
    mirrored = shm.mirror_x(lobe_x)
    print("lobe_pos_x sh[3, r]", f"{lobe_x[3, 0]:.6f}", "mirrored", f"{mirrored[3, 0]:.6f}")
    print("  mirrored", _dominant(mirrored))
    turned = shm.rotate_sh(lobe_x, shm.camera_rotation(-math.pi / 2, 0.0, 0.0))
    print("  camera yawed -90 deg: sh[2, r]", f"{turned[2, 0]:.6f},", _dominant(turned))

    warm = environment_by_name("lobe_oblique_warm").sh()
    axis = np.array([2.0, 2.0, 1.0]) / 3.0
    e = shm.irradiance(warm, axis)
    print("lobe_oblique_warm", _dominant(warm))
    print("  E(axis)", _fmt(e, 4), "E/pi", _fmt(e / math.pi, 4), "sRGB8", shm.to_srgb8(e / math.pi))


if __name__ == "__main__":
    main()
