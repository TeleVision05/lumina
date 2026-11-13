"""Minimal line plots drawn with OpenCV (no plotting dependency), for docs/media figures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

COLOURS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]  # RGB
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45) -> None:
    cv2.putText(img, text, org, _FONT, scale, (40, 40, 40), 1, cv2.LINE_AA)


def _nice_ticks(lo: float, hi: float, count: int = 5) -> np.ndarray:
    raw = (hi - lo) / max(1, count - 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    return np.arange(np.ceil(lo / step) * step, hi + 0.5 * step, step)


def line_panel(
    xs: np.ndarray, series: dict[str, np.ndarray], title: str, size: tuple[int, int] = (520, 300)
) -> np.ndarray:
    """One panel: white background, axes with ticks, one line per series, legend. RGB uint8."""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    left, right, top, bottom = 60, 12, 30, 40
    ys = np.concatenate(list(series.values()))
    y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
    pad = 0.05 * (y_hi - y_lo or 1.0)
    y_lo, y_hi = y_lo - pad, y_hi + pad
    x_lo, x_hi = float(xs.min()), float(xs.max())

    def px(x: float, y: float) -> tuple[int, int]:
        u = left + (x - x_lo) / (x_hi - x_lo or 1.0) * (w - left - right)
        v = top + (1.0 - (y - y_lo) / (y_hi - y_lo)) * (h - top - bottom)
        return round(u), round(v)

    cv2.rectangle(img, (left, top), (w - right, h - bottom), (120, 120, 120), 1)
    for t in _nice_ticks(y_lo, y_hi):
        _, v = px(x_lo, t)
        cv2.line(img, (left, v), (w - right, v), (230, 230, 230), 1)
        _text(img, f"{t:g}", (4, v + 4), 0.38)
    for t in _nice_ticks(x_lo, x_hi):
        u, _ = px(t, y_lo)
        _text(img, f"{t / 1000:g}k", (u - 10, h - bottom + 16), 0.38)
    _text(img, "training step", (w // 2 - 40, h - 8), 0.4)
    _text(img, title, (left, 20), 0.5)
    for i, (name, values) in enumerate(series.items()):
        colour = COLOURS[i % len(COLOURS)]
        pts = np.array([px(x, y) for x, y in zip(xs, values, strict=True)], np.int32)
        cv2.polylines(img, [pts], False, colour, 2, cv2.LINE_AA)
        ly = top + 16 + 16 * i
        cv2.line(img, (w - 170, ly - 4), (w - 150, ly - 4), colour, 2)
        _text(img, name, (w - 145, ly), 0.4)
    return img


def save_panels(panels: list[np.ndarray], path: Path) -> None:
    gap = np.full((panels[0].shape[0], 10, 3), 255, np.uint8)
    parts: list[np.ndarray] = []
    for p in panels:
        parts += [p, gap]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(parts[:-1], axis=1)[:, :, ::-1])
