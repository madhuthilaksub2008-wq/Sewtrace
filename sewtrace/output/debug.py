"""Debug image of one traced photo: what the outline step found, to diagnose a failed real run.

Left: the photo with the whole outline (yellow), the body with the sleeves cut off (green) and the
invented armholes (magenta). Right: the half panel with each labelled edge in its own colour and
its length in mm, which is what the seam check compares."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sewtrace.types import Edge, Outline
from sewtrace.vision.marker import Homography

COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (0, 128, 128)]  # BGR, one per edge
PANEL_PX = 900


def edge_run(points: np.ndarray, edge: Edge) -> np.ndarray:
    if edge.start <= edge.end:
        return points[edge.start : edge.end + 1]
    return np.vstack([points[edge.start :], points[: edge.end + 1]])


def _length(run: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())


def _poly(img: np.ndarray, pts_px: np.ndarray, colour: tuple[int, int, int], width: int, closed: bool) -> None:
    cv2.polylines(img, [np.round(pts_px).astype(np.int32)], closed, colour, width, cv2.LINE_AA)


def write_debug(
    path: Path, photo: str | Path, h: Homography, outline: Outline, half: Outline, edges: list[Edge],
    body: Outline | None = None, invented: np.ndarray | None = None, note: str | None = None,
) -> Path:
    image = cv2.imread(str(photo))
    width = max(3, round(max(image.shape[:2]) / 600))
    _poly(image, h.mm_to_px(outline.array()), (0, 220, 255), width, True)
    if body is not None:
        b = body.array()
        _poly(image, h.mm_to_px(b), (60, 200, 60), width, True)
        if invented is not None and invented.any():
            for p in h.mm_to_px(b[invented]):
                cv2.circle(image, (round(p[0]), round(p[1])), width + 1, (200, 0, 200), -1)
    scale = PANEL_PX / image.shape[0]
    left = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    panel = np.full((PANEL_PX, PANEL_PX, 3), 255, np.uint8)
    pts = half.array()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    s = (PANEL_PX - 260) / float(max(hi - lo))
    to_px = lambda p: (p - lo) * s + [40, 40]  # noqa: E731
    _poly(panel, to_px(pts), (200, 200, 200), 1, True)
    lines = []
    for i, e in enumerate(edges):
        colour = COLOURS[i % len(COLOURS)]
        run = edge_run(pts, e)
        _poly(panel, to_px(run), colour, 4, False)
        mid = to_px(run[len(run) // 2])
        cv2.putText(panel, e.name, (round(mid[0]) + 6, round(mid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        lines.append((f"{e.name}: {_length(run):.1f} mm", colour))
    for i, (text, colour) in enumerate(lines):
        cv2.putText(panel, text, (PANEL_PX - 250, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(panel, "half panel (fold on x=0), stitch line lengths", (40, PANEL_PX - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    if note:  # the lay correction, wrapped under the edge lengths
        words, line, y = note.split(), "", 40 + 30 * len(lines) + 20
        for word in words:
            if len(line) + len(word) > 34:
                cv2.putText(panel, line, (PANEL_PX - 330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 90, 200), 1)
                line, y = "", y + 22
            line += word + " "
        cv2.putText(panel, line, (PANEL_PX - 330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 90, 200), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack([left, panel]))
    return path
