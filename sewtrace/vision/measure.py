"""T15.4: the six proof measurements from labelled half outlines."""

from __future__ import annotations

import numpy as np

from sewtrace.config import get_config
from sewtrace.types import Edge, Outline


def _edge(pts: np.ndarray, edges: list[Edge], name: str) -> np.ndarray:
    e = next(e for e in edges if e.name == name)
    if e.start <= e.end:
        return pts[e.start : e.end + 1]
    return np.vstack([pts[e.start :], pts[: e.end + 1]])


def _interior(run: np.ndarray, trim: float = 0.2) -> np.ndarray:
    n = len(run)
    a, b = int(n * trim), max(int(n * (1 - trim)), int(n * trim) + 2)
    return run[a:b]


def _x_at(run: np.ndarray, y: float) -> float:
    """x of a straight, roughly vertical run (line fit to its interior) at height y."""
    mid = _interior(run)
    slope, icept = np.polyfit(mid[:, 1], mid[:, 0], 1)
    return float(slope * y + icept)


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit direction of the total least squares line through pts."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[0]


def _near_end(run: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Points of run whose arc distance from its last point is within [lo, hi]."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(run[::-1], axis=0), axis=1))][::-1]
    pick = run[(s >= lo) & (s <= hi)]
    return pick if len(pick) >= 2 else run[-3:]


def _corner(into: np.ndarray, out_of: np.ndarray, reach_in: float = 8.0, reach_out: float = 8.0) -> np.ndarray:
    """Corner between two edges: intersection of tangent lines fitted to each edge
    over arc lengths 1..reach mm from the corner."""
    c1, d1 = _fit_line(_near_end(into, 1.0, reach_in))
    c2, d2 = _fit_line(_near_end(out_of[::-1], 1.0, reach_out))
    A = np.c_[d1, -d2]
    if abs(np.linalg.det(A)) < 1e-6:
        return into[-1]
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def _neck_point(neck: np.ndarray, shoulder: np.ndarray) -> np.ndarray:
    """Neck point: where the neck curve meets the shoulder line.

    A straight tangent is biased on a tight neck curve (a back neckline seen through the
    neck opening on a front photo has a radius of about 7 mm at the neck point), so the
    neck curve is fitted as x = f(y), a parabola, and intersected with the shoulder line.
    Falls back to the tangent corner when the fit is not possible or the best intersection
    lies at the edge of the search window (then the window did not bracket it)."""
    point = _neck_point_fit(neck, shoulder)
    if point is None:
        return _corner(neck, shoulder, reach_in=4.0, reach_out=30.0)
    return point


def _neck_point_fit(neck: np.ndarray, shoulder: np.ndarray) -> np.ndarray | None:
    """Parabola fit of the neck curve against the shoulder line; None if it is not usable.

    The search runs +/- ``measure.neck_search_mm`` along the shoulder line from the neck end of
    the shoulder edge (the line is fitted over its first 30 mm, so its centroid is about 15 mm
    away from the neck point and is not a good centre)."""
    cfg = get_config().measure
    run = _near_end(neck, 1.0, cfg.neck_fit_mm)
    if len(run) < 4 or np.ptp(run[:, 1]) < 1.0:
        return None
    c, d = _fit_line(_near_end(shoulder[::-1], 1.0, 30.0))
    start = c + float((shoulder[0] - c) @ d) * d  # neck end of the shoulder edge, on the fitted line
    fit = np.polyfit(run[:, 1], run[:, 0], 2)
    t = np.linspace(-cfg.neck_search_mm, cfg.neck_search_mm, 6001)
    line = start + t[:, None] * d
    t_best = int(np.argmin(np.abs(np.polyval(fit, line[:, 1]) - line[:, 0])))
    if t_best in (0, len(t) - 1):
        return None
    return np.array([np.polyval(fit, line[t_best, 1]), line[t_best, 1]])


def _shoulder_point(pts: np.ndarray, edges: list[Edge]) -> np.ndarray:
    """Shoulder seam point: where the shoulder edge meets the armhole. Corners come from
    end tangents, so smoothing that rounds a vertex does not shift them."""
    return _corner(_edge(pts, edges, "shoulder"), _edge(pts, edges, "armhole"), reach_in=30.0, reach_out=4.0)


def measure_tee(
    front: Outline,
    front_edges: list[Edge],
    back: Outline,
    back_edges: list[Edge],
    sleeve: Outline,
    sleeve_edges: list[Edge],
) -> dict[str, float]:
    """Keys = sewtrace.types.MEASUREMENT_KEYS, values in mm (SPEC 10.2, REQ-GEO-04).

    Chest, body length, neck width and armhole depth come from the front, the shoulder
    from the back and the sleeve length from the sleeve fold, as SPEC 10.2 says."""
    f = front.array()
    neck = _edge(f, front_edges, "neck")
    shoulder = _edge(f, front_edges, "shoulder")
    armhole = _edge(f, front_edges, "armhole")
    side = _edge(f, front_edges, "side")
    neck_pt = _neck_point(neck, shoulder)
    shoulder_pt = _shoulder_point(f, front_edges)
    underarm = _corner(armhole, side, reach_in=4.0, reach_out=30.0)
    hem_y = float(np.median(_interior(_edge(f, front_edges, "hem"))[:, 1]))
    chest_y = float(underarm[1]) + get_config().pattern.chest_drop_mm

    back_shoulder_pt = _shoulder_point(back.array(), back_edges)

    s = sleeve.array()
    cap_top = s[next(e for e in sleeve_edges if e.name == "cap").start]
    sleeve_hem_y = float(np.median(_interior(_edge(s, sleeve_edges, "hem"))[:, 1]))

    return {
        "chest_width": 2 * _x_at(side, chest_y),
        "body_length": hem_y - float(neck_pt[1]),
        "shoulder_width": 2 * float(back_shoulder_pt[0]),
        "sleeve_length": sleeve_hem_y - float(cap_top[1]),
        "neck_width": 2 * float(neck_pt[0]),
        "armhole_depth": float(underarm[1] - shoulder_pt[1]),
    }
