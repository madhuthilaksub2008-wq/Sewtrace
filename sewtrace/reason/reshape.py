"""Deterministic reshaping of the two inferred edges: the neck curve and the sleeve cap.

The model never returns polylines (REQ-RSN-04). It returns a few numbers, and this
module turns them into geometry. Everything else in a panel stays as measured.

Sleeve cap shape: a cubic Bezier from the cap top P0 to the cap end P3 (the underarm
corner, which comes from the photo). Both end tangents are horizontal, as in the
draft. The two numbers are the handle lengths:
    P1 = P0 + (crown_run_mm, 0)      P2 = P3 - (underarm_run_mm, 0)
So the model never moves the cap end points or the sleeve width. Only the code fix may move the cap end
(``with_flat_cap``): when no curve between the photo's end points reaches the arm hole length, the
sleeve's top edge is drawn straight and widened to it.

Neck depth: the draft neck curve is scaled in y about the neck point. The neck point
and every x value stay as measured, so the neck width is bound by the photo.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sewtrace.types import TEE_EDGES, Edge, Panel

CAP_STEP_MM = 2.0
FOLD_GAP_MM = 0.5  # a fold vertex closer than this to the new neck start is dropped
_DEFAULT_RUN_FRAC = 0.45  # handle length as a fraction of the cap width when none is known


class SleeveCap(BaseModel):
    """Sleeve cap shape: the two Bezier handle lengths, in mm (half sleeve)."""

    model_config = ConfigDict(extra="forbid")

    crown_run_mm: float = Field(ge=0, description="Handle length at the cap top; larger is a flatter, wider crown.")
    underarm_run_mm: float = Field(ge=0, description="Handle length at the underarm end of the cap.")


def cap_curve(start: np.ndarray, end: np.ndarray, cap: SleeveCap, step_mm: float = CAP_STEP_MM) -> np.ndarray:
    """Cap polyline from `start` (cap top) to `end` (underarm corner)."""
    p1 = start + np.array([cap.crown_run_mm, 0.0])
    p2 = end - np.array([cap.underarm_run_mm, 0.0])
    ctrl = np.array([start, p1, p2, end])
    n = max(int(np.ceil(np.linalg.norm(np.diff(ctrl, axis=0), axis=1).sum() / step_mm)), 8)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1]
            + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])


def cap_width_mm(panel: Panel) -> float:
    """Horizontal distance from the cap top to the cap end: the largest handle that makes sense."""
    cap = panel.edge_points("cap")
    return float(abs(cap[-1, 0] - cap[0, 0]))


def fit_cap(points: np.ndarray) -> SleeveCap:
    """Handle lengths whose Bezier is closest (two-way max distance) to a cap polyline."""
    start, end = points[0], points[-1]
    width = abs(end[0] - start[0])
    ref = _resample(points, 60)

    def error(a: float, b: float) -> float:
        curve = cap_curve(start, end, SleeveCap(crown_run_mm=a, underarm_run_mm=b), step_mm=1.0)
        d = np.linalg.norm(ref[:, None, :] - curve[None, :, :], axis=2)
        return float(max(d.min(axis=1).max(), d.min(axis=0).max()))

    best, step = (0.5 * width, 0.5 * width), width / 10
    for _ in range(6):
        grid = [(float(np.clip(best[0] + i * step, 0, width)), float(np.clip(best[1] + j * step, 0, width)))
                for i in range(-5, 6) for j in range(-5, 6)]
        best = min(grid, key=lambda ab: error(*ab))
        step /= 4
    return SleeveCap(crown_run_mm=round(best[0], 1), underarm_run_mm=round(best[1], 1))


def scale_cap_to_length(points: np.ndarray, cap: SleeveCap, target_mm: float, iters: int = 50) -> SleeveCap:
    """Scale both handles by one factor until the cap is `target_mm` long (closest reachable).

    Handles stay within [0, cap width], so the result can miss the target; callers
    check the seam afterwards and fail loudly.
    """
    start, end = points[0], points[-1]
    width = abs(end[0] - start[0])
    if cap.crown_run_mm <= 0 and cap.underarm_run_mm <= 0:
        cap = SleeveCap(crown_run_mm=_DEFAULT_RUN_FRAC * width, underarm_run_mm=_DEFAULT_RUN_FRAC * width)
    longest = max(cap.crown_run_mm, cap.underarm_run_mm)
    k_max = width / longest

    def scaled(k: float) -> SleeveCap:
        return SleeveCap(crown_run_mm=cap.crown_run_mm * k, underarm_run_mm=cap.underarm_run_mm * k)

    def length(k: float) -> float:
        return _length(cap_curve(start, end, scaled(k)))

    lo, hi = 0.0, k_max
    if length(hi) <= target_mm:
        return scaled(hi)
    if length(lo) >= target_mm:
        return scaled(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if length(mid) < target_mm else (lo, mid)
    return scaled((lo + hi) / 2)


def cap_length_at(points: np.ndarray, frac: float) -> float:
    """Cap length between these end points with both handles at ``frac`` of the cap width."""
    start, end = points[0], points[-1]
    run = frac * abs(end[0] - start[0])
    return _length(cap_curve(start, end, SleeveCap(crown_run_mm=run, underarm_run_mm=run)))


def with_straight_hem(panel: Panel) -> Panel:
    """Sleeve with its hem (cuff) drawn straight between its two corners. The photo's cuff edge carries
    creases and wrinkles of the fabric (real kid tee: a kink near the underarm); a cuff is cut straight."""
    hem = panel.edge_points("hem")
    chord, rel = hem[-1] - hem[0], hem - hem[0]
    if np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]).max() / max(np.linalg.norm(chord), 1e-9) < 1.0:
        return panel  # already straight: keep the points as they are
    n = max(int(np.ceil(np.linalg.norm(hem[-1] - hem[0]) / CAP_STEP_MM)), 2)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return replace_edges(panel, {"hem": hem[0] + t * (hem[-1] - hem[0])})


def max_cap_length(points: np.ndarray) -> float:
    """Longest cap the handles can give between these end points (both handles at the cap width)."""
    return cap_length_at(points, 1.0)


def with_flat_cap(panel: Panel, half_cap_mm: float, rise_mm: float) -> Panel:
    """Sleeve with a straight top edge (the half sleeve hole) ``half_cap_mm`` long, rising ``rise_mm`` from
    the cap top on the fold to the underarm corner, and the underarm seam straight from there to the cuff
    corner. The fold and the cuff stay as measured. A T-shirt's sleeve is set in along a nearly straight
    seam; a flat photo shows that seam bunched, so its length comes from the arm hole on the body."""
    cap = panel.edge_points("cap")
    cuff = panel.edge_points("underarm")[-1]
    top = cap[0]
    across = float(np.sqrt(max(half_cap_mm**2 - rise_mm**2, 1.0)))
    end = top + np.array([across, rise_mm])
    if end[1] > cuff[1] - 1.0:
        raise ValueError("sleeve: the cap end would lie below the cuff")

    def line(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = max(int(np.ceil(np.linalg.norm(b - a) / CAP_STEP_MM)), 2)
        return a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)

    return replace_edges(panel, {"cap": line(top, end), "underarm": line(end, cuff)})


def with_sleeve_cap(panel: Panel, cap: SleeveCap) -> Panel:
    """Sleeve panel with its cap edge rebuilt from `cap`; the cap end points are kept."""
    pts = panel.edge_points("cap")
    return replace_edges(panel, {"cap": cap_curve(pts[0], pts[-1], cap)})


def neck_depth_mm(panel: Panel) -> float:
    """Neck depth: vertical distance from the neck point to the neck curve's fold end."""
    neck = panel.edge_points("neck")
    return float(neck[0, 1] - neck[-1, 1])


def with_neck_depth(panel: Panel, depth_mm: float) -> Panel:
    """Body panel with the neck curve scaled in y so its fold end sits `depth_mm` below the neck point.

    The fold ends where the neck curve starts, so a deeper neck trims the top of the fold."""
    neck = panel.edge_points("neck")
    current = neck[0, 1] - neck[-1, 1]
    if current <= 0:
        raise ValueError(f"{panel.name}: neck curve has no depth to scale")
    scaled = neck.copy()
    scaled[:, 1] = neck[-1, 1] + (neck[:, 1] - neck[-1, 1]) * depth_mm / current
    fold = panel.edge_points("fold")
    below = fold[fold[:, 1] > scaled[0, 1] + FOLD_GAP_MM]  # y grows downward
    new_fold = np.vstack([below, scaled[0]]) if len(below) else np.vstack([fold[0], scaled[0]])
    return replace_edges(panel, {"neck": scaled, "fold": new_fold})


def replace_edges(panel: Panel, new: dict[str, np.ndarray]) -> Panel:
    """Panel with some edge polylines swapped. Each edge starts where the previous one ends,
    so a changed first point also moves the previous edge's last point (the fold's neck end)."""
    names = TEE_EDGES[panel.name]
    segs = [np.asarray(new[n] if n in new else panel.edge_points(n), dtype=float) for n in names]
    pts: list[np.ndarray] = []
    edges: list[Edge] = []
    for name, seg in zip(names, segs):
        start = len(pts)
        pts.extend(seg[:-1])
        edges.append(Edge(name=name, start=start, end=len(pts)))
    edges[-1] = Edge(name=edges[-1].name, start=edges[-1].start, end=0)
    return panel.model_copy(update={"outline_mm": [(float(x), float(y)) for x, y in pts], "edges": edges})


def _length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, cum[-1], n)
    return np.c_[np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]
