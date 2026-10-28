"""Test helper: lengthen or shorten an edge by sliding its end vertex, to build broken specs.

Production code never does this (REQ-RSN-05): it would edit a measured edge.
"""

from __future__ import annotations

import numpy as np

from sewtrace.reason.seams import edge_length
from sewtrace.types import Panel


def set_edge_length(panel: Panel, edge: str, target: float, iters: int = 50) -> Panel:
    """Slide the edge's end vertex along its tangent until the edge is `target` mm long."""
    pts = panel.edge_points(edge)
    tangent = pts[-1] - pts[-2]
    tangent /= np.linalg.norm(tangent)
    span = 0.5 * edge_length(panel, edge)
    lo, hi = -span, span

    def length_at(t: float) -> float:
        return edge_length(move_edge_end(panel, edge, t * tangent), edge)

    if length_at(lo) >= target:
        hi = lo
    elif length_at(hi) <= target:
        lo = hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if length_at(mid) < target else (lo, mid)
    return move_edge_end(panel, edge, (lo + hi) / 2 * tangent)


def move_edge_end(panel: Panel, edge: str, delta: np.ndarray) -> Panel:
    """Displace the edge's end vertex by delta; this edge and the next follow linearly."""
    pts = np.asarray(panel.outline_mm, dtype=float)
    e = next(x for x in panel.edges if x.name == edge)
    nxt = next(x for x in panel.edges if x.start == e.end and x.name != edge)
    new = pts.copy()
    for ed, rising in ((e, True), (nxt, False)):
        idx = _indices(ed.start, ed.end, len(pts))
        seg = pts[idx]
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        frac = cum / cum[-1] if cum[-1] > 0 else np.linspace(0, 1, len(seg))
        w = frac if rising else 1 - frac
        new[idx] = seg + w[:, None] * delta
    return panel.model_copy(update={"outline_mm": [(float(x), float(y)) for x, y in new]})


def _indices(start: int, end: int, n: int) -> list[int]:
    return list(range(start, end + 1)) if start <= end else list(range(start, n)) + list(range(end + 1))


def bulge_edge(panel: Panel, edge: str, extra_mm: float, iters: int = 50) -> Panel:
    """Push the edge's inner points sideways (end points fixed) until it is `extra_mm` longer."""
    pts = np.asarray(panel.outline_mm, dtype=float)
    e = next(x for x in panel.edges if x.name == edge)
    idx = _indices(e.start, e.end, len(pts))
    seg = pts[idx]
    chord = seg[-1] - seg[0]
    normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    u = np.linspace(0, 1, len(seg))
    shape = np.sin(np.pi * u)[:, None] * normal

    def bulged(h: float) -> Panel:
        new = pts.copy()
        new[idx] = seg + h * shape
        return panel.model_copy(update={"outline_mm": [(float(x), float(y)) for x, y in new]})

    target = edge_length(panel, edge) + extra_mm
    lo, hi = 0.0, 200.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if edge_length(bulged(mid), edge) < target else (lo, mid)
    return bulged((lo + hi) / 2)
