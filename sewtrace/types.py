"""Shared data types. Every module talks in these.

Coordinate convention (everywhere, unless a function says otherwise):
  * units are millimetres
  * x grows to the right, y grows DOWN (same as image rows)
  * "on fold" panels (front, back, sleeve half) are stored as the HALF panel,
    with the fold line on x = 0 and the panel at x >= 0
  * outlines are closed polylines; the last point is NOT repeated
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

Point = tuple[float, float]

# The six ground-truth measurements (proof metric, parent page section 4).
MEASUREMENT_KEYS = (
    "chest_width",     # full body width at underarm level
    "body_length",     # high point shoulder (neck point) to hem, vertical
    "shoulder_width",  # shoulder point to shoulder point
    "sleeve_length",   # sleeve cap top to sleeve hem, along the fold
    "neck_width",      # neck point to neck point
    "armhole_depth",   # shoulder point to underarm, vertical
)

TEE_PANELS = ("front", "back", "sleeve", "neckband")

# Named edges per panel, in order around the outline.
TEE_EDGES = {
    "front": ("neck", "shoulder", "armhole", "side", "hem", "fold"),
    "back": ("neck", "shoulder", "armhole", "side", "hem", "fold"),
    "sleeve": ("cap", "underarm", "hem", "fold"),
}


class Outline(BaseModel):
    """Closed polyline in mm."""

    points: list[Point]

    def array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Outline":
        return cls(points=[(float(x), float(y)) for x, y in np.asarray(arr)])


class Edge(BaseModel):
    """A named run of an outline: points[start..end] inclusive, wrapping allowed."""

    name: str
    start: int
    end: int


class Panel(BaseModel):
    name: str
    cut: str  # e.g. "1 on fold", "2"
    outline_mm: list[Point]  # stitch line (no seam allowance), half panel if on fold
    edges: list[Edge] = Field(default_factory=list)
    seam_allowance_mm: float = 10.0
    grain: Literal["vertical", "horizontal"] = "vertical"
    notches: list[Point] = Field(default_factory=list)
    on_fold: bool = False
    length_rule: str | None = None  # neckband only, e.g. "85% of neckline"

    def outline(self) -> Outline:
        return Outline(points=self.outline_mm)

    def edge_points(self, name: str) -> np.ndarray:
        """Points of a named edge, in outline order (handles wrap-around)."""
        edge = next(e for e in self.edges if e.name == name)
        pts = np.asarray(self.outline_mm, dtype=float)
        if edge.start <= edge.end:
            return pts[edge.start : edge.end + 1]
        return np.vstack([pts[edge.start :], pts[: edge.end + 1]])


class Seam(BaseModel):
    """Two edges that get sewn together. Refs are "panel.edge"."""

    a: str
    b: str
    len_a_mm: float = 0.0
    len_b_mm: float = 0.0
    # Allowed b - a, in mm. Default is "must match within 3 mm".
    # Sleeve cap vs armhole carries ease, so it gets a positive range.
    min_diff_mm: float = -3.0
    max_diff_mm: float = 3.0

    @property
    def diff_mm(self) -> float:
        return self.len_b_mm - self.len_a_mm

    @property
    def ok(self) -> bool:
        return self.min_diff_mm <= self.diff_mm <= self.max_diff_mm


class PanelSpec(BaseModel):
    """The structured output of the reasoning step (parent page section 3)."""

    garment: str = "kids_round_neck_tee"
    panels: list[Panel]
    seams: list[Seam] = Field(default_factory=list)
    sewing_order: list[str] = Field(
        default_factory=lambda: ["shoulders", "neckband", "sleeves", "side seams", "hems"]
    )

    def panel(self, name: str) -> Panel:
        return next(p for p in self.panels if p.name == name)


class PhotoSet(BaseModel):
    """The three capture photos for one garment."""

    front: str
    back: str
    sleeve: str
