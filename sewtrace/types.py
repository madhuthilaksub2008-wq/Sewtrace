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
from pydantic import BaseModel, Field, computed_field

from sewtrace.config import get_config

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
    seam_allowance_mm: float = Field(default_factory=lambda: get_config().pattern.seam_allowance_mm)
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
    """Two edges that get sewn together. Refs are "panel.edge".

    `a` may sum edges ("front.armhole+back.armhole"). `b_count` is how many times
    the half-panel edge `b` occurs in the whole garment (2 for an on-fold sleeve cap).
    A ":mirror" suffix measures the edge on the mirrored half instead.
    """

    a: str
    b: str
    b_count: int = Field(default=1, ge=1)
    len_a_mm: float = 0.0
    len_b_mm: float = 0.0
    # Allowed b - a, in mm. Required: default_tee_seams() fills them from config.
    min_diff_mm: float
    max_diff_mm: float

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
    @computed_field  # type: ignore[prop-decorator]
    @property
    def sewing_order(self) -> list[str]:
        """One step per seam, in sewing order (REQ-OUT-06). Always follows ``seams``."""
        return sewing_steps(self)

    def panel(self, name: str) -> Panel:
        return next(p for p in self.panels if p.name == name)


class PhotoSet(BaseModel):
    """The three capture photos for one garment."""

    front: str
    back: str
    sleeve: str


# Sewing rank of a seam by the edge it joins: shoulders first (the neckband follows),
# then the sleeve goes in, then the long side and underarm seams.
_SEAM_RANK = (("shoulder", 0), ("armhole", 1), ("cap", 1), ("side", 2), ("underarm", 3))


def _ref_label(ref: str) -> str:
    """"front.armhole+back.armhole" -> "front armhole and back armhole"."""
    parts = [r.removesuffix(":mirror").replace(".", " ") for r in ref.split("+")]
    return " and ".join(parts)


def _seam_rank(seam: Seam) -> int:
    return next((rank for key, rank in _SEAM_RANK if key in seam.a or key in seam.b), len(_SEAM_RANK))


def sewing_steps(spec: PanelSpec) -> list[str]:
    """Sewing sequence as text. Each seam is its own step (the side seam and the sleeve
    underarm are not merged), then the neckband after the shoulders and the hems last."""
    steps: list[tuple[int, str]] = []
    for seam in spec.seams:
        if seam.a == seam.b.removesuffix(":mirror"):
            text = f"Sew the {_ref_label(seam.a)} seam on both sides"
        else:
            text = f"Sew {_ref_label(seam.a)} to {_ref_label(seam.b)}"
        steps.append((_seam_rank(seam), text))
    if any(p.name == "neckband" for p in spec.panels):
        steps.append((0, "Join the neckband ends, then sew the neckband to the neckline"))
    steps.sort(key=lambda t: t[0])  # stable: the neckband follows the shoulder seams
    return [text for _, text in steps] + ["Turn up and sew the body hem and the sleeve hems"]
