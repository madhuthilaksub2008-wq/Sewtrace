"""Ground-truth PanelSpec for a synthetic tee, so downstream modules (seam check,
renderers) can be built and tested before the vision chain exists."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sewtrace.config import get_config
from sewtrace.types import Edge, Panel, PanelSpec, Seam

NECKBAND_RATIO = 0.85
NECKBAND_WIDTH_MM = 40.0  # cut width, folded in half


def _edges(raw: dict) -> list[Edge]:
    return [Edge(name=k, start=v[0], end=v[1]) for k, v in raw.items()]


def truth_spec(tee_dir: str | Path) -> PanelSpec:
    truth = json.loads((Path(tee_dir) / "truth.json").read_text())
    photos = truth["photos"]
    panels = []
    for name in ("front", "back"):
        panels.append(Panel(
            name=name, cut="1 on fold", on_fold=True,
            outline_mm=[tuple(p) for p in photos[name]["half_outline_mm"]],
            edges=_edges(photos[name]["edges"]),
        ))
    panels.append(Panel(
        name="sleeve", cut="2 on fold", on_fold=True,
        outline_mm=[tuple(p) for p in photos["sleeve"]["half_outline_mm"]],
        edges=_edges(photos["sleeve"]["edges"]),
    ))
    neck = sum(_len(panels[i].edge_points("neck")) for i in (0, 1)) * 2  # both halves
    length = NECKBAND_RATIO * neck
    panels.append(Panel(
        name="neckband", cut="1", seam_allowance_mm=get_config().pattern.neckband_allowance_mm, grain="horizontal",
        outline_mm=[(0, 0), (length, 0), (length, NECKBAND_WIDTH_MM), (0, NECKBAND_WIDTH_MM)],
        length_rule=f"{int(NECKBAND_RATIO * 100)}% of neckline",
    ))
    return PanelSpec(panels=panels, seams=default_tee_seams())


def default_tee_seams() -> list[Seam]:
    """Seam pairs of SPEC 7.6.1 with the tolerances from config."""
    c = get_config().seam
    ease = {"min_diff_mm": c.cap_ease_min_mm, "max_diff_mm": c.cap_ease_max_mm}
    return [
        Seam(a="front.shoulder", b="back.shoulder", min_diff_mm=-c.shoulder_tol_mm, max_diff_mm=c.shoulder_tol_mm),
        Seam(a="front.side", b="back.side", min_diff_mm=-c.side_tol_mm, max_diff_mm=c.side_tol_mm),
        # Half sleeve cap sews to one armhole.
        Seam(a="front.armhole", b="sleeve.cap", **ease),
        Seam(a="back.armhole", b="sleeve.cap", **ease),
    ]


def _len(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
