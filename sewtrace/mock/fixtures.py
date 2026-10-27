"""Ground-truth PanelSpec for a synthetic tee, so downstream modules (seam check,
renderers) can be built and tested before the vision chain exists."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
        name="neckband", cut="1", seam_allowance_mm=6.0, grain="horizontal",
        outline_mm=[(0, 0), (length, 0), (length, NECKBAND_WIDTH_MM), (0, NECKBAND_WIDTH_MM)],
        length_rule=f"{int(NECKBAND_RATIO * 100)}% of neckline",
    ))
    return PanelSpec(panels=panels, seams=default_tee_seams())


def default_tee_seams() -> list[Seam]:
    return [
        Seam(a="front.shoulder", b="back.shoulder"),
        Seam(a="front.side", b="back.side"),
        # Half sleeve cap sews to one armhole; knit cap ease 0 to 12 mm (pending decision).
        Seam(a="front.armhole", b="sleeve.cap", min_diff_mm=0.0, max_diff_mm=12.0),
        Seam(a="back.armhole", b="sleeve.cap", min_diff_mm=0.0, max_diff_mm=12.0),
    ]


def _len(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
