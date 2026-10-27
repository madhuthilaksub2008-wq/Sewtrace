"""T19: one garment, photos to pattern files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from sewtrace.api.client import MetaClient
from sewtrace.output.pdf import render_pdf
from sewtrace.output.vector import render_dxf, render_svg
from sewtrace.reason.panels import draft_spec, infer_panels
from sewtrace.reason.seams import adjust_loop
from sewtrace.types import Edge, Outline, PanelSpec, PhotoSet
from sewtrace.vision import contour
from sewtrace.vision.marker import photo_homography
from sewtrace.vision.measure import measure_tee


@dataclass
class RunResult:
    spec: PanelSpec
    measurements: dict[str, float]
    seam_log: list[dict]
    files: dict[str, Path] = field(default_factory=dict)
    model_calls: int = 0


def photo_set(garment_dir: Path) -> PhotoSet:
    return PhotoSet(**{v: str(garment_dir / f"{v}.jpg") for v in ("front", "back", "sleeve")})


def trace(client: MetaClient, photo: str, region: str, sleeve: bool = False) -> tuple[Outline, list[Edge]]:
    """Photo -> labelled half outline in mm."""
    h = photo_homography(cv2.imread(photo))
    outline = contour.smooth_outline(contour.mask_to_outline(client.segment(photo, region), h))
    if sleeve:
        half = contour.sleeve_half(outline)
        return half, contour.label_sleeve_edges(half)
    half = contour.body_half(outline)
    return half, contour.label_body_edges(half)


def run(garment_dir: str | Path, out_dir: str | Path, client: MetaClient | None = None) -> RunResult:
    garment_dir, out_dir = Path(garment_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or MetaClient()
    calls_before = client.calls
    photos = photo_set(garment_dir)

    front, front_edges = trace(client, photos.front, "body")
    back, back_edges = trace(client, photos.back, "body")
    sleeve, sleeve_edges = trace(client, photos.sleeve, "sleeve", sleeve=True)
    measurements = measure_tee(front, front_edges, sleeve, sleeve_edges)

    draft = draft_spec(front, front_edges, back, back_edges, sleeve, sleeve_edges)
    spec = infer_panels(client, photos, draft, measurements)
    spec, seam_log = adjust_loop(client, spec)

    files = {
        "pdf": render_pdf(spec, out_dir / "pattern.pdf"),
        "svg": render_svg(spec, out_dir / "pattern.svg"),
        "dxf": render_dxf(spec, out_dir / "pattern.dxf"),
        "spec": out_dir / "spec.json",
        "measurements": out_dir / "measurements.json",
        "seam_log": out_dir / "seam_log.json",
    }
    files["spec"].write_text(spec.model_dump_json(indent=1))
    files["measurements"].write_text(json.dumps(measurements, indent=1))
    files["seam_log"].write_text(json.dumps(seam_log, indent=1, default=str))
    return RunResult(spec, measurements, seam_log, files, client.calls - calls_before)
