"""T25: overlay proof. Paper FRONT laid on the tee; how far is its edge from the tee's?

Owner: Madhu. SAM masks "paper" and "body" on the proof photo, both outlines go
to the marker mm frame, and every paper edge point gets its distance to the
nearest body edge point.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import shapely
from pydantic import BaseModel

from sewtrace.api.client import MetaClient
from sewtrace.types import Point
from sewtrace.vision.marker import Homography

GAP_TOLERANCE_MM = 3.0


class ProofResult(BaseModel):
    """Paper-to-body edge gaps, mm. points_mm[i] has gap distances_mm[i]."""

    mean_mm: float
    max_mm: float
    p95_mm: float
    frac_within_tol: float
    tol_mm: float = GAP_TOLERANCE_MM
    points_mm: list[Point]
    distances_mm: list[float]


def mask_outline_mm(mask: np.ndarray, homography: Homography) -> np.ndarray:
    """(N, 2) largest external contour of a boolean mask, in mm."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    biggest = max(contours, key=cv2.contourArea)
    return homography.px_to_mm(biggest.reshape(-1, 2).astype(float))


def edge_gaps(paper_mm: np.ndarray, body_mm: np.ndarray, step_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Paper edge resampled every ``step_mm``, and each point's distance to the body edge."""
    ring = shapely.LinearRing(paper_mm)
    pts = shapely.line_interpolate_point(ring, np.arange(0, ring.length, step_mm))
    dist = shapely.distance(pts, shapely.LinearRing(body_mm))
    return shapely.get_coordinates(pts), np.asarray(dist, float)


def gap_result(points: np.ndarray, distances: np.ndarray, tol_mm: float = GAP_TOLERANCE_MM) -> ProofResult:
    return ProofResult(
        mean_mm=float(distances.mean()),
        max_mm=float(distances.max()),
        p95_mm=float(np.percentile(distances, 95)),
        frac_within_tol=float((distances < tol_mm).mean()),
        tol_mm=tol_mm,
        points_mm=[(float(x), float(y)) for x, y in points],
        distances_mm=[float(d) for d in distances],
    )


def verify_proof(
    photo: str | Path,
    client: MetaClient,
    homography: Homography,
    tol_mm: float = GAP_TOLERANCE_MM,
    step_mm: float = 2.0,
) -> ProofResult:
    """Gap between the paper edge and the tee body edge on a proof photo."""
    paper = mask_outline_mm(client.segment(photo, "paper"), homography)
    body = mask_outline_mm(client.segment(photo, "body"), homography)
    points, distances = edge_gaps(paper, body, step_mm)
    return gap_result(points, distances, tol_mm)


def overlay_image(photo: str | Path, result: ProofResult, homography: Homography, out_path: str | Path) -> Path:
    """Paper edge drawn on the photo, green where the gap is within tolerance, red elsewhere."""
    image = cv2.imread(str(photo))
    if image is None:
        raise FileNotFoundError(photo)
    px = homography.mm_to_px(np.asarray(result.points_mm))
    ok = np.asarray(result.distances_mm) < result.tol_mm
    thick = max(2, round(image.shape[1] / 600))
    for i in range(len(px)):
        a, b = px[i], px[(i + 1) % len(px)]
        colour = (40, 200, 40) if ok[i] else (30, 30, 230)
        cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, thick, cv2.LINE_AA)
    label = (f"mean gap {result.mean_mm:.1f} mm  max {result.max_mm:.1f} mm  "
             f"{100 * result.frac_within_tol:.0f}% within {result.tol_mm:g} mm")
    font_scale = image.shape[1] / 1600
    org = (int(0.03 * image.shape[1]), int(0.05 * image.shape[0]))
    cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3 * thick, cv2.LINE_AA)
    cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    out_path = Path(out_path)
    cv2.imwrite(str(out_path), image)
    return out_path
