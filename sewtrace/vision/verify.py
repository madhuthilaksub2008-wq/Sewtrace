"""T25: overlay proof. Paper FRONT laid on the tee; how far is its edge from the tee's?

Owner: Madhu. SPEC 9.4: the paper is a HALF front panel cut on the stitch line, with its
fold on the centre of the t-shirt. SAM masks "paper" and "t-shirt" (REQ-SEG-02) go to the
marker mm frame. The t-shirt outline includes the sleeves, so they are cut off first, as
for the front and back photos. Then each paper edge point gets its distance to the nearest
body edge point, with four rules:

  * The fold edge of the paper is not a cut edge. It lies on the centre line, far inside
    the t-shirt edge, so it is found and left out (drawn grey in the overlay).
  * A paper point is compared with the body edge on its own side of the fold line only.
  * The armhole of the body outline is a default shape, because the sleeve hides the real
    one. Paper points nearest to it are left out, not compared.
  * On a front flat-lay the "t-shirt" mask includes the back panel seen through the neck
    opening, so the outer outline follows the BACK neckline, which sits higher than the
    front one. The paper's front neck edge is lower by (front depth - back depth), so its
    gap is that difference, not a cutting error. Paper points nearest to the body neck
    are left out of the mean and reported on their own as information (REQ-VER-02).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import shapely
from pydantic import BaseModel
from scipy.spatial import cKDTree

from sewtrace.api.client import MetaClient
from sewtrace.config import get_config
from sewtrace.types import Outline, Point
from sewtrace.vision import contour
from sewtrace.vision.marker import Homography


class ProofResult(BaseModel):
    """Paper-to-body edge gaps, mm. points_mm[i] has gap distances_mm[i]."""

    mean_mm: float
    max_mm: float
    p95_mm: float
    frac_within_tol: float
    tol_mm: float
    points_mm: list[Point]
    distances_mm: list[float]
    fold_mm: tuple[Point, Point] | None = None  # excluded fold edge, end points
    step_mm: float = 2.0  # spacing of points_mm along the paper edge
    # Paper neck points: left out of the numbers above, kept as information. Their gap is
    # front neck depth minus back neck depth, not a cutting error.
    neck_points_mm: list[Point] = []
    neck_distances_mm: list[float] = []

    @property
    def neck_mean_mm(self) -> float | None:
        return float(np.mean(self.neck_distances_mm)) if self.neck_distances_mm else None


def mask_outline_mm(mask: np.ndarray, homography: Homography) -> np.ndarray:
    """(N, 2) largest external contour of a boolean mask, in mm."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    biggest = max(contours, key=cv2.contourArea)
    return homography.px_to_mm(biggest.reshape(-1, 2).astype(float))


def find_fold_edge(paper_mm: np.ndarray, body_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End points of the paper's fold edge: its long straight edge that lies deepest inside
    the t-shirt outline (the centre line is about half a chest away from any t-shirt edge)."""
    cfg = get_config().proof
    simple = np.asarray(shapely.simplify(shapely.LinearRing(paper_mm), cfg.fold_straight_tol_mm).coords)
    body_ring = shapely.LinearRing(body_mm)
    min_length = 0.5 * float(np.ptp(paper_mm, axis=0).max())
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for a, b in zip(simple[:-1], simple[1:]):
        if np.linalg.norm(b - a) < min_length:
            continue
        samples = shapely.line_interpolate_point(shapely.LineString([a, b]), np.linspace(0, 1, 21), normalized=True)
        inset = float(np.median(shapely.distance(samples, body_ring)))
        if inset >= cfg.fold_min_inset_mm and (best is None or inset > best[0]):
            best = (inset, a, b)
    if best is None:
        raise ValueError(
            "No fold edge found on the paper: put the fold of the half panel on the centre of the t-shirt (SPEC 9.4)"
        )
    return best[1], best[2]


def _body_side_runs(body_mm: np.ndarray, keep: np.ndarray) -> shapely.Geometry:
    """Body edge restricted to the points flagged ``keep``, as open lines."""
    if keep.all():
        return shapely.LinearRing(body_mm)
    order = np.roll(np.arange(len(body_mm)), -int(np.argmin(keep)))  # start on a dropped point
    lines: list[list[np.ndarray]] = []
    run: list[np.ndarray] = []
    for point, flag in zip(body_mm[order], keep[order]):
        if flag:
            run.append(point)
        elif run:
            lines.append(run)
            run = []
    if run:
        lines.append(run)
    return shapely.MultiLineString([r for r in lines if len(r) >= 2])


def edge_gaps(
    paper_mm: np.ndarray,
    body_mm: np.ndarray,
    step_mm: float = 2.0,
    hidden: np.ndarray | None = None,
    fold: tuple[np.ndarray, np.ndarray] | None = None,
    neck: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper edge resampled every ``step_mm``, and each point's distance to the body edge.

    ``hidden`` flags body points that are not a visible edge (the armhole, which the
    sleeve covers). ``neck`` flags body points on the neck edge, which on a front photo is
    the back neckline. Paper points whose nearest body point is flagged are left out.
    ``fold`` is the paper's fold edge: its points are left out, and the body edge is cut
    to the paper's side of the fold line (``proof.centre_margin_mm`` past it).
    """
    points, distances, _, _ = edge_gaps_split(paper_mm, body_mm, step_mm, hidden, fold, neck)
    return points, distances


def edge_gaps_split(
    paper_mm: np.ndarray,
    body_mm: np.ndarray,
    step_mm: float = 2.0,
    hidden: np.ndarray | None = None,
    fold: tuple[np.ndarray, np.ndarray] | None = None,
    neck: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """edge_gaps(), plus the left-out neck points: (points, gaps, neck points, neck gaps)."""
    cfg = get_config().proof
    ring = shapely.LinearRing(paper_mm)
    pts = shapely.line_interpolate_point(ring, np.arange(0, ring.length, step_mm))
    xy = shapely.get_coordinates(pts)
    body_pts = np.asarray(body_mm, float)
    keep_body = np.ones(len(body_pts), bool)
    keep_paper = np.ones(len(xy), bool)
    if fold is not None:
        p0, p1 = (np.asarray(f, float) for f in fold)
        direction = (p1 - p0) / np.linalg.norm(p1 - p0)

        def side(q: np.ndarray) -> np.ndarray:  # signed mm from the fold line
            return direction[0] * (q[:, 1] - p0[1]) - direction[1] * (q[:, 0] - p0[0])

        paper_side = np.sign(side(np.asarray(ring.centroid.coords)))[0]
        keep_body = paper_side * side(body_pts) >= -cfg.centre_margin_mm
        keep_paper = shapely.distance(pts, shapely.LineString([p0, p1])) >= cfg.fold_exclude_mm
    body_edge = _body_side_runs(body_pts, keep_body)
    dist = np.asarray(shapely.distance(pts, body_edge), float)
    near = cKDTree(body_pts[keep_body]).query(xy)[1]
    if hidden is not None:
        keep_paper &= ~np.asarray(hidden)[keep_body][near]
    on_neck = np.zeros(len(xy), bool)
    if neck is not None:
        on_neck = keep_paper & np.asarray(neck)[keep_body][near]
        keep_paper &= ~on_neck
    return xy[keep_paper], dist[keep_paper], xy[on_neck], dist[on_neck]


def gap_result(
    points: np.ndarray,
    distances: np.ndarray,
    tol_mm: float | None = None,
    fold: tuple[np.ndarray, np.ndarray] | None = None,
    step_mm: float = 2.0,
    neck_points: np.ndarray | None = None,
    neck_distances: np.ndarray | None = None,
) -> ProofResult:
    tol_mm = get_config().proof.gap_tolerance_mm if tol_mm is None else tol_mm
    return ProofResult(
        mean_mm=float(distances.mean()),
        max_mm=float(distances.max()),
        p95_mm=float(np.percentile(distances, 95)),
        frac_within_tol=float((distances < tol_mm).mean()),
        tol_mm=tol_mm,
        points_mm=[(float(x), float(y)) for x, y in points],
        distances_mm=[float(d) for d in distances],
        fold_mm=None if fold is None else ((float(fold[0][0]), float(fold[0][1])), (float(fold[1][0]), float(fold[1][1]))),
        step_mm=step_mm,
        neck_points_mm=[] if neck_points is None else [(float(x), float(y)) for x, y in neck_points],
        neck_distances_mm=[] if neck_distances is None else [float(d) for d in neck_distances],
    )


def verify_proof(
    photo: str | Path,
    client: MetaClient,
    homography: Homography,
    tol_mm: float | None = None,
    step_mm: float = 2.0,
) -> ProofResult:
    """Gap between the paper edge (fold edge excluded) and the tee body edge on a proof photo."""
    prompts = client.config.sam.prompts
    paper = mask_outline_mm(client.segment(photo, prompts.paper), homography)
    garment = mask_outline_mm(client.segment(photo, prompts.garment), homography)
    body, armhole, neck = contour.cut_sleeves_flagged(contour.smooth_outline(Outline.from_array(garment)))
    fold = find_fold_edge(paper, body.array())
    points, distances, neck_pts, neck_dist = edge_gaps_split(
        paper, body.array(), step_mm, hidden=armhole, fold=fold, neck=neck)
    return gap_result(points, distances, tol_mm, fold, step_mm, neck_pts, neck_dist)


def overlay_image(photo: str | Path, result: ProofResult, homography: Homography, out_path: str | Path) -> Path:
    """Paper edge drawn on the photo, green where the gap is within tolerance, red elsewhere.
    The fold edge and the neck are drawn grey: they are not compared.
    Points left out (hidden armhole) are not drawn."""
    image = cv2.imread(str(photo))
    if image is None:
        raise FileNotFoundError(photo)
    px = homography.mm_to_px(np.asarray(result.points_mm))
    ok = np.asarray(result.distances_mm) < result.tol_mm
    thick = max(2, round(image.shape[1] / 600))
    for i in range(len(px)):
        a, b = px[i], px[(i + 1) % len(px)]
        if np.linalg.norm(np.asarray(result.points_mm[i]) - result.points_mm[(i + 1) % len(px)]) > 2.5 * result.step_mm:
            continue  # a stretch that was left out lies between these points
        colour = (40, 200, 40) if ok[i] else (30, 30, 230)
        cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, thick, cv2.LINE_AA)
    for x, y in np.round(homography.mm_to_px(np.asarray(result.neck_points_mm).reshape(-1, 2))).astype(int):
        cv2.circle(image, (int(x), int(y)), thick, (150, 150, 150), -1, cv2.LINE_AA)  # information only
    if result.fold_mm is not None:
        f0, f1 = np.round(homography.mm_to_px(np.asarray(result.fold_mm))).astype(int)
        cv2.line(image, tuple(f0), tuple(f1), (150, 150, 150), thick, cv2.LINE_AA)
    label = (f"mean gap {result.mean_mm:.1f} mm  max {result.max_mm:.1f} mm  "
             f"{100 * result.frac_within_tol:.0f}% within {result.tol_mm:g} mm")
    font_scale = image.shape[1] / 1600
    org = (int(0.03 * image.shape[1]), int(0.05 * image.shape[0]))
    cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3 * thick, cv2.LINE_AA)
    cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    out_path = Path(out_path)
    cv2.imwrite(str(out_path), image)
    return out_path
