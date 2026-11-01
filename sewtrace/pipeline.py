"""T19: one garment, photos to pattern files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sewtrace.api.client import CacheMiss, MetaClient
from sewtrace.output.pdf import render_pdf
from sewtrace.output.vector import render_dxf, render_svg
from sewtrace.reason.panels import draft_spec, infer_panels, refresh_derived
from sewtrace.reason.seams import SeamCheckFailed, adjust_loop, require_seams_ok, set_measured_pairs
from sewtrace.types import Edge, Outline, PanelSpec, PhotoSet
from sewtrace.vision import contour
from sewtrace.vision.marker import Homography, photo_homography
from sewtrace.vision.measure import measure_tee

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    spec: PanelSpec
    measurements: dict[str, float]
    seam_log: list[dict]
    files: dict[str, Path] = field(default_factory=dict)
    model_calls: int = 0


def photo_set(garment_dir: Path) -> PhotoSet:
    return PhotoSet(**{v: str(garment_dir / f"{v}.jpg") for v in ("front", "back", "sleeve")})


class PhotoMissing(FileNotFoundError):
    """A capture photo is missing or is not an image (REQ-CAP-01)."""


class PanelInferenceFailed(ValueError):
    """The panels cannot be inferred from the measured outlines (REQ-RSN-03)."""


class ContourFailed(ValueError):
    """The outline of a photo could not be split into panels (REQ-GEO-01). Names the photo."""


def load_photo(view: str, path: str | Path) -> np.ndarray:
    """Read one capture photo; the error names the view and the path."""
    image = cv2.imread(str(path)) if Path(path).is_file() else None
    if image is None:
        raise PhotoMissing(f"{view} photo {path} is missing or is not an image. Put the file there and run again.")
    return image


def photo_scale(view: str, path: str | Path) -> Homography:
    """Photo px -> mm for one capture photo. Raises, naming the photo, when the file is
    missing (REQ-CAP-01), the marker is not found (REQ-CAP-02) or it is off by more than
    marker.tolerance_mm (REQ-SCL-03)."""
    return photo_homography(load_photo(view, path), f"{view} photo {path}")


def sleeve_seams(client: MetaClient, photo: str | Path, garment: np.ndarray, h: Homography) -> list | None:
    """Shoulder and underarm points from where SAM's two sleeve masks meet the body (REQ-GEO-01).
    One extra SAM call per photo (cached). None, so the outline method is used, in mock mode,
    offline without a stored answer, or when SAM does not give two sleeves that touch the body."""
    if client.settings.mock:
        return None  # mock SAM has no per-record sleeve masks
    try:
        return contour.seam_points(garment, client.segment_all(photo, client.config.sam.prompts.sleeve), h)
    except (CacheMiss, ValueError) as err:
        log.warning("sleeve seams from SAM not used for %s (%s); finding them from the outline", photo, err)
        return None


def body_without_sleeves(
    client: MetaClient, photo: str | Path, garment: np.ndarray, h: Homography, outline: Outline,
) -> tuple[Outline, np.ndarray, np.ndarray]:
    """``contour.cut_sleeves_flagged`` with the sleeve seams from SAM when they can be trusted, else
    from the outline shape. A SAM seam that disagrees with the outline is logged and not used; a
    refusal of the corner search (ContourRefused) is raised, since the outline method needs it too."""
    seams = sleeve_seams(client, photo, garment, h)
    if seams is not None:
        try:
            cut = contour.cut_sleeves_flagged(outline, seams)
            log.info("sleeve seams from SAM used for %s", photo)
            return cut
        except contour.ContourRefused:
            raise
        except ValueError as err:
            log.warning("sleeve seams from SAM rejected for %s (%s); finding them from the outline", photo, err)
    return contour.cut_sleeves_flagged(outline)


def _debug(debug_dir: Path | None, view: str, photo: str, h: Homography, *args) -> None:
    """Write debug_<view>.png; a failure to draw never fails the run."""
    if debug_dir is None:
        return
    try:
        from sewtrace.output.debug import write_debug
        write_debug(debug_dir / f"debug_{view}.png", photo, h, *args)
    except Exception as err:  # noqa: BLE001 - a debug picture is never worth a failed run
        log.warning("debug image for %s not written: %s", photo, err)


def trace(
    client: MetaClient, photo: str, h: Homography, prompt: str, sleeve: bool = False, debug_dir: Path | None = None,
) -> tuple[Outline, list[Edge]]:
    """Photo -> labelled half outline in mm. Front and back masks hold the whole t-shirt.
    With ``debug_dir``, also writes debug_<view>.png there (see sewtrace/output/debug.py)."""
    mask = client.segment(photo, prompt)
    view = Path(photo).stem
    try:
        outline = contour.smooth_outline(contour.mask_to_outline(mask, h))
        if sleeve:
            half = contour.sleeve_half(outline)
            edges = contour.label_sleeve_edges(half)
            _debug(debug_dir, view, photo, h, outline, half, edges)
            return half, edges
        body, invented, _ = body_without_sleeves(client, photo, mask, h, outline)
        half = contour.body_half(body)
        edges = contour.label_body_edges(half)
        _debug(debug_dir, view, photo, h, outline, half, edges, body, invented)
        return half, edges
    except contour.ContourRefused as err:
        # The numbers let refusals on real photos recalibrate the limits in contour.joints.
        log.warning("%s: %s, %s", err.label, photo, err.log_details())
        raise ContourFailed(f"{view} photo {photo}: {err}. Take the {view} photo again.") from err
    except ValueError as err:
        raise ContourFailed(
            f"{view} photo {photo}: {err}. Take the photo again with the whole t-shirt flat and in view."
        ) from err


def run(
    garment_dir: str | Path, out_dir: str | Path, client: MetaClient | None = None,
    overrides: dict[str, float] | None = None,
) -> RunResult:
    """Photos -> pattern files. ``overrides`` sets the shoulder or side seam to the user's own length
    (a tape measure on the tee), in place of the value picked from the front and back photos."""
    garment_dir, out_dir = Path(garment_dir), Path(out_dir)
    client = client or MetaClient()
    calls_before = client.calls
    photos = photo_set(garment_dir)
    prompts = client.config.sam.prompts
    # Check all three photos before the first paid call (REQ-CAP-01, REQ-CAP-02, REQ-SCL-03).
    scale = {view: photo_scale(view, path) for view, path in photos}

    # Debug images go in <out>_debug, written before the seam check so a failed run still shows what
    # the outline step found. Not in out_dir: a failed run leaves that empty (REQ-SEW-04).
    debug = out_dir.parent / f"{out_dir.name}_debug"
    front, front_edges = trace(client, photos.front, scale["front"], prompts.garment, debug_dir=debug)
    back, back_edges = trace(client, photos.back, scale["back"], prompts.garment, debug_dir=debug)
    sleeve, sleeve_edges = trace(client, photos.sleeve, scale["sleeve"], prompts.sleeve, sleeve=True, debug_dir=debug)
    measurements = measure_tee(front, front_edges, back, back_edges, sleeve, sleeve_edges)

    draft = draft_spec(front, front_edges, back, back_edges, sleeve, sleeve_edges)
    try:
        spec = infer_panels(client, photos, draft, measurements)
    except ValueError as err:  # e.g. a draft neck curve with no depth to scale
        raise PanelInferenceFailed(
            f"panel inference: {err}. Take the front and back photos again with the whole neck in view."
        ) from err
    user_rows: list[dict] = []
    if overrides:
        spec, applied = set_measured_pairs(spec, overrides)
        user_rows = [{"iteration": -1, "source": "user", "applied": applied}]
    spec, seam_log = adjust_loop(client, spec)
    seam_log = user_rows + seam_log
    spec = refresh_derived(spec)  # the cap may have changed: notches follow it
    try:
        require_seams_ok(spec, seam_log)  # REQ-SEW-04: before any file is written
    except SeamCheckFailed as err:
        err.measurements = measurements  # lets eval report what was measured
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
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
