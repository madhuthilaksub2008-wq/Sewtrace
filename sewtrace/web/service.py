"""What a web job does: run the pipeline or the proof check, and turn the result into JSON.

Everything here goes through the same MetaClient as the command line, so the budget guard,
the ledger and the cache apply."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sewtrace import pipeline
from sewtrace.api.client import MetaClient
from sewtrace.mock import proof as mock_proof
from sewtrace.mock import synth
from sewtrace.reason.seams import seam_name
from sewtrace.types import MEASUREMENT_KEYS, PanelSpec
from sewtrace.output.layout import render_preview
from sewtrace.vision.marker import photo_homography
from sewtrace.vision.verify import overlay_image, verify_proof
from sewtrace.web.jobs import Job

SAMPLE_TEE = "tee1"
VIEWS = ("front", "back", "sleeve")
SAMPLE_PROOF_SCALE = 1.01  # the sample paper is cut 1 % too big, so the gap is not zero
OVERLAY_MAX_PX = 1100
DOWNLOADS = {"pdf": "pattern.pdf", "svg": "pattern.svg", "dxf": "pattern.dxf", "spec": "spec.json"}


@dataclass
class Context:
    client: MetaClient
    last_spec: PanelSpec | None = None  # the pattern the proof is checked against
    last_sample_dir: Path | None = None  # set when the last run was the sample tee
    last_source: str | None = None  # "sample" or the id of the run that made ``last_spec``


def ensure_sample(sample_root: Path) -> Path:
    """The synthetic sample tee folder; written if it is missing (mock SAM needs its mask files)."""
    tee = sample_root / SAMPLE_TEE
    if not (tee / "truth.json").exists():
        synth.make_tee(synth.TEES[0], tee, seed=0)
    return tee


def mask_overlay(photo: Path, mask: np.ndarray, out: Path) -> Path:
    """The photo with the outline of the SAM mask drawn on it, scaled down for the page."""
    image = cv2.imread(str(photo))
    if image is None:
        raise pipeline.PhotoMissing(f"{photo.name} is not an image.")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thick = max(2, round(image.shape[1] / 400))
    cv2.polylines(image, contours, True, (255, 200, 0), thick, cv2.LINE_AA)
    k = min(1.0, OVERLAY_MAX_PX / max(image.shape[:2]))
    if k < 1.0:
        image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def run_pattern(ctx: Context, job: Job, garment_dir: Path, sample: bool) -> dict:
    # A failed run must not leave the old pattern for a later proof check to use without a word.
    ctx.last_spec, ctx.last_sample_dir, ctx.last_source = None, None, None
    job.step = "Running the pipeline: marker, SAM, Muse, seam check"
    result = pipeline.run(garment_dir, job.dir / "out", ctx.client)

    job.step = "Drawing the results"
    prompts = ctx.client.config.sam.prompts
    photos = pipeline.photo_set(garment_dir)
    for view, path in photos:
        prompt = prompts.sleeve if view == "sleeve" else prompts.garment
        job.files[f"mask_{view}"] = mask_overlay(Path(path), ctx.client.segment(path, prompt), job.dir / f"mask_{view}.jpg")
    job.files["preview"] = render_preview(result.spec, job.dir / "preview.png")
    for name, filename in DOWNLOADS.items():
        job.files[name] = job.dir / "out" / filename

    truth_file = garment_dir / "truth.json"
    truth = json.loads(truth_file.read_text())["measurements"] if sample and truth_file.exists() else {}
    ctx.last_spec = result.spec
    ctx.last_sample_dir = garment_dir if sample else None
    ctx.last_source = "sample" if sample else job.id
    return {
        "synthetic": bool(truth),
        "photos": [{"view": v, "file": f"mask_{v}"} for v in VIEWS],
        "measurements": [
            {"key": k, "value_mm": result.measurements[k], "truth_mm": truth.get(k),
             "error_mm": result.measurements[k] - truth[k] if k in truth else None}
            for k in MEASUREMENT_KEYS],
        "seams": [{"name": seam_name(s), "len_a_mm": s.len_a_mm, "len_b_mm": s.len_b_mm, "diff_mm": s.diff_mm,
                   "min_diff_mm": s.min_diff_mm, "max_diff_mm": s.max_diff_mm, "ok": s.ok}
                  for s in result.spec.seams],
        "seam_log": json.loads(json.dumps(result.seam_log, default=str)),
        "model_calls": result.model_calls,
        "preview": "preview",
        "downloads": list(DOWNLOADS),
    }


def run_proof(ctx: Context, job: Job, photo: Path | None) -> dict:
    """Proof check against the spec of the last run. ``photo`` None: draw the sample proof."""
    spec = ctx.last_spec
    if spec is None:
        raise ValueError("proof: there is no pattern yet. Run a pattern first, then check the proof.")
    calls_before = ctx.client.calls
    if photo is None:
        if ctx.last_sample_dir is None:
            raise ValueError("proof: the sample proof fits the sample tee only. Run the sample tee first.")
        job.step = "Drawing the sample proof photo"
        shutil.copy(ctx.last_sample_dir / "truth.json", job.dir / "truth.json")
        mock_proof.make_proof(job.dir, scale=SAMPLE_PROOF_SCALE)
        photo = job.dir / "proof.jpg"
    job.step = "Reading the marker"
    homography = photo_homography(pipeline.load_photo("proof", photo), "proof photo")
    job.step = "Measuring the gap: SAM paper and t-shirt masks"
    result = verify_proof(photo, ctx.client, homography, spec=spec)
    job.files["proof_overlay"] = overlay_image(photo, result, homography, job.dir / "proof_overlay.png")
    neck = None
    if result.neck_mean_mm is not None:
        neck = {"points": len(result.neck_distances_mm), "mean_mm": result.neck_mean_mm,
                "expected_mm": result.expected_neck_gap_mm, "difference_mm": result.neck_gap_difference_mm,
                "flagged": result.neck_gap_flagged, "tol_mm": result.neck_gap_tol_mm}
    return {"mean_mm": result.mean_mm, "max_mm": result.max_mm, "p95_mm": result.p95_mm,
            "pct_within": 100 * result.frac_within_tol, "tol_mm": result.tol_mm, "neck": neck,
            "side_offset_mm": result.side_offset_mm,
            "overlay": "proof_overlay", "spec_source": ctx.last_source, "model_calls": ctx.client.calls - calls_before}
