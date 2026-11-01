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
from sewtrace.output.layout import render_preview, unfold_spec
from sewtrace.output.manufacture import BUNDLE_NAME, FILE_NAMES
from sewtrace.reason.edits import EditProposal, PatternState, apply_proposal, proposal_rows, propose_edit
from sewtrace.reason.seams import seam_name
from sewtrace.types import MEASUREMENT_KEYS, PanelSpec
from sewtrace.vision.marker import photo_homography
from sewtrace.vision.verify import overlay_image, verify_proof
from sewtrace.web.jobs import Job

SAMPLE_TEE = "tee1"
VIEWS = ("front", "back", "sleeve")
SAMPLE_PROOF_SCALE = 1.01  # the sample paper is cut 1 % too big, so the gap is not zero
OVERLAY_MAX_PX = 1100
# cutplan.json is not a download: the page fetches it for the cutter animation.
DOWNLOADS = {"pdf": "pattern.pdf", "svg": "pattern.svg", "dxf": "pattern.dxf", "spec": "spec.json",
             **{k: v for k, v in FILE_NAMES.items() if k != "cutplan_json"}, "bundle": BUNDLE_NAME}
CUTPLAN_JSON = "cutplan_json"


@dataclass
class Context:
    client: MetaClient
    last_spec: PanelSpec | None = None  # the pattern the proof is checked against
    last_sample_dir: Path | None = None  # set when the last run was the sample tee
    last_source: str | None = None  # "sample" or the id of the run that made ``last_spec``
    last_state: PatternState | None = None  # the pattern the Co-pilot edits (the same pattern as ``last_spec``)
    last_job: str | None = None  # id of the job whose page shows ``last_state``: an edit must name it
    last_full: bool = False  # whole pieces were printed


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


def run_pattern(
    ctx: Context, job: Job, garment_dir: Path, sample: bool, overrides: dict[str, float] | None = None,
    full: bool = False,
) -> dict:
    # A failed run must not leave the old pattern for a later proof check to use without a word.
    ctx.last_spec, ctx.last_sample_dir, ctx.last_source = None, None, None
    ctx.last_state, ctx.last_job = None, None
    job.step = "Running the pipeline: marker, SAM, Muse, seam check"
    try:
        result = pipeline.run(garment_dir, job.dir / "out", ctx.client, overrides=overrides or None, full=full)
    finally:  # the debug images are written before the seam check, so a failed run still shows them
        for view in VIEWS:
            if (job.dir / "out_debug" / f"debug_{view}.png").is_file():
                job.files[f"debug_{view}"] = job.dir / "out_debug" / f"debug_{view}.png"

    job.step = "Drawing the results"
    prompts = ctx.client.config.sam.prompts
    photos = pipeline.photo_set(garment_dir)
    for view, path in photos:
        prompt = prompts.sleeve if view == "sleeve" else prompts.garment
        job.files[f"mask_{view}"] = mask_overlay(Path(path), ctx.client.segment(path, prompt), job.dir / f"mask_{view}.jpg")
    job.files["preview"] = render_preview(unfold_spec(result.spec) if full else result.spec, job.dir / "preview.png")
    debug = [v for v in VIEWS if f"debug_{v}" in job.files]
    for name, filename in DOWNLOADS.items():
        job.files[name] = job.dir / "out" / filename
    job.files[CUTPLAN_JSON] = job.dir / "out" / FILE_NAMES[CUTPLAN_JSON]
    cutplan = json.loads(job.files[CUTPLAN_JSON].read_text())["summary"]

    truth_file = garment_dir / "truth.json"
    truth = json.loads(truth_file.read_text())["measurements"] if sample and truth_file.exists() else {}
    state = PatternState(result.spec, result.measurements, result.seam_log, result.provenance)
    ctx.last_spec = result.spec
    ctx.last_sample_dir = garment_dir if sample else None
    ctx.last_source = "sample" if sample else job.id
    ctx.last_state, ctx.last_job, ctx.last_full = state, job.id, full
    return {
        "synthetic": bool(truth),
        "photos": [{"view": v, "file": f"mask_{v}"} for v in VIEWS],
        **pattern_view(state, truth),
        "lay_notes": result.lay_notes,  # how crooked each photo was, and that it was evened out
        "debug": [{"view": v, "file": f"debug_{v}"} for v in debug],
        "sample": sample,
        "full": full,
        "model_calls": result.model_calls,
        "preview": "preview",
        "downloads": list(DOWNLOADS),
        "download_files": DOWNLOADS,
        "cutplan": {"summary": cutplan, "image": "cutplan", "paths": CUTPLAN_JSON},
        "edits": [],
    }


def pattern_view(state: PatternState, truth: dict[str, float]) -> dict:
    """The parts of a run result that follow from the pattern: measurements with their labels, seams,
    seam log. ``truth`` is empty once the pattern was edited, since the truth no longer applies."""
    seam_log = state.seam_log
    return {
        "measurements": [
            {"key": k, "value_mm": state.measurements[k], "truth_mm": truth.get(k),
             "error_mm": state.measurements[k] - truth[k] if k in truth else None,
             "provenance": state.provenance.get(k)}
            for k in MEASUREMENT_KEYS],
        "inferred": pipeline.inferred_values(state.spec, state.provenance),
        "provenance": state.provenance,
        "seams": [{"name": seam_name(s), "len_a_mm": s.len_a_mm, "len_b_mm": s.len_b_mm, "diff_mm": s.diff_mm,
                   "min_diff_mm": s.min_diff_mm, "max_diff_mm": s.max_diff_mm, "ok": s.ok}
                  for s in state.spec.seams],
        "seam_log": json.loads(json.dumps(seam_log, default=str)),
        # Seams the front and back photos measured differently: the length picked, to show and correct.
        "review": pipeline.pair_review(seam_log),
        "cap_note": (pipeline.cap_note(cap) if (cap := pipeline.cap_review(seam_log)) else None),
    }


def current_pattern(ctx: Context, run: str) -> PatternState:
    """The pattern an edit starts from. Only the latest pattern of this server can be edited."""
    if ctx.last_state is None or ctx.last_job != run:
        raise ValueError("edit: this is not the latest pattern of the server. Run the pattern again, then edit it.")
    return ctx.last_state


def propose(ctx: Context, job: Job, run: str, text: str) -> dict:
    """The Co-pilot proposal for a request: one Spark call (none when it is cached). Nothing changes yet."""
    state = current_pattern(ctx, run)
    job.step = "Asking the Co-pilot (one Spark call)"
    calls_before = ctx.client.calls
    proposal = propose_edit(ctx.client, text, state)
    return {"run": run, "text": text, "explanation": proposal.explanation, "proposal": proposal.model_dump(),
            "rows": proposal_rows(proposal, state.measurements), "model_calls": ctx.client.calls - calls_before}


def apply_edit(ctx: Context, job: Job, earlier: Job, run: str, proposal: EditProposal) -> dict:
    """Apply an approved proposal by code (no model call) and make a new pattern job beside the old one:
    the files, the preview and a result in the shape of ``run_pattern``."""
    state = current_pattern(ctx, run)
    if earlier.result is None:
        raise ValueError("edit: the pattern page has no result. Run the pattern again.")
    job.step = "Changing the pattern, checking the seams"
    calls_before = ctx.client.calls
    edited = apply_proposal(ctx.client, state, proposal)
    job.step = "Drawing the results"
    out = job.dir / "out"
    for name, path in earlier.files.items():  # the photo overlays do not change: copy them into this job
        if name.startswith("mask_") and path.is_file():
            job.files[name] = Path(shutil.copyfile(path, job.dir / path.name))
        elif name.startswith("debug_") and path.is_file():  # beside out/, so the new bundle holds them too
            (job.dir / "out_debug").mkdir(exist_ok=True)
            job.files[name] = Path(shutil.copyfile(path, job.dir / "out_debug" / path.name))
    pipeline.save_state(edited.state, out, ctx.last_full)
    job.files["preview"] = render_preview(unfold_spec(edited.state.spec) if ctx.last_full else edited.state.spec,
                                          job.dir / "preview.png")
    for name, filename in DOWNLOADS.items():
        job.files[name] = out / filename
    job.files[CUTPLAN_JSON] = out / FILE_NAMES[CUTPLAN_JSON]
    cutplan = json.loads(job.files[CUTPLAN_JSON].read_text())["summary"]
    ctx.last_spec, ctx.last_state, ctx.last_job, ctx.last_source = edited.state.spec, edited.state, job.id, job.id
    ctx.last_sample_dir = None  # the sample proof photo is drawn for the unedited sample tee
    return {**earlier.result, **pattern_view(edited.state, {}), "synthetic": False,
            "model_calls": ctx.client.calls - calls_before, "edits": edited.edits, "edit_changes": edited.changes,
            "cutplan": {"summary": cutplan, "image": "cutplan", "paths": CUTPLAN_JSON}}


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
