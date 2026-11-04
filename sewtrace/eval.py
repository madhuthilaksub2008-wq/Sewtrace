"""T26.1: run every garment, compare the 6 measurements with ground truth.

Ground truth per garment (SPEC 10.1, 10.3):
  * synthetic tee (data/synth/<tee>/truth.json): exact values from the drawn geometry
  * real tee (data/raw/<tee>): tape measure values in data/ground_truth/<tee>.json
A real tee with no such file is an error. There is no fallback, so a synthetic
parameter can never stand in for a tape measure.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from sewtrace.api.client import MetaClient, SegmentationFailed
from sewtrace.config import get_config
from sewtrace.pipeline import ContourFailed, PhotoMissing, run
from sewtrace.reason.seams import SeamCheckFailed, seam_name
from sewtrace.types import MEASUREMENT_KEYS
from sewtrace.vision.marker import MarkerNotFound, MarkerRejected

log = logging.getLogger(__name__)


class GroundTruthMissing(FileNotFoundError):
    """A real tee has no tape measure file."""


def _as_readings(value: object) -> object:
    return value if isinstance(value, list) else [value]


# One number, or the list of tape readings for one point (SPEC 10.3 steps 3 to 5).
Readings = Annotated[list[float], BeforeValidator(_as_readings), Field(min_length=1, max_length=3)]


class GroundTruth(BaseModel):
    """data/ground_truth/<tee>.json: the six SPEC 10.2 values in mm.

    Each value is one number or a list of readings. Two readings that differ by more
    than ground_truth.repeat_tol_mm need a third (SPEC 10.3 step 5). The truth is the mean."""

    model_config = ConfigDict(extra="forbid")

    chest_width: Readings
    body_length: Readings
    shoulder_width: Readings
    sleeve_length: Readings
    neck_width: Readings
    armhole_depth: Readings

    @model_validator(mode="after")
    def _repeats_agree(self) -> "GroundTruth":
        tol = get_config().ground_truth.repeat_tol_mm
        for key in MEASUREMENT_KEYS:
            values = getattr(self, key)
            if any(v <= 0 for v in values):
                raise ValueError(f"{key}: readings must be more than 0 mm")
            if len(values) == 2 and abs(values[0] - values[1]) > tol:
                raise ValueError(f"{key}: the two readings differ by more than {tol:g} mm. Measure a third time.")
        return self

    def means_mm(self) -> dict[str, float]:
        return {k: sum(v) / len(v) for k in MEASUREMENT_KEYS for v in [getattr(self, k)]}


def ground_truth(garment_dir: Path, truth_dir: Path) -> dict[str, float]:
    truth = garment_dir / "truth.json"
    if truth.exists():
        return json.loads(truth.read_text())["measurements"]
    tape = truth_dir / f"{garment_dir.name}.json"
    if not tape.exists():
        raise GroundTruthMissing(
            f"{garment_dir.name} has no truth.json and no {tape}. Measure the tee with a tape "
            "(SPEC 10.3) and write the six values in mm to that file."
        )
    try:
        return GroundTruth.model_validate_json(tape.read_text()).means_mm()
    except ValidationError as err:
        raise ValueError(f"{tape}: {err}") from err


# One bad tee must not stop the eval. BudgetExceeded and CacheMiss are not here: they abort.
TEE_ERRORS = (SeamCheckFailed, PhotoMissing, MarkerNotFound, MarkerRejected, SegmentationFailed, ContourFailed)
ROW_COLUMNS = ["garment", "run", *[f"err_{k}" for k in MEASUREMENT_KEYS], "mean_abs_err",
               "seams_ok", "model_calls", "synthetic", "error"]


def _error_text(err: Exception) -> str:
    if isinstance(err, SeamCheckFailed):
        return f"SeamCheckFailed: {', '.join(seam_name(s) for s in err.failed)}"
    return f"{type(err).__name__}: {' '.join(str(err).split())}".replace("|", "/")


def is_synthetic(garment_dir: Path) -> bool:
    """True for a tee drawn by the synth command: its truth.json says so. A tape measure tee is real."""
    truth = garment_dir / "truth.json"
    return truth.is_file() and json.loads(truth.read_text()).get("synthetic") is True


def _row(garment: str, run_no: int, truth: dict[str, float], measurements: dict[str, float] | None,
         seams_ok: bool, model_calls: int, error: str = "", *, synthetic: bool) -> dict:
    row: dict = {"garment": garment, "run": run_no}
    if measurements is None:
        row.update({f"err_{k}": "" for k in MEASUREMENT_KEYS}, mean_abs_err="")
    else:
        errors = {k: measurements[k] - truth[k] for k in MEASUREMENT_KEYS}
        row.update({f"err_{k}": round(v, 2) for k, v in errors.items()},
                   mean_abs_err=round(sum(abs(v) for v in errors.values()) / len(errors), 2))
    row.update(seams_ok=seams_ok, model_calls=model_calls, synthetic=synthetic, error=error)
    return row


def tee_failed(row: dict) -> bool:
    return bool(row["error"]) or not row["seams_ok"]


def evaluate(raw_dir: str | Path, out_dir: str | Path, runs: int = 1) -> list[dict]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    client = MetaClient()
    garments = sorted(p for p in raw_dir.iterdir() if (p / "front.jpg").exists()) if raw_dir.is_dir() else []
    if not garments:
        raise FileNotFoundError(f"no tee folders with front.jpg in {raw_dir}. Run the synth command or add photos.")
    # Read every truth first: a missing file must stop the eval before any paid call.
    truths = {g.name: ground_truth(g, raw_dir.parent / "ground_truth") for g in garments}
    rows = []
    for garment in garments:
        truth = truths[garment.name]
        synthetic = is_synthetic(garment)
        for i in range(runs):
            calls_before = client.calls
            try:
                result = run(garment, out_dir / garment.name / f"run{i + 1}", client)
            except TEE_ERRORS as err:
                log.error("eval: %s run %d failed: %s", garment.name, i + 1, err)
                measured = err.measurements if isinstance(err, SeamCheckFailed) else None
                rows.append(_row(garment.name, i + 1, truth, measured, False,
                                 client.calls - calls_before, _error_text(err), synthetic=synthetic))
                continue
            rows.append(_row(garment.name, i + 1, truth, result.measurements,
                             all(s.ok for s in result.spec.seams), result.model_calls, synthetic=synthetic))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "results.md").write_text(to_markdown(rows))
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = ROW_COLUMNS
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    measured = [r["mean_abs_err"] for r in rows if r["mean_abs_err"] != ""]
    failed = sum(tee_failed(r) for r in rows)
    overall = f"{sum(measured) / len(measured):.2f} mm" if measured else "n/a"
    lines += ["", f"Mean absolute error over all runs: {overall}",
              f"Failed runs: {failed} of {len(rows)}"]
    return "\n".join(lines) + "\n"
