"""Day 1 test (REQ-API-09): exactly one real SAM call and one real Spark call on one photo.

Prints the model ID, input tokens, latency and cost of each call, and the ledger total before
and after. It never loops and never retries by itself. It refuses to run without
``--yes-spend``, and prints the worst-case cost first. Run it only after the user approves."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from sewtrace.api.client import CHARS_PER_TOKEN, ApiError, MetaClient, Usage, cost_usd
from sewtrace.reason.panels import PanelInference

# The Spark call sends the production schema (PanelInference through strict_json_schema, the same
# path as the pipeline), so Day 1 shows whether the API accepts it. The prompt is short on purpose.
DAY1_PROMPT = (
    "The photo shows one kids' t-shirt laid flat. Estimate these four fields: front_neck_depth_mm, "
    "back_neck_depth_mm, sleeve_cap (crown_run_mm and underarm_run_mm) and grain_line. "
    "Return only JSON with exactly these fields."
)


def worst_case_usd(client: MetaClient) -> dict[str, float]:
    """The cost the budget guard assumes for each of the two calls."""
    cfg = client.config
    spark = Usage(
        input_tokens=len(DAY1_PROMPT) // CHARS_PER_TOKEN + cfg.reasoning.est_image_tokens,
        output_tokens=cfg.reasoning.max_tokens, images=1,
    )
    return {
        cfg.models.sam.id: cost_usd(cfg.models.sam, Usage(images=1)),
        cfg.reasoning_model.id: cost_usd(cfg.reasoning_model, spark),
    }


def _ledger_rows(client: MetaClient) -> list[dict[str, str]]:
    if not client.ledger_path.exists():
        return []
    with client.ledger_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _report(label: str, rows: list[dict[str, str]], seconds: float, text: str) -> None:
    if not rows:
        print(f"{label}: no ledger row. The answer came from the cache, so nothing was spent.")
        return
    for r in rows:
        estimated = r["estimated"] == "1"
        print(f"{label}: model {r['model']}  input tokens {r['input_tokens']}  cached {r['cached_tokens']}  "
              f"output tokens {r['output_tokens']}  latency {seconds:.1f} s  cost ${float(r['cost_usd']):.6f}"
              f"{'  (ESTIMATED: the call failed, timed out or reported no usage)' if estimated else ''}")
        if estimated or int(r["input_tokens"]) == 0:
            print(f"{label}: image tokens unknown (no usage reported)")
            continue
        # Input tokens minus the text we sent (4 characters per token). The schema and the
        # system overhead are counted as image tokens, so the number is an upper bound.
        image_tokens = int(r["input_tokens"]) - len(text) // CHARS_PER_TOKEN
        print(f"{label}: image tokens APPROXIMATELY {image_tokens} "
              "(input tokens minus estimated text tokens). Use it to update reasoning.est_image_tokens.")


def _write_overlay(client: MetaClient, photo: Path, prompt: str, mask: np.ndarray, out_dir: Path) -> Path:
    """Mask (red) and SAM boxes (green) over the photo, in the frame the pipeline reads, so the
    user can check by eye that they line up. Boxes are also printed next to the image size."""
    (width, height), boxes = client.sam_boxes(photo, prompt)
    print(f"SAM frame check: image {width} x {height} px (as cv2.imread reads it, EXIF applied)")
    inspection = client.sam_inspect(photo, prompt)
    sent, source = inspection.sent_wh, inspection.source_wh
    print(f"  sent to SAM {sent[0]} x {sent[1]} px; SAM reported source "
          + (f"{source[0]} x {source[1]} px" + ("" if source == sent else "  (DIFFERS from sent: resized on the server)")
             if source else "size unreadable (sent size assumed)"))
    for r in inspection.records:
        conf = "none" if r.confidence is None else f"{r.confidence:.3f}"
        print(f"  record {r.object_id}: confidence {conf}  bounds {r.bounds[0]:g},{r.bounds[1]:g} to "
              f"{r.bounds[2]:g},{r.bounds[3]:g}  raster {r.raster_wh[0]} x {r.raster_wh[1]}")
    for object_id, left, top, right, bottom in boxes:
        print(f"  box {object_id}: left {left:g}  top {top:g}  right {right:g}  bottom {bottom:g}")
    if not boxes:
        print("  no box records in the answer")
    image = cv2.imread(str(photo))
    shaded = image.copy()
    shaded[mask] = (0, 0, 255)
    overlay = cv2.addWeighted(shaded, 0.5, image, 0.5, 0)
    for object_id, left, top, right, bottom in boxes:
        cv2.rectangle(overlay, (round(left), round(top)), (round(right), round(bottom)), (0, 200, 0), 3)
        cv2.putText(overlay, str(object_id), (round(left) + 4, round(top) + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 0), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{photo.stem}.sam_overlay.png"
    cv2.imwrite(str(path), overlay)
    return path


def run_day1(
    photo: str | Path, *, yes_spend: bool, client: MetaClient | None = None, out_dir: str | Path = "out/day1",
) -> None:
    client = client or MetaClient()
    photo = Path(photo)
    if not photo.is_file():
        raise ApiError(f"Photo {photo} not found. Check the path.")
    worst = worst_case_usd(client)
    print(f"Day 1 test on {photo.name}: one call to each model.")
    for model_id, usd in worst.items():
        print(f"  {model_id}: worst case ${usd:.4f}")
    print(f"  total worst case ${sum(worst.values()):.4f}. Ledger limit ${client.config.api.budget_limit_usd:.2f}.")
    if not yes_spend:
        raise ApiError("Nothing was called. Add --yes-spend to make the two paid calls.")
    if client.settings.mock or client.settings.offline:
        raise ApiError("Mock or offline mode is on. Set META_API_KEY and unset SEWTRACE_MOCK and SEWTRACE_OFFLINE.")

    before_total = client.ledger_total_usd()
    print(f"Ledger total before: ${before_total:.6f}")
    try:
        # One call each, in this order, no loop. A failure of the first stops the run.
        sam_prompt = client.config.sam.prompts.garment
        n = len(_ledger_rows(client))
        start = time.monotonic()
        mask = client.segment(photo, sam_prompt)
        _report("SAM", _ledger_rows(client)[n:], time.monotonic() - start, sam_prompt)
        print(f"SAM mask: {int(mask.sum())} px of {mask.size}")
        print(f"SAM overlay: {_write_overlay(client, photo, sam_prompt, mask, Path(out_dir))}")

        n = len(_ledger_rows(client))
        start = time.monotonic()
        answer = client.reason(DAY1_PROMPT, [photo], schema=PanelInference.model_json_schema())
        _report("Spark", _ledger_rows(client)[n:], time.monotonic() - start, DAY1_PROMPT)
        try:
            print(f"Spark answer: {PanelInference.model_validate(answer)}")
        except ValidationError as err:
            print("Spark schema check: FAIL. The answer does not match PanelInference.")
            raise ApiError(f"Spark answered, but not with PanelInference: {err}") from err
        print("Spark schema check: PASS. The API accepted the production schema and the answer validates.")
    finally:
        after_total = client.ledger_total_usd()
        print(f"Ledger total after: ${after_total:.6f}  (this run: ${after_total - before_total:.6f})")
        print("Compare with the Usage page: they must agree within 10% (REQ-API-09).")
