"""Day 1 test (REQ-API-09): exactly one real SAM call and one real Spark call on one photo.

Prints the model ID, input tokens, latency and cost of each call, and the ledger total before
and after. It never loops and never retries by itself. It refuses to run without
``--yes-spend``, and prints the worst-case cost first. Run it only after the user approves."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sewtrace.api.client import ApiError, MetaClient, Usage, cost_usd

DAY1_PROMPT = (
    "The photo shows one kids' t-shirt laid flat. Say whether it is a front view or a back view, "
    "and whether the neck opening is visible."
)


class Day1Answer(BaseModel):
    """Tiny schema: the Day 1 call tests the request shape, not the pattern logic."""

    model_config = ConfigDict(extra="forbid")

    view: str
    neck_visible: bool


def worst_case_usd(client: MetaClient) -> dict[str, float]:
    """The cost the budget guard assumes for each of the two calls."""
    cfg = client.config
    spark = Usage(
        input_tokens=len(DAY1_PROMPT) // 4 + cfg.reasoning.est_image_tokens,
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


def _report(label: str, rows: list[dict[str, str]], seconds: float) -> None:
    if not rows:
        print(f"{label}: no ledger row. The answer came from the cache, so nothing was spent.")
        return
    for r in rows:
        print(f"{label}: model {r['model']}  input tokens {r['input_tokens']}  cached {r['cached_tokens']}  "
              f"output tokens {r['output_tokens']}  latency {seconds:.1f} s  cost ${float(r['cost_usd']):.6f}"
              f"{'  (ESTIMATED, the call timed out)' if r['estimated'] == '1' else ''}")


def run_day1(photo: str | Path, *, yes_spend: bool, client: MetaClient | None = None) -> None:
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
        n = len(_ledger_rows(client))
        start = time.monotonic()
        mask = client.segment(photo, client.config.sam.prompts.garment)
        _report("SAM", _ledger_rows(client)[n:], time.monotonic() - start)
        print(f"SAM mask: {int(mask.sum())} px of {mask.size}")

        n = len(_ledger_rows(client))
        start = time.monotonic()
        answer = client.reason(DAY1_PROMPT, [photo], schema=Day1Answer.model_json_schema())
        _report("Spark", _ledger_rows(client)[n:], time.monotonic() - start)
        print(f"Spark answer: {Day1Answer.model_validate(answer)}")
    finally:
        after_total = client.ledger_total_usd()
        print(f"Ledger total after: ${after_total:.6f}  (this run: ${after_total - before_total:.6f})")
        print("Compare with the Usage page: they must agree within 10% (REQ-API-09).")
