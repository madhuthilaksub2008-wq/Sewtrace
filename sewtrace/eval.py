"""T26.1: run every garment, compare the 6 measurements with ground truth.

Ground truth per garment comes from truth.json (synthetic) or, for real tees,
data/ground_truth.csv (tape measure, board T18).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sewtrace.api.client import MetaClient
from sewtrace.pipeline import run
from sewtrace.types import MEASUREMENT_KEYS


def ground_truth(garment_dir: Path, csv_path: Path | None) -> dict[str, float]:
    truth = garment_dir / "truth.json"
    if truth.exists():
        return json.loads(truth.read_text())["measurements"]
    if csv_path and csv_path.exists():
        for row in csv.DictReader(csv_path.open()):
            if row["tee"] == garment_dir.name:
                return {k: float(row[k]) for k in MEASUREMENT_KEYS}
    raise FileNotFoundError(f"no ground truth for {garment_dir.name}")


def evaluate(raw_dir: str | Path, out_dir: str | Path, runs: int = 1) -> list[dict]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    client = MetaClient()
    rows = []
    for garment in sorted(p for p in raw_dir.iterdir() if (p / "front.jpg").exists()):
        truth = ground_truth(garment, raw_dir.parent / "ground_truth.csv")
        for i in range(runs):
            result = run(garment, out_dir / garment.name / f"run{i + 1}", client)
            errors = {k: result.measurements[k] - truth[k] for k in MEASUREMENT_KEYS}
            rows.append({
                "garment": garment.name,
                "run": i + 1,
                **{f"err_{k}": round(v, 2) for k, v in errors.items()},
                "mean_abs_err": round(sum(abs(v) for v in errors.values()) / len(errors), 2),
                "seams_ok": all(s.ok for s in result.spec.seams),
                "model_calls": result.model_calls,
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "results.md").write_text(to_markdown(rows))
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    overall = sum(r["mean_abs_err"] for r in rows) / len(rows)
    lines += ["", f"Mean absolute error over all runs: {overall:.2f} mm"]
    return "\n".join(lines) + "\n"
