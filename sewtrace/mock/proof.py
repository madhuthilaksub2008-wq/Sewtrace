"""T24 stand-in: the proof photo, a printed-and-cut paper FRONT laid on the tee.

The paper is the full (unfolded) front stitch outline from truth.json. Its error
is controllable (scale about the panel centroid, rotation, offset), so the
overlay check (vision/verify.py) can be tested against a known gap:

  <tee_dir>/proof.jpg                photo, marker in frame
  <tee_dir>/proof.masks/paper.png    the paper sheet
  <tee_dir>/proof.masks/t-shirt.png  whole tee with sleeves, as if seen through the paper
  <tee_dir>/proof.masks/body.png     ground truth only: the tee body without sleeves
  <tee_dir>/proof.json               paper params, outlines, expected gap

    python -m sewtrace.mock.proof data/raw/tee1 --scale 1.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from sewtrace.mock import synth

PAPER_BGR = (238, 241, 243)


def full_panel(half: np.ndarray, edges: dict[str, list[int]]) -> np.ndarray:
    """Unfolded panel from a half: mirror without tracing the fold line twice."""
    side = half[: edges["fold"][0] + 1]  # neck point (0, depth) round to the hem at x = 0
    other = side[::-1][1:-1] * [-1, 1]
    return np.vstack([side, other])


def paper_outline(full: np.ndarray, scale: float = 1.0, rotation_deg: float = 0.0,
                  offset_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Full panel outline with a known cutting/placement error, about its centroid."""
    c = np.asarray(shapely.Polygon(full).centroid.coords[0])
    a = np.radians(rotation_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (full - c) @ rot.T * scale + c + np.asarray(offset_mm, float)


def expected_gaps(paper: np.ndarray, body: np.ndarray, step_mm: float = 2.0) -> np.ndarray:
    """Exact distance from evenly spaced paper edge points to the body edge, mm."""
    ring = shapely.LinearRing(paper)
    pts = shapely.line_interpolate_point(ring, np.arange(0, ring.length, step_mm))
    return shapely.distance(pts, shapely.LinearRing(body))


def make_proof(
    tee_dir: str | Path,
    seed: int = 0,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    offset_mm: tuple[float, float] = (0.0, 0.0),
    name: str = "proof",
) -> dict:
    """Render ``<name>.jpg`` + masks next to a synthetic tee; returns the proof truth."""
    tee_dir = Path(tee_dir)
    truth = json.loads((tee_dir / "truth.json").read_text())
    params = dict(truth["params"], colour_bgr=tuple(truth["params"]["colour_bgr"]))
    p = synth.TeeParams(**params)
    front = truth["photos"]["front"]
    origin = np.asarray(front["origin_mm"], float)
    full = full_panel(np.asarray(front["half_outline_mm"], float), front["edges"])
    body = full + origin
    sleeves = [synth.flatlay_sleeve(p, s) + origin for s in (1, -1)]
    paper = paper_outline(full, scale, rotation_deg, offset_mm) + origin

    rng = np.random.default_rng(seed)
    photo, masks, H, marker_mm = synth.render_layers(
        {"body": [body], "sleeve": sleeves}, p.colour_bgr, rng, overlays={"paper": ([paper], PAPER_BGR)}
    )
    synth._save(tee_dir, name, photo, masks)
    gaps = expected_gaps(paper, body)
    proof = {
        "paper": {"scale": scale, "rotation_deg": rotation_deg, "offset_mm": list(offset_mm)},
        "paper_outline_mm": paper.tolist(),
        "body_outline_mm": body.tolist(),
        "world_mm_to_photo_px": H.tolist(),
        "marker_corners_mm": marker_mm.tolist(),
        "expected_gap_mean_mm": float(gaps.mean()),
        "expected_gap_max_mm": float(gaps.max()),
    }
    (tee_dir / f"{name}.json").write_text(json.dumps(proof, indent=1))
    return proof


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tee_dir")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rotation", type=float, default=0.0)
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    proof = make_proof(args.tee_dir, args.seed, args.scale, args.rotation, tuple(args.offset))
    print(f"wrote {Path(args.tee_dir) / 'proof.jpg'}, expected mean gap {proof['expected_gap_mean_mm']:.2f} mm")


if __name__ == "__main__":
    main()
