"""T24 stand-in: the proof photo, a paper HALF front laid on the tee (SPEC 9.4).

The paper is the half front stitch outline from truth.json, cut on the stitch line, with
its fold on the centre of the t-shirt (the right half of the tee, fold at the tee's
x origin). The tee is whole, with sleeves. The paper error is controllable (scale about
the half-panel centroid, rotation, offset), so the overlay check (vision/verify.py) can be
tested against an exact gap computed on the comparable edges only: the fold edge, the
paper points nearest the (hidden) armhole and the paper points nearest the body neck are
left out. The tee mask has the BACK neckline, as on a real front photo: the back panel
shows through the neck opening, so the front neck of the paper lies lower than the outline.
The gap of those neck points is reported on its own (expected_neck_gap_mean_mm).

  <tee_dir>/<name>.jpg                photo, marker in frame
  <tee_dir>/<name>.masks/paper.png    the paper sheet
  <tee_dir>/<name>.masks/t-shirt.png  whole tee with sleeves, as if seen through the paper
  <tee_dir>/<name>.masks/body.png     ground truth only: the tee body without sleeves
  <tee_dir>/<name>.json               paper params, outlines, expected gap

    python -m sewtrace.mock.proof data/synth/tee1 --scale 1.02
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import shapely

from sewtrace.config import get_config
from sewtrace.mock import synth

PAPER_BGR = (238, 241, 243)


def paper_outline(half: np.ndarray, scale: float = 1.0, rotation_deg: float = 0.0,
                  offset_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Half panel outline with a known cutting/placement error, about its centroid."""
    c = np.asarray(shapely.Polygon(half).centroid.coords[0])
    a = np.radians(rotation_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (half - c) @ rot.T * scale + c + np.asarray(offset_mm, float)


def expected_gaps(
    paper: np.ndarray,
    body: np.ndarray,
    fold: tuple[np.ndarray, np.ndarray],
    armhole: np.ndarray,
    centre_x_mm: float,
    step_mm: float = 2.0,
    neck: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact distance from evenly spaced paper edge points to the body edge, mm.

    Returns (compared gaps, neck gaps). Compared points are not the fold edge, not points
    whose nearest body edge is the armhole (hidden under the sleeve on a real tee) and not
    points whose nearest body edge is the neck (the back neckline). Neck gaps are the
    left-out neck points. The body edge is the half on the paper's side of the tee centre
    line (plus the config margin).
    """
    cfg = get_config().proof
    ring = shapely.LinearRing(paper)
    pts = shapely.line_interpolate_point(ring, np.arange(0, ring.length, step_mm))
    side = shapely.box(centre_x_mm - cfg.centre_margin_mm, -1e5, 1e5, 1e5)
    body_edge = shapely.intersection(shapely.LinearRing(body), side)
    d_body = np.asarray(shapely.distance(pts, body_edge), float)
    d_armhole = np.asarray(shapely.distance(pts, shapely.LineString(armhole)), float)
    d_fold = np.asarray(shapely.distance(pts, shapely.LineString(list(fold))), float)
    d_neck = np.full(len(d_body), np.inf) if neck is None else np.asarray(
        shapely.distance(pts, shapely.LineString(neck)), float)
    comparable = (d_fold >= cfg.fold_exclude_mm) & (d_armhole > d_body)
    on_neck = comparable & (d_neck <= d_body)
    return d_body[comparable & ~on_neck], d_body[on_neck]


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
    half = np.asarray(front["half_outline_mm"], float)  # the pattern: front neck
    visible = np.asarray(front["visible_half_outline_mm"], float)  # the mask: back neck
    edges = front["edges"]
    body = synth.mirror_full(visible) + origin
    sleeves = [synth.flatlay_sleeve(p, s) + origin for s in (1, -1)]
    placed = paper_outline(half, scale, rotation_deg, offset_mm)
    paper = placed + origin
    fold = (paper[edges["fold"][0]], paper[0])  # hem corner to neck point; the last edge wraps to 0

    rng = np.random.default_rng(seed)
    photo, masks, H, marker_mm = synth.render_layers(
        {"body": [body], "sleeve": sleeves}, p.colour_bgr, rng, overlays={"paper": ([paper], PAPER_BGR)}
    )
    synth._save(tee_dir, name, photo, masks)
    armhole = half[edges["armhole"][0] : edges["armhole"][1] + 1] + origin
    neck = visible[edges["neck"][0] : edges["neck"][1] + 1] + origin
    # The rendered body has a zero-width sliver along the fold; the gap must not see it.
    clean_body = synth.mirror_full(visible[: edges["fold"][0] + 1]) + origin
    gaps, neck_gaps = expected_gaps(paper, clean_body, fold, armhole, origin[0], neck=neck)
    proof = {
        "paper": {"scale": scale, "rotation_deg": rotation_deg, "offset_mm": list(offset_mm)},
        "paper_outline_mm": paper.tolist(),
        "body_outline_mm": body.tolist(),
        "fold_mm": [f.tolist() for f in fold],
        "world_mm_to_photo_px": H.tolist(),
        "marker_corners_mm": marker_mm.tolist(),
        "expected_gap_mean_mm": float(gaps.mean()),
        "expected_gap_max_mm": float(gaps.max()),
        "expected_neck_gap_mean_mm": float(neck_gaps.mean()) if len(neck_gaps) else None,
    }
    (tee_dir / f"{name}.json").write_text(json.dumps(proof, indent=1))
    return proof


def make_proof_hard(
    tee_dir: str | Path,
    seed: int = 0,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    offset_mm: tuple[float, float] = (0.0, 0.0),
    name: str = "proof",
    cut_wave_mm: float = 1.5,
) -> dict:
    """make_proof() for a hard tee (``synth --hard``): the paper is cut by hand (waves of
    ``cut_wave_mm`` on every edge but the fold), a lifted paper edge casts a shadow, and the
    photo has every fault of the tee (lens, tilt, light, noisy masks). The expected gap is
    computed from the exact wavy paper and the exact, rounded, back-neck tee body."""
    from sewtrace.mock import synth_hard

    tee_dir = Path(tee_dir)
    truth = json.loads((tee_dir / "truth.json").read_text())
    hp = synth_hard.params_from_truth(truth)
    p = synth.TeeParams(**dict(truth["params"], colour_bgr=tuple(truth["params"]["colour_bgr"])))
    front = truth["photos"]["front"]
    origin = np.asarray(front["origin_mm"], float)
    half = np.asarray(front["half_outline_mm"], float)  # the pattern: front neck
    visible = np.asarray(front["visible_half_outline_mm"], float)  # the mask: back neck
    edges = {k: tuple(v) for k, v in front["edges"].items()}
    regions = {k: np.asarray(v) for k, v in front["regions_mm"].items() if k != "sleeves"}
    cut = synth_hard.wave_edges(half, edges, replace(hp, wave_mm=cut_wave_mm), view=3, side=0, skip=("fold",))
    paper = paper_outline(cut, scale, rotation_deg, offset_mm) + origin
    fold = (paper[edges["fold"][0]], paper[0])

    rng = np.random.default_rng(seed)
    photo, masks, H, marker_mm = synth_hard.render_hard(
        {"body": [regions["body"]], "sleeve": [np.asarray(s) for s in front["regions_mm"]["sleeves"]]},
        [regions["garment"]], p.colour_bgr, rng, hp, overlays={"paper": ([paper], PAPER_BGR)})
    synth_hard.save(tee_dir, name, photo, masks, hp.jpeg_quality)
    armhole = half[edges["armhole"][0] : edges["armhole"][1] + 1] + origin
    neck = visible[edges["neck"][0] : edges["neck"][1] + 1] + origin
    gaps, neck_gaps = expected_gaps(paper, regions["body_rounded"], fold, armhole, origin[0], neck=neck)
    proof = {
        "paper": {"scale": scale, "rotation_deg": rotation_deg, "offset_mm": list(offset_mm),
                  "cut_wave_mm": cut_wave_mm},
        "paper_outline_mm": paper.tolist(),
        "fold_mm": [f.tolist() for f in fold],
        "world_mm_to_photo_px": H.tolist(),
        "marker_corners_mm": marker_mm.tolist(),
        "expected_gap_mean_mm": float(gaps.mean()),
        "expected_gap_max_mm": float(gaps.max()),
        "expected_neck_gap_mean_mm": float(neck_gaps.mean()) if len(neck_gaps) else None,
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
    ap.add_argument("--hard", action="store_true", help="tee_dir is a hard tee: hand-cut wavy paper, lift shadow")
    args = ap.parse_args(argv)
    make = make_proof_hard if args.hard else make_proof
    proof = make(args.tee_dir, args.seed, args.scale, args.rotation, tuple(args.offset))
    print(f"wrote {Path(args.tee_dir) / 'proof.jpg'}, expected mean gap {proof['expected_gap_mean_mm']:.2f} mm")


if __name__ == "__main__":
    main()
