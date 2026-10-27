"""Synthetic capture photos, standing in for real tees until Bala's captures land.

Each tee is drawn from known geometry, so every downstream number can be
checked against exact ground truth:

  data/raw/teeN/front.jpg, back.jpg, sleeve.jpg   photos with ArUco marker
  data/raw/teeN/<photo>.masks/<region>.png         ground-truth region masks
                                                    (the mock SAM serves these)
  data/raw/teeN/truth.json                         params, measurements, half
                                                    outlines in mm, world->photo H
  data/ground_truth.csv                            the 6 tape-measure points

Photos are rendered top-down at PX_PER_MM, then warped by a small random
camera homography (tilt, rotation, scale), so the marker step has real work.

    python -m sewtrace.mock.synth --out data/raw
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from sewtrace.config import MARKER_ID, MARKER_SIZE_MM

PX_PER_MM = 3.0
MARKER_BORDER_MM = 15.0
BACKDROP_BGR = (45, 95, 40)


@dataclass
class TeeParams:
    name: str
    chest_half: float      # W
    length: float          # L, neck point to hem
    shoulder_half: float   # S
    neck_half: float       # N
    front_neck_depth: float
    back_neck_depth: float
    armhole_depth: float   # A
    shoulder_drop: float
    hem_flare: float
    sleeve_length: float
    sleeve_hem_half: float
    cap_ease: float        # extra sleeve cap length per half, mm
    colour_bgr: tuple[int, int, int]


# Five kids tees across sizes (roughly age 4 to 12), flat-lay mm.
TEES = [
    TeeParams("tee1", 160, 380, 140, 62, 58, 20, 140, 12, 4, 110, 70, 6, (40, 40, 200)),
    TeeParams("tee2", 170, 420, 148, 65, 62, 21, 150, 13, 4, 120, 74, 6, (30, 200, 230)),
    TeeParams("tee3", 180, 460, 156, 68, 66, 22, 160, 14, 5, 130, 78, 6, (190, 110, 30)),
    TeeParams("tee4", 195, 500, 168, 72, 70, 23, 170, 15, 5, 140, 82, 6, (30, 140, 240)),
    TeeParams("tee5", 210, 540, 180, 76, 74, 24, 180, 16, 6, 150, 86, 6, (160, 60, 160)),
]


# ---- geometry (mm, y down, half panel with fold at x = 0) -----------------

def _bezier(p0, p1, p2, p3, n=40) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def _length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _line(a, b, n=20) -> np.ndarray:
    return np.linspace(a, b, n)


def body_half(p: TeeParams, back: bool = False) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Half body outline and its named edges (index ranges, inclusive)."""
    depth = p.back_neck_depth if back else p.front_neck_depth
    W, L, S, N, A, d = p.chest_half, p.length, p.shoulder_half, p.neck_half, p.armhole_depth, p.shoulder_drop
    t = np.linspace(0, np.pi / 2, 30)
    parts = {
        "neck": np.c_[N * np.sin(t), depth * np.cos(t)],
        "shoulder": _line((N, 0), (S, d)),
        "armhole": armhole_curve(p),
        "side": _line((W, d + A), (W + p.hem_flare, L)),
        "hem": _line((W + p.hem_flare, L), (0, L), 40),
        "fold": _line((0, L), (0, depth), 30),
    }
    return _join(parts)


def armhole_curve(p: TeeParams) -> np.ndarray:
    S, W, A, d = p.shoulder_half, p.chest_half, p.armhole_depth, p.shoulder_drop
    return _bezier((S, d), (S - 8, d + 0.45 * A), (S + 0.2 * (W - S), d + A), (W, d + A))


def sleeve_half(p: TeeParams) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Half sleeve (folded flat), cap length = armhole length + cap_ease."""
    target = _length(armhole_curve(p)) + p.cap_ease
    height = 0.3 * p.armhole_depth

    def cap(width):
        return _bezier((0, 0), (0.45 * width, 0), (0.55 * width, height), (width, height))

    lo, hi = 10.0, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _length(cap(mid)) < target else (lo, mid)
    width = (lo + hi) / 2
    hem = sleeve_hem_half(p)
    parts = {
        "cap": cap(width),
        "underarm": _line((width, height), (hem, p.sleeve_length)),
        "hem": _line((hem, p.sleeve_length), (0, p.sleeve_length)),
        "fold": _line((0, p.sleeve_length), (0, 0)),
    }
    return _join(parts)


def sleeve_hem_half(p: TeeParams) -> float:
    """Half the sleeve opening: the larger of the param and 70% of the armhole length."""
    return max(p.sleeve_hem_half, 0.7 * _length(armhole_curve(p)))


def _join(parts: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Concatenate edge polylines sharing end points; return outline + edge index ranges."""
    pts, edges = [], {}
    for name, seg in parts.items():
        start = len(pts)
        pts.extend(seg[:-1])  # next edge starts with this edge's last point
        edges[name] = (start, len(pts))
    edges[list(parts)[-1]] = (edges[list(parts)[-1]][0], 0)  # last edge wraps to point 0
    return np.asarray(pts), edges


def measurements(p: TeeParams) -> dict[str, float]:
    return {
        "chest_width": 2 * p.chest_half,
        "body_length": p.length,
        "shoulder_width": 2 * p.shoulder_half,
        "sleeve_length": p.sleeve_length,
        "neck_width": 2 * p.neck_half,
        "armhole_depth": p.armhole_depth,
    }


def mirror_full(half: np.ndarray) -> np.ndarray:
    """Full symmetric outline from a half (fold at x = 0)."""
    other = half[::-1] * [-1, 1]
    return np.vstack([half, other])


def flatlay_sleeve(p: TeeParams, side: int) -> np.ndarray:
    """Sleeve as it lies on a flat tee, attached at the armhole (side = +1 right, -1 left)."""
    arm = armhole_curve(p)  # inner edge follows the armhole, so no gap to the body
    top, under = arm[0], arm[-1]
    direction = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    normal = np.array([direction[1], -direction[0]])  # points up and out
    mid_out = (top + under) / 2 + direction * p.sleeve_length
    hem = sleeve_hem_half(p)
    poly = np.vstack([arm, [mid_out - normal * hem, mid_out + normal * hem]])
    return poly * [side, 1]


# ---- rendering -------------------------------------------------------------

def _marker_image(size_px: int) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.generateImageMarker(d, MARKER_ID, size_px)


def _camera_homography(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.025, 0.025, (4, 2)) * [w, h]
    angle = np.radians(rng.uniform(-4, 4))
    c = np.array([w / 2, h / 2])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = ((src - c) @ rot.T) * rng.uniform(0.92, 1.0) + c + jitter
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def render(
    regions: dict[str, list[np.ndarray]],
    colour: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Draw regions (mm polygons, already placed in world coords) plus marker.

    Returns photo, region masks (photo space), world_mm->photo_px homography,
    and marker corners in world mm (TL, TR, BR, BL).
    """
    all_pts = np.vstack([p for polys in regions.values() for p in polys])
    marker_total = MARKER_SIZE_MM + 2 * MARKER_BORDER_MM
    width_mm = all_pts[:, 0].max() + 120
    height_mm = all_pts[:, 1].max() + marker_total + 120
    w, h = int(width_mm * PX_PER_MM), int(height_mm * PX_PER_MM)

    # Backdrop with lighting gradient and noise.
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.85 + 0.25 * (xx / w) - 0.1 * (yy / h)
    world = (np.array(BACKDROP_BGR, float) * light[..., None]).clip(0, 255)

    masks_world = {}
    garment = np.zeros((h, w), np.uint8)
    for name, polys in regions.items():
        m = np.zeros((h, w), np.uint8)
        for poly in polys:
            cv2.fillPoly(m, [np.round(poly * PX_PER_MM).astype(np.int32)], 255)
        masks_world[name] = m
        garment |= m
    masks_world["garment"] = garment
    fabric = np.array(colour, float) * light[..., None] * (1 + 0.04 * rng.normal(size=(h, w, 1)))
    world = np.where(garment[..., None] > 0, fabric, world)
    # Faint seam shadow where sleeves meet the body.
    if "sleeve" in regions and "body" in regions:
        seam = cv2.dilate(masks_world["sleeve"], np.ones((3, 3))) & masks_world["body"]
        world[seam > 0] *= 0.8

    # Marker on a white card, bottom left, below the garment.
    m0 = np.array([60.0, all_pts[:, 1].max() + 60])
    card = (m0 * PX_PER_MM).astype(int)
    card_px = int(marker_total * PX_PER_MM)
    world[card[1] : card[1] + card_px, card[0] : card[0] + card_px] = 245
    mk_px = int(MARKER_SIZE_MM * PX_PER_MM)
    off = card + int(MARKER_BORDER_MM * PX_PER_MM)
    world[off[1] : off[1] + mk_px, off[0] : off[0] + mk_px] = _marker_image(mk_px)[..., None]
    tl = off / PX_PER_MM
    marker_mm = np.array([tl, tl + [MARKER_SIZE_MM, 0], tl + MARKER_SIZE_MM, tl + [0, MARKER_SIZE_MM]])

    world += rng.normal(0, 4, world.shape)
    world = world.clip(0, 255).astype(np.uint8)

    cam = _camera_homography(w, h, rng)
    photo = cv2.warpPerspective(world, cam, (w, h), borderMode=cv2.BORDER_REPLICATE)
    photo = cv2.GaussianBlur(photo, (0, 0), 0.7)
    masks = {k: cv2.warpPerspective(v, cam, (w, h), flags=cv2.INTER_NEAREST) for k, v in masks_world.items()}
    scale = np.diag([PX_PER_MM, PX_PER_MM, 1.0])
    return photo, masks, cam @ scale, marker_mm


def make_tee(p: TeeParams, out: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    truth: dict = {"params": asdict(p), "measurements": measurements(p), "photos": {}}

    for view in ("front", "back"):
        half, edges = body_half(p, back=(view == "back"))
        origin = np.array([p.chest_half + p.sleeve_length + 120, 80])
        body = mirror_full(half) + origin
        sleeves = [flatlay_sleeve(p, s) + origin for s in (1, -1)]
        photo, masks, H, marker_mm = render({"body": [body], "sleeve": sleeves}, p.colour_bgr, rng)
        _save(out, view, photo, masks)
        truth["photos"][view] = {
            "half_outline_mm": half.tolist(),
            "edges": edges,
            "origin_mm": origin.tolist(),
            "world_mm_to_photo_px": H.tolist(),
            "marker_corners_mm": marker_mm.tolist(),
        }

    half, edges = sleeve_half(p)
    origin = np.array([80.0, 80.0])
    sleeve = half + origin
    photo, masks, H, marker_mm = render({"sleeve": [sleeve]}, p.colour_bgr, rng)
    _save(out, "sleeve", photo, masks)
    truth["photos"]["sleeve"] = {
        "half_outline_mm": half.tolist(),
        "edges": edges,
        "origin_mm": origin.tolist(),
        "world_mm_to_photo_px": H.tolist(),
        "marker_corners_mm": marker_mm.tolist(),
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def _save(out: Path, view: str, photo: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    cv2.imwrite(str(out / f"{view}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    mdir = out / f"{view}.masks"
    mdir.mkdir(exist_ok=True)
    for name, m in masks.items():
        cv2.imwrite(str(mdir / f"{name}.png"), m)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--only", nargs="*", help="tee names, e.g. tee1")
    args = ap.parse_args(argv)
    out = Path(args.out)
    rows = []
    for i, p in enumerate(TEES):
        if args.only and p.name not in args.only:
            continue
        truth = make_tee(p, out / p.name, seed=i)
        rows.append({"tee": p.name, **truth["measurements"]})
        print(f"wrote {out / p.name}")
    if rows:
        with (out.parent / "ground_truth.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
