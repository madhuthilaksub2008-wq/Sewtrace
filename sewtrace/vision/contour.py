"""T15: SAM mask to a clean half outline in mm.

Owner: Madhu. See sewtrace/types.py for the coordinate convention.

Pipeline per photo:
    outline = smooth_outline(mask_to_outline(mask, H))
    half = body_half(remove_sleeves(outline))   # or sleeve_half(outline)
    edges = label_body_edges(half)              # or label_sleeve_edges(half)

SAM masks the whole t-shirt, sleeves included (REQ-SEG-02), so the front and back
outlines pass through remove_sleeves() first.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from sewtrace.config import get_config
from sewtrace.types import TEE_EDGES, Edge, Outline
from sewtrace.vision.marker import Homography
from sewtrace.vision.measure import _corner

UPSAMPLE = 2           # mask upsampling before tracing, for a sub-pixel edge
CORNER_DEG = 30.0      # turning angle that counts as a corner
CORNER_WINDOW_MM = 4.0
SMOOTH_SIGMA_MM = 1.5
FOLD_TOL_MM = 1.0
JOINT_SPACING_MM = 1.0   # outline spacing for the sleeve joint search
JOINT_TRIM_MM = 4.0      # least length of a sleeve-top or shoulder run ignored at its ends by the fit
JOINT_MIN_RUN_MM = 15.0  # shortest straight run on each side of the shoulder point
JOINT_REACH_MM = 20.0    # tangent fit length at the underarm point


# ---- polyline helpers -------------------------------------------------------

def _closed(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts, pts[:1]])


def _cumlen(pts: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def _resample_open(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Even resampling of an open polyline; both end points kept."""
    s = _cumlen(pts)
    if s[-1] <= 0:
        return pts[:1].copy()
    n = max(int(np.ceil(s[-1] / spacing)), 1)
    t = np.linspace(0, s[-1], n + 1)
    return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]


def _resample_closed(pts: np.ndarray, spacing: float) -> np.ndarray:
    return _resample_open(_closed(pts), spacing)[:-1]


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * a)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _turning(pts: np.ndarray, k: int) -> np.ndarray:
    """Absolute turning angle (deg) at each point of a closed polyline, chord window k."""
    v1 = pts - np.roll(pts, k, axis=0)
    v2 = np.roll(pts, -k, axis=0) - pts
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(axis=1)
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def _peaks(score: np.ndarray, threshold: float, radius: int) -> list[int]:
    """Indices of local maxima above threshold, strongest first, circular suppression."""
    score = score.copy()
    n, out = len(score), []
    while True:
        i = int(np.argmax(score))
        if score[i] < threshold:
            return out
        out.append(i)
        score[[(i + d) % n for d in range(-radius, radius + 1)]] = -np.inf


def _spacing(pts: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(np.diff(_closed(pts), axis=0), axis=1)))


def _orient_like_synth(pts: np.ndarray) -> np.ndarray:
    """Synth outlines run neck -> shoulder -> ... -> fold: positive shoelace area in y-down."""
    return pts if _signed_area(pts) > 0 else pts[::-1]


# ---- public API -------------------------------------------------------------

def px_per_mm(homography: Homography, at: np.ndarray) -> float:
    """Photo pixels per mm near the pixel ``at`` (x, y)."""
    step = 10.0
    mm = homography.px_to_mm(np.array([at, at + [step, 0.0], at + [0.0, step]]))
    return step / float(np.mean(np.linalg.norm(mm[1:] - mm[0], axis=1)))


def clean_mask(mask: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Binary mask -> its largest piece with holes filled and nicks and whiskers up to
    ``contour.mask.cleanup_mm`` closed or removed (REQ-SEG-03). A real SAM mask has small
    holes, edge nicks and stray blobs that must not turn into corners of the outline."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    # Round half up with an epsilon: at 3 px/mm the radius is 1.5 px, and float noise (or banker's
    # rounding) must not give 1 for one photo and 2 for another.
    radius = max(int(np.floor(get_config().contour.mask.cleanup_mm * px_per_mm + 0.5 + 1e-6)), 1)
    pad = radius + 2
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
    crop = np.pad(mask[y0 : ys.max() + pad + 1, x0 : xs.max() + pad + 1], pad)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    def largest_filled(m: np.ndarray) -> np.ndarray:
        found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = np.zeros_like(m)
        cv2.drawContours(out, [max(found, key=cv2.contourArea)], -1, 1, cv2.FILLED)
        return out

    # Whiskers first: closing would thicken them. Close before the largest piece is taken, so a
    # crack of a pixel or two across a sleeve seam is bridged and the sleeve stays part of the tee.
    opened = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
    crop = largest_filled(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel))
    out = np.zeros_like(mask)
    out[y0 : y0 + crop.shape[0] - 2 * pad, x0 : x0 + crop.shape[1] - 2 * pad] = crop[pad:-pad, pad:-pad]
    return out


def mask_to_outline(mask: np.ndarray, homography: Homography) -> Outline:
    """Largest external contour of a boolean mask, in the marker mm frame."""
    mask = np.asarray(mask).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    mask = clean_mask(mask, px_per_mm(homography, np.array([xs.mean(), ys.mean()])))
    ys, xs = np.nonzero(mask)
    pad = 4
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    crop = mask[y0 : ys.max() + pad + 1, x0 : xs.max() + pad + 1] * 255
    big = cv2.resize(crop, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours((big > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    px = (c + 0.5) / UPSAMPLE - 0.5 + [x0, y0]
    mm = homography.px_to_mm(px)
    return Outline.from_array(_orient_like_synth(mm))


def smooth_outline(outline: Outline, spacing_mm: float = 2.0) -> Outline:
    """Resample at even spacing and smooth, keeping sharp corners.

    The output starts at the input's first point, which is always kept as a vertex.
    """
    pts = outline.array()
    step = min(0.5, spacing_mm / 2)
    origin = pts[0]
    dense = _resample_closed(pts, step)
    k = max(int(round(CORNER_WINDOW_MM / step)), 1)
    turn = gaussian_filter1d(_turning(dense, k), k / 2, mode="wrap")
    corners = set(_peaks(turn, CORNER_DEG, 2 * k))
    corners.add(int(np.argmin(np.linalg.norm(dense - origin, axis=1))))
    breaks = sorted(corners)
    first = int(np.argmin(np.linalg.norm(dense[breaks] - origin, axis=1)))
    breaks = breaks[first:] + breaks[:first]
    sigma = SMOOTH_SIGMA_MM / step
    n, out = len(dense), []
    for a, b in zip(breaks, breaks[1:] + breaks[:1]):
        idx = np.arange(a, b + (n if b <= a else 0) + 1) % n
        seg = _smooth_pinned(dense[idx], sigma)
        out.append(_resample_open(seg, spacing_mm)[:-1])
    return Outline.from_array(np.vstack(out))


def _smooth_pinned(seg: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of an open run with both end points fixed (odd reflection)."""
    if len(seg) < 5:
        return seg
    m = min(len(seg) - 1, int(3 * sigma) + 1)
    head = 2 * seg[0] - seg[m:0:-1]
    tail = 2 * seg[-1] - seg[-2 : -m - 2 : -1]
    ext = np.vstack([head, seg, tail])
    sm = gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[m : m + len(seg)]
    sm[0], sm[-1] = seg[0], seg[-1]
    return sm


def _signed_turning(pts: np.ndarray, k: int) -> np.ndarray:
    """Turning angle (deg) at each point of a closed polyline, positive where the
    outline is convex when its shoelace area is positive."""
    v1 = pts - np.roll(pts, k, axis=0)
    v2 = np.roll(pts, -k, axis=0) - pts
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.degrees(np.arctan2(cross, (v1 * v2).sum(axis=1)))


@dataclass(frozen=True)
class _Joints:
    """Sleeve joints of a whole-garment outline, found by walking it once.

    ``pts`` is the outline resampled at JOINT_SPACING_MM and rolled so that it starts
    at the first underarm point, then runs along the hem side to the second one
    (index ``other``), then over the sleeves and the neck back to the start.
    """

    pts: np.ndarray
    other: int
    underarm_a: np.ndarray    # at index ``other``
    underarm_b: np.ndarray    # at index 0
    shoulder_a: np.ndarray    # between the first two sleeve-top corners and the neck
    shoulder_b: np.ndarray
    index_a: int              # outline samples nearest the shoulder points
    index_b: int
    neck_a: int               # the two neck corners (where the shoulder seams meet the neck)
    neck_b: int
    shoulder_dir_a: np.ndarray  # unit vectors along the shoulder seam, towards the neck
    shoulder_dir_b: np.ndarray
    side_dir_a: np.ndarray      # unit vectors along the side seam, towards the hem
    side_dir_b: np.ndarray


@dataclass(frozen=True)
class _HingeScan:
    """Broken-stick fits of one sleeve-top-plus-shoulder run, one per candidate bend position."""

    origin: np.ndarray
    e: np.ndarray           # unit vector along the chord of the fitted part
    n: np.ndarray           # unit normal
    bend: np.ndarray        # candidate bend positions along the chord, mm
    seam_mm: np.ndarray     # shoulder seam length (bend to neck corner) for each candidate
    ssr: np.ndarray         # sum of squared residuals of each fit
    coef: np.ndarray        # (candidates, 3): offset, slope, slope change

    def at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Shoulder point and unit direction of the shoulder seam (towards the neck) for candidate i."""
        c0, c1, c2 = self.coef[i]
        point = self.origin + self.bend[i] * self.e + (c0 + c1 * self.bend[i]) * self.n
        d = self.e + (c1 + c2) * self.n
        d /= np.linalg.norm(d)
        return point, d


def _hinge_scan(run: np.ndarray, trim_start: float, trim_end: float) -> _HingeScan:
    """Fit the run from a sleeve's top hem corner to the neck corner (REQ-GEO-01).

    The run is two straight lines (sleeve top, shoulder seam) that meet at a kink of only
    a few degrees (3 to 25), too soft for a corner detector, and the sleeve end is rounded
    by up to 15 mm. Fit a broken stick through the run, with the ends that the rounding
    covers left out: one line that bends once, continuous at the bend (fewer parameters
    than two free lines, so a faint kink cannot throw the bend point far).
    """
    s = _cumlen(run)
    keep = (s >= trim_start) & (s <= s[-1] - trim_end)
    pts = run[keep]
    if len(pts) < 2 * JOINT_MIN_RUN_MM / JOINT_SPACING_MM or np.ptp(s[keep]) < 2 * JOINT_MIN_RUN_MM:
        raise ValueError("sleeve top and shoulder seam are too short to find the shoulder point")
    chord = pts[-1] - pts[0]
    e = chord / np.linalg.norm(chord)
    n = np.array([-e[1], e[0]])
    u, v = (pts - pts[0]) @ e, (pts - pts[0]) @ n
    bend = np.arange(u[0] + JOINT_MIN_RUN_MM, u[-1] - JOINT_MIN_RUN_MM + 1e-9, JOINT_SPACING_MM / 2)
    coefs, ssrs = [], []
    for u0 in bend:
        design = np.c_[np.ones_like(u), u, np.maximum(u - u0, 0.0)]
        coef, *_ = np.linalg.lstsq(design, v, rcond=None)
        coefs.append(coef)
        ssrs.append(float(((design @ coef - v) ** 2).sum()))
    return _HingeScan(pts[0], e, n, bend, (u[-1] - bend) + trim_end, np.array(ssrs), np.array(coefs))


class ShoulderPointNotClear(ValueError):
    """The sleeve top bends too little away from the shoulder seam to place the shoulder point.

    ``spread_mm`` is the measured joint plateau and ``limit_mm`` the config limit it passed. They
    are kept so that refusals on real photos can be used to recalibrate the limit."""

    def __init__(self, message: str, spread_mm: float = float("nan"), limit_mm: float = float("nan")) -> None:
        super().__init__(message)
        self.spread_mm = spread_mm
        self.limit_mm = limit_mm


def _joint_fit(a: _HingeScan, b: _HingeScan) -> tuple[np.ndarray, np.ndarray]:
    """For each bend position i of the first shoulder: the best partner j of the second one (the
    shoulder seams within ``contour.joints.shoulder_symmetry_mm`` of each other in length) and the
    joint cost ssr_a[i] + ssr_b[j] (inf when no partner is that close)."""
    tol = get_config().contour.joints.shoulder_symmetry_mm
    cost = np.full(len(a.bend), np.inf)
    partner = np.zeros(len(a.bend), int)
    for i in range(len(a.bend)):
        near = np.nonzero(np.abs(b.seam_mm - a.seam_mm[i]) <= tol)[0]
        if len(near) == 0:
            continue
        partner[i] = near[np.argmin(b.ssr[near])]
        cost[i] = a.ssr[i] + b.ssr[partner[i]]
    return cost, partner


def _joint_bend(a: _HingeScan, b: _HingeScan) -> tuple[int, int]:
    """Best pair of bend positions (i, j), one per shoulder, from _joint_fit(). A tee is nearly
    symmetric, so the two runs give independent evidence for one seam length: a faint kink on
    one side is placed by the other.

    The pair is refused when the fit is a plateau: the pairs that fit nearly as well as the best
    one (``contour.joints.near_best``) put the shoulder seam at lengths that spread over more
    than ``contour.joints.shoulder_plateau_mm``. The spread is measured on the joint cost, the
    one that was fitted, not on each side alone: one faint side that the other side places
    well is not a plateau."""
    cfg = get_config().contour.joints
    cost, partner = _joint_fit(a, b)
    if not np.isfinite(cost).any():
        raise ValueError("the two shoulder seams are not about equally long: is the whole t-shirt in the mask?")
    i = int(np.argmin(cost))
    possible = np.nonzero(cost <= cfg.near_best * cost[i])[0]
    spread = max(float(np.ptp(a.seam_mm[possible])), float(np.ptp(b.seam_mm[partner[possible]])))
    if spread > cfg.shoulder_plateau_mm:
        raise ShoulderPointNotClear(
            "the shoulder point is not clear: the sleeve top and the shoulder seam are nearly in line. "
            "Put each sleeve at an angle down from the shoulder. Keep a gap between the sleeve and the side "
            f"of the t-shirt (spread {spread:.0f} mm, limit {cfg.shoulder_plateau_mm:g} mm)",
            spread_mm=spread, limit_mm=cfg.shoulder_plateau_mm,
        )
    return i, int(partner[i])


def _ring_run(pts: np.ndarray, first: int, last: int) -> np.ndarray:
    """Samples first..last inclusive of a closed polyline, wrapping."""
    n = len(pts)
    return pts[np.arange(first, last + 1 + (n if last < first else 0)) % n]


def _net_turning(pts: np.ndarray, first: int, last: int) -> float:
    """Total turning (deg) of a closed polyline over the vertices strictly between first and last,
    walking forward. Unlike a corner count it does not change when a corner is rounded."""
    n = len(pts)
    v_in = pts - np.roll(pts, 1, axis=0)
    v_out = np.roll(pts, -1, axis=0) - pts
    turn = np.degrees(np.arctan2(v_in[:, 0] * v_out[:, 1] - v_in[:, 1] * v_out[:, 0], (v_in * v_out).sum(axis=1)))
    return float(turn[np.arange(first + 1, last + (n if last < first else 0)) % n].sum())


def _rounding_trim(turn_fine: np.ndarray, corner: int, direction: int, limit_mm: float) -> float:
    """Arc length (mm, at JOINT_SPACING_MM) from a corner to the end of its rounding, walking
    ``direction`` (+1 or -1): where the outline turns less than ``contour.joints.rounding_deg``
    over the fine chord. At least JOINT_TRIM_MM, at most ``limit_mm``."""
    cfg = get_config().contour.joints
    n = len(turn_fine)
    steps = int(limit_mm / JOINT_SPACING_MM)
    run = 0
    for i in range(1, steps + 1):
        if turn_fine[(corner + direction * i) % n] < cfg.rounding_deg:
            break
        run = i
    return max(run * JOINT_SPACING_MM + cfg.rounding_margin_mm, JOINT_TRIM_MM)


def _six_corners(score: np.ndarray, other: int, n: int, radius: int) -> list[int]:
    """The six convex corners of the top arc (``other`` .. ``n``): sleeve hem bottom and top and
    neck, left and right. The strongest peaks are candidates, and the six are chosen to be
    mirror pairs about the middle of the arc (outer pair: sleeve hem bottoms, then sleeve hem
    tops, then the neck corners), because a corner that is rounded or beside a scoop can be
    weaker than a wiggle in the edge, but a wiggle is not mirrored by another one. The two hem
    pairs may be off from mirrored by ``pair_tol_hem_mm``, the neck pair by ``pair_tol_neck_mm``:
    the neck pair also carries the difference of the two sleeve tops, on top of the difference of
    the two sleeve lengths, so it can be off by more."""
    cfg = get_config().contour.joints
    cands = sorted(_peaks(score, cfg.corner_min_deg, radius)[:12])
    arc = n - other
    hem_tol, neck_tol = cfg.pair_tol_hem_mm / JOINT_SPACING_MM, cfg.pair_tol_neck_mm / JOINT_SPACING_MM
    pairs = [(i, j, abs((i - other) + (j - other) - arc)) for ii, i in enumerate(cands) for j in cands[ii + 1 :]]
    hem_pairs = [(i, j) for i, j, off in pairs if off <= hem_tol]
    neck_pairs = [(i, j) for i, j, off in pairs if off <= neck_tol]
    best: tuple[float, list[int]] | None = None
    for p0 in hem_pairs:
        for p1 in hem_pairs:
            if not p0[0] < p1[0] < p1[1] < p0[1]:
                continue
            for p2 in neck_pairs:
                if not p1[0] < p2[0] < p2[1] < p1[1]:
                    continue
                strength = sum(min(score[i], score[j]) for i, j in (p0, p1, p2))
                if best is None or strength > best[0]:
                    best = (strength, [p0[0], p1[0], p2[0], p2[1], p1[1], p0[1]])
    if best is None:
        raise ValueError(
            f"found {len(cands)} convex corners over the sleeves and neck, but no 6 that mirror each other "
            "(2 neck, 4 sleeve hem): is the whole t-shirt, with both sleeves, in the mask, neck up?"
        )
    return best[1]


def _find_joints(outline: Outline) -> _Joints:
    """Shoulder and underarm points of a whole t-shirt outline (REQ-GEO-01).

    Underarm points are the two sharpest concave corners. They split the outline into the
    hem arc and the top arc; the top arc is the one that turns more (sleeves and neck add
    about 140 degrees over the hem and its two corners), a total that rounding a corner
    does not change. The top arc holds six convex corners: the strongest six found over a
    chord that is long enough for rounded corners (``contour.joints``). Their order is fixed:
    sleeve hem (2), neck (2) ... The shoulder point of each sleeve lies on the run between its
    top hem corner and the neck corner. A different count or order is a ValueError that says so.
    """
    cfg = get_config().contour.joints
    pts = _resample_closed(_orient_like_synth(outline.array()), JOINT_SPACING_MM)
    n = len(pts)
    k = max(int(round(cfg.corner_window_mm / JOINT_SPACING_MM)), 1)
    turn = gaussian_filter1d(_signed_turning(pts, k), k / 2, mode="wrap")
    concave = sorted(_peaks(-turn, cfg.underarm_min_deg, 2 * k)[:2])
    if len(concave) < 2:
        raise ValueError("found no underarm corners: is the whole t-shirt, with both sleeves, in the mask?")
    u1, u2 = concave
    # The arc u1 -> u2 is the hem arc when it turns less than the other one.
    start, other_abs = (u1, u2) if _net_turning(pts, u1, u2) < _net_turning(pts, u2, u1) else (u2, u1)
    pts = np.roll(pts, -start, axis=0)
    turn = np.roll(turn, -start)
    other = (other_abs - start) % n

    # Two peaks of one rounded corner can be 17 mm apart, so suppress over 3 windows.
    convex = sorted(_peaks(turn, cfg.corner_min_deg, 3 * k))
    hem = [c for c in convex if 0 < c < other]
    inner = np.full(n, -np.inf)
    inner[other + k : n - k] = turn[other + k : n - k]
    top = _six_corners(inner, other, n, 3 * k)
    c0, c1, c2, c3, c4, c5 = top
    reach = int(JOINT_REACH_MM / JOINT_SPACING_MM)
    h = hem[0] if hem else other // 2

    def underarm(into_lo: int, into_hi: int, out_lo: int, out_hi: int) -> np.ndarray:
        return _corner(_ring_run(pts, into_lo, into_hi), _ring_run(pts, out_lo, out_hi), JOINT_REACH_MM, JOINT_REACH_MM)

    prev = hem[-1] if hem else 0
    under_a = underarm(other - min(reach, other - prev - 2), other, other, other + min(reach, c0 - other - 2))
    under_b = underarm(n - min(reach, n - c5 - 2), n, 0, min(reach, h - 2))

    fine = gaussian_filter1d(_signed_turning(pts, max(int(round(4.0 / JOINT_SPACING_MM)), 1)), 2, mode="wrap")
    fine = np.abs(fine)

    scan_a = _shoulder_scan(pts, fine, sleeve=c1, neck=c2)
    scan_b = _shoulder_scan(pts, fine, sleeve=c4, neck=c3)
    i, j = _joint_bend(scan_a, scan_b)
    shoulder_a, dir_a = scan_a.at(i)
    shoulder_b, dir_b = scan_b.at(j)
    ia = c1 + int(np.argmin(np.linalg.norm(pts[c1 : c2 + 1] - shoulder_a, axis=1)))
    ib = c3 + int(np.argmin(np.linalg.norm(pts[c3 : c4 + 1] - shoulder_b, axis=1)))
    side_a = pts[max(other - reach, 0)] - under_a
    side_b = pts[min(reach, other)] - under_b
    return _Joints(pts, other, under_a, under_b, shoulder_a, shoulder_b, ia, ib, c2, c3,
                   dir_a, dir_b, side_a / np.linalg.norm(side_a), side_b / np.linalg.norm(side_b))


def _shoulder_scan(pts: np.ndarray, fine: np.ndarray, sleeve: int, neck: int) -> _HingeScan:
    """_hinge_scan of the run between a sleeve top hem corner and the neck corner, walked from
    the sleeve. ``fine`` is the turning over a short chord: the rounding at each end is left out."""
    forward = sleeve < neck
    run = pts[sleeve : neck + 1] if forward else pts[neck : sleeve + 1][::-1]
    span = abs(neck - sleeve) * JOINT_SPACING_MM / 3
    step = 1 if forward else -1
    return _hinge_scan(run, _rounding_trim(fine, sleeve, step, span), _rounding_trim(fine, neck, -step, span))


def find_sleeve_joints(outline: Outline) -> tuple[np.ndarray, np.ndarray]:
    """(shoulder points, underarm points) of a whole t-shirt outline: two (2, 2) arrays, mm.

    Row 0 is the sleeve met first on the outline walk (a), row 1 the other (b).
    """
    j = _find_joints(outline)
    return np.vstack([j.shoulder_a, j.shoulder_b]), np.vstack([j.underarm_a, j.underarm_b])


def _perp_towards(direction: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to ``direction`` that points to the side of ``toward``."""
    perp = np.array([-direction[1], direction[0]])
    perp /= np.linalg.norm(perp)
    return perp if perp @ toward > 0 else -perp


def default_armhole(
    shoulder: np.ndarray,
    underarm: np.ndarray,
    shoulder_dir: np.ndarray,
    side_dir: np.ndarray,
    centre: np.ndarray,
) -> np.ndarray:
    """Armhole from the shoulder point to the underarm point, about 1 mm apart.

    The flat-lay photo shows only the two end points. The default is a cubic Bezier
    that leaves the shoulder seam at a right angle and meets the side seam at a right
    angle, both bending towards the body centre (handle lengths in config).
    """
    cfg = get_config().contour.armhole
    chord = float(np.linalg.norm(underarm - shoulder))
    ctrl1 = shoulder + cfg.shoulder_handle * chord * _perp_towards(shoulder_dir, centre - shoulder)
    ctrl2 = underarm + cfg.underarm_handle * chord * _perp_towards(side_dir, centre - underarm)
    t = np.linspace(0, 1, max(int(np.ceil(chord)), 8))[:, None]
    return ((1 - t) ** 3 * shoulder + 3 * (1 - t) ** 2 * t * ctrl1
            + 3 * (1 - t) * t**2 * ctrl2 + t**3 * underarm)


def cut_sleeves(outline: Outline) -> tuple[Outline, np.ndarray]:
    """Whole t-shirt outline -> body outline: cut each sleeve off at the shoulder point
    and the underarm point and close the gap with the default armhole (REQ-GEO-01).

    Only the two joints are visible in a flat-lay photo, so the armhole shape here is
    a default, not a measurement. The model infers the real shape (REQ-RSN-04).
    Also returns a boolean array, True for each point of the body outline that lies on
    a default armhole, so callers can leave those points out of an edge comparison.
    """
    body, invented, _ = cut_sleeves_flagged(outline)
    return body, invented


def cut_sleeves_flagged(outline: Outline) -> tuple[Outline, np.ndarray, np.ndarray]:
    """cut_sleeves(), plus a boolean array that is True for each body point on the neck.

    The neck is the outline between the two neck corners. On a front flat-lay it follows
    the BACK neckline: the back panel is seen through the neck opening, so the front neck
    edge is not on the outline (front neck depth is inferred, REQ-RSN-04).
    """
    j = _find_joints(outline)
    pts, other = j.pts, j.other
    centre = _centroid(pts)
    arm_a = default_armhole(j.shoulder_a, j.underarm_a, j.shoulder_dir_a, j.side_dir_a, centre)
    arm_b = default_armhole(j.shoulder_b, j.underarm_b, j.shoulder_dir_b, j.side_dir_b, centre)
    parts = [
        (j.underarm_b[None], False),
        (pts[1:other], False),
        (j.underarm_a[None], True),
        (arm_a[::-1][1:-1], True),
        (j.shoulder_a[None], True),
        (pts[j.index_a + 1 : j.index_b], False),
        (j.shoulder_b[None], True),
        (arm_b[1:-1], True),
    ]
    body = np.vstack([p for p, _ in parts])
    invented = np.concatenate([np.full(len(p), flag) for p, flag in parts])
    neck = np.zeros(len(body), bool)
    run_start = sum(len(p) for p, _ in parts[:5])  # first point of pts[index_a + 1 : index_b]
    samples = np.arange(j.index_a + 1, j.index_b)
    neck[run_start : run_start + len(samples)] = (samples >= j.neck_a) & (samples <= j.neck_b)
    return Outline.from_array(body), invented, neck


def remove_sleeves(outline: Outline) -> Outline:
    """cut_sleeves() without the armhole flags."""
    return cut_sleeves(outline)[0]


def body_half(outline: Outline) -> Outline:
    """Full body outline -> half panel: find the vertical symmetry axis, mirror
    the two sides into one averaged half, fold on x = 0, neck point at y = 0."""
    pts = _resample_closed(outline.array(), 1.0)
    c = _centroid(pts)
    centred = pts - c
    tree = cKDTree(centred)

    def cost(v: np.ndarray) -> float:
        theta, dx = v
        q = centred @ _rot(-theta).T
        mirrored = np.c_[2 * dx - q[:, 0], q[:, 1]] @ _rot(theta).T
        d, _ = tree.query(mirrored)
        return float(np.mean(np.minimum(d, 20.0)))

    grid = np.radians(np.arange(-30, 30.5, 1.0))
    t0 = grid[int(np.argmin([cost(np.array([t, 0.0])) for t in grid]))]
    res = minimize(cost, np.array([t0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6, "initial_simplex": [[t0, 0], [t0 + 0.01, 0], [t0, 2.0]]})
    theta, dx = res.x
    q = centred @ _rot(-theta).T - [dx, 0]

    top, bottom = _axis_crossings(q)
    if (q[:, 1].max() - bottom) > (top - q[:, 1].min()):  # neck recess is at the bottom: flip
        q = -q
    q = _orient_like_synth(q)
    right, left = _split_sides(q)
    avg = _average_sides(right, left * [-1, 1])
    return _close_half(avg, spacing=2.0)


def _axis_crossings(q: np.ndarray) -> tuple[float, float]:
    """y of the top and bottom places where a closed outline crosses x = 0."""
    x, y = q[:, 0], q[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    hit = np.nonzero((x <= 0) != (xn <= 0))[0]
    ys = y[hit] + (yn[hit] - y[hit]) * (0 - x[hit]) / (xn[hit] - x[hit])
    return float(ys.min()), float(ys.max())


def _split_sides(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right (x > 0) and left runs, each ordered top to bottom, with axis end points."""
    n = len(q)
    x = q[:, 0]
    hit = np.nonzero((x <= 0) != (np.roll(x, -1) <= 0))[0]
    cross = []
    for i in hit:
        a, b = q[i], q[(i + 1) % n]
        t = a[0] / (a[0] - b[0])
        cross.append((i, a + t * (b - a)))
    cross.sort(key=lambda c: c[1][1])
    (i_top, p_top), (i_bot, p_bot) = cross[0], cross[-1]

    def run(i: int, j: int) -> np.ndarray:  # points strictly after i up to j, cyclic
        idx = np.arange(i + 1, j + 1 + (n if j < i else 0)) % n
        return q[idx]

    a = np.vstack([p_top, run(i_top, i_bot), p_bot])
    b = np.vstack([p_bot, run(i_bot, i_top), p_top])[::-1]
    if np.median(a[:, 0]) < np.median(b[:, 0]):
        a, b = b, a
    return a, b


def _average_sides(right: np.ndarray, left_mirrored: np.ndarray) -> np.ndarray:
    """Midpoint of each right point and its nearest point on the mirrored left run."""
    dense = _resample_open(left_mirrored, 0.25)
    _, j = cKDTree(dense).query(right)
    avg = (right + dense[j]) / 2
    avg[0, 0] = avg[-1, 0] = 0.0
    return avg


def _close_half(chain: np.ndarray, spacing: float) -> Outline:
    """Chain from the top fold point round to the bottom fold point -> closed half outline,
    smoothed, fold on x = 0, the topmost point (neck point / cap top) on y = 0."""
    chain = chain.copy()
    chain[0, 0] = chain[-1, 0] = 0.0
    fold = _resample_open(np.vstack([chain[-1], chain[0]]), spacing)[1:-1]
    half = np.vstack([chain, fold])
    half = smooth_outline(Outline.from_array(half), spacing).array()
    half[:, 0] = np.maximum(half[:, 0], 0)
    half[:, 1] -= half[:, 1].min()
    return Outline.from_array(half)


def sleeve_half(outline: Outline) -> Outline:
    """Folded sleeve outline (fold on the left, roughly vertical) -> half panel
    with the fold on x = 0 and the cap top at y = 0."""
    pts = _resample_closed(outline.array(), 1.0)
    c = pts.mean(axis=0)
    centred = pts - c

    def support(theta: float) -> tuple[int, np.ndarray]:
        q = centred @ _rot(-theta).T
        near = q[:, 0] < q[:, 0].min() + FOLD_TOL_MM
        return int(near.sum()), near

    grid = np.radians(np.arange(-30, 30.05, 0.25))
    theta = grid[int(np.argmax([support(t)[0] for t in grid]))]
    _, near = support(theta)
    # Refine: total least squares line through the fold points.
    fp = pts[near]
    _, _, vt = np.linalg.svd(fp - fp.mean(axis=0))
    d = vt[0] if vt[0][1] > 0 else -vt[0]
    theta = np.arctan2(-d[0], d[1])  # angle of the fold from +y
    q = (pts - fp.mean(axis=0)) @ _rot(-theta).T
    q = _orient_like_synth(q)

    on_fold = np.abs(q[:, 0]) < FOLD_TOL_MM
    chain = _longest_run(q, ~on_fold)
    if chain[0, 1] > chain[-1, 1]:
        chain = chain[::-1]
    return _close_half(chain, spacing=2.0)


def _longest_run(q: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Longest cyclic run of points where keep is True."""
    n = len(q)
    if keep.all():
        return q
    start = int(np.argmin(keep))  # a False index
    order = (np.arange(n) + start) % n
    best, cur = (0, 0), None
    for pos, i in enumerate(order):
        if keep[i]:
            cur = pos if cur is None else cur
            if pos - cur + 1 > best[1] - best[0]:
                best = (cur, pos + 1)
        else:
            cur = None
    return q[order[best[0] : best[1]]]


def _edges(names: tuple[str, ...], corners: list[int]) -> list[Edge]:
    bounds = list(corners) + [0]
    return [Edge(name=nm, start=bounds[i], end=bounds[i + 1]) for i, nm in enumerate(names)]


def label_body_edges(half: Outline) -> list[Edge]:
    """Edges in TEE_EDGES["front"] order: neck, shoulder, armhole, side, hem, fold."""
    pts = half.array()
    n = len(pts)
    k = max(int(round(get_config().contour.labels.corner_window_mm / _spacing(pts))), 1)
    turn = _turning(pts, k)
    top = int(np.argmin(pts[:, 1]))  # neck point is the highest point; snap to the sharpest corner near it
    near = np.arange(max(top - k - 1, 1), top + k + 2)
    neck_pt = int(near[np.argmax(turn[near])])
    hem_fold = _bottom_fold_corner(pts)
    inner = np.full(n, -np.inf)
    inner[neck_pt + 2 * k : hem_fold - 2 * k] = turn[neck_pt + 2 * k : hem_fold - 2 * k]
    shoulder, underarm, hem_side = sorted(_peaks(inner, 0.0, 2 * k)[:3])
    return _edges(TEE_EDGES["front"], [0, neck_pt, shoulder, underarm, hem_side, hem_fold])


def label_sleeve_edges(half: Outline) -> list[Edge]:
    """Edges in TEE_EDGES["sleeve"] order: cap, underarm, hem, fold."""
    pts = half.array()
    n = len(pts)
    k = max(int(round(get_config().contour.labels.corner_window_mm / _spacing(pts))), 1)
    turn = _turning(pts, k)
    hem_fold = _bottom_fold_corner(pts)
    inner = np.full(n, -np.inf)
    inner[2 * k : hem_fold - 2 * k] = turn[2 * k : hem_fold - 2 * k]
    cap_end, hem_side = sorted(_peaks(inner, 0.0, 2 * k)[:2])
    return _edges(TEE_EDGES["sleeve"], [0, cap_end, hem_side, hem_fold])


def _bottom_fold_corner(pts: np.ndarray) -> int:
    """Index of the fold's bottom end (x = 0, largest y)."""
    on_fold = pts[:, 0] < 0.5
    idx = np.nonzero(on_fold)[0]
    return int(idx[np.argmax(pts[idx, 1])])
