"""Hard synthetic tees: the clean set plus the faults real photos have (REQ-GEO-01, REQ-SCL-03).

The clean set (sewtrace/mock/synth.py) draws sharp polygons that no real tee looks like.
This module draws the same tees with the faults that most likely break the vision chain
on real photos. Each fault is a field of HardParams, so each one is controllable and seeded:

  * soft shoulder-to-sleeve kink (3 to 10 degrees, rounded over a few cm)
  * rounded sleeve hem, sleeve cap and body hem corners (radius 3 to 15 mm)
  * wavy edges (0.5 to 2 mm), different on the left and the right
  * left and right sleeves at different angles and lengths, a left/right body offset
  * fabric texture, wrinkle shading, edge shadow, uneven lighting
  * barrel lens distortion on the whole photo (the marker too), stronger perspective tilt
  * a curled marker card, low JPEG quality
  * mock SAM masks with edge noise, small holes and stray blobs

Truth is exact: the measurements are read off the perturbed half outlines at the design
vertices (the tee a tape measure would see), never from sewtrace/vision. Waves and corner
rounding are zero at those vertices, except the chest width, which is read from the wavy
side seam where the chest line crosses it, as a tape would.

    python -m sewtrace.mock.synth --hard --out data/synth_hard
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import shapely

from sewtrace.config import MARKER_SIZE_MM, get_config
from sewtrace.mock import synth
from sewtrace.mock.synth import BACKDROP_BGR, MARKER_BORDER_MM, PX_PER_MM

EDGE_ID = {"neck": 0, "shoulder": 1, "armhole": 2, "side": 3, "hem": 4, "cap": 5, "underarm": 6,
           "sleeve_under": 7, "sleeve_hem": 8, "sleeve_top": 9, "paper": 10}
MASK_NOISE = ("t-shirt", "sleeve", "paper")   # masks the pipeline asks for; "body" stays exact


@dataclass(frozen=True)
class HardParams:
    """Every fault of one hard tee. Lengths in mm, angles in degrees."""

    seed: int
    kink_deg: tuple[float, float]     # shoulder-to-sleeve turning angle, (right, left), 3 to 10
    kink_span_mm: float               # tangent length over which the kink is rounded
    corner_radius_mm: float           # sleeve hem, sleeve cap and body hem corners, 3 to 15
    wave_mm: float                    # peak edge waviness, 0.5 to 2
    wave_length_mm: float
    sleeve_length_delta_mm: float     # right sleeve longer, left shorter, by this
    asymmetry_mm: float               # left body wider than the right at the chest, 1 to 3
    shade: float                      # wrinkle shading strength
    shadow: float                     # edge shadow strength on the backdrop
    light_tilt: float                 # uneven lighting
    barrel_k1: float                  # x_d = x_u (1 + k1 r^2), r = 1 at the half diagonal; k1 < 0 is barrel
    tilt: float                       # camera corner jitter, fraction of the photo size
    rotation_deg: float
    curl_mm: float                    # marker card lift at its far edge
    jpeg_quality: int
    mask_blur_px: float               # mock SAM mask edge: blur and jitter, 2 to 4 px
    mask_jitter_px: float
    mask_holes: int
    mask_strays: int


def hard_params(index: int) -> HardParams:
    """Hard faults of tee ``index`` (0 based). The main faults are spread over their full
    ranges across five consecutive tees, in a different order for each fault."""
    rng = np.random.default_rng(1000 + index)
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    order = ((0.0, 0.5, 1.0, 0.25, 0.75), (1.0, 0.25, 0.75, 0.0, 0.5), (0.5, 1.0, 0.0, 0.75, 0.25),
             (0.25, 0.75, 0.5, 1.0, 0.0), (0.75, 0.0, 0.25, 0.5, 1.0))
    s = [o[index % 5] for o in order]
    kink_r = 3 + 7 * s[0]
    kink_l = float(np.clip(kink_r + rng.choice([-1, 1]) * u(1.5, 4.0), 3, 10))
    return HardParams(
        seed=index,
        kink_deg=(kink_r, kink_l),
        kink_span_mm=u(8, 20),
        corner_radius_mm=3 + 12 * s[1],
        wave_mm=0.5 + 1.5 * s[2],
        wave_length_mm=u(70, 160),
        sleeve_length_delta_mm=u(2, 6),
        asymmetry_mm=1 + 2 * s[3],
        shade=u(0.06, 0.14),
        shadow=u(0.25, 0.45),
        light_tilt=u(0.10, 0.25),
        barrel_k1=-(0.003 + 0.009 * s[4]),
        tilt=u(0.04, 0.06),
        rotation_deg=u(3, 7) * float(rng.choice([-1, 1])),
        curl_mm=u(0.5, 2.0),
        jpeg_quality=int(rng.integers(55, 76)),
        mask_blur_px=u(2.0, 4.0),
        mask_jitter_px=u(1.5, 3.0),
        mask_holes=int(rng.integers(3, 7)),
        mask_strays=int(rng.integers(3, 9)),
    )


# ---- polyline helpers --------------------------------------------------------

def _cumlen(pts: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def wave_polyline(pts: np.ndarray, amp_mm: float, wavelength_mm: float, rng: np.random.Generator,
                  taper_mm: float = 12.0) -> np.ndarray:
    """Push an open polyline sideways by a smooth random wave of peak ``amp_mm``.

    The wave dies out at both ends, so the vertices the edge joins do not move."""
    if amp_mm <= 0 or len(pts) < 4:
        return pts.copy()
    s = _cumlen(pts)
    f = np.zeros_like(s)
    for _ in range(3):
        f += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * s / (wavelength_mm * rng.uniform(0.7, 1.4))
                                            + rng.uniform(0, 2 * np.pi))
    f *= _smoothstep(np.minimum(s, s[-1] - s) / min(taper_mm, s[-1] / 4))
    peak = np.abs(f).max()
    if peak < 1e-9:
        return pts.copy()
    t = np.gradient(pts, axis=0)
    normal = np.c_[-t[:, 1], t[:, 0]] / np.linalg.norm(t, axis=1, keepdims=True)
    return pts + (amp_mm * f / peak)[:, None] * normal


def _edge_rng(hp: HardParams, view: int, side: int, edge: str) -> np.random.Generator:
    return np.random.default_rng([hp.seed, view, side, EDGE_ID[edge]])


def wave_edges(pts: np.ndarray, edges: dict[str, tuple[int, int]], hp: HardParams, view: int, side: int,
                skip: tuple[str, ...] = ("fold", "armhole")) -> np.ndarray:
    """Wave every edge of a half outline except the fold (straight) and the hidden armhole."""
    out = pts.copy()
    for name, (a, b) in edges.items():
        if name in skip or b <= a:
            continue
        out[a : b + 1] = wave_polyline(pts[a : b + 1], hp.wave_mm, hp.wave_length_mm,
                                       _edge_rng(hp, view, side, name))
    return out


def _arc_index(ring: np.ndarray, i: int, reach_mm: float, direction: int) -> int:
    """Index of the ring point ``reach_mm`` of arc length from point i, walking ``direction``."""
    n = len(ring)
    walked, j = 0.0, i
    for _ in range(n):
        k = (j + direction) % n
        walked += float(np.linalg.norm(ring[k] - ring[j]))
        j = k
        if walked >= reach_mm:
            break
    return j


def fillet(ring: np.ndarray, target: np.ndarray, reach_mm: float) -> np.ndarray:
    """Round the vertex of a dense closed ring nearest to ``target``: replace the stretch
    ``reach_mm`` of arc length either side with a quadratic Bezier whose control point is
    the vertex, which is tangent to both edges of a sharp corner."""
    i = int(np.argmin(np.linalg.norm(ring - target, axis=1)))
    a, b = _arc_index(ring, i, reach_mm, -1), _arc_index(ring, i, reach_mm, +1)
    p0, p1, p2 = ring[a], ring[i], ring[b]
    t = np.linspace(0, 1, 2 * max(int(reach_mm), 4) + 1)[:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    rolled = np.roll(ring, -b, axis=0)                      # starts at b, ends just before a... then a
    keep = rolled[1 : (a - b) % len(ring)]                  # points strictly between b and a, going forward
    return np.vstack([curve, keep])


def turning_deg(ring: np.ndarray, target: np.ndarray, probe_mm: float = 3.0) -> float:
    """Turning angle (deg) of a dense ring at the vertex nearest ``target``."""
    i = int(np.argmin(np.linalg.norm(ring - target, axis=1)))
    v1 = ring[i] - ring[_arc_index(ring, i, probe_mm, -1)]
    v2 = ring[_arc_index(ring, i, probe_mm, +1)] - ring[i]
    return float(np.degrees(np.abs(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2))))


def round_corner(ring: np.ndarray, target: np.ndarray, radius_mm: float) -> np.ndarray:
    """Fillet of radius ``radius_mm`` at the corner nearest ``target`` (tangent length r tan(turn/2))."""
    turn = min(turning_deg(ring, target), 150.0)
    return fillet(ring, target, max(radius_mm * np.tan(np.radians(turn) / 2), 1.0))


def _dense(poly: np.ndarray, spacing_mm: float = 1.0) -> np.ndarray:
    ring = shapely.segmentize(shapely.Polygon(poly), spacing_mm)
    return np.asarray(ring.exterior.coords)[:-1]


# ---- the hard geometry (mm, world coordinates once the origin is added) --------

def _shoulder_deg(p: synth.TeeParams) -> float:
    return float(np.degrees(np.arctan2(p.shoulder_drop, p.shoulder_half - p.neck_half)))


def _sleeve_corners(p: synth.TeeParams, axis_deg: float, length_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """(bottom, top) outer corners of a flat-lay sleeve whose axis points ``axis_deg`` below +x."""
    arm = synth.armhole_curve(p)
    direction = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))])
    normal = np.array([direction[1], -direction[0]])
    mid_out = (arm[0] + arm[-1]) / 2 + direction * length_mm
    return mid_out - normal * p.sleeve_hem_half, mid_out + normal * p.sleeve_hem_half


def sleeve_axis_for_kink(p: synth.TeeParams, kink_deg: float, length_mm: float) -> float:
    """Sleeve axis angle that makes the sleeve top edge turn ``kink_deg`` away from the shoulder line."""
    arm0 = synth.armhole_curve(p)[0]
    want = _shoulder_deg(p) + kink_deg

    def top_edge_deg(axis: float) -> float:
        top = _sleeve_corners(p, axis, length_mm)[1] - arm0
        return float(np.degrees(np.arctan2(top[1], top[0])))

    lo, hi = -20.0, 70.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if top_edge_deg(mid) < want else (lo, mid)
    return (lo + hi) / 2


def flatlay_sleeve_hard(p: synth.TeeParams, hp: HardParams, view: int, side: int) -> tuple[np.ndarray, dict]:
    """Right-hand flat-lay sleeve polygon (mm, relative to the body origin) with wavy outer
    edges, for side 0 (right) or 1 (left, mirrored by the caller). Also its record."""
    length = p.sleeve_length + (hp.sleeve_length_delta_mm if side == 0 else -hp.sleeve_length_delta_mm)
    axis = sleeve_axis_for_kink(p, hp.kink_deg[side], length)
    arm = synth.armhole_curve(p)
    bottom, top = _sleeve_corners(p, axis, length)

    def wave(a, b, name):
        return wave_polyline(synth._line(a, b, 24), hp.wave_mm, hp.wave_length_mm, _edge_rng(hp, view, side, name))

    under, hem, top_edge = wave(arm[-1], bottom, "sleeve_under"), wave(bottom, top, "sleeve_hem"), wave(top, arm[0], "sleeve_top")
    poly = np.vstack([arm, under[1:], hem[1:], top_edge[1:-1]])
    return poly, {"axis_deg": axis, "length_mm": length, "bottom_mm": bottom.tolist(), "top_mm": top.tolist()}


def hard_body_sides(p: synth.TeeParams, hp: HardParams, view: int, back_neck: bool):
    """(right half, left half, edges): the same tee half twice, waved independently and with
    the left side wider by ``asymmetry_mm / chest_half`` (as a fraction)."""
    half, edges = synth.body_half(p, back=back_neck)
    right = wave_edges(half, edges, hp, view, 0)
    left = wave_edges(half, edges, hp, view, 1) * [1 + hp.asymmetry_mm / p.chest_half, 1]
    return right, left, edges


def _full_ring(right: np.ndarray, left: np.ndarray, fold_start: int) -> np.ndarray:
    """Closed outline of a whole body from its two halves, without the fold sliver."""
    r, lft = right[: fold_start + 1], left[: fold_start + 1][::-1] * [-1, 1]
    return np.vstack([r, lft[1:-1]])


def silhouette(body: np.ndarray, sleeves: list[np.ndarray], corners: list[tuple[np.ndarray, float]],
               kinks: list[np.ndarray], hp: HardParams) -> np.ndarray:
    """Outline of body + sleeves as the photo shows it: union, then the corners and the
    shoulder kinks rounded."""
    union = shapely.union_all([shapely.make_valid(shapely.Polygon(g)) for g in [body, *sleeves]])
    if union.geom_type != "Polygon":
        union = max(union.geoms, key=lambda g: g.area)
    ring = _dense(np.asarray(union.exterior.coords))
    for xy, radius in corners:
        ring = round_corner(ring, xy, radius)
    for xy in kinks:
        ring = fillet(ring, xy, hp.kink_span_mm)
    return ring


def hard_measurements(p, front, fe, back, be, sleeve, se) -> dict[str, float]:
    """The six SPEC 10.2 values of the hard tee, from its averaged half outlines (mm)."""
    underarm = front[fe["armhole"][1]]
    side = front[fe["side"][0] : fe["side"][1] + 1]
    side = side[np.argsort(side[:, 1])]
    chest_y = underarm[1] + get_config().pattern.chest_drop_mm
    neck_point = front[fe["shoulder"][0]]
    return {
        "chest_width": 2 * float(np.interp(chest_y, side[:, 1], side[:, 0])),
        "body_length": float(front[fe["hem"][1], 1] - neck_point[1]),   # hem at the fold
        "shoulder_width": 2 * float(back[be["armhole"][0], 0]),
        "sleeve_length": float(sleeve[se["fold"][0], 1] - sleeve[0, 1]),
        "neck_width": 2 * float(front[fe["neck"][1], 0]),
        "armhole_depth": float(underarm[1] - front[fe["armhole"][0], 1]),
    }


# ---- rendering ----------------------------------------------------------------

def _smooth_field(h: int, w: int, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random field, unit std, (h, w) float32, built at 1/16 size."""
    f = 16
    small = rng.normal(size=(h // f + 2, w // f + 2)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), max(sigma_px / f, 0.5))
    small /= max(float(small.std()), 1e-6)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _blurred(mask: np.ndarray, sigma_px: float, shift_px: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Soft version of a 0/1 mask, blurred at 1/4 size and shifted by (dy, dx) full-size px."""
    f = 4
    h, w = mask.shape
    small = cv2.resize(mask, (w // f, h // f), interpolation=cv2.INTER_AREA)
    small = np.roll(small, (shift_px[0] // f, shift_px[1] // f), axis=(0, 1))
    small = cv2.GaussianBlur(small, (0, 0), sigma_px / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _fill(polys: list[np.ndarray], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for poly in polys:
        cv2.fillPoly(m, [np.round(poly * PX_PER_MM).astype(np.int32)], 255)
    return m


def _card_patch(hp: HardParams, rng: np.random.Generator, view_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """White card with the marker, curled: its right edge lifts by ``curl_mm``. Returns the
    BGR patch and its alpha, in world pixels. A lifted point looks bigger and shifts away
    from the photo centre (``view_dir``), so the card is compressed along x and sheared."""
    total = MARKER_SIZE_MM + 2 * MARKER_BORDER_MM
    pad = 12
    card_px = int(total * PX_PER_MM)
    size = card_px + 2 * pad * int(PX_PER_MM)
    flat = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)
    o = pad * int(PX_PER_MM)
    flat[o : o + card_px, o : o + card_px] = 245
    mk_px = int(MARKER_SIZE_MM * PX_PER_MM)
    off = o + int(MARKER_BORDER_MM * PX_PER_MM)
    flat[off : off + mk_px, off : off + mk_px] = synth._marker_image(mk_px)[..., None]
    alpha[o : o + card_px, o : o + card_px] = 1
    if hp.curl_mm <= 0:
        return flat, alpha
    radius = total**2 / (2 * hp.curl_mm)
    view_ratio = 0.5                                     # card offset from the axis over camera height
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x_dst, y_dst = (xx - o) / PX_PER_MM, (yy - o) / PX_PER_MM          # mm from the card corner
    u = x_dst.copy()
    for _ in range(12):                                   # invert x_dst = R sin(u/R) + shift_x(u)
        lift = u**2 / (2 * radius)
        u = x_dst - (radius * np.sin(u / radius) - u) - view_ratio * lift * view_dir[0]
    lift = u**2 / (2 * radius)
    v = y_dst - view_ratio * lift * view_dir[1]
    maps = (u * PX_PER_MM + o).astype(np.float32), (v * PX_PER_MM + o).astype(np.float32)
    patch = cv2.remap(flat, *maps, cv2.INTER_LINEAR, borderValue=0)
    alpha = cv2.remap(alpha, *maps, cv2.INTER_LINEAR, borderValue=0)
    patch *= (1 - 0.05 * np.clip(u / total, 0, 1))[..., None]    # the lifted edge tips away from the light
    return patch, alpha


def distort_points(px: np.ndarray, w: int, h: int, k1: float) -> np.ndarray:
    """Photo px before the lens distortion -> photo px after it (the forward law of _barrel_maps)."""
    centre, norm = np.array([w / 2, h / 2]), np.hypot(w, h) / 2
    d = (np.asarray(px, float) - centre) / norm
    return centre + d * (1 + k1 * (d**2).sum(axis=1, keepdims=True)) * norm


def _barrel_maps(h: int, w: int, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """remap() maps that apply x_d = x_u (1 + k1 r^2) (r = 1 at the half diagonal): for each
    output pixel the source pixel, found by fixed-point iteration."""
    cx, cy, norm = w / 2, h / 2, np.hypot(w, h) / 2
    dx = ((np.arange(w, dtype=np.float32) - cx) / norm)[None, :]
    dy = ((np.arange(h, dtype=np.float32) - cy) / norm)[:, None]
    ux, uy = np.broadcast_to(dx, (h, w)).copy(), np.broadcast_to(dy, (h, w)).copy()
    for _ in range(12):
        scale = 1 + k1 * (ux**2 + uy**2)
        ux, uy = dx / scale, dy / scale
    return (ux * norm + cx).astype(np.float32), (uy * norm + cy).astype(np.float32)


def noisy_mask(mask: np.ndarray, hp: HardParams, rng: np.random.Generator) -> np.ndarray:
    """A mock SAM mask: edge blurred and jittered, a few small holes and notches, stray blobs."""
    ys, xs = np.nonzero(mask)
    pad = 60
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    m = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    soft = cv2.GaussianBlur(m, (0, 0), hp.mask_blur_px)
    n = cv2.GaussianBlur(rng.normal(size=m.shape).astype(np.float32), (0, 0), 6.0)
    n /= max(float(n.std()), 1e-6)
    gain = hp.mask_jitter_px / (hp.mask_blur_px * 2.507)       # edge shift per unit of noise, px
    out = ((soft + gain * n * 4 * soft * (1 - soft)) > 0.5).astype(np.uint8)
    inner = cv2.distanceTransform(out, cv2.DIST_L2, 3)
    inside = np.argwhere(inner > 14)
    for _ in range(hp.mask_holes if len(inside) else 0):        # holes inside
        y, x = inside[rng.integers(len(inside))]
        cv2.circle(out, (int(x), int(y)), int(rng.integers(2, 9)), 0, -1)
    edge = np.argwhere((inner > 0) & (inner <= 1.5))
    for _ in range(2 if len(edge) else 0):                      # notches on the edge
        y, x = edge[rng.integers(len(edge))]
        cv2.circle(out, (int(x), int(y)), int(rng.integers(3, 7)), 0, -1)
    far = cv2.distanceTransform(1 - out, cv2.DIST_L2, 3)
    for _ in range(hp.mask_strays):                             # stray blobs, not touching the mask
        for _try in range(50):
            y, x = int(rng.integers(m.shape[0])), int(rng.integers(m.shape[1]))
            if far[y, x] > 25:
                cv2.circle(out, (x, y), int(rng.integers(1, 7)), 1, -1)
                break
    result = np.zeros_like(mask)
    result[y0:y1, x0:x1] = out * 255
    return result


def render_hard(
    regions: dict[str, list[np.ndarray]],
    garment: list[np.ndarray],
    colour: tuple[int, int, int],
    rng: np.random.Generator,
    hp: HardParams,
    overlays: dict[str, tuple[list[np.ndarray], tuple[int, int, int]]] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """synth.render_layers() with every fault of ``hp``. ``garment`` is the t-shirt silhouette
    (the mask SAM would return); ``regions`` are the exact body and sleeve parts (truth only).

    Returns photo, masks (photo space), world_mm -> UNDISTORTED photo px homography and the
    marker corners in world mm. The photo itself carries the barrel distortion."""
    overlays = overlays or {}
    all_pts = np.vstack([p for polys in (*regions.values(), garment) for p in polys]
                        + [p for polys, _ in overlays.values() for p in polys])
    total = MARKER_SIZE_MM + 2 * MARKER_BORDER_MM
    width_mm = all_pts[:, 0].max() + 160
    height_mm = all_pts[:, 1].max() + total + 200
    w, h = int(width_mm * PX_PER_MM), int(height_mm * PX_PER_MM)

    # Uneven lighting: a tilted plane, a lamp blob and a gentle falloff.
    angle = rng.uniform(0, 2 * np.pi)
    xs = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :]
    ys = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None]
    light = 0.9 + 2 * hp.light_tilt * (np.cos(angle) * xs + np.sin(angle) * ys) + 0.06 * _smooth_field(h, w, 500, rng)
    light -= 0.12 * (xs**2 + ys**2) * 4

    garment_mask = _fill(garment, h, w)
    g = (garment_mask > 0).astype(np.float32)
    backdrop = np.array(BACKDROP_BGR, np.float32) * light[..., None]
    shadow = _blurred(g, 6 * PX_PER_MM, (int(5 * PX_PER_MM), int(3 * PX_PER_MM)))
    backdrop *= (1 - hp.shadow * np.clip(shadow, 0, 1))[..., None]

    # Fabric: weave, fibre noise, smooth shading and crease lines where a second field crosses zero.
    weave = np.sin(2 * np.pi * np.arange(w, dtype=np.float32) / 4.1)[None, :] * np.sin(
        2 * np.pi * np.arange(h, dtype=np.float32) / 4.1)[:, None]
    smooth = _smooth_field(h, w, 60 * PX_PER_MM / 3, rng)
    crease = np.exp(-((_smooth_field(h, w, 40 * PX_PER_MM / 3, rng) / 0.18) ** 2))
    shading = 1 + 0.5 * hp.shade * smooth - hp.shade * crease + 0.03 * weave
    shading *= 0.85 + 0.15 * np.clip(_blurred(g, 3 * PX_PER_MM), 0, 1)       # darker near the edge
    fabric = np.array(colour, np.float32) * (light * shading)[..., None]
    fabric *= 1 + 0.04 * rng.normal(size=(h, w, 1)).astype(np.float32)
    world = np.where(garment_mask[..., None] > 0, fabric, backdrop)
    del fabric, backdrop

    masks_world = {name: _fill(polys, h, w) for name, polys in regions.items()}
    masks_world["t-shirt"] = garment_mask
    if "sleeve" in regions and "body" in regions:
        seam = cv2.dilate(masks_world["sleeve"], np.ones((3, 3))) & masks_world["body"]
        world[seam > 0] *= 0.8
    for name, (polys, bgr) in overlays.items():
        m = _fill(polys, h, w)
        masks_world[name] = m
        mf = (m > 0).astype(np.float32)
        lift = np.clip(_blurred(mf, 4 * PX_PER_MM, (int(3.5 * PX_PER_MM), int(2.5 * PX_PER_MM))) - mf, 0, 1)
        world *= (1 - 0.3 * lift)[..., None]                       # a lifted paper edge casts a wider shadow
        tilt = 1 + 0.03 * ((np.arange(w, dtype=np.float32) / w - 0.5)[None, :] * 2)
        sheet = np.array(bgr, np.float32) * (0.9 + 0.1 * light[..., None]) * tilt[..., None]
        world = np.where(m[..., None] > 0, sheet * (1 + 0.01 * rng.normal(size=(h, w, 1)).astype(np.float32)), world)

    # Marker card, bottom left, curled.
    m0 = np.array([90.0, all_pts[:, 1].max() + 90])
    centre = np.array([w, h]) / PX_PER_MM / 2
    away = m0 + total / 2 - centre
    patch, alpha = _card_patch(hp, rng, away / np.linalg.norm(away))
    pad_px = 12 * int(PX_PER_MM)
    x0, y0 = (m0 * PX_PER_MM).astype(int) - pad_px
    sl = (slice(y0, y0 + patch.shape[0]), slice(x0, x0 + patch.shape[1]))
    lit = light[sl][..., None]
    world[sl] = np.where(alpha[..., None] > 0.5, patch * (0.85 + 0.15 * lit), world[sl])
    tl = m0 + MARKER_BORDER_MM
    marker_mm = np.array([tl, tl + [MARKER_SIZE_MM, 0], tl + MARKER_SIZE_MM, tl + [0, MARKER_SIZE_MM]])

    world += rng.normal(0, 5, world.shape).astype(np.float32)
    world = world.clip(0, 255).astype(np.uint8)

    # Camera: stronger perspective, kept inside the frame, then barrel distortion.
    content = np.float32([[x0, y0], [x0 + patch.shape[1], y0 + patch.shape[0]],
                          *(all_pts.min(axis=0) * PX_PER_MM, all_pts.max(axis=0) * PX_PER_MM)])
    box = np.float32([[content[:, 0].min(), content[:, 1].min()], [content[:, 0].max(), content[:, 1].min()],
                      [content[:, 0].max(), content[:, 1].max()], [content[:, 0].min(), content[:, 1].max()]])
    cam = _camera(w, h, rng, hp, box)
    photo = cv2.warpPerspective(world, cam, (w, h), borderMode=cv2.BORDER_REPLICATE)
    mx, my = _barrel_maps(h, w, hp.barrel_k1)
    photo = cv2.remap(photo, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    photo = cv2.GaussianBlur(photo, (0, 0), 0.8)
    masks = {}
    for k, v in masks_world.items():
        warped = cv2.remap(cv2.warpPerspective(v, cam, (w, h), flags=cv2.INTER_NEAREST), mx, my, cv2.INTER_NEAREST)
        masks[k] = noisy_mask(warped, hp, rng) if k in MASK_NOISE else warped
    scale = np.diag([PX_PER_MM, PX_PER_MM, 1.0])
    return photo, masks, cam @ scale, marker_mm


def _camera(w: int, h: int, rng: np.random.Generator, hp: HardParams, content: np.ndarray) -> np.ndarray:
    """Random perspective with corner jitter ``hp.tilt``; shrunk until ``content`` stays in frame."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-hp.tilt, hp.tilt, (4, 2)) * [w, h]
    angle = np.radians(hp.rotation_deg)
    c = np.array([w / 2, h / 2])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shrink = rng.uniform(0.88, 0.95)
    margin = 30.0
    for _ in range(40):
        dst = ((src - c) @ rot.T) * shrink + c + jitter * shrink
        cam = cv2.getPerspectiveTransform(src, np.float32(dst))
        pts = cv2.perspectiveTransform(content[None], cam)[0]
        if pts.min() >= margin and (pts[:, 0] <= w - margin).all() and (pts[:, 1] <= h - margin).all():
            return cam
        shrink *= 0.97
    raise RuntimeError("hard camera: the content does not fit in the frame")


# ---- tees -----------------------------------------------------------------------

def save(out: Path, view: str, photo: np.ndarray, masks: dict[str, np.ndarray], quality: int) -> None:
    cv2.imwrite(str(out / f"{view}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, quality])
    mdir = out / f"{view}.masks"
    mdir.mkdir(exist_ok=True)
    for name, m in masks.items():
        cv2.imwrite(str(mdir / f"{name}.png"), m)


def hard_views(p: synth.TeeParams, hp: HardParams) -> dict:
    """Exact geometry of the front and back photos, world mm: halves, silhouettes, regions."""
    views = {}
    e = hp.asymmetry_mm / p.chest_half
    for view, back in (("front", False), ("back", True)):
        idx = 0 if view == "front" else 1
        right, left, edges = hard_body_sides(p, hp, idx, back_neck=back)
        vis_r, vis_l, _ = hard_body_sides(p, hp, idx, back_neck=True)       # back neck seen through the front opening
        origin = np.array([p.chest_half + p.sleeve_length + 120, 80])
        fold_start = edges["fold"][0]
        body = _full_ring(vis_r, vis_l, fold_start) + origin
        sleeve_polys, records = [], []
        for side, sign in ((0, 1), (1, -(1 + e))):
            poly, rec = flatlay_sleeve_hard(p, hp, idx, side)
            sleeve_polys.append(poly * [sign, 1] + origin)
            records.append(rec)
        # Corners to round: body hem corners and the four sleeve hem corners; the kink at each shoulder point.
        corners = [(vis_r[edges["side"][1]] + origin, hp.corner_radius_mm),
                   (vis_l[edges["side"][1]] * [-1, 1] + origin, hp.corner_radius_mm)]
        for poly_rec, sign in zip(records, (1, -(1 + e))):
            for key in ("bottom_mm", "top_mm"):
                corners.append((np.asarray(poly_rec[key]) * [sign, 1] + origin, hp.corner_radius_mm))
        arm0 = synth.armhole_curve(p)[0]
        kinks = [arm0 * [1, 1] + origin, arm0 * [-(1 + e), 1] + origin]
        outline = silhouette(body, sleeve_polys, corners, kinks, hp)
        body_round = _dense(body)   # the body alone with rounded hem corners, for the proof's exact gaps
        for xy, radius in corners[:2]:
            body_round = round_corner(body_round, xy, radius)
        avg = (right + left) / 2
        vis_avg = (vis_r + vis_l) / 2
        views[view] = {
            "half": avg, "visible_half": vis_avg, "visible_right": vis_r, "edges": edges, "origin": origin,
            "body": body, "body_rounded": body_round, "sleeves": sleeve_polys, "garment": outline,
            "sleeve_records": records,
            "right": right, "left": left,
        }
    return views


def make_tee_hard(p: synth.TeeParams, out: Path, seed: int, hard: HardParams | None = None) -> dict:
    """Write one hard tee (photos, masks, truth.json) in the layout of synth.make_tee()."""
    hp = hard or hard_params(seed)
    rng = np.random.default_rng(10_000 + seed)
    out.mkdir(parents=True, exist_ok=True)
    views = hard_views(p, hp)
    truth: dict = {"synthetic": True, "params": asdict(p), "photos": {}}

    for view in ("front", "back"):
        v = views[view]
        photo, masks, H, marker_mm = render_hard(
            {"body": [v["body"]], "sleeve": v["sleeves"]}, [v["garment"]], p.colour_bgr, rng, hp)
        save(out, view, photo, masks, hp.jpeg_quality)
        truth["photos"][view] = {
            "half_outline_mm": v["half"].tolist(),
            "visible_half_outline_mm": v["visible_half"].tolist() if view == "front" else v["half"].tolist(),
            "visible_right_half_mm": v["visible_right"].tolist(),   # the right side of the body ring, not averaged
            "edges": v["edges"],
            "origin_mm": v["origin"].tolist(),
            "world_mm_to_photo_px": H.tolist(),   # before the lens distortion
            "size_px": [photo.shape[1], photo.shape[0]],
            "marker_corners_mm": marker_mm.tolist(),
            "regions_mm": {"body": v["body"].tolist(), "body_rounded": v["body_rounded"].tolist(),
                           "garment": v["garment"].tolist(),
                           "sleeves": [s.tolist() for s in v["sleeves"]]},
            "sleeves": v["sleeve_records"],
        }

    half, edges = synth.sleeve_half(p)
    half = wave_edges(half, edges, hp, 2, 0, skip=("fold",))
    ring = _dense(half)
    for key in ("underarm", "hem"):               # sleeve cap end and hem corner: the two free convex corners
        ring = round_corner(ring, half[edges[key][0]], hp.corner_radius_mm)
    cap = half[edges["cap"][0] : edges["cap"][1] + 1]
    body = synth.sleeve_scene_body(p, cap, ring, band_mm=10.0)   # clean body: the whole t-shirt around the sleeve
    origin = synth.place_in_frame([ring, *body])
    body = [b + origin for b in body]
    ring = ring + origin
    garment = shapely.union_all([shapely.make_valid(shapely.Polygon(g)) for g in [ring, *body]])
    garment = [np.asarray(g.exterior.coords) for g in getattr(garment, "geoms", [garment])]
    photo, masks, H, marker_mm = render_hard({"body": body, "sleeve": [ring]}, garment, p.colour_bgr, rng, hp)
    save(out, "sleeve", photo, masks, hp.jpeg_quality)
    truth["photos"]["sleeve"] = {
        "half_outline_mm": half.tolist(), "edges": edges, "origin_mm": origin.tolist(),
        "world_mm_to_photo_px": H.tolist(), "marker_corners_mm": marker_mm.tolist(),
        "size_px": [photo.shape[1], photo.shape[0]],
    }
    truth["measurements"] = hard_measurements(
        p, views["front"]["half"], views["front"]["edges"], views["back"]["half"], views["back"]["edges"],
        half, edges)
    truth["hard"] = {**asdict(hp), "k1_convention": "x_d = x_u (1 + k1 r^2), r = 1 at the half diagonal",
                     "truth_note": "measurements are read at the design vertices; chest_width at the wavy side seam"}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def params_from_truth(truth: dict) -> HardParams:
    """The HardParams that a hard tee's truth.json records."""
    rec = {k: v for k, v in truth["hard"].items() if k in HardParams.__dataclass_fields__}
    rec["kink_deg"] = tuple(rec["kink_deg"])
    return HardParams(**rec)


def make_hard_set(out: Path, count: int = 10, only: list[str] | None = None, lens_scale: float = 1.0,
                  curl_scale: float = 1.0) -> list[Path]:
    """``count`` hard tees: tee1..tee5 are the five clean sizes, tee6.. repeat them with other faults.
    ``lens_scale`` and ``curl_scale`` multiply the barrel distortion and the marker curl (0 = none):
    both scale the photo as seen from the marker, which no outline code can undo, so setting
    them to 0 tells those errors from the rest."""
    written = []
    for i in range(count):
        base = synth.TEES[i % len(synth.TEES)]
        p = synth.TeeParams(**{**asdict(base), "name": f"tee{i + 1}", "colour_bgr": tuple(base.colour_bgr)})
        if only and p.name not in only:
            continue
        hp = hard_params(i)
        make_tee_hard(p, out / p.name, seed=i, hard=replace(hp, barrel_k1=hp.barrel_k1 * lens_scale,
                                                                   curl_mm=hp.curl_mm * curl_scale))
        written.append(out / p.name)
    return written
