"""Hard synthetic tees (sewtrace/mock/synth_hard.py): the faults of real photos.

Targets below were chosen from measured results on the hard set (REQ-GEO-01, REQ-SCL-03,
REQ-SEG-03, REQ-VER-02), with margin. They are NEW targets for the hard set only. No clean-set
tolerance was changed.
"""

import json
import shutil
from dataclasses import replace

import cv2
import numpy as np
import pytest
import shapely

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings, get_config
from sewtrace.mock import proof, synth, synth_hard
from sewtrace.pipeline import ContourFailed, run
from sewtrace.types import MEASUREMENT_KEYS
from sewtrace.vision import contour
from sewtrace.vision.marker import Homography, photo_homography

PX = synth.PX_PER_MM


def _tee(index: int) -> synth.TeeParams:
    base = synth.TEES[index % 5]
    return synth.TeeParams(**{**base.__dict__, "name": f"tee{index + 1}"})


def _joints(p: synth.TeeParams, hp: synth_hard.HardParams, noisy: bool = True):
    """Shoulder and underarm points found on the rasterised hard silhouette, and the true ones."""
    v = synth_hard.hard_views(p, hp)["front"]
    ring = v["garment"]
    x0, y0 = ring.min(axis=0) - 30
    shape = (int((ring[:, 1].max() - y0 + 30) * PX), int((ring[:, 0].max() - x0 + 30) * PX))
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round((ring - [x0, y0]) * PX).astype(np.int32)], 255)
    if noisy:
        mask = synth_hard.noisy_mask(mask, hp, np.random.default_rng(3))
    hom = Homography(np.array([[1 / PX, 0, x0], [0, 1 / PX, y0], [0, 0, 1.0]]))
    outline = contour.smooth_outline(contour.mask_to_outline(mask > 0, hom))
    e = hp.asymmetry_mm / p.chest_half
    arm = synth.armhole_curve(p)
    origin = v["origin"]
    true_shoulder = np.array([arm[0], arm[0] * [-(1 + e), 1]]) + origin
    true_underarm = np.array([arm[-1], arm[-1] * [-(1 + e), 1]]) + origin
    return outline, true_shoulder, true_underarm


def _nearest_errors(found: np.ndarray, truth: np.ndarray) -> list[float]:
    return [float(np.linalg.norm(found - t, axis=1).min()) for t in truth]


def _calm(index: int, **kw) -> synth_hard.HardParams:
    """Hard params of tee ``index`` with the main outline faults set by ``kw``."""
    return replace(synth_hard.hard_params(index), wave_mm=1.0, asymmetry_mm=2.0, **kw)


def test_req_geo_01_clean_mask_drops_strays_fills_holes_and_nicks():
    """REQ-GEO-01: a SAM mask with holes, stray blobs and an edge nick gives one clean piece."""
    mask = np.zeros((300, 300), np.uint8)
    mask[60:240, 60:240] = 1
    mask[100:106, 100:106] = 0           # hole
    mask[58:62, 140:143] = 0             # nick in the top edge
    mask[10:14, 10:14] = 1               # stray blob
    mask[280:282, 20:22] = 1             # stray pixels
    mask[150:153, 238:246] = 1           # whisker on the right edge
    out = contour.clean_mask(mask, px_per_mm=3.0)
    assert out[100:106, 100:106].all()
    assert not out[10:14, 10:14].any() and not out[280:282, 20:22].any()
    assert not out[150:153, 242:246].any()
    n, _ = cv2.connectedComponents(out)
    assert n == 2  # background and one piece
    assert abs(int(out.sum()) - 180 * 180) < 0.03 * 180 * 180


def test_req_geo_01_clean_mask_bridges_a_one_pixel_crack_across_a_sleeve_seam():
    """REQ-GEO-01: a crack of 1 px between the body and a sleeve must not cut the sleeve off.
    The old order (largest piece, then close) kept the body only."""
    mask = np.zeros((300, 400), np.uint8)
    mask[60:240, 60:240] = 1             # body
    mask[100:200, 241:330] = 1           # sleeve, one pixel column of gap (x = 240)
    out = contour.clean_mask(mask, px_per_mm=3.0)
    assert out[150, 300] == 1
    n, _ = cv2.connectedComponents(out)
    assert n == 2


def test_req_geo_01_proof_masks_are_cleaned_like_the_photo_masks():
    """REQ-GEO-01, REQ-VER-02: the proof's mask outline goes through clean_mask too: a nick in
    the edge of the mask is not an inward corner of the outline."""
    from sewtrace.vision.verify import mask_outline_mm

    mask = np.zeros((300, 300), np.uint8)
    mask[60:240, 60:240] = 1
    mask[58:64, 140:144] = 0             # nick, 4 px wide and 4 px deep
    hom = Homography(np.diag([1 / 4, 1 / 4, 1.0]))   # 4 px per mm: a 2 px cleanup radius
    ring = mask_outline_mm(mask > 0, hom)
    in_nick = ring[(np.abs(ring[:, 0] - 142 / 4) < 0.2) & (ring[:, 1] < 40)]   # top edge above the middle of the nick
    assert len(in_nick) > 0 and in_nick[:, 1].max() < 15.6   # edge at y = 60 px = 15 mm; the raw nick floor is at 16


def test_req_geo_01_hard_truth_matches_clean_truth_without_faults():
    """REQ-GEO-01: with no wave and no offset the hard truth is the clean truth, so the
    hard measurements are read the same way as the clean ones."""
    p = _tee(2)
    hp = replace(synth_hard.hard_params(2), wave_mm=0.0, asymmetry_mm=0.0)
    views = synth_hard.hard_views(p, hp)
    sleeve, se = synth.sleeve_half(p)
    got = synth_hard.hard_measurements(
        p, views["front"]["half"], views["front"]["edges"], views["back"]["half"], views["back"]["edges"], sleeve, se)
    want = synth.measurements(p)
    for k in MEASUREMENT_KEYS:
        assert got[k] == pytest.approx(want[k], abs=1e-6), k


@pytest.mark.parametrize("radius", [3.0, 15.0])
def test_req_geo_01_joint_finder_tolerates_rounded_corners(radius):
    """REQ-GEO-01: sleeve hem corners rounded to 3 and 15 mm, noisy mask: underarm points
    within 3 mm and shoulder points within 15 mm (kink 10 degrees, measured 12)."""
    p = _tee(2)
    hp = _calm(2, kink_deg=(10.0, 10.0), corner_radius_mm=radius)
    outline, shoulder, underarm = _joints(p, hp)
    found_shoulder, found_underarm = contour.find_sleeve_joints(outline)
    assert max(_nearest_errors(found_underarm, underarm)) < 3.0
    assert max(_nearest_errors(found_shoulder, shoulder)) < 15.0


def test_req_geo_01_faint_kink_is_refused_with_a_clear_error():
    """REQ-GEO-01: a 3 degree kink with waves on both sides cannot place the shoulder point: the error
    says so and what to do, instead of a guess that is tens of mm out."""
    p = _tee(0)
    hp = _calm(0, kink_deg=(3.0, 3.0), corner_radius_mm=10.0)
    outline, _, _ = _joints(p, hp)
    with pytest.raises(ValueError, match="shoulder point is not clear.*Let the sleeves hang down more"):
        contour.find_sleeve_joints(outline)


@pytest.mark.parametrize("kinks", [(3.0, 10.0), (10.0, 3.0)])
def test_req_geo_01_one_faint_shoulder_is_placed_by_the_clear_one(kinks):
    """REQ-GEO-01: a tee with one faint (3 degree) and one clear (10 degree) shoulder is accepted
    and both shoulder points are near the truth. Measured: the faint side alone has a plateau of 54
    and 132 mm (the old per-side gate refused), the joint fit one of 28 and 35 mm; shoulder error
    3.3 and 8.6 mm."""
    p = _tee(1)
    hp = _calm(1, kink_deg=kinks, corner_radius_mm=8.0)
    outline, shoulder, _ = _joints(p, hp)
    found_shoulder, _ = contour.find_sleeve_joints(outline)
    assert max(_nearest_errors(found_shoulder, shoulder)) < 15.0


def test_req_geo_01_two_faint_shoulders_are_refused():
    """REQ-GEO-01: both shoulders at 3 degrees: the joint fit is a plateau of 117 mm, refused
    (a truly ambiguous tee), with the action to take."""
    outline, _, _ = _joints(_tee(1), _calm(1, kink_deg=(3.0, 3.0), corner_radius_mm=8.0))
    with pytest.raises(contour.ShoulderPointNotClear, match="Let the sleeves hang down more"):
        contour.find_sleeve_joints(outline)


def _peak_scores(positions: list[int], n: int = 400) -> np.ndarray:
    score = np.zeros(n)
    score[positions] = 40.0
    return score


@pytest.mark.parametrize("neck_offset, ok", [(30, True), (55, False)])
def test_req_geo_01_neck_pair_may_be_further_from_mirrored_than_the_hem_pairs(neck_offset, ok):
    """REQ-GEO-01: six corners on an arc of 300 mm. Hem pairs are off from mirrored by 4 and 10 mm,
    the neck pair by ``neck_offset``: the neck pair carries the sleeve top difference too, so it has
    its own, larger tolerance (hem 25 mm, neck 40 mm)."""
    other, arc = 100, 300
    tol = get_config().contour.joints
    assert tol.pair_tol_hem_mm < 30 < tol.pair_tol_neck_mm < 55   # the two cases sit between the tolerances
    # positions on the arc, from the first underarm point: (outer), (middle), (neck)
    pairs = [(20, 284), (60, 250), (100, 200 + neck_offset)]
    score = _peak_scores([other + a for a, _ in pairs] + [other + b for _, b in pairs])
    if ok:
        got = contour._six_corners(score, other, other + arc, radius=5)
        assert got == [other + 20, other + 60, other + 100, other + 230, other + 250, other + 284]
    else:
        with pytest.raises(ValueError, match="no 6 that mirror each other"):
            contour._six_corners(score, other, other + arc, radius=5)


def test_req_geo_01_whole_garment_with_no_underarm_is_an_error():
    """REQ-GEO-01: a plain rectangle has no sleeves: a clear error, not a wrong answer."""
    pts = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], float)
    with pytest.raises(ValueError, match="underarm"):
        contour.find_sleeve_joints(contour.Outline.from_array(contour._resample_closed(pts, 1.0)))


# ---- generated hard tees ------------------------------------------------------

@pytest.fixture(scope="module")
def hard_dir(tmp_path_factory):
    """tee3 and tee8 (kink 10 degrees on a side, wavy, rounded, noisy) and tee1 (kink 3).
    Lens distortion and marker curl are off here: they scale the whole photo as seen from the
    marker, which no outline code can undo (see test_req_scl_03_*)."""
    out = tmp_path_factory.mktemp("hard")
    for i in (0, 2, 7):
        hp = replace(synth_hard.hard_params(i), barrel_k1=0.0, curl_mm=0.0)
        synth.make_tee_hard(_tee(i), out / f"tee{i + 1}", i, hp)
    return out


@pytest.fixture
def client(tmp_path):
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))


def test_req_geo_01_hard_truth_records_its_faults(hard_dir):
    """REQ-GEO-01: truth.json says synthetic and records every fault; masks carry strays and holes."""
    truth = json.loads((hard_dir / "tee3" / "truth.json").read_text())
    assert truth["synthetic"] is True
    hard = truth["hard"]
    assert 3 <= hard["kink_deg"][0] <= 10 and 3 <= hard["corner_radius_mm"] <= 15
    assert 0.5 <= hard["wave_mm"] <= 2 and 1 <= hard["asymmetry_mm"] <= 3
    mask = cv2.imread(str(hard_dir / "tee3" / "front.masks" / "t-shirt.png"), cv2.IMREAD_GRAYSCALE) > 127
    n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    assert n > 3  # background, the tee and at least two stray blobs


@pytest.mark.parametrize("name", ["tee3", "tee8"])
def test_req_geo_04_hard_tee_measurements_and_seams(hard_dir, client, tmp_path, name):
    """REQ-GEO-01, REQ-GEO-04, REQ-SEW-04: hard tees with a clear shoulder kink run end to end.
    New hard-set targets, from measured errors: every measurement within 2.5 mm, the shoulder
    width within 10 mm (the shoulder point is read to about +/-5 mm at a 10 degree kink, doubled
    across the width), mean error under 2.5 mm, all seams pass."""
    result = run(hard_dir / name, tmp_path / "out", client)
    truth = json.loads((hard_dir / name / "truth.json").read_text())["measurements"]
    errors = {k: abs(result.measurements[k] - truth[k]) for k in MEASUREMENT_KEYS}
    print(name, {k: round(v, 2) for k, v in errors.items()})
    for k, err in errors.items():
        assert err < (10.0 if k == "shoulder_width" else 2.5), k
    assert sum(errors.values()) / len(errors) < 2.5
    assert all(s.ok for s in result.spec.seams)


def test_req_geo_01_hard_tee_with_faint_kink_names_the_photo(hard_dir, client, tmp_path):
    """REQ-GEO-01: tee1 (3 degree kink) is refused with ContourFailed that names the photo, says
    what to do and names the photo to take again."""
    with pytest.raises(ContourFailed, match=r"front photo .*not clear.*Let the sleeves hang down more.*Take the front photo again"):
        run(hard_dir / "tee1", tmp_path / "out", client)


def test_req_scl_03_marker_passes_on_hard_photos_with_lens_and_curl(tmp_path):
    """REQ-SCL-03 (documents a gap): with barrel distortion and a curled marker card the marker
    check passes (error well under 1 mm), yet the scale at the garment is off by 0.3 to 2 percent,
    because the marker error is a residual inside the marker. The check cannot see this."""
    hp = replace(synth_hard.hard_params(2), curl_mm=1.5)
    truth = synth.make_tee_hard(_tee(2), tmp_path / "tee3", 2, hp)
    photo = truth["photos"]["front"]
    hom = photo_homography(cv2.imread(str(tmp_path / "tee3" / "front.jpg")))
    assert hom.marker_error_mm < 0.2
    ring = np.asarray(photo["regions_mm"]["garment"])
    px = cv2.perspectiveTransform(ring[None], np.asarray(photo["world_mm_to_photo_px"]))[0]
    px = synth_hard.distort_points(px, *photo["size_px"], hp.barrel_k1)
    mm = hom.px_to_mm(px)
    want = ring - np.asarray(photo["marker_corners_mm"][0])
    span_error = np.ptp(mm[:, 0]) / np.ptp(want[:, 0]) - 1
    assert 0.003 < abs(span_error) < 0.03


@pytest.mark.parametrize("angle_deg", [11.0, 17.0, 29.0])
def test_req_ver_02_expected_gap_leaves_out_every_point_nearest_the_armhole(angle_deg):
    """REQ-VER-02: a paper point whose nearest body edge is the armhole ties with it (the same
    distance, up to float noise). The tie must go to 'armhole' every time. Before, it went either
    way, and the expected gap of a hard tee read 1.3 mm too high (and neck points of 40 mm counted)."""
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    body = (shapely.segmentize(shapely.box(0, 0, 100, 100), 1.0).exterior.coords)
    body = np.asarray(body)[:-1] @ rot.T
    armhole = np.array([[100.0, 0.0], [100.0, 50.0], [100.0, 100.0]]) @ rot.T   # the right edge
    paper = np.array([[10.0, 20.0], [70.0, 20.0], [70.0, 80.0], [10.0, 80.0]]) @ rot.T
    gaps, neck_gaps = proof.expected_gaps(paper, body, (np.array([-9e3, 0.0]), np.array([-9e3, 1.0])),
                                          armhole, centre_x_mm=-9e3)
    assert len(neck_gaps) == 0
    assert not np.isclose(gaps, 30.0, atol=1e-6).any()   # the paper's right edge is 30 mm from the armhole


@pytest.mark.parametrize("tee, scale, seed", [
    ("tee3", 1.02, 5),   # the first case: measured -0.23 mm
    ("tee3", 0.98, 6),   # a 3 to 4 mm bulge cut the found fold to 244 mm of 400: +15.4 mm before
    ("tee8", 1.0, 5),    # the same bulge: +14.6 mm before
    ("tee8", 1.02, 5),   # mask wobble put 13 fold points outside the 1.5 mm band: +1.8 mm before
])
def test_req_ver_02_hard_proof_fold_found_and_gap_matches_exact(hard_dir, tmp_path, client, tee, scale, seed):
    """REQ-VER-02: wavy hand-cut paper with a lift shadow on a hard tee. The whole fold is found
    and the measured mean gap is within 0.5 mm of the exact one (the clean proof uses 0.5 mm too).
    The tolerance is the same as before the fold fix: only the way the fold is found changed.

    History: the fold used to be followed along the outline in chunks, and left out within a fixed
    1.5 mm. A bulge of 3 to 4 mm cut the fold short and a wobble of 2 to 3 mm left fold points
    in, both counted as cut edge with gaps of 30 to 140 mm. The fold is now a RANSAC line fit
    and a band around the fitted line (proof.fold_band_mm).

    Known open gap (SPEC 10.1): other seeds scatter by -0.5 to +0.3 mm."""
    from sewtrace.vision.verify import verify_proof

    shutil.copytree(hard_dir / tee, tmp_path / tee)   # no lens, no curl; proof files stay out of the fixture
    truth = proof.make_proof_hard(tmp_path / tee, seed=seed, scale=scale, name="proof")
    photo = tmp_path / tee / "proof.jpg"
    result = verify_proof(photo, client, photo_homography(cv2.imread(str(photo))))
    print(f"{tee} scale {scale} seed {seed}: measured {result.mean_mm:.2f} expected {truth['expected_gap_mean_mm']:.2f}")
    assert result.fold_mm is not None
    true_fold = np.asarray(truth["fold_mm"])
    found_len = np.linalg.norm(np.diff(np.asarray(result.fold_mm), axis=0))
    assert found_len == pytest.approx(np.linalg.norm(np.diff(true_fold, axis=0)), abs=8.0)
    assert result.mean_mm == pytest.approx(truth["expected_gap_mean_mm"], abs=0.5)
