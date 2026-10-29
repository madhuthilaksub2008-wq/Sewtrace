"""Hard synthetic tees (sewtrace/mock/synth_hard.py): the faults of real photos.

Targets below were chosen from measured results on the hard set (REQ-GEO-01, REQ-SCL-03,
REQ-SEG-03, REQ-VER-02), with margin. They are NEW targets for the hard set only. No clean-set
tolerance was changed.
"""

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
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


def test_req_seg_03_clean_mask_drops_strays_fills_holes_and_nicks():
    """REQ-SEG-03: a SAM mask with holes, stray blobs and an edge nick gives one clean piece."""
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
    """REQ-GEO-01: a 3 degree kink with waves cannot place the shoulder point: ContourFailed's cause
    says so and what to do, instead of a guess that is tens of mm out."""
    p = _tee(0)
    hp = _calm(0, kink_deg=(3.0, 3.0), corner_radius_mm=10.0)
    outline, _, _ = _joints(p, hp)
    with pytest.raises(ValueError, match="shoulder point is not clear.*wider angle"):
        contour.find_sleeve_joints(outline)


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
    """REQ-GEO-01: tee1 (3 degree kink) is refused with ContourFailed that names the photo."""
    with pytest.raises(ContourFailed, match=r"front photo .*not clear.*wider angle"):
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


def test_req_ver_02_hard_proof_fold_found_and_gap_matches_exact(tmp_path, client):
    """REQ-VER-02: wavy hand-cut paper with a lift shadow on a hard tee. The fold is found and the
    measured mean gap is within 1.5 mm of the exact one (measured: 1.3 mm; the clean proof uses 0.5 mm)."""
    from sewtrace.vision.verify import verify_proof

    hp = replace(synth_hard.hard_params(2), barrel_k1=0.0, curl_mm=0.0)
    synth.make_tee_hard(_tee(2), tmp_path / "tee3", 2, hp)
    truth = proof.make_proof_hard(tmp_path / "tee3", seed=5, scale=1.02, name="proof")
    photo = tmp_path / "tee3" / "proof.jpg"
    result = verify_proof(photo, client, photo_homography(cv2.imread(str(photo))))
    print(f"measured {result.mean_mm:.2f} expected {truth['expected_gap_mean_mm']:.2f}")
    assert result.fold_mm is not None
    p0, p1 = np.asarray(result.fold_mm)
    assert np.linalg.norm(p1 - p0) > 250
    assert result.mean_mm == pytest.approx(truth["expected_gap_mean_mm"], abs=1.5)
