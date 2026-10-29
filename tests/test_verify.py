import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import shapely

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.mock.proof import make_proof
from sewtrace.vision.marker import photo_homography
from sewtrace.config import get_config
from sewtrace.vision import contour
from sewtrace.vision.verify import (PaperFoldNotStraight, ProofResult, edge_gaps, find_fold_edge, gap_result, mask_outline_mm,
                                    overlay_image, verify_proof)
from sewtrace.types import Outline

CASES = {
    "perfect": {},
    "scaled": {"scale": 1.02},
    "shifted": {"offset_mm": (4.0, 0.0)},
}


@pytest.fixture(scope="session")
def proofs(tmp_path_factory) -> tuple[Path, dict[str, dict]]:
    tee_dir = tmp_path_factory.mktemp("proof") / "tee1"
    synth.make_tee(synth.TEES[0], tee_dir, seed=0)
    return tee_dir, {name: make_proof(tee_dir, seed=7, name=name, **kw) for name, kw in CASES.items()}


@pytest.fixture
def client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))


def _verify(proofs, client, name: str) -> tuple[ProofResult, dict]:
    tee_dir, truth = proofs
    photo = tee_dir / f"{name}.jpg"
    hom = photo_homography(cv2.imread(str(photo)))
    return verify_proof(photo, client, hom), truth[name]


def test_proof_writes_photo_and_masks(proofs):
    """REQ-VER-01: proof writes photo and masks."""
    tee_dir, _ = proofs
    for mask in ("paper", "body"):
        m = cv2.imread(str(tee_dir / "perfect.masks" / f"{mask}.png"), cv2.IMREAD_GRAYSCALE)
        assert m is not None and (m > 127).sum() > 10_000
    assert (tee_dir / "perfect.jpg").exists() and (tee_dir / "perfect.json").exists()


def test_edge_gaps_square():
    """REQ-VER-02: edge gaps square."""
    inner = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    outer = inner * 1.04 - 2.0
    pts, d = edge_gaps(outer, inner, step_mm=2.0)
    corner = np.isclose(np.abs(pts - 50), 52).all(axis=1)  # outer corners sit 2*sqrt(2) out
    assert corner.sum() == 4
    assert np.allclose(d[~corner], 2.0) and np.allclose(d[corner], 2 * np.sqrt(2))
    r = gap_result(pts, d, tol_mm=3.0)
    assert r.frac_within_tol == 1.0 and r.max_mm == pytest.approx(2 * np.sqrt(2))


def test_req_ver_02_perfect_paper_has_small_gap(proofs, client):
    """REQ-VER-02: a perfect half paper on the centre of the tee, fold edge excluded."""
    r, truth = _verify(proofs, client, "perfect")
    print(f"perfect: mean {r.mean_mm:.2f} max {r.max_mm:.2f} p95 {r.p95_mm:.2f} mm")
    assert truth["expected_gap_mean_mm"] < 1e-6
    assert r.mean_mm < 0.5
    assert r.frac_within_tol > 0.98
    assert len(r.points_mm) == len(r.distances_mm) > 200
    # Same sampling as before the neck rule: compared points plus the left-out neck points.
    assert len(r.points_mm) + len(r.neck_points_mm) > 250
    assert r.fold_mm is not None
    p0, p1 = np.asarray(r.fold_mm)
    assert np.linalg.norm(p1 - p0) > 250  # the fold edge runs from the neck point to the hem
    fold = shapely.LineString([p0, p1])
    assert shapely.distance(shapely.points(r.points_mm), fold).min() >= get_config().proof.fold_exclude_mm - 1e-6


@pytest.mark.parametrize("name", ["scaled", "shifted"])
def test_req_ver_02_gap_matches_known_paper_error(proofs, client, name):
    """REQ-VER-02: scale 1.02 and a 4 mm shift: measured mean within 0.5 mm of the exact gap."""
    r, truth = _verify(proofs, client, name)
    expected = truth["expected_gap_mean_mm"]
    print(f"{name}: measured mean {r.mean_mm:.2f} mm, expected {expected:.2f} mm")
    assert expected > 2.0
    assert r.mean_mm == pytest.approx(expected, abs=0.5)
    assert r.max_mm == pytest.approx(truth["expected_gap_max_mm"], abs=1.0)
    assert r.frac_within_tol < 0.9


def test_overlay_marks_gaps_red_and_green(proofs, client, tmp_path):
    """REQ-VER-03: overlay marks gaps red and green."""
    tee_dir, _ = proofs
    r, _ = _verify(proofs, client, "scaled")
    hom = photo_homography(cv2.imread(str(tee_dir / "scaled.jpg")))
    out = overlay_image(tee_dir / "scaled.jpg", r, hom, tmp_path / "overlay.png")
    img = cv2.imread(str(out))
    b, g, rr = (img[..., i].astype(int) for i in range(3))
    assert ((rr > 200) & (g < 60) & (b < 60)).sum() > 100  # red: gap over tolerance
    assert ((g > 180) & (rr < 80) & (b < 80)).sum() > 100  # green: within tolerance


def test_req_ver_02_fold_edge_is_not_compared(proofs, client):
    """Regression: the fold edge lies on the tee centre line, about half a chest from the
    tee edge. Measured like any cut edge it would put the mean gap far over 3 mm even for
    a perfect paper. The old code (no fold handling) fails this."""
    tee_dir, _ = proofs
    photo = tee_dir / "perfect.jpg"
    hom = photo_homography(cv2.imread(str(photo)))
    paper = mask_outline_mm(client.segment(photo, "paper"), hom)
    garment = mask_outline_mm(client.segment(photo, "t-shirt"), hom)
    body, armhole, neck = contour.cut_sleeves_flagged(contour.smooth_outline(Outline.from_array(garment)))
    _, naive = edge_gaps(paper, body.array(), hidden=armhole, neck=neck)
    assert naive.mean() > get_config().proof.gap_tolerance_mm  # every edge point counted
    _, handled = edge_gaps(paper, body.array(), hidden=armhole, fold=find_fold_edge(paper, body.array()), neck=neck)
    assert handled.mean() < 0.5 and handled.max() < get_config().proof.gap_tolerance_mm


def test_req_ver_02_back_neck_is_visible_in_the_tee_mask(proofs):
    """The tee mask of a front flat-lay has the BACK neckline: the front neck of the paper lies lower."""
    tee_dir, truth = proofs
    spec = json.loads((tee_dir / "truth.json").read_text())
    front = spec["photos"]["front"]
    p = spec["params"]
    visible = np.asarray(front["visible_half_outline_mm"])
    assert visible[0, 1] == pytest.approx(p["back_neck_depth"], abs=1e-6)
    assert np.asarray(front["half_outline_mm"])[0, 1] == pytest.approx(p["front_neck_depth"], abs=1e-6)
    paper = np.asarray(truth["perfect"]["paper_outline_mm"])
    body = np.asarray(truth["perfect"]["body_outline_mm"])
    origin = np.asarray(front["origin_mm"])
    assert paper[0, 1] - origin[1] == pytest.approx(p["front_neck_depth"], abs=1e-6)
    assert body[0, 1] - origin[1] == pytest.approx(p["back_neck_depth"], abs=1e-6)


def test_req_ver_02_neck_gap_is_reported_separately_not_in_the_mean(proofs, client):
    """Regression (SPEC 10.1): with the back neck visible, a perfect paper has a neck gap of
    front depth minus back depth. Old behaviour counted it: mean gap over 0.5 mm for a perfect
    paper and more than 3 mm for a 1.02 paper whose cutting error is small. Now it is information."""
    tee_dir, truth = proofs
    for name in ("perfect", "scaled"):
        photo = tee_dir / f"{name}.jpg"
        hom = photo_homography(cv2.imread(str(photo)))
        paper = mask_outline_mm(client.segment(photo, "paper"), hom)
        garment = mask_outline_mm(client.segment(photo, "t-shirt"), hom)
        body, armhole, neck = contour.cut_sleeves_flagged(contour.smooth_outline(Outline.from_array(garment)))
        fold = find_fold_edge(paper, body.array())
        _, old = edge_gaps(paper, body.array(), hidden=armhole, fold=fold)  # no neck rule
        _, new = edge_gaps(paper, body.array(), hidden=armhole, fold=fold, neck=neck)
        r, t = _verify(proofs, client, name)
        print(f"{name}: mean without neck rule {old.mean():.2f} mm, with {new.mean():.2f} mm, "
              f"neck info {r.neck_mean_mm:.1f} mm, expected {t['expected_neck_gap_mean_mm']:.1f} mm")
        assert old.mean() > new.mean() + 1.0
        if name == "perfect":  # the old code turned a perfect paper red
            assert old.mean() > get_config().proof.gap_tolerance_mm
        assert r.mean_mm == pytest.approx(new.mean(), abs=1e-9)
        assert r.mean_mm == pytest.approx(t["expected_gap_mean_mm"], abs=0.5)
        assert r.neck_mean_mm == pytest.approx(t["expected_neck_gap_mean_mm"], abs=2.0)
        assert len(r.neck_points_mm) == len(r.neck_distances_mm) > 20
        depth_diff = truth_depth_diff(tee_dir)
        assert r.neck_mean_mm > 0.5 * depth_diff  # the neck gap is of the order front minus back depth
    perfect, _ = _verify(proofs, client, "perfect")
    assert perfect.mean_mm < 0.5 and perfect.frac_within_tol > 0.98  # the old code: mean over 0.5 mm


def truth_depth_diff(tee_dir: Path) -> float:
    p = json.loads((tee_dir / "truth.json").read_text())["params"]
    return p["front_neck_depth"] - p["back_neck_depth"]


def test_req_ver_02_find_fold_edge_needs_a_fold():
    """A paper with no straight edge deep inside the tee is reported, not guessed."""
    body = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], float)
    circle = np.c_[150 + 100 * np.cos(np.linspace(0, 2 * np.pi, 200, endpoint=False)),
                   150 + 100 * np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False))]
    with pytest.raises(ValueError, match="fold"):
        find_fold_edge(circle, body)
    half = np.array([[150, 20], [280, 20], [280, 280], [150, 280]], float)  # fold at x = 150
    p0, p1 = find_fold_edge(half, body)
    assert p0[0] == pytest.approx(150) and p1[0] == pytest.approx(150)


def test_req_ver_01_cli_verify_prints_gap_and_writes_overlay(proofs, tmp_path, monkeypatch, capsys):
    """REQ-VER-01: sewtrace verify <photo> --out overlay.png on a synthetic proof, offline mock."""
    from sewtrace import cli

    tee_dir, _ = proofs
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "overlay.png"
    cli.main(["verify", str(tee_dir / "scaled.jpg"), "--out", str(out)])
    text = capsys.readouterr().out
    assert "mean gap" in text and "max" in text and "p95" in text and "within 3 mm" in text
    assert "neck (information, not in the mean)" in text
    assert cv2.imread(str(out)) is not None


def test_req_ver_01_cli_verify_missing_photo_names_it(tmp_path, capsys):
    from sewtrace import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", str(tmp_path / "nope.jpg")])
    assert exc.value.code not in (0, None)
    assert "nope.jpg" in str(exc.value.code)


def test_req_ver_01_cli_verify_marker_not_found_names_photo(tmp_path):
    from sewtrace import cli

    blank = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank), np.full((200, 300, 3), 200, np.uint8))
    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", str(blank)])
    assert "blank.jpg" in str(exc.value.code) and "marker" in str(exc.value.code)


# ---- expected neck gap from the spec (REQ-VER-02) ------------------------------------

def _spec_with_front_depth(tee_dir: Path, depth_mm: float):
    from sewtrace.mock.fixtures import truth_spec
    from sewtrace.reason.reshape import with_neck_depth
    spec = truth_spec(tee_dir)
    front = with_neck_depth(spec.panel("front"), depth_mm)
    return spec.model_copy(update={"panels": [front if p.name == "front" else p for p in spec.panels]})


def test_req_ver_02_expected_neck_gap_matches_the_synthetic_truth(proofs, client):
    """The expected neck gap from the spec's two neck curves matches the measured one (flag off)
    for the right spec, and the gap flag is on for a spec whose front neck is as shallow as the back."""
    from sewtrace.mock.fixtures import truth_spec
    from sewtrace.vision.verify import expected_neck_gap_mm
    tee_dir, truth = proofs
    spec = truth_spec(tee_dir)
    assert expected_neck_gap_mm(spec) == pytest.approx(truth["perfect"]["expected_neck_gap_mean_mm"], abs=2.0)
    tol = get_config().proof.neck_gap_tolerance_mm
    for name in ("perfect", "scaled"):
        photo = tee_dir / f"{name}.jpg"
        r = verify_proof(photo, client, photo_homography(cv2.imread(str(photo))), spec=spec)
        print(f"{name}: neck gap measured {r.neck_mean_mm:.1f}, expected {r.expected_neck_gap_mm:.1f} mm")
        assert abs(r.neck_gap_difference_mm) < tol and not r.neck_gap_flagged
    shallow = _spec_with_front_depth(tee_dir, 22.0)  # the bug of SPEC 10.1: front depth about the back depth
    photo = tee_dir / "perfect.jpg"
    r = verify_proof(photo, client, photo_homography(cv2.imread(str(photo))), spec=shallow)
    assert r.neck_gap_flagged and abs(r.neck_gap_difference_mm) > tol
    r = verify_proof(photo, client, photo_homography(cv2.imread(str(photo))))
    assert r.expected_neck_gap_mm is None and not r.neck_gap_flagged  # no spec: nothing to compare


def test_req_ver_02_cli_verify_prints_expected_neck_gap_and_flags_a_difference(proofs, tmp_path, monkeypatch, capsys):
    from sewtrace import cli
    from sewtrace.mock.fixtures import truth_spec

    tee_dir, _ = proofs
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)  # no out/<folder>/spec.json to find
    cli.main(["verify", str(tee_dir / "perfect.jpg"), "--out", str(tmp_path / "o.png")])
    assert "expected: unknown" in capsys.readouterr().out
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(truth_spec(tee_dir).model_dump_json())
    bad.write_text(_spec_with_front_depth(tee_dir, 22.0).model_dump_json())
    cli.main(["verify", str(tee_dir / "perfect.jpg"), "--out", str(tmp_path / "o.png"), "--spec", str(good)])
    text = capsys.readouterr().out
    assert "neck gap expected from the spec" in text and "WARNING" not in text
    cli.main(["verify", str(tee_dir / "perfect.jpg"), "--out", str(tmp_path / "o.png"), "--spec", str(bad)])
    assert "WARNING: the neck gap differs from the spec" in capsys.readouterr().out
    # Default lookup: out/<photo folder name>/spec.json, the folder of `sewtrace run`.
    found = tmp_path / "out" / tee_dir.name
    found.mkdir(parents=True)
    (found / "spec.json").write_text(good.read_text())
    cli.main(["verify", str(tee_dir / "perfect.jpg"), "--out", str(tmp_path / "o.png")])
    assert "neck gap expected from the spec" in capsys.readouterr().out


# ---- fold edge: bulge, wobble, cut edge next to it (REQ-VER-02) ----------------------

def _rect_case(fold_shape):
    """Body 600 x 700 mm; half paper cut 3 mm inside the body edge on three sides; its fold on
    x = 300 follows ``fold_shape(y)`` (mm off the line, towards the left)."""
    body = np.asarray(shapely.segmentize(shapely.box(0, 0, 600, 700), 1.0).exterior.coords)[:-1]
    ys = np.arange(697.0, 2.9, -1.0)
    fold = np.column_stack([300 - fold_shape(ys), ys])
    paper = np.vstack([[[300, 3.0], [597, 3.0], [597, 697.0]], fold])
    return paper, body


@pytest.mark.parametrize("name, shape", [
    ("straight", lambda y: 0 * y),
    ("bulge", lambda y: 3.5 * np.sin(np.clip((y - 250) / 150, 0, 1) * np.pi)),   # 3.5 mm over 150 mm
    ("wobble", lambda y: 2.5 * np.sin(y / 6.0)),                                      # 2.5 mm, short waves
])
def test_req_ver_02_fold_found_whole_and_cut_edge_next_to_it_keeps_its_gap(name, shape):
    """REQ-VER-02: the whole fold is found (694 mm) and left out, even with a bulge or a wobble of
    2 to 4 mm; a paper cut 3 mm too small still shows 3 mm everywhere except within the fold band."""
    paper, body = _rect_case(shape)
    p0, p1 = find_fold_edge(paper, body)
    assert abs(np.linalg.norm(p1 - p0) - 694) < 8
    pts, d = edge_gaps(paper, body, fold=(p0, p1))
    assert len(d) > 300
    assert np.allclose(d, 3.0, atol=0.2), f"{name}: max gap {d.max():.1f} mm"
    band = get_config().proof.fold_band_mm
    top = pts[np.isclose(pts[:, 1], 3.0)]
    assert top[:, 0].min() < 300 + 2 * band   # the cut edge right next to the fold is compared


def _bow(sagitta_mm: float):
    """A fold that bows by ``sagitta_mm`` in the middle (a parabola) and meets x = 300 at both ends."""
    return lambda y: sagitta_mm * 4 * ((y - 3) / 694) * (1 - (y - 3) / 694)


@pytest.mark.parametrize("sagitta_mm", [3.0, 5.0])
def test_req_ver_02_a_small_bow_in_the_fold_is_absorbed_with_correct_gaps(sagitta_mm):
    """REQ-VER-02: a fold that bows 3 or 5 mm stays inside the band: the whole fold is found and a
    paper cut 3 mm too small shows 3 mm everywhere, also at the fold ends."""
    paper, body = _rect_case(_bow(sagitta_mm))
    p0, p1 = find_fold_edge(paper, body)
    assert abs(np.linalg.norm(p1 - p0) - 694) < 8
    _, d = edge_gaps(paper, body, fold=(p0, p1))
    assert len(d) > 300 and np.allclose(d, 3.0, atol=0.2), f"max gap {d.max():.1f} mm"


@pytest.mark.parametrize("sagitta_mm", [8.0, 12.0, -8.0])
def test_req_ver_02_a_fold_that_bows_over_the_band_is_refused(sagitta_mm):
    """REQ-VER-02: an 8 mm bow keeps only the straight part of the fold (556 mm of 700) and the rest
    would count as cut edge (mean gap 7.6 mm). The user gets the action instead of a wrong gap."""
    paper, body = _rect_case(_bow(sagitta_mm))
    with pytest.raises(PaperFoldNotStraight, match=r"The paper fold is not straight\. Make the paper flat "
                                                    r"on the t-shirt and take the proof photo again\."):
        find_fold_edge(paper, body)


def test_req_ver_02_cli_verify_names_the_photo_when_the_fold_is_not_straight(proofs, monkeypatch):
    from sewtrace import cli
    from sewtrace.vision import verify

    def refuse(*_):
        raise PaperFoldNotStraight("The paper fold is not straight. Make the paper flat on the t-shirt "
                                   "and take the proof photo again.")

    tee_dir, _ = proofs
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setattr(verify, "find_fold_edge", refuse)
    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", str(tee_dir / "perfect.jpg")])
    assert "perfect.jpg" in str(exc.value.code) and "take the proof photo again" in str(exc.value.code)


def test_req_geo_01_cli_verify_tells_to_take_the_proof_photo_again_on_an_unclear_shoulder(proofs, monkeypatch, caplog):
    """A ShoulderPointNotClear on the proof photo says which photo to take again, keeps the numbers
    of the refusal, and logs them."""
    from sewtrace import cli

    def refuse(_):
        raise contour.ShoulderPointNotClear("the shoulder point is not clear: x (spread 52 mm, limit 40 mm)", 52.0, 40.0)

    tee_dir, _ = proofs
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setattr(contour, "cut_sleeves_flagged", refuse)
    with caplog.at_level("WARNING"), pytest.raises(SystemExit) as exc:
        cli.main(["verify", str(tee_dir / "perfect.jpg")])
    text = str(exc.value.code)
    assert "perfect.jpg" in text and "spread 52 mm, limit 40 mm" in text and text.endswith("Take the proof photo again.")
    assert "spread 52.0 mm" in caplog.text


def test_req_ver_02_fold_band_and_end_margin_are_separate_settings():
    """Nit: the width of the band and the margin at the fold ends are two settings with their own effect."""
    proof_cfg = get_config().proof
    assert proof_cfg.fold_band_mm > 0 and proof_cfg.fold_end_margin_mm >= 0
    assert "fold_end_margin_mm" in type(proof_cfg).model_fields


def _side_result(side_gap_of_y):
    """Proof result for a half paper whose top edge is 3 mm inside the body and whose right side is
    3 mm + ``side_gap_of_y`` mm inside it."""
    ys = np.arange(3.0, 697.1, 2.0)
    top = np.column_stack([np.arange(300, 597, 2.0), np.full(149, 3.0)])
    points = np.vstack([top, np.column_stack([597 - side_gap_of_y(ys), ys])])
    distances = np.concatenate([np.full(len(top), 3.0), 3.0 + side_gap_of_y(ys)])
    return gap_result(points, distances, fold=(np.array([300.0, 3.0]), np.array([300.0, 697.0])))


def test_req_ver_02_same_gap_along_the_side_gives_an_info_line_and_changes_no_number():
    """A fold off the centre moves the whole side: a flat side gap. A gap that grows along the side
    (a paper that is tilted) is not flat. Information only: the gap numbers are the same."""
    flat = _side_result(lambda y: 0 * y + 5.0)
    assert flat.side_offset_mm == pytest.approx(8.0, abs=0.1)
    tilted = _side_result(lambda y: y / 100.0)
    assert tilted.side_offset_mm is None
    assert _side_result(lambda y: 0 * y - 2.0).side_offset_mm is None   # a flat gap of 1 mm: nothing to report
    plain = gap_result(np.asarray(flat.points_mm), np.asarray(flat.distances_mm))
    assert (flat.mean_mm, flat.max_mm, flat.p95_mm) == (plain.mean_mm, plain.max_mm, plain.p95_mm)


def test_req_ver_02_cli_verify_prints_the_same_side_gap_hint_for_a_shifted_paper(proofs, tmp_path, monkeypatch, capsys):
    from sewtrace import cli

    tee_dir, _ = proofs
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    cli.main(["verify", str(tee_dir / "shifted.jpg"), "--out", str(tmp_path / "o.png")])
    assert "The gap is the same along the side" in capsys.readouterr().out
    cli.main(["verify", str(tee_dir / "perfect.jpg"), "--out", str(tmp_path / "o.png")])
    assert "same along the side" not in capsys.readouterr().out


def test_req_ver_02_fold_line_must_be_deep_inside_the_tee():
    """REQ-VER-02: a straight paper edge near the tee edge is a cut edge, not a fold."""
    body = np.asarray(shapely.segmentize(shapely.box(0, 0, 600, 700), 1.0).exterior.coords)[:-1]
    paper = np.array([[10, 10], [590, 10], [590, 690], [10, 690]], float)
    with pytest.raises(ValueError, match="fold"):
        find_fold_edge(paper, body)


def test_req_geo_01_clean_mask_radius_is_stable_at_exactly_3_px_per_mm():
    """REQ-GEO-01: cleanup_mm * px_per_mm = 1.5 at 3 px/mm must give a radius of 2 px, also when
    the product is 1.4999999999999998. A whisker 3 px wide survives a 1 px radius, not a 2 px one."""
    from sewtrace.vision import contour

    mask = np.zeros((200, 200), np.uint8)
    mask[40:160, 40:160] = 1
    mask[98:101, 160:168] = 1   # whisker 3 px wide, 8 px long
    for px_per_mm in (3.0, np.nextafter(3.0, 0), 3.0 * (1 - 1e-12)):
        assert not contour.clean_mask(mask, px_per_mm)[98:101, 162:168].any(), px_per_mm
