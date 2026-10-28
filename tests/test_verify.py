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
from sewtrace.vision.verify import (ProofResult, edge_gaps, find_fold_edge, gap_result, mask_outline_mm,
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
