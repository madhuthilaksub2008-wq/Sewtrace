import numpy as np
import pytest

from sewtrace.mock import synth
from sewtrace.types import MEASUREMENT_KEYS, Edge, Outline
from sewtrace.vision.measure import measure_tee
from test_contour import (  # noqa: F401  (shared session fixtures)
    TEE_NAMES,
    extract,
    halves,
    rotate_photo,
    session_client,
    tees,
    truth_homography,
)

TOL_MM = 2.0


def _edges(ranges: dict) -> list[Edge]:
    return [Edge(name=k, start=a, end=b) for k, (a, b) in ranges.items()]


@pytest.mark.parametrize("name", TEE_NAMES)
def test_measurements_match_truth(halves, tees, name):
    """REQ-GEO-04: measurements match truth."""
    got = measure_tee(*halves[name, "front"], *halves[name, "back"], *halves[name, "sleeve"])
    want = tees[name][1]["measurements"]
    assert set(got) == set(MEASUREMENT_KEYS)
    for k in MEASUREMENT_KEYS:
        assert got[k] == pytest.approx(want[k], abs=TOL_MM), k


def test_req_geo_04_whole_garment_mean_error_within_gate(halves, tees):
    """REQ-GEO-04, SPEC 10.1: on the whole-garment path the mean absolute error over the
    6 measurements and 5 tees is 0.5 mm or less (the synthetic gate, now 0.37 mm). The single
    measurement bound is TOL_MM. SPEC 10.2 takes the shoulder across the back, where the
    sleeve joint of tee2 sits 0.5 mm off, so the worst single value is 1.58 mm (was 0.95 mm
    on the front)."""
    errors = []
    for name in TEE_NAMES:
        got = measure_tee(*halves[name, "front"], *halves[name, "back"], *halves[name, "sleeve"])
        errors += [abs(got[k] - tees[name][1]["measurements"][k]) for k in MEASUREMENT_KEYS]
    assert np.mean(errors) <= 0.5 and max(errors) < TOL_MM


@pytest.mark.parametrize("name", TEE_NAMES)
def test_req_geo_04_front_photo_shows_back_neck_width_still_measured(halves, tees, name):
    """REQ-GEO-04, REQ-RSN-04: the front mask shows the back neckline through the opening, so the
    neck DEPTH read from the front outline is the back depth (it is inferred, not measured).
    The neck WIDTH is still measured: within 1 mm on a neck with a radius of about 7 mm at the
    neck point (a straight tangent there was 1.2 to 1.7 mm off)."""
    params = tees[name][1]["params"]
    front, edges = halves[name, "front"]
    neck = front.array()[edges[0].start : edges[0].end + 1]
    assert neck[0, 1] - neck[-1, 1] == pytest.approx(params["back_neck_depth"], abs=1.0)
    assert abs(neck[0, 1] - neck[-1, 1] - params["front_neck_depth"]) > 30.0
    got = measure_tee(*halves[name, "front"], *halves[name, "back"], *halves[name, "sleeve"])
    assert got["neck_width"] == pytest.approx(tees[name][1]["measurements"]["neck_width"], abs=1.0)


def test_measurements_on_rotated_photos(tees, session_client):
    """REQ-GEO-04: measurements on rotated photos."""
    d, truth = tees["tee4"]
    parts = []
    for view, deg, frame_deg in (("front", 6.0, -2.0), ("back", 3.0, 1.0), ("sleeve", -4.0, 3.0)):
        mask = session_client.segment(d / f"{view}.jpg", "sleeve" if view == "sleeve" else "t-shirt")
        rmask, H = rotate_photo(mask, truth_homography(truth, view), deg, frame_deg)
        parts.extend(extract(rmask, H, view))
    got = measure_tee(*parts)
    for k in MEASUREMENT_KEYS:
        assert got[k] == pytest.approx(truth["measurements"][k], abs=TOL_MM), k


def test_measurements_on_exact_truth_outlines():
    """REQ-GEO-04: Exact synth geometry with its own edge labels gives the parameters back."""
    p = synth.TEES[2]
    body, be = synth.body_half(p)
    back, bke = synth.body_half(p, back=True)
    sleeve, se = synth.sleeve_half(p)
    got = measure_tee(Outline.from_array(body), _edges(be), Outline.from_array(back), _edges(bke),
                      Outline.from_array(sleeve), _edges(se))
    for k, v in synth.measurements(p).items():
        assert got[k] == pytest.approx(v, abs=0.5), k
    assert np.isfinite(list(got.values())).all()


def test_req_geo_04_definitions_follow_spec_10_2():
    """REQ-GEO-04, SPEC 10.2: chest is 25 mm below the underarm points (not at them) and the
    shoulder is taken across the back. A tee whose back shoulder and side flare differ from
    the front shows both, with the truth read off the exact geometry."""
    from dataclasses import replace

    from sewtrace.config import get_config

    p = synth.TEES[1]
    flared = replace(p, hem_flare=20.0)
    body, be = synth.body_half(flared)
    wide_back, bke = synth.body_half(replace(flared, shoulder_half=p.shoulder_half + 7.0), back=True)
    sleeve, se = synth.sleeve_half(p)
    got = measure_tee(Outline.from_array(body), _edges(be), Outline.from_array(wide_back), _edges(bke),
                      Outline.from_array(sleeve), _edges(se))
    drop = get_config().pattern.chest_drop_mm
    side_run = flared.length - (flared.shoulder_drop + flared.armhole_depth)
    want_chest = 2 * (flared.chest_half + flared.hem_flare * drop / side_run)
    assert want_chest - 2 * flared.chest_half > 3.0  # the definition matters on this tee
    assert got["chest_width"] == pytest.approx(want_chest, abs=0.5)
    assert got["shoulder_width"] == pytest.approx(2 * (p.shoulder_half + 7.0), abs=0.5)
    assert got["armhole_depth"] == pytest.approx(p.armhole_depth, abs=0.5)
    assert got["neck_width"] == pytest.approx(2 * p.neck_half, abs=0.5)


def test_req_geo_04_synthetic_truth_is_independent_of_measure_code():
    """REQ-GEO-04: truth.json values come from the exact geometry in SPEC 10.2 words."""
    p = synth.TEES[0]
    truth = synth.measurements(p)
    side_run = p.length - (p.shoulder_drop + p.armhole_depth)
    assert truth["chest_width"] == pytest.approx(2 * (p.chest_half + p.hem_flare * 25 / side_run), abs=1e-6)
    assert truth["chest_width"] > 2 * p.chest_half  # not measured at the underarm level
    assert truth["body_length"] == pytest.approx(p.length, abs=1e-6)
    assert truth["shoulder_width"] == pytest.approx(2 * p.shoulder_half, abs=1e-6)
    assert truth["sleeve_length"] == pytest.approx(p.sleeve_length, abs=1e-6)
    assert truth["neck_width"] == pytest.approx(2 * p.neck_half, abs=1e-6)
    assert truth["armhole_depth"] == pytest.approx(p.armhole_depth, abs=1e-6)
