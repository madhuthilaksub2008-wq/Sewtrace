from pathlib import Path

import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings, get_config
from sewtrace.mock import synth
from sewtrace.mock.fixtures import truth_spec
from sewtrace.reason import seams
from sewtrace.reason.seams import adjust_loop, check_seams, edge_length, set_edge_length
from sewtrace.types import Panel, PanelSpec

MAX_ADJUST_CALLS = get_config().seam.max_adjust_calls


@pytest.fixture(scope="session")
def tees(tmp_path_factory) -> list[Path]:
    out = tmp_path_factory.mktemp("raw")
    dirs = []
    for i in (0, 4):
        synth.make_tee(synth.TEES[i], out / synth.TEES[i].name, seed=i)
        dirs.append(out / synth.TEES[i].name)
    return dirs


@pytest.fixture
def client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))


def _replace(spec: PanelSpec, panel: Panel) -> PanelSpec:
    return spec.model_copy(update={"panels": [panel if p.name == panel.name else p for p in spec.panels]})


def perturb(spec: PanelSpec) -> PanelSpec:
    """Sleeve cap 10% narrower; back side seam 8 mm shorter (hem corner pulled up the seam)."""
    sleeve = spec.panel("sleeve")
    pts = np.asarray(sleeve.outline_mm, dtype=float)
    cap = next(e for e in sleeve.edges if e.name == "cap")
    pts[cap.start : cap.end + 1, 0] *= 0.9
    spec = _replace(spec, sleeve.model_copy(update={"outline_mm": [tuple(p) for p in pts]}))

    back = spec.panel("back")
    pts = np.asarray(back.outline_mm, dtype=float)
    side = next(e for e in back.edges if e.name == "side")
    seg = pts[side.start : side.end + 1]
    k = (edge_length(back, "side") - 8.0) / edge_length(back, "side")
    pts[side.start : side.end + 1] = seg[0] + (seg - seg[0]) * k
    return _replace(spec, back.model_copy(update={"outline_mm": [tuple(p) for p in pts]}))


def test_edge_length_of_straight_shoulder(tees):
    p = synth.TEES[0]
    front = truth_spec(tees[0]).panel("front")
    expected = np.hypot(p.shoulder_half - p.neck_half, p.shoulder_drop)
    assert edge_length(front, "shoulder") == pytest.approx(expected, abs=1e-6)


def test_truth_passes_all_seams(tees, client):
    for tee in tees:
        spec = truth_spec(tee)
        checked = check_seams(spec)
        assert all(s.ok for s in checked)
        assert all(s.len_a_mm > 0 and s.len_b_mm > 0 for s in checked)
        out, log = adjust_loop(client, spec)
        assert len(log) == 1 and log[0]["failing"] == [] and client.calls == 0


def test_perturbed_loop_converges_with_fallback(tees, client):
    spec = perturb(truth_spec(tees[0]))
    failing = [seams.seam_name(s) for s in check_seams(spec) if not s.ok]
    assert "front.side~back.side" in failing and "front.armhole~sleeve.cap" in failing

    out, log = adjust_loop(client, spec, max_calls=MAX_ADJUST_CALLS)
    for row in log:
        print(row["iteration"], row["source"], row["diffs"], row["applied"])
    assert all(s.ok for s in check_seams(out))
    assert all(s.ok for s in out.seams)
    assert client.calls <= MAX_ADJUST_CALLS
    sources = [r["source"] for r in log]
    assert sources[0] == "initial" and sources[-1] == "fallback"
    assert sources.count("model") == MAX_ADJUST_CALLS
    # Each model iteration improved the worst seam it touched.
    assert log[1]["diffs"]["front.armhole~sleeve.cap"] > log[0]["diffs"]["front.armhole~sleeve.cap"]
    # Geometry stays sane: fold corners untouched, polygons still simple.
    from shapely.geometry import Polygon
    for name in ("front", "back", "sleeve"):
        assert Polygon(out.panel(name).outline_mm).is_valid
    assert out.panel("back").outline_mm[0] == spec.panel("back").outline_mm[0]


def test_perturbed_loop_is_cached(tees, client):
    spec = perturb(truth_spec(tees[1]))
    out1, log1 = adjust_loop(client, spec)
    calls = client.calls
    out2, log2 = adjust_loop(client, spec)
    assert client.calls == calls
    assert log1 == log2 and out1 == out2


def test_perfect_model_needs_no_fallback(tees, client, monkeypatch):
    monkeypatch.setattr(seams, "MOCK_FIX_FRACTION", 1.0)
    spec = truth_spec(tees[0])
    back = set_edge_length(spec.panel("back"), "shoulder", edge_length(spec.panel("back"), "shoulder") + 6)
    out, log = adjust_loop(client, _replace(spec, back))
    assert log[-1]["source"] == "model" and log[-1]["failing"] == []
    assert client.calls == 1


def test_set_edge_length_hits_target(tees):
    sleeve = truth_spec(tees[0]).panel("sleeve")
    for delta in (-15.0, 7.5):
        target = edge_length(sleeve, "cap") + delta
        new = set_edge_length(sleeve, "cap", target)
        assert edge_length(new, "cap") == pytest.approx(target, abs=0.01)
        assert new.outline_mm[0] == sleeve.outline_mm[0]  # cap top stays put


def test_bad_model_answers_are_skipped(tees):
    spec = truth_spec(tees[0])
    out, applied = seams.apply_adjustments(spec, {"adjustments": [
        {"panel": "back", "edge": "hem", "target_length_mm": 100},
        {"panel": "sleeve", "edge": "cap", "target_length_mm": 1e6},
        {"panel": "nope", "edge": "cap", "target_length_mm": 10},
        {"oops": 1},
    ]})
    assert applied == [] and out == spec
