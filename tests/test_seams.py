import json
from pathlib import Path

import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings, get_config
from sewtrace.mock import synth
from sewtrace.mock.fixtures import truth_spec
from sewtrace.reason import seams
from sewtrace.reason.reshape import cap_width_mm, fit_cap, scale_cap_to_length
from sewtrace.reason.seams import SeamCheckFailed, adjust_loop, require_seams_ok, check_seams, edge_length
from perturb import bulge_edge, set_edge_length
from sewtrace.types import Panel, PanelSpec

MAX_ADJUST_CALLS = get_config().seam.max_adjust_calls
CAP = "front.armhole+back.armhole~sleeve.cap"


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


def test_edge_length_of_straight_shoulder(tees):
    """REQ-SEW-01: edge length of straight shoulder."""
    p = synth.TEES[0]
    front = truth_spec(tees[0]).panel("front")
    expected = np.hypot(p.shoulder_half - p.neck_half, p.shoulder_drop)
    assert edge_length(front, "shoulder") == pytest.approx(expected, abs=1e-6)


def test_truth_passes_all_seams(tees, client):
    """REQ-SEW-01: truth passes all seams."""
    for tee in tees:
        spec = truth_spec(tee)
        checked = check_seams(spec)
        assert all(s.ok for s in checked)
        assert all(s.len_a_mm > 0 and s.len_b_mm > 0 for s in checked)
        out, log = adjust_loop(client, spec)
        assert len(log) == 1 and log[0]["failing"] == [] and client.calls == 0


def _longer_front_armhole(spec: PanelSpec, extra_mm: float = 16.0) -> PanelSpec:
    """Armhole bulges outward: end points (so shoulder and side) stay as measured."""
    return _replace(spec, bulge_edge(spec.panel("front"), "armhole", extra_mm))


def _back_shoulder_6mm_long(spec: PanelSpec) -> PanelSpec:
    back = spec.panel("back")
    return _replace(spec, set_edge_length(back, "shoulder", edge_length(back, "shoulder") + 6))


def _measured_edges(spec: PanelSpec) -> dict[str, np.ndarray]:
    """Every edge except the hidden ones (neck curve, armhole, sleeve cap), as arrays."""
    hidden = {"neck", "armhole", "cap"}
    return {f"{p.name}.{e.name}": p.edge_points(e.name)
            for p in spec.panels if p.name in ("front", "back", "sleeve") for e in p.edges if e.name not in hidden}


def _assert_measured_unchanged(before: PanelSpec, after: PanelSpec) -> None:
    a, b = _measured_edges(before), _measured_edges(after)
    assert a.keys() == b.keys()
    for key in a:
        assert np.allclose(a[key], b[key], atol=1e-6), key
    cap_a, cap_b = before.panel("sleeve").edge_points("cap"), after.panel("sleeve").edge_points("cap")
    assert np.allclose([cap_a[0], cap_a[-1]], [cap_b[0], cap_b[-1]], atol=1e-6)


def test_req_sew_02_cap_loop_converges_and_measured_edges_stay(tees, client):
    """REQ-SEW-02/03, REQ-RSN-05: a too-short cap is fixed by cap shape alone, within 3 model calls."""
    spec = _longer_front_armhole(truth_spec(tees[0]))
    assert [seams.seam_name(s) for s in check_seams(spec) if not s.ok] == [CAP]

    out, log = adjust_loop(client, spec, max_calls=MAX_ADJUST_CALLS)
    for row in log:
        print(row["iteration"], row["source"], row["diffs"], row["applied"])
    assert all(s.ok for s in check_seams(out)) and all(s.ok for s in out.seams)
    assert client.calls <= MAX_ADJUST_CALLS
    assert log[0]["source"] == "initial"
    assert log[1]["diffs"][CAP] > log[0]["diffs"][CAP]  # the imperfect mock still moves toward the target
    _assert_measured_unchanged(spec, out)
    from shapely.geometry import Polygon
    for name in ("front", "back", "sleeve"):
        assert Polygon(out.panel(name).outline_mm).is_valid
    require_seams_ok(out, log)  # does not raise


def test_perturbed_loop_is_cached(tees, client):
    """REQ-API-02: perturbed loop is cached."""
    spec = _longer_front_armhole(truth_spec(tees[1]))
    out1, log1 = adjust_loop(client, spec)
    calls = client.calls
    out2, log2 = adjust_loop(client, spec)
    assert client.calls == calls
    assert log1 == log2 and out1 == out2


def test_perfect_model_needs_no_fallback(tees, client, monkeypatch):
    """REQ-SEW-02: perfect model needs no fallback."""
    monkeypatch.setattr(seams, "MOCK_FIX_FRACTION", 1.0)
    out, log = adjust_loop(client, _longer_front_armhole(truth_spec(tees[0])))
    assert log[-1]["source"] == "model" and log[-1]["failing"] == []
    assert client.calls == 1


def test_req_sew_01_all_four_pairs_measured_with_whole_cap(tees, client):
    """REQ-SEW-01: pairs of SPEC 7.6.1; cap is 2x half cap against front + back armhole."""
    spec = truth_spec(tees[0])
    by = {seams.seam_name(s): s for s in check_seams(spec)}
    assert set(by) == {"front.shoulder~back.shoulder", "front.side~back.side", CAP,
                       "sleeve.underarm~sleeve.underarm:mirror"}
    arm = edge_length(spec.panel("front"), "armhole") + edge_length(spec.panel("back"), "armhole")
    assert by[CAP].len_a_mm == pytest.approx(arm, abs=1e-6)
    assert by[CAP].len_b_mm == pytest.approx(2 * edge_length(spec.panel("sleeve"), "cap"), abs=1e-6)
    # Synthetic truth: 2 x 4 mm ease, inside 0..12 with margin.
    assert by[CAP].diff_mm == pytest.approx(8.0, abs=0.5)
    under = by["sleeve.underarm~sleeve.underarm:mirror"]
    assert under.len_a_mm == pytest.approx(under.len_b_mm, abs=1e-6) and under.ok


def test_req_sew_01_cap_range_is_for_the_whole_cap(tees):
    """REQ-SEW-01: 8 mm per half (16 mm whole) fails; the old per-half rule would allow it."""
    spec = truth_spec(tees[0])
    sleeve = spec.panel("sleeve")
    wide = set_edge_length(sleeve, "cap", edge_length(sleeve, "cap") + 4.0)
    cap = next(s for s in check_seams(_replace(spec, wide)) if seams.seam_name(s) == CAP)
    assert cap.diff_mm > get_config().seam.cap_ease_max_mm and not cap.ok


def _script_model(monkeypatch, misses_mm: list[float]) -> None:
    """Model answers: a cap whose half length is the ideal one plus a miss. A miss of 0 fixes it.

    The misses differ per call so each prompt differs and nothing is served from cache.
    """
    answers = iter(misses_mm)

    def scripted(spec_, failing):
        pts = spec_.panel("sleeve").edge_points("cap")
        target = seams._target_half_cap(failing[0]) + next(answers)
        return seams.SeamAdjustment(sleeve_cap=scale_cap_to_length(pts, fit_cap(pts), target)).model_dump()

    monkeypatch.setattr(seams, "_mock_adjust", scripted)


def test_req_sew_02_mock_fails_twice_then_succeeds(tees, client, monkeypatch):
    """REQ-SEW-02: two bad model answers, the third fixes it; no code fix needed."""
    _script_model(monkeypatch, [-5.0, -4.5, 0.0])
    out, log = adjust_loop(client, _longer_front_armhole(truth_spec(tees[0]), 10.0))
    assert [r["source"] for r in log] == ["initial", "model", "model", "model"]
    assert [bool(r["failing"]) for r in log] == [True, True, True, False]
    assert client.calls == 3


def test_req_sew_03_05_mock_always_fails_then_code_fix_and_log(tees, client, monkeypatch):
    """REQ-SEW-03 and REQ-SEW-05: 3 useless calls, then the code fix; log has checks per iteration."""
    _script_model(monkeypatch, [-5.0, -4.5, -4.0])
    spec = _longer_front_armhole(truth_spec(tees[0]), 10.0)
    out, log = adjust_loop(client, spec)
    assert client.calls == MAX_ADJUST_CALLS
    assert [r["source"] for r in log] == ["initial", "model", "model", "model", "fallback"]
    assert all({"iteration", "diffs", "failing"} <= set(r) for r in log)
    assert log[-1]["failing"] == [] and log[-1]["applied"]
    assert set(log[0]["diffs"]) == {seams.seam_name(s) for s in out.seams}
    assert all(s.ok for s in check_seams(out))
    _assert_measured_unchanged(spec, out)


def test_req_sew_03_rsn_05_measured_pair_failure_is_not_edited(tees, client):
    """REQ-RSN-05: back shoulder 6 mm long. No model call, no code fix; require_seams_ok raises."""
    spec = _back_shoulder_6mm_long(truth_spec(tees[0]))
    out, log = adjust_loop(client, spec)
    assert client.calls == 0 and out == spec.model_copy(update={"seams": out.seams})
    assert log[-1]["failing"] and "front.shoulder~back.shoulder" in log[-1]["failing"]
    with pytest.raises(SeamCheckFailed, match=r"front\.shoulder~back\.shoulder"):
        require_seams_ok(out, log)


def test_req_sew_03_cap_that_cannot_reach_range_fails_loudly(tees, client):
    """A cap cannot grow without limit (handles stop at the cap width): the seam stays failing, no silent edit."""
    spec = _longer_front_armhole(truth_spec(tees[0]), extra_mm=60.0)
    out, log = adjust_loop(client, spec)
    assert log[-1]["source"] == "fallback" and log[-1]["failing"] == [CAP]
    _assert_measured_unchanged(spec, out)
    with pytest.raises(SeamCheckFailed):
        require_seams_ok(out, log)


def test_req_rsn_04_adjust_schema_allows_only_the_cap_shape():
    """Adjust schema: no panel name, no front shoulder, armhole or side."""
    schema = seams.ADJUST_SCHEMA
    assert set(schema["properties"]) == {"sleeve_cap"} and schema["additionalProperties"] is False
    text = json.dumps(schema)
    for word in ("front", "back", "shoulder", "armhole", "side", "panel", "edge"):
        assert word not in text.replace("sleeve_cap", "").replace("SleeveCap", ""), word
    assert set(schema["$defs"]["SleeveCap"]["properties"]) == {"crown_run_mm", "underarm_run_mm"}


def test_req_rsn_04_adjust_call_sends_the_schema(tees, client, monkeypatch):
    """REQ-RSN-02: the adjust call passes the pydantic schema as structured output."""
    seen = []
    real = client.reason
    monkeypatch.setattr(client, "reason", lambda *a, **k: (seen.append(k.get("schema")), real(*a, **k))[1])
    adjust_loop(client, _longer_front_armhole(truth_spec(tees[0])))
    assert seen and all(s == seams.ADJUST_SCHEMA for s in seen)


def test_req_rsn_04_bad_adjust_answers_are_skipped(tees):
    spec = truth_spec(tees[0])
    width = cap_width_mm(spec.panel("sleeve"))
    for bad in (
        {"sleeve_cap": {"crown_run_mm": 10, "underarm_run_mm": 10}, "panel": "front"},  # extra field
        {"adjustments": [{"panel": "front", "edge": "side", "target_length_mm": 100}]},  # old shape
        {"sleeve_cap": {"crown_run_mm": width + 1, "underarm_run_mm": 10}},  # outside the sleeve
        {"sleeve_cap": {"crown_run_mm": -1, "underarm_run_mm": 10}},
        {"oops": 1},
    ):
        out, applied = seams.apply_adjustment(spec, bad)
        assert applied == [] and out == spec


def test_req_sew_02_prompt_states_whole_cap_range_from_config(tees):
    spec = _longer_front_armhole(truth_spec(tees[0]))
    prompt = seams.build_adjust_prompt(spec, [s for s in check_seams(spec) if not s.ok])
    cfg = get_config().seam
    assert "2 x the half cap" in prompt and f"{cfg.cap_ease_max_mm:g} mm longer" in prompt
    assert f"{CAP.split('~')[0]}" in prompt


def test_req_sew_04_require_seams_ok_names_pair_and_diff(tees):
    spec = _back_shoulder_6mm_long(truth_spec(tees[0]))
    with pytest.raises(SeamCheckFailed) as err:
        require_seams_ok(spec)
    text = str(err.value)
    assert "front.shoulder~back.shoulder" in text and "mm" in text
    assert "front.shoulder~back.shoulder" in [seams.seam_name(s) for s in err.value.failed]
