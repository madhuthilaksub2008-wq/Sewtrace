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
from sewtrace.reason.seams import SeamCheckFailed, adjust_loop, require_seams_ok, check_seams, edge_length, seam_name
from perturb import bulge_edge, set_edge_length
from sewtrace.types import Panel, PanelSpec, Seam

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
    """REQ-RSN-05: back shoulder 6 mm long, over the code fix limit (4 mm). No model call, no edit; raises."""
    spec = _back_shoulder_6mm_long(truth_spec(tees[0]))
    out, log = adjust_loop(client, spec)
    assert client.calls == 0 and out == spec.model_copy(update={"seams": out.seams})
    assert log[-1]["failing"] and "front.shoulder~back.shoulder" in log[-1]["failing"]
    with pytest.raises(SeamCheckFailed, match=r"front\.shoulder~back\.shoulder"):
        require_seams_ok(out, log)


PAIR_LIMIT = get_config().seam.measured_pair_max_fix_mm


def _pair_seam(spec: PanelSpec, edge: str) -> "Seam":
    return next(s for s in check_seams(spec) if s.a == f"front.{edge}")


def _side_2mm_long(spec: PanelSpec, mm: float = 2.0) -> PanelSpec:
    back = spec.panel("back")
    return _replace(spec, set_edge_length(back, "side", edge_length(back, "side") + mm))


def test_req_sew_03_side_mismatch_is_balanced_by_code_and_logged(tees, client):
    """REQ-SEW-03, D-02: back side 2 mm long. Code sets both sides to the mean; no model call; logged in mm."""
    spec = _side_2mm_long(truth_spec(tees[0]))
    assert not _pair_seam(spec, "side").ok
    front0, back0 = (edge_length(spec.panel(n), "side") for n in ("front", "back"))
    out, log = adjust_loop(client, spec)
    assert client.calls == 0
    assert all(s.ok for s in check_seams(out))
    front1, back1 = (edge_length(out.panel(n), "side") for n in ("front", "back"))
    mean = (front0 + back0) / 2
    assert front1 == pytest.approx(mean, abs=1e-6) and back1 == pytest.approx(mean, abs=1e-6)
    assert [r["source"] for r in log] == ["initial", "pair_fix"]
    row = log[1]
    assert row["failing"] == [] and "front.side~back.side" in log[0]["failing"]
    changes = {(a["panel"], a["edge"]): a for a in row["applied"]}
    assert changes[("front", "side")]["from_mm"] == pytest.approx(front0, abs=0.01)
    assert changes[("front", "side")]["to_mm"] == pytest.approx(mean, abs=0.01)
    assert changes[("back", "side")]["end_point_shift_mm"] == pytest.approx(1.0, abs=0.05)
    # Only the hem corner moves: neck, shoulder, armhole and fold are as measured.
    for name in ("front", "back"):
        for edge in ("neck", "shoulder", "armhole", "fold"):
            assert np.allclose(spec.panel(name).edge_points(edge), out.panel(name).edge_points(edge), atol=1e-6)
    assert np.allclose(spec.panel("sleeve").outline_mm, out.panel("sleeve").outline_mm)


def test_req_sew_03_shoulder_fix_keeps_the_neck_point_and_underarm(tees, client):
    """REQ-SEW-03, D-02: shoulder mismatch moves the shoulder point; neck point and underarm stay."""
    spec = truth_spec(tees[0])
    back = spec.panel("back")
    spec = _replace(spec, set_edge_length(back, "shoulder", edge_length(back, "shoulder") + 3.0))
    assert 1.0 < abs(_pair_seam(spec, "shoulder").diff_mm) <= PAIR_LIMIT
    out, log = adjust_loop(client, spec)
    assert client.calls == 0 and all(s.ok for s in check_seams(out))
    assert abs(_pair_seam(out, "shoulder").diff_mm) < 1e-6
    for name in ("front", "back"):
        before, after = spec.panel(name), out.panel(name)
        assert np.allclose(before.edge_points("neck"), after.edge_points("neck"), atol=1e-6)
        assert np.allclose(before.edge_points("shoulder")[0], after.edge_points("shoulder")[0], atol=1e-6)
        assert np.allclose(before.edge_points("armhole")[-1], after.edge_points("armhole")[-1], atol=1e-6)
        assert np.allclose(before.edge_points("side"), after.edge_points("side"), atol=1e-6)
        # The armhole stays attached to the moved shoulder point.
        assert np.allclose(after.edge_points("shoulder")[-1], after.edge_points("armhole")[0], atol=1e-9)
    assert abs(next(a for a in log[1]["applied"] if a["panel"] == "front")["end_point_shift_mm"]) < 1.6


def test_req_sew_03_pair_over_limit_is_not_edited_and_raises(tees, client):
    """D-02: a mismatch just over the limit is left alone, so the run fails loudly."""
    spec = _side_2mm_long(truth_spec(tees[0]), PAIR_LIMIT + 0.5)
    out, log = adjust_loop(client, spec)
    assert client.calls == 0 and [r["source"] for r in log] == ["initial"]
    assert out == spec.model_copy(update={"seams": out.seams})
    with pytest.raises(SeamCheckFailed, match=r"front\.side~back\.side"):
        require_seams_ok(out, log)
    at_limit = _side_2mm_long(truth_spec(tees[0]), PAIR_LIMIT - 0.05)
    assert all(s.ok for s in check_seams(adjust_loop(client, at_limit)[0]))


def test_req_sew_03_pair_fix_then_cap_loop_still_converges(tees, client):
    """The side pair fix (side 2 mm long) does not touch the cap seam; a failing cap is still fixed after it."""
    spec = _longer_front_armhole(_side_2mm_long(truth_spec(tees[0])))
    out, log = adjust_loop(client, spec)
    assert log[1]["source"] == "pair_fix" and log[-1]["failing"] == []
    require_seams_ok(out, log)


def test_req_sew_03_unfixable_measured_pair_spends_no_model_call_on_the_cap(tees, client):
    """REQ-SEW-02, REQ-SEW-04: back shoulder 6 mm long (over the limit) AND a failing cap: stop at once."""
    spec = _longer_front_armhole(_back_shoulder_6mm_long(truth_spec(tees[0])))
    failing = {seam_name(s) for s in check_seams(spec) if not s.ok}
    assert failing == {"front.shoulder~back.shoulder", CAP}
    out, log = adjust_loop(client, spec)
    assert client.calls == 0 and [r["source"] for r in log] == ["initial"]
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


def test_req_sew_02_instructions_prefix_is_fixed_and_data_changes(tees):
    """REQ-SEW-02: every seam-loop call sends the same developer prefix (prompt cache); only the
    user text changes with the failing seam."""
    spec_a = _longer_front_armhole(truth_spec(tees[0]), 16.0)
    spec_b = _longer_front_armhole(truth_spec(tees[0]), 24.0)
    ia, da = seams.build_adjust_messages(spec_a, [s for s in check_seams(spec_a) if not s.ok])
    ib, db = seams.build_adjust_messages(spec_b, [s for s in check_seams(spec_b) if not s.ok])
    assert ia == ib and da != db
    assert "Failing seam" not in ia and "Failing seam" in da
    assert "2 x the half cap" in ia


def test_req_sew_04_require_seams_ok_names_pair_and_diff(tees):
    spec = _back_shoulder_6mm_long(truth_spec(tees[0]))
    with pytest.raises(SeamCheckFailed) as err:
        require_seams_ok(spec)
    text = str(err.value)
    assert "front.shoulder~back.shoulder" in text and "mm" in text
    assert "front.shoulder~back.shoulder" in [seams.seam_name(s) for s in err.value.failed]


class _FailingClient:
    def __init__(self, client, error):
        self.client, self.error, self.reason_calls = client, error, 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def reason(self, *a, **k):
        self.reason_calls += 1
        self.client.calls += 1  # a live call that fails was sent, so the real client counts it
        raise self.error


def test_req_sew_03_failed_model_call_is_a_skipped_answer_then_code_fix(tees, client):
    """REQ-SEW-02/03, REQ-RSN-06: ApiError counts as a used call; after 3 the code fix runs."""
    from sewtrace.api.client import ApiError
    spec = _longer_front_armhole(truth_spec(tees[0]))
    flaky = _FailingClient(client, ApiError("cut off at max_tokens"))
    out, log = adjust_loop(flaky, spec)
    assert flaky.reason_calls == MAX_ADJUST_CALLS
    assert log[-1]["source"] == "fallback" and all(s.ok for s in check_seams(out))


def test_req_sew_03_budget_cache_policy_errors_propagate(tees, client):
    from sewtrace.api.client import BudgetExceeded, CacheMiss, PolicyViolation
    spec = _longer_front_armhole(truth_spec(tees[0]))
    for err in (BudgetExceeded("over"), CacheMiss("miss"), PolicyViolation("no")):
        with pytest.raises(type(err)):
            adjust_loop(_FailingClient(client, err), spec)


def test_req_rsn_03_rejected_request_stops_the_seam_loop(tees, tmp_path):
    """REQ-API-06: a mocked HTTP 400 is not a skipped answer: it propagates after one call."""
    from sewtrace.api.client import ApiRequestRejected, HttpError
    from test_api_client import FakeTransport
    transport = FakeTransport(HttpError(400, "unknown parameter"))
    real = MetaClient(Settings(api_key="k", mock=False, cache_dir=tmp_path / "cache", config=get_config(),
                               ledger_path=tmp_path / "ledger.csv"), transport=transport, sleep=lambda s: None)
    with pytest.raises(ApiRequestRejected):
        adjust_loop(real, _longer_front_armhole(truth_spec(tees[0])))
    assert len(transport.calls) == 1


def test_seam_log_marks_underarm_pair_equal_by_construction(tees, client):
    """The sleeve underarm is compared with its own mirror image: always 0, so the log says so."""
    _, log = adjust_loop(client, truth_spec(tees[0]))
    assert log[0]["equal_by_construction"] == ["sleeve.underarm~sleeve.underarm:mirror"]


def test_req_sew_02_03_stored_rejected_answer_goes_straight_to_the_code_fix(tees, client, monkeypatch):
    """REQ-SEW-02, REQ-SEW-03: a replay of a stored rejected answer is not a model call; no more replays."""
    from sewtrace.api.client import StoredAnswerRejected
    attempts: list[int] = []

    def replay(*a, **k):
        attempts.append(1)
        raise StoredAnswerRejected("not valid JSON", "k.rejected.json")  # client.calls does not move

    monkeypatch.setattr(client, "reason", replay)
    spec = _longer_front_armhole(truth_spec(tees[0]), 10.0)
    out, log = adjust_loop(client, spec)
    assert len(attempts) == 1 and client.calls == 0
    assert [r["source"] for r in log] == ["initial", "fallback"]
    assert [r["stored_rejected_replayed"] for r in log] == [False, True]
    assert log[-1]["failing"] == [] and all(s.ok for s in check_seams(out))
