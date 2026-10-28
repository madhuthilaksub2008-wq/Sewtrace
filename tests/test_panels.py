import json
from pathlib import Path

import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings, get_config
from sewtrace.mock import synth
from sewtrace.mock.fixtures import truth_spec
from sewtrace.reason import panels
from sewtrace.reason.panels import (
    PanelInference, apply_answer, build_prompt, draft_spec, estimate_tokens, infer_panels, measured_changes,
    validate_spec,
)
from sewtrace.reason.reshape import neck_depth_mm
from sewtrace.types import Edge, Outline, PhotoSet
from sewtrace.vision.measure import measure_tee

FIELDS = {"front_neck_depth_mm", "back_neck_depth_mm", "sleeve_cap", "grain_line"}


@pytest.fixture(scope="session")
def tees(tmp_path_factory) -> list[Path]:
    out = tmp_path_factory.mktemp("raw")
    dirs = []
    for i in range(2):
        synth.make_tee(synth.TEES[i], out / synth.TEES[i].name, seed=i)
        dirs.append(out / synth.TEES[i].name)
    return dirs


@pytest.fixture
def client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path))


def measured(tee_dir: Path) -> list:
    """What the vision chain hands over: half outline + edges per photo."""
    photos = json.loads((tee_dir / "truth.json").read_text())["photos"]
    args = []
    for view in ("front", "back", "sleeve"):
        args.append(Outline(points=[tuple(p) for p in photos[view]["half_outline_mm"]]))
        args.append([Edge(name=k, start=v[0], end=v[1]) for k, v in photos[view]["edges"].items()])
    return args


def photo_set(tee_dir: Path) -> PhotoSet:
    return PhotoSet(front=str(tee_dir / "front.jpg"), back=str(tee_dir / "back.jpg"),
                    sleeve=str(tee_dir / "sleeve.jpg"))


def meas(tee_dir: Path) -> dict:
    return json.loads((tee_dir / "truth.json").read_text())["measurements"]


def test_draft_matches_truth(tees):
    """REQ-GEO-03: draft matches truth."""
    for tee in tees:
        draft, truth = draft_spec(*measured(tee)), truth_spec(tee)
        assert [p.name for p in draft.panels] == ["front", "back", "sleeve", "neckband"]
        for name in ("front", "back", "sleeve"):
            assert draft.panel(name).outline_mm == truth.panel(name).outline_mm
            assert draft.panel(name).cut == truth.panel(name).cut
        nb, tnb = draft.panel("neckband"), truth.panel("neckband")
        assert np.allclose(nb.outline_mm, tnb.outline_mm)
        assert nb.seam_allowance_mm == 6.0 and draft.panel("front").seam_allowance_mm == 10.0
        assert all(s.ok and s.len_a_mm > 0 for s in draft.seams)
        assert validate_spec(draft, draft) == []


def test_notches_sit_on_their_edges(tees):
    """REQ-OUT-04: notches sit on their edges."""
    draft = draft_spec(*measured(tees[0]))
    assert len(draft.panel("front").notches) == 1
    assert len(draft.panel("back").notches) == 2
    assert draft.panel("sleeve").notches[0] == pytest.approx((0.0, 0.0))  # shoulder point
    for name, edge in (("front", "armhole"), ("back", "armhole"), ("sleeve", "cap")):
        pts = draft.panel(name).edge_points(edge)
        for n in draft.panel(name).notches:
            assert np.min(np.linalg.norm(pts - n, axis=1)) < 3.0


def _replace(spec, panel):
    return spec.model_copy(update={"panels": [panel if p.name == panel.name else p for p in spec.panels]})


def _shift_points(panel, edge_name, dx):
    """Move every point of an edge sideways (an edit of a measured edge)."""
    pts = [list(p) for p in panel.outline_mm]
    e = next(x for x in panel.edges if x.name == edge_name)
    for i in range(e.start, e.end + 1):
        pts[i][0] += dx
    return panel.model_copy(update={"outline_mm": [tuple(p) for p in pts]})


def test_req_rsn_04_prompt_lists_only_the_four_fields(tees):
    """REQ-RSN-04: the prompt lists front/back neck depth, sleeve cap and grain line as model output."""
    import re
    draft = draft_spec(*measured(tees[0]))
    prompt = build_prompt(draft, meas(tees[0]))
    listed = set(re.findall(r"^\d\. (\w+):", prompt, flags=re.M))
    assert listed == FIELDS == set(PanelInference.model_fields)
    assert "ONLY JSON" in prompt
    assert not re.search(r"\[\[", prompt)  # no edge polylines for the model to rewrite
    tokens = estimate_tokens(prompt)
    print(f"prompt ~{tokens} tokens, {len(prompt)} chars")
    assert tokens < 600


def test_req_rsn_02_schema_is_the_pydantic_model_and_is_sent(tees, client, monkeypatch):
    """REQ-RSN-02: PanelInference.model_json_schema() goes to MetaClient.reason as structured output."""
    schema = PanelInference.model_json_schema()
    assert set(schema["properties"]) == FIELDS and schema["additionalProperties"] is False
    tee = tees[0]
    draft = draft_spec(*measured(tee))
    sent = []
    real = client.reason
    monkeypatch.setattr(client, "reason", lambda *a, **k: (sent.append(k.get("schema")), real(*a, **k))[1])
    infer_panels(client, photo_set(tee), draft, meas(tee))
    assert sent == [schema]
    # The schema is part of the cache key: a different schema is a different request.
    cached = list(client.cache_dir.glob("*.json"))
    assert len(cached) == 1
    client.reason(build_prompt(draft, meas(tee)), [photo_set(tee).front, photo_set(tee).back, photo_set(tee).sleeve],
                  mock=lambda: panels._mock_answer(draft), schema={"type": "object"})
    assert len(list(client.cache_dir.glob("*.json"))) == 2


def test_req_rsn_02_extra_fields_are_rejected(tees):
    """REQ-RSN-02/05: the pydantic model forbids any field beyond the four."""
    draft = draft_spec(*measured(tees[0]))
    answer = panels._mock_answer(draft)
    assert apply_answer(answer, draft)[1] == []
    for extra in ({"side_shift_mm": 2.0}, {"panels": []}, {"notes": "hi"}):
        spec, problems = apply_answer({**answer, **extra}, draft)
        assert spec is None and "schema" in problems[0], extra
    nested = {**answer, "sleeve_cap": {**answer["sleeve_cap"], "extra": 1}}
    assert apply_answer(nested, draft)[0] is None
    missing = {k: v for k, v in answer.items() if k != "grain_line"}
    assert apply_answer(missing, draft)[0] is None
    assert apply_answer({**answer, "grain_line": "diagonal"}, draft)[0] is None


def test_req_rsn_04_out_of_range_values_are_rejected(tees):
    draft = draft_spec(*measured(tees[0]))
    answer = panels._mock_answer(draft)
    cfg = get_config().inference
    spec, problems = apply_answer({**answer, "back_neck_depth_mm": cfg.back_neck_depth_max_mm + 1}, draft)
    assert spec is None and "back_neck_depth_mm" in problems[0]
    spec, problems = apply_answer({**answer, "sleeve_cap": {"crown_run_mm": 1e4, "underarm_run_mm": 1.0}}, draft)
    assert spec is None and "cap width" in problems[0]


def test_req_rsn_04_05_valid_answer_changes_only_inferred_quantities(tees):
    """A valid answer moves neck depths, cap shape and grain. Measured edges move 0.5 mm at most."""
    cfg = get_config().inference
    for tee in tees:
        draft = draft_spec(*measured(tee))
        answer = panels._mock_answer(draft)
        answer["front_neck_depth_mm"] += 12.0
        answer["back_neck_depth_mm"] -= 4.0
        answer["sleeve_cap"]["crown_run_mm"] *= 0.5
        answer["grain_line"] = "horizontal"
        spec, problems = apply_answer(answer, draft)
        assert problems == [] and measured_changes(draft, spec) == []
        assert neck_depth_mm(spec.panel("front")) == pytest.approx(answer["front_neck_depth_mm"], abs=1e-6)
        assert neck_depth_mm(spec.panel("back")) == pytest.approx(answer["back_neck_depth_mm"], abs=1e-6)
        assert all(spec.panel(n).grain == "horizontal" for n in ("front", "back", "sleeve"))
        for name in ("front", "back"):  # neck width bound by the photo
            assert spec.panel(name).edge_points("neck")[-1] == pytest.approx(draft.panel(name).edge_points("neck")[-1])
        for name, edges in panels.MEASURED_EDGES.items():
            for edge in edges:
                a, b = draft.panel(name).edge_points(edge), spec.panel(name).edge_points(edge)
                assert np.allclose(a, b, atol=cfg.measured_tol_mm), (name, edge)
        assert not np.allclose(draft.panel("sleeve").edge_points("cap")[5], spec.panel("sleeve").edge_points("cap")[5])
        f0, f1 = draft.panel("front"), spec.panel("front")
        before = measure_tee(f0.outline(), f0.edges, draft.panel("back").outline(), draft.panel("back").edges,
                             draft.panel("sleeve").outline(), draft.panel("sleeve").edges)
        after = measure_tee(f1.outline(), f1.edges, spec.panel("back").outline(), spec.panel("back").edges,
                            spec.panel("sleeve").outline(), spec.panel("sleeve").edges)
        assert after == pytest.approx(before, abs=cfg.measured_tol_mm)


def test_req_rsn_05_measured_edge_moved_is_caught(tees):
    """REQ-RSN-05: 2 mm on the side seam fails; 0.3 mm passes; so do neck width and cap end points."""
    draft = draft_spec(*measured(tees[0]))
    side = measured_changes(draft, _replace(draft, _shift_points(draft.panel("back"), "side", 2.0)))
    assert any("back.side" in p for p in side)
    assert measured_changes(draft, _replace(draft, _shift_points(draft.panel("back"), "side", 0.3))) == []
    front = draft.panel("front")
    wide = [(x * 1.05, y) for x, y in front.outline_mm]
    assert measured_changes(draft, _replace(draft, front.model_copy(update={"outline_mm": wide})))
    sleeve = draft.panel("sleeve")
    cap = next(e for e in sleeve.edges if e.name == "cap")
    pts = list(sleeve.outline_mm)
    pts[cap.end] = (pts[cap.end][0] + 3.0, pts[cap.end][1])
    assert any("cap end points" in p for p in measured_changes(
        draft, _replace(draft, sleeve.model_copy(update={"outline_mm": pts}))))


def test_req_rsn_03_05_side_seam_moved_2mm_is_rejected_retried_then_fallback(tees, client, monkeypatch):
    """A response whose applied spec moves the side seam 2 mm is rejected, retried once, then the draft is used."""
    tee = tees[0]
    draft = draft_spec(*measured(tee))
    real_apply = panels.apply_inference
    monkeypatch.setattr(panels, "apply_inference", lambda d, inf: _replace(
        real_apply(d, inf), _shift_points(real_apply(d, inf).panel("back"), "side", 2.0)))
    history: list[dict] = []
    spec = infer_panels(client, photo_set(tee), draft, meas(tee), history=history)
    assert [h["result"] for h in history] == ["retry", "fallback"]
    assert any("back.side" in p for p in history[0]["problems"])
    assert client.calls == 2  # exactly one retry
    assert spec == panels.fallback_spec(draft)


def test_infer_panels_mock_valid_and_cached(tees, client):
    """REQ-RSN-02: infer panels mock valid and cached."""
    tee = tees[1]
    draft = draft_spec(*measured(tee))
    history: list[dict] = []
    spec = infer_panels(client, photo_set(tee), draft, meas(tee), history=history)
    assert validate_spec(spec, draft) == []
    assert all(s.ok for s in spec.seams)
    assert history[-1]["result"] == "accepted" and client.calls == 1
    again: list[dict] = []
    infer_panels(client, photo_set(tee), draft, meas(tee), history=again)
    assert client.calls == 1 and again[0]["new_call"] is False


def test_req_rsn_03_retries_then_falls_back(tees, client, monkeypatch):
    tee = tees[0]
    draft = draft_spec(*measured(tee))
    answers = iter([{"panels": []}, panels._mock_answer(draft)])
    monkeypatch.setattr(panels, "_mock_answer", lambda d: next(answers))
    history: list[dict] = []
    spec = infer_panels(client, photo_set(tee), draft, meas(tee), history=history)
    assert [h["result"] for h in history] == ["retry", "accepted"]
    assert validate_spec(spec, draft) == []

    monkeypatch.setattr(panels, "_mock_answer", lambda d: {"notes": "garbage"})
    history = []
    spec = infer_panels(client, photo_set(tee), draft, {**meas(tee), "chest_width": 1.0}, history=history)
    assert [h["result"] for h in history] == ["retry", "fallback"]
    assert spec == panels.fallback_spec(draft)


def test_validate_catches_self_intersection_and_missing_panel(tees):
    """REQ-RSN-02: validate catches self intersection and missing panel."""
    draft = draft_spec(*measured(tees[0]))
    front = draft.panel("front")
    pts = list(front.outline_mm)
    i, j = front.edges[1].start, front.edges[3].start  # swap shoulder start with side start
    pts[i], pts[j] = pts[j], pts[i]
    rest = [p for p in draft.panels if p.name not in ("front", "neckband")]
    bad = draft.model_copy(update={"panels": [front.model_copy(update={"outline_mm": pts}), *rest]})
    problems = validate_spec(bad, draft)
    assert "front: outline self-intersects" in problems
    assert "missing panel neckband" in problems


def test_validate_catches_moved_sleeve_and_bad_edges(tees):
    """REQ-RSN-05: validate catches moved sleeve and bad edges."""
    draft = draft_spec(*measured(tees[0]))
    sleeve = draft.panel("sleeve")
    moved = sleeve.model_copy(update={
        "outline_mm": [(x * 1.3, y) for x, y in sleeve.outline_mm],
        "edges": [e for e in sleeve.edges if e.name != "hem"] + [Edge(name="hem", start=0, end=9999)],
    })
    problems = validate_spec(_replace(draft, moved), draft)
    assert any("index out of range" in p for p in problems)
    scaled = sleeve.model_copy(update={"outline_mm": [(x * 1.3, y) for x, y in sleeve.outline_mm]})
    assert any("moved" in p for p in validate_spec(_replace(draft, scaled), draft))


class _FailingClient:
    """Stand-in client whose reason() raises the queued errors, then answers like the mock."""

    def __init__(self, client, *errors):
        self.client, self.errors, self.reason_calls = client, list(errors), 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def reason(self, *a, **k):
        self.reason_calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.client.reason(*a, **k)


def test_req_rsn_06_failed_model_call_is_a_rejected_attempt_then_retry(tees, client):
    """REQ-RSN-03, REQ-RSN-06: a truncated answer (ApiError) is retried, then the draft is the fallback."""
    from sewtrace.api.client import ApiError
    tee = tees[0]
    draft = draft_spec(*measured(tee))
    flaky = _FailingClient(client, ApiError("cut off at max_tokens"))
    history: list = []
    spec = infer_panels(flaky, photo_set(tee), draft, meas(tee), history=history)
    assert flaky.reason_calls == 2 and history[0]["result"] == "retry" and history[1]["result"] == "accepted"
    assert "cut off" in history[0]["problems"][0] and spec != draft

    dead = _FailingClient(client, ApiError("cut off"), ApiError("not JSON"))
    history = []
    assert infer_panels(dead, photo_set(tee), draft, meas(tee), history=history) == panels.fallback_spec(draft)
    assert [h["result"] for h in history] == ["retry", "fallback"]


def test_req_rsn_03_budget_cache_and_policy_errors_propagate(tees, client):
    """REQ-API-04, REQ-API-08: stop errors are not rejected attempts."""
    from sewtrace.api.client import BudgetExceeded, CacheMiss, PolicyViolation
    tee = tees[0]
    draft = draft_spec(*measured(tee))
    for err in (BudgetExceeded("over"), CacheMiss("miss"), PolicyViolation("real photo")):
        flaky = _FailingClient(client, err)
        with pytest.raises(type(err)):
            infer_panels(flaky, photo_set(tee), draft, meas(tee))
        assert flaky.reason_calls == 1


# ---- front neck depth (REQ-RSN-03, REQ-RSN-04, SPEC 10.1) --------------------------------

def measured_as_photographed(tee_dir: Path) -> list:
    """Like measured(), but the front outline is the one the mask shows: the BACK neckline."""
    args = measured(tee_dir)
    photos = json.loads((tee_dir / "truth.json").read_text())["photos"]
    args[0] = Outline(points=[tuple(p) for p in photos["front"]["visible_half_outline_mm"]])
    return args


def _params(tee_dir: Path) -> dict:
    return json.loads((tee_dir / "truth.json").read_text())["params"]


def test_req_rsn_04_photographed_draft_has_front_depth_equal_to_back_depth(tees):
    """The premise: from the photo the draft front neck is as shallow as the back neck."""
    draft = draft_spec(*measured_as_photographed(tees[0]))
    assert neck_depth_mm(draft.panel("front")) == pytest.approx(neck_depth_mm(draft.panel("back")), abs=0.1)


def test_req_rsn_03_04_mock_front_neck_depth_is_near_truth(tees, client):
    """SPEC 10.1: on a synthetic tee the mock model gives the true front depth (within 3 mm)."""
    for tee in tees:
        draft = draft_spec(*measured_as_photographed(tee))
        history: list[dict] = []
        spec = infer_panels(client, photo_set(tee), draft, meas(tee), history=history)
        assert history[-1]["result"] == "accepted"
        assert neck_depth_mm(spec.panel("front")) == pytest.approx(_params(tee)["front_neck_depth"], abs=3.0)
        assert neck_depth_mm(spec.panel("back")) == pytest.approx(_params(tee)["back_neck_depth"], abs=1.0)


def test_req_rsn_03_mock_does_not_read_truth_of_a_non_synthetic_photo(tees, client, tmp_path):
    """The truth is only for photos marked "synthetic": true. Otherwise the mock gives the default depth."""
    import shutil
    copy = tmp_path / "real_looking"
    shutil.copytree(tees[0], copy)
    truth = json.loads((copy / "truth.json").read_text())
    (copy / "truth.json").write_text(json.dumps({**truth, "synthetic": False}))
    draft = draft_spec(*measured_as_photographed(copy))
    spec = infer_panels(client, photo_set(copy), draft, meas(copy))
    expected = panels.default_front_depth_mm(neck_depth_mm(draft.panel("back")))
    assert neck_depth_mm(spec.panel("front")) == pytest.approx(expected, abs=1e-6)
    assert abs(expected - truth["params"]["front_neck_depth"]) > 1.0  # it did not copy the truth


def test_req_rsn_03_fallback_front_neck_is_deeper_than_back(tees, client, monkeypatch):
    """The fallback is not the draft: front depth = back depth + config offset, measured edges unchanged."""
    cfg = get_config().inference
    tee = tees[0]
    draft = draft_spec(*measured_as_photographed(tee))
    monkeypatch.setattr(panels, "_mock_answer", lambda d: {"notes": "garbage"})
    history: list[dict] = []
    spec = infer_panels(client, photo_set(tee), draft, meas(tee), history=history)
    assert [h["result"] for h in history] == ["retry", "fallback"]
    front, back = neck_depth_mm(spec.panel("front")), neck_depth_mm(spec.panel("back"))
    assert front > back + cfg.front_minus_back_min_mm
    assert front == pytest.approx(back + cfg.fallback_front_minus_back_mm, abs=1e-6)
    assert front > neck_depth_mm(draft.panel("front")) + 10.0
    assert validate_spec(spec, draft) == [] and all(s.ok for s in spec.seams)
    assert spec == panels.fallback_spec(draft)


def test_req_rsn_04_front_not_deeper_than_back_is_rejected(tees):
    """An answer with front depth <= back depth (or less than front_minus_back_min_mm deeper) is rejected."""
    cfg = get_config().inference
    draft = draft_spec(*measured(tees[0]))
    answer = panels._mock_answer(draft)
    assert apply_answer(answer, draft)[1] == []
    for front in (answer["back_neck_depth_mm"] + 5.0, answer["back_neck_depth_mm"],
                  answer["back_neck_depth_mm"] + cfg.front_minus_back_min_mm - 0.1):
        spec, problems = apply_answer({**answer, "front_neck_depth_mm": front}, draft)
        assert spec is None and "deeper than back_neck_depth_mm" in problems[0], front
    exact = answer["back_neck_depth_mm"] + cfg.front_minus_back_min_mm
    assert apply_answer({**answer, "front_neck_depth_mm": exact}, draft)[0] is not None


def test_req_rsn_04_prompt_gives_no_draft_value_for_the_front_neck(tees):
    """The prompt must not anchor the model on the back neckline depth for the front neck."""
    draft = draft_spec(*measured_as_photographed(tees[0]))
    front_line = next(ln for ln in build_prompt(draft, meas(tees[0])).splitlines() if ln.startswith("1. "))
    assert "Draft" not in front_line
    assert f"{neck_depth_mm(draft.panel('front')):.1f}" not in front_line
    assert f"{get_config().inference.front_minus_back_min_mm:g} mm deeper" in front_line


def _real_client_with_400(tmp_path):
    from sewtrace.api.client import HttpError
    from test_api_client import FakeTransport
    transport = FakeTransport(HttpError(400, "unknown parameter reasoning_effort"))
    settings = Settings(api_key="k", mock=False, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.csv")
    return MetaClient(settings, transport=transport, sleep=lambda s: None), transport


def test_req_rsn_03_http_400_is_not_a_bad_answer(tees, tmp_path):
    """REQ-RSN-03, REQ-API-06: a mocked 400 stops the run (ApiRequestRejected), with one call and no fallback."""
    from sewtrace.api.client import ApiRequestRejected
    client, transport = _real_client_with_400(tmp_path)
    draft = draft_spec(*measured_as_photographed(tees[0]))
    with pytest.raises(ApiRequestRejected, match="400"):
        infer_panels(client, photo_set(tees[0]), draft, meas(tees[0]))
    assert len(transport.calls) == 1


def test_req_rsn_03_request_rejected_propagates_from_the_loop(tees, client):
    from sewtrace.api.client import ApiRequestRejected
    draft = draft_spec(*measured(tees[0]))
    flaky = _FailingClient(client, ApiRequestRejected("HTTP 400"))
    with pytest.raises(ApiRequestRejected):
        infer_panels(flaky, photo_set(tees[0]), draft, meas(tees[0]))
    assert flaky.reason_calls == 1
