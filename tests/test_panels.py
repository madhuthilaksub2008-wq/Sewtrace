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
    assert spec == draft


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
    assert spec == draft


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
