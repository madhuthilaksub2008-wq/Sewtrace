import json
from pathlib import Path

import numpy as np
import pytest

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings
from sewtrace.mock import synth
from sewtrace.mock.fixtures import truth_spec
from sewtrace.reason import panels
from sewtrace.reason.panels import (
    build_prompt, draft_spec, estimate_tokens, infer_panels, parse_answer, validate_spec,
)
from sewtrace.types import Edge, Outline, PhotoSet


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
    return MetaClient(Settings(api_key="", api_base_url="", mock=True, reasoning_model="spark", cache_dir=tmp_path))


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
    draft = draft_spec(*measured(tees[0]))
    assert len(draft.panel("front").notches) == 1
    assert len(draft.panel("back").notches) == 2
    assert draft.panel("sleeve").notches[0] == pytest.approx((0.0, 0.0))  # shoulder point
    for name, edge in (("front", "armhole"), ("back", "armhole"), ("sleeve", "cap")):
        pts = draft.panel(name).edge_points(edge)
        for n in draft.panel(name).notches:
            assert np.min(np.linalg.norm(pts - n, axis=1)) < 3.0


def test_prompt_is_compact_and_round_trips(tees):
    draft = draft_spec(*measured(tees[0]))
    prompt = build_prompt(draft, meas(tees[0]))
    tokens = estimate_tokens(prompt)
    print(f"prompt ~{tokens} tokens, {len(prompt)} chars")
    assert tokens < 3000
    assert "ONLY JSON" in prompt
    spec, problems = parse_answer(panels._mock_answer(draft), draft)
    assert problems == []
    assert validate_spec(spec, draft) == []
    assert all(s.ok for s in spec.seams)


def test_validate_catches_self_intersection_and_missing_panel(tees):
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


def test_validate_catches_drift_and_bad_edges(tees):
    draft = draft_spec(*measured(tees[0]))
    sleeve = draft.panel("sleeve")
    moved = sleeve.model_copy(update={
        "outline_mm": [(x * 1.3, y) for x, y in sleeve.outline_mm],
        "edges": [e for e in sleeve.edges if e.name != "hem"] + [Edge(name="hem", start=0, end=9999)],
    })
    bad = draft.model_copy(update={"panels": [moved if p.name == "sleeve" else p for p in draft.panels]})
    problems = validate_spec(bad, draft)
    assert any("drifts" in p for p in problems)
    assert any("index out of range" in p for p in problems)


def test_infer_panels_mock_valid_and_cached(tees, client):
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


def test_infer_panels_retries_then_falls_back(tees, client, monkeypatch):
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
