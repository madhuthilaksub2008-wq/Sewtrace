"""Co-pilot edits (sewtrace/reason/edits.py, `sewtrace edit`, /api/edit) and provenance labels.

No test makes a paid call or needs a key. The REAL-mode tests use a stub transport.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from perturb import set_edge_length
from pydantic import ValidationError
from test_web import CSRF, make_client, run_sample, wait_job

from sewtrace import cli, pipeline
from sewtrace.api.client import BudgetExceeded, MetaClient
from sewtrace.config import Settings, get_config, load_config
from sewtrace.mock import synth
from sewtrace.reason import panels, provenance
from sewtrace.reason.edits import (
    EDIT_SCHEMA,
    EditChange,
    EditProposal,
    EditRejected,
    PatternState,
    apply_proposal,
    mock_proposal,
    projected_measurements,
    proposal_rows,
    propose_edit,
    validate_proposal,
)
from sewtrace.reason.seams import SeamCheckFailed, edge_length
from sewtrace.types import MEASUREMENT_KEYS

CFG = get_config().edits
TOL_MM = 0.05  # a length edit is exact to float noise; the side seam is slightly slanted


def _client(tmp_path) -> MetaClient:
    return MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.csv"))


@pytest.fixture(scope="module")
def finished(tmp_path_factory) -> dict:
    """One finished mock run of the sample tee: its folder and state."""
    root = tmp_path_factory.mktemp("edits")
    synth.make_tee(synth.TEES[0], root / "tee1", seed=0)
    client = _client(root)
    result = pipeline.run(root / "tee1", root / "out", client)
    return {"root": root, "dir": root / "out", "tee": root / "tee1", "result": result,
            "state": pipeline.load_state(root / "out"), "client": client}


def _proposal(field: str, value: float) -> EditProposal:
    if field == "overall_scale_pct":
        change = EditChange(field=field, delta_mm=None, scale_pct=value)
    else:
        change = EditChange(field=field, delta_mm=value, scale_pct=None)
    return EditProposal(changes=[change], explanation="test")


# ---- the schema and the bounds ------------------------------------------------------------

def test_req_rsn_02_edit_proposal_schema_is_strict():
    """REQ-RSN-02: the model answer is checked against a strict pydantic schema."""
    ok = {"changes": [{"field": "body_length", "delta_mm": 20, "scale_pct": None}], "explanation": "longer"}
    assert EditProposal.model_validate(ok).changes[0].delta_mm == 20
    for bad in (
        {**ok, "extra": 1},                                                                    # no other field
        {"changes": [{"field": "neck_depth", "delta_mm": 5, "scale_pct": None}], "explanation": ""},  # unknown field
        {"changes": [{"field": "body_length", "delta_mm": 5}], "explanation": ""},              # both keys needed
        {"changes": [{"field": "body_length", "delta_mm": None, "scale_pct": None}], "explanation": ""},
        {"changes": [{"field": "body_length", "delta_mm": 5, "scale_pct": 105}], "explanation": ""},
        {"changes": [{"field": "overall_scale_pct", "delta_mm": 5, "scale_pct": None}], "explanation": ""},
        {"changes": [{"field": "overall_scale_pct", "delta_mm": None, "scale_pct": None}], "explanation": ""},
        {"changes": []},                                                                       # explanation needed
    ):
        with pytest.raises(ValidationError):
            EditProposal.model_validate(bad)
    change = EDIT_SCHEMA["$defs"]["EditChange"]
    assert set(change["required"]) == {"field", "delta_mm", "scale_pct"}
    assert change["properties"]["field"]["enum"] == ["body_length", "chest_width", "sleeve_length", "overall_scale_pct"]


@pytest.mark.parametrize("field,value", [
    ("body_length", CFG.max_delta_mm + 0.1), ("chest_width", -CFG.max_delta_mm - 5), ("sleeve_length", 0.0),
    ("overall_scale_pct", CFG.scale_min_pct - 0.1), ("overall_scale_pct", CFG.scale_max_pct + 0.1),
])
def test_req_rsn_04_proposal_outside_the_config_bounds_is_rejected(field, value):
    with pytest.raises(EditRejected, match="smaller|between"):
        validate_proposal(_proposal(field, value), CFG)


def test_req_rsn_04_bounds_edges_empty_and_repeated_fields():
    validate_proposal(_proposal("body_length", CFG.max_delta_mm), CFG)
    validate_proposal(_proposal("overall_scale_pct", CFG.scale_max_pct), CFG)
    validate_proposal(_proposal("overall_scale_pct", CFG.scale_min_pct), CFG)
    with pytest.raises(EditRejected, match="Nothing to change. why not"):
        validate_proposal(EditProposal(changes=[], explanation="why not"), CFG)
    twice = EditProposal(changes=_proposal("body_length", 5).changes * 2, explanation="")
    with pytest.raises(EditRejected, match="twice"):
        validate_proposal(twice, CFG)


# ---- the mock parser ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("make it 2 cm longer", {"body_length": 20.0}),
    ("sleeves 1 cm shorter", {"sleeve_length": -10.0}),
    ("a bit wider at the chest", {"chest_width": CFG.mock_a_bit_mm}),
    ("size up 5%", {"overall_scale_pct": 105.0}),
    ("Size down 3 percent", {"overall_scale_pct": 97.0}),
    ("size up", {"overall_scale_pct": 100.0 + CFG.mock_size_step_pct}),
    ("1 inch longer", {"body_length": 25.4}),
    ("make it 15 mm shorter", {"body_length": -15.0}),
    ("make it 2 longer", {"body_length": 20.0}),                                 # no unit means cm
    ("narrower chest by 1.5 cm", {"chest_width": -15.0}),
    ("3 cm shorter and sleeves 1 cm longer", {"body_length": -30.0, "sleeve_length": 10.0}),
    ("make it blue", {}),
])
def test_req_rsn_02_mock_parser_reads_plain_requests(text, expected):
    proposal = EditProposal.model_validate(mock_proposal(text, CFG))
    got = {c.field: (c.scale_pct if c.field == "overall_scale_pct" else c.delta_mm) for c in proposal.changes}
    assert got == pytest.approx(expected)
    assert proposal.explanation


# ---- apply: the geometry ------------------------------------------------------------------

def _no_model_call(client: MetaClient, monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(client, "reason", lambda *a, **k: calls.append("reason"))
    return calls


def test_req_sew_04_body_length_moves_the_hem_of_front_and_back(finished, monkeypatch):
    state, client = finished["state"], finished["client"]
    calls = _no_model_call(client, monkeypatch)
    out = apply_proposal(client, state, _proposal("body_length", 20.0)).state
    assert calls == []                                   # apply makes no model call
    for name in ("front", "back"):
        before, after = state.spec.panel(name), out.spec.panel(name)
        assert edge_length(after, "fold") - edge_length(before, "fold") == pytest.approx(20.0, abs=TOL_MM)
        assert edge_length(after, "side") - edge_length(before, "side") == pytest.approx(20.0, abs=TOL_MM)
        assert edge_length(after, "hem") == pytest.approx(edge_length(before, "hem"), abs=TOL_MM)
        assert after.edge_points("hem")[:, 1].mean() - before.edge_points("hem")[:, 1].mean() == pytest.approx(20.0, abs=TOL_MM)
        assert edge_length(after, "armhole") == pytest.approx(edge_length(before, "armhole"), abs=TOL_MM)
    assert edge_length(out.spec.panel("sleeve"), "fold") == pytest.approx(edge_length(state.spec.panel("sleeve"), "fold"))
    assert all(s.ok for s in out.spec.seams)
    assert out.measurements["body_length"] == pytest.approx(state.measurements["body_length"] + 20.0)
    assert panels._measure(out.spec)["body_length"] - panels._measure(state.spec)["body_length"] == pytest.approx(20.0, abs=0.1)


def test_req_sew_04_body_length_shorter_keeps_pairs_equal(finished):
    state, client = finished["state"], finished["client"]
    out = apply_proposal(client, state, _proposal("body_length", -35.0)).state
    side = {n: edge_length(out.spec.panel(n), "side") for n in ("front", "back")}
    before = {n: edge_length(state.spec.panel(n), "side") for n in ("front", "back")}
    assert side["front"] - side["back"] == pytest.approx(before["front"] - before["back"], abs=TOL_MM)
    assert side["front"] - before["front"] == pytest.approx(-35.0, abs=TOL_MM)
    assert all(s.ok for s in out.spec.seams)


def test_req_sew_04_chest_width_moves_the_side_seams_and_keeps_the_armhole_joined(finished, monkeypatch):
    state, client = finished["state"], finished["client"]
    calls = _no_model_call(client, monkeypatch)
    out = apply_proposal(client, state, _proposal("chest_width", 30.0)).state
    assert calls == []
    for name in ("front", "back"):
        before, after = state.spec.panel(name), out.spec.panel(name)
        assert after.edge_points("side")[:, 0].mean() - before.edge_points("side")[:, 0].mean() == pytest.approx(15.0, abs=TOL_MM)
        assert after.edge_points("hem")[0, 0] - before.edge_points("hem")[0, 0] == pytest.approx(15.0, abs=TOL_MM)
        assert edge_length(after, "side") == pytest.approx(edge_length(before, "side"), abs=TOL_MM)
        assert edge_length(after, "fold") == pytest.approx(edge_length(before, "fold"), abs=TOL_MM)
        assert after.edge_points("shoulder")[-1] == pytest.approx(before.edge_points("shoulder")[-1])  # shoulder point stays
        assert after.edge_points("armhole")[0] == pytest.approx(after.edge_points("shoulder")[-1])      # joined at the top
        assert after.edge_points("armhole")[-1] == pytest.approx(after.edge_points("side")[0])          # and at the underarm
    assert all(s.ok for s in out.spec.seams)
    assert panels._measure(out.spec)["chest_width"] - panels._measure(state.spec)["chest_width"] == pytest.approx(30.0, abs=0.1)
    assert out.measurements["chest_width"] == pytest.approx(state.measurements["chest_width"] + 30.0)


def test_req_sew_04_sleeve_length_moves_the_sleeve_hem(finished, monkeypatch):
    state, client = finished["state"], finished["client"]
    calls = _no_model_call(client, monkeypatch)
    out = apply_proposal(client, state, _proposal("sleeve_length", -12.0)).state
    assert calls == []
    before, after = state.spec.panel("sleeve"), out.spec.panel("sleeve")
    assert edge_length(after, "fold") - edge_length(before, "fold") == pytest.approx(-12.0, abs=TOL_MM)
    assert after.edge_points("hem")[:, 1].mean() - before.edge_points("hem")[:, 1].mean() == pytest.approx(-12.0, abs=TOL_MM)
    assert edge_length(after, "underarm") < edge_length(before, "underarm")        # the underarm seam shortens too
    assert edge_length(after, "cap") == pytest.approx(edge_length(before, "cap"), abs=TOL_MM)
    assert all(s.ok for s in out.spec.seams)
    assert panels._measure(out.spec)["sleeve_length"] - panels._measure(state.spec)["sleeve_length"] == pytest.approx(-12.0, abs=0.1)


def test_req_sew_04_overall_scale_scales_every_panel_about_the_fold_origin(finished):
    state, client = finished["state"], finished["client"]
    out = apply_proposal(client, state, _proposal("overall_scale_pct", 105.0)).state
    for name in ("front", "back", "sleeve"):
        before, after = state.spec.panel(name), out.spec.panel(name)
        for edge in ("fold", "hem") + (("side", "armhole") if name != "sleeve" else ("cap", "underarm")):
            assert edge_length(after, edge) == pytest.approx(1.05 * edge_length(before, edge), rel=1e-6)
        assert min(x for x, _ in after.outline_mm) == pytest.approx(1.05 * min(x for x, _ in before.outline_mm))  # fold stays on x = 0
        assert after.seam_allowance_mm == before.seam_allowance_mm
    assert all(s.ok for s in out.spec.seams)
    for k in MEASUREMENT_KEYS:
        assert out.measurements[k] == pytest.approx(1.05 * state.measurements[k])


def test_req_sew_04_edit_with_several_fields_and_scale_projects_the_table(finished):
    state, client = finished["state"], finished["client"]
    proposal = EditProposal.model_validate(mock_proposal("3 cm shorter and sleeves 1 cm longer, size down 3%", CFG))
    rows = {r["field"]: r for r in proposal_rows(proposal, state.measurements)}
    assert rows["body_length"]["proposed"] == pytest.approx((state.measurements["body_length"] - 30.0) * 0.97)
    assert rows["overall_scale_pct"]["now"] == 100.0 and rows["overall_scale_pct"]["proposed"] == pytest.approx(97.0)
    result = apply_proposal(client, state, proposal)
    assert result.state.measurements == pytest.approx(projected_measurements(state.measurements, proposal))
    assert [e["edit"] for e in result.edits] == ["body_length", "sleeve_length", "overall_scale_pct"]
    assert all(s.ok for s in result.state.spec.seams)


def test_req_sew_04_edit_is_logged_in_the_seam_log_and_the_result(finished):
    state, client = finished["state"], finished["client"]
    result = apply_proposal(client, state, _proposal("body_length", 20.0))
    new_rows = result.state.seam_log[len(state.seam_log):]
    assert new_rows[0]["source"] == "edit" and new_rows[0]["iteration"] == len(state.seam_log)
    change = new_rows[0]["applied"][0]
    assert change["edit"] == "body_length" and change["unit"] == "mm"
    assert change["to"] - change["from"] == pytest.approx(20.0, abs=0.01)
    assert result.edits == new_rows[0]["applied"]
    assert result.changes == [{"key": "body_length", "from_mm": state.measurements["body_length"],
                               "to_mm": state.measurements["body_length"] + 20.0}]
    iterations = [r["iteration"] for r in result.state.seam_log]
    assert iterations == sorted(set(iterations))


def test_req_sew_04_a_shape_that_crosses_itself_is_refused(tmp_path):
    # A curved sleeve top crosses its shortened underarm; the default straight top would not (seam.cap_style).
    synth.make_tee(synth.TEES[0], tmp_path / "tee1", seed=0)
    client = _client(tmp_path)
    cfg = client.config
    client.config = cfg.model_copy(update={"seam": cfg.seam.model_copy(update={"cap_style": "curve"})})
    pipeline.run(tmp_path / "tee1", tmp_path / "out", client)
    with pytest.raises(EditRejected, match="sleeve panel"):
        apply_proposal(client, pipeline.load_state(tmp_path / "out"), _proposal("sleeve_length", -CFG.max_delta_mm))


def test_req_sew_04_edit_never_exports_a_failing_seam_pair(finished, tmp_path):
    """REQ-SEW-04: a pair beyond the code fix limit stays failing, so the edit raises and nothing is written."""
    state = finished["state"]
    back = state.spec.panel("back")
    broken = set_edge_length(back, "side", edge_length(back, "side") + 3 * get_config().seam.measured_pair_max_fix_mm)
    spec = state.spec.model_copy(update={"panels": [broken if p.name == "back" else p for p in state.spec.panels]})
    with pytest.raises(SeamCheckFailed):
        apply_proposal(finished["client"], PatternState(spec, state.measurements, state.seam_log, state.provenance),
                       _proposal("body_length", 10.0))


def test_req_sew_03_a_rebalance_after_a_chest_edit_keeps_the_edit_label():
    prov = provenance.build_provenance(
        [{"source": "edit", "applied": [{"edit": "chest_width"}]},
         {"source": "pair_fix", "applied": [{"panel": "front", "edge": "side"}]}], None,
        start=provenance.initial_provenance("model"))
    assert prov["chest_width"] == provenance.EDIT


# ---- REAL mode: every call goes through MetaClient.reason ----------------------------------

class _Transport:
    """Stub for the chat completions endpoint: queues one answer, counts the requests."""

    def __init__(self, answer: dict):
        self.answer, self.calls = answer, []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.answer)}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 700,
                          "prompt_tokens_details": {"cached_tokens": 0}}}


def _real_client(tmp_path, transport, **api) -> MetaClient:
    cfg = load_config()
    if api:
        cfg = cfg.model_copy(update={"api": cfg.api.model_copy(update=api)})
    settings = Settings(api_key="test-key-not-real", mock=False, cache_dir=tmp_path / "cache",
                        ledger_path=tmp_path / "ledger" / "ledger.csv", config=cfg)
    return MetaClient(settings, transport=transport, sleep=lambda s: None)


def test_req_api_01_one_reason_call_per_proposal_and_none_on_apply(finished, tmp_path, monkeypatch):
    """REQ-API-01..04: the proposal is one MetaClient.reason call (cache, ledger, budget); apply makes none."""
    state = finished["state"]
    transport = _Transport(mock_proposal("make it 2 cm longer", CFG))
    client = _real_client(tmp_path, transport)
    seen: list[str] = []
    real_reason = client.reason
    monkeypatch.setattr(client, "reason", lambda *a, **k: (seen.append("reason"), real_reason(*a, **k))[1])

    proposal = propose_edit(client, "make it 2 cm longer", state)
    assert seen == ["reason"] and len(transport.calls) == 1 and client.calls == 1
    body = transport.calls[0]
    assert body["model"] == "muse-spark-1.3" and body["reasoning_effort"] == load_config().reasoning.effort
    assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    user = body["messages"][-1]["content"]
    assert len(user) == 1 and user[0]["type"] == "text"                      # no image is sent
    assert "make it 2 cm longer" in user[0]["text"] and f"{state.measurements['body_length']:.1f}" in user[0]["text"]
    assert "body_length" in body["messages"][0]["content"]                   # the allowed fields, as the stable prefix
    with (tmp_path / "ledger" / "ledger.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and float(rows[0]["cost_usd"]) > 0

    result = apply_proposal(client, state, proposal)
    assert seen == ["reason"] and len(transport.calls) == 1 and client.calls == 1   # apply: zero calls
    assert result.state.measurements["body_length"] == pytest.approx(state.measurements["body_length"] + 20.0)

    again = propose_edit(client, "make it 2 cm longer", state)                # same request: served from the cache
    assert again == proposal and len(transport.calls) == 1 and client.calls == 1


def test_req_api_04_the_budget_guard_stops_a_proposal_before_any_call(finished, tmp_path):
    transport = _Transport(mock_proposal("make it 2 cm longer", CFG))
    client = _real_client(tmp_path, transport, budget_limit_usd=0.01)
    with pytest.raises(BudgetExceeded):
        propose_edit(client, "make it 2 cm longer", finished["state"])
    assert transport.calls == []


def test_req_rsn_02_a_model_answer_outside_the_bounds_is_rejected_after_the_one_call(finished, tmp_path):
    transport = _Transport({"changes": [{"field": "body_length", "delta_mm": 500, "scale_pct": None}], "explanation": "x"})
    client = _real_client(tmp_path, transport)
    with pytest.raises(EditRejected, match="smaller change"):
        propose_edit(client, "make it a lot longer", finished["state"])
    assert len(transport.calls) == 1
    bad_shape = _Transport({"changes": "none"})
    with pytest.raises(EditRejected, match="schema"):
        propose_edit(_real_client(tmp_path / "other", bad_shape), "make it longer", finished["state"])


def test_req_rsn_02_request_text_is_checked_before_any_call(finished, tmp_path):
    transport = _Transport(mock_proposal("make it 2 cm longer", CFG))
    client = _real_client(tmp_path, transport)
    for text in ("", "   \n ", "x" * (CFG.max_text_chars + 1)):
        with pytest.raises(EditRejected):
            propose_edit(client, text, finished["state"])
    assert transport.calls == []


# ---- the command line -----------------------------------------------------------------------

def _digest(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("SEWTRACE_MOCK", "1")
    monkeypatch.setenv("SEWTRACE_OFFLINE", "0")
    monkeypatch.setenv("SEWTRACE_CACHE_DIR", str(tmp_path / "cli-cache"))


def _fresh_run(finished, tmp_path) -> Path:
    out = tmp_path / "out"
    pipeline.run(finished["tee"], out, _client(tmp_path))
    return out


def test_req_sew_04_cli_edit_without_yes_only_prints(finished, tmp_path, cli_env, capsys):
    out = _fresh_run(finished, tmp_path)
    before = _digest(out)
    cli.main(["edit", str(out), "make it 2 cm longer"])
    text = capsys.readouterr().out
    assert "body_length" in text and "now" in text and "proposed" in text and "No file was changed" in text
    assert _digest(out) == before


def test_req_sew_04_cli_edit_with_yes_rewrites_the_pattern_files(finished, tmp_path, cli_env, capsys):
    out = _fresh_run(finished, tmp_path)
    before = pipeline.load_state(out)
    cli.main(["edit", str(out), "make it 2 cm longer", "--yes"])
    text = capsys.readouterr().out
    assert "co-pilot edit" in text and "seams: all ok" in text
    after = pipeline.load_state(out)
    assert after.measurements["body_length"] == pytest.approx(before.measurements["body_length"] + 20.0)
    assert after.provenance["body_length"] == "co-pilot edit"
    assert [r["source"] for r in after.seam_log][-1] == "edit"
    stored = json.loads((out / "measurements.json").read_text())
    assert stored["body_length"] == pytest.approx(after.measurements["body_length"])
    for name in ("pattern.pdf", "pattern.svg", "pattern.dxf", "spec.json", "seam_log.json"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0
    cli.main(["edit", str(out), "sleeves 1 cm shorter", "--yes"])           # edits stack
    twice = pipeline.load_state(out)
    assert twice.measurements["sleeve_length"] == pytest.approx(before.measurements["sleeve_length"] - 10.0)
    assert twice.measurements["body_length"] == pytest.approx(after.measurements["body_length"])
    assert twice.provenance["body_length"] == twice.provenance["sleeve_length"] == "co-pilot edit"


@pytest.mark.parametrize("text,message", [("make it blue", "Nothing to change"), ("50 cm longer", "smaller change"),
                                           ("", "empty")])
def test_req_sew_04_cli_edit_refusal_changes_nothing_and_exits_nonzero(finished, tmp_path, cli_env, capsys, text, message):
    out = _fresh_run(finished, tmp_path)
    before = _digest(out)
    with pytest.raises(SystemExit):
        cli.main(["edit", str(out), text, "--yes"])
    assert message in capsys.readouterr().err
    assert _digest(out) == before


def test_req_cap_01_cli_edit_names_a_folder_that_is_not_a_run(tmp_path, cli_env, capsys):
    with pytest.raises(SystemExit):
        cli.main(["edit", str(tmp_path / "nothing"), "make it longer"])
    assert "not a finished run folder" in capsys.readouterr().err


def test_req_out_01_cli_edit_full_prints_whole_pieces(finished, tmp_path, cli_env):
    out = _fresh_run(finished, tmp_path)
    cli.main(["edit", str(out), "make it 2 cm longer", "--yes", "--full"])
    assert (out / "pattern.pdf").stat().st_size > 0


# ---- the web page -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("web-synth")
    synth.make_tee(synth.TEES[0], root / "tee1", seed=0)
    return root


@pytest.fixture
def web(tmp_path, sample_root):
    return make_client(tmp_path, sample_root)


def _post_json(client, url, body, **kw):
    return client.post(url, json=body, **kw)


def test_req_sew_04_web_edit_proposes_then_applies_on_the_sample_tee(web):
    run = run_sample(web)
    assert run["state"] == "done"
    rid, before = run["id"], {m["key"]: m["value_mm"] for m in run["result"]["measurements"]}
    res = _post_json(web, "/api/edit", {"run": rid, "text": "make it 2 cm longer"})
    assert res.status_code == 202, res.text
    proposal_job = wait_job(web, res.json()["id"])
    assert proposal_job["state"] == "done", proposal_job
    proposal = proposal_job["result"]
    assert proposal["model_calls"] == 1                                      # mock mode counts the one call
    (row,) = proposal["rows"]
    assert row["field"] == "body_length" and row["unit"] == "mm"
    assert row["now"] == pytest.approx(before["body_length"]) and row["proposed"] == pytest.approx(before["body_length"] + 20.0)
    assert web.get(f"/api/jobs/{rid}").json()["result"]["measurements"][1]["value_mm"] == pytest.approx(before["body_length"])  # not applied yet

    applied = _post_json(web, "/api/edit/apply", {"run": rid, "edit": proposal_job["id"]})
    assert applied.status_code == 202, applied.text
    job = wait_job(web, applied.json()["id"])
    assert job["state"] == "done", job
    result = job["result"]
    assert result["model_calls"] == 0                                        # apply makes no model call
    assert result["edits"] == [{"edit": "body_length", "from": pytest.approx(before["body_length"], abs=0.01),
                                "to": pytest.approx(before["body_length"] + 20.0, abs=0.01), "unit": "mm"}]
    now = {m["key"]: m for m in result["measurements"]}
    assert now["body_length"]["value_mm"] == pytest.approx(before["body_length"] + 20.0)
    assert now["body_length"]["provenance"] == "co-pilot edit" and now["chest_width"]["provenance"] == "from photo"
    assert result["edit_changes"][0]["key"] == "body_length"
    assert all(s["ok"] for s in result["seams"]) and result["seam_log"][-1]["source"] == "edit"
    for name in ("preview", "pdf", "svg", "dxf", "spec", "mask_front"):
        got = web.get(f"/api/jobs/{job['id']}/files/{name}")
        assert got.status_code == 200 and len(got.content) > 0, name
    assert web.get("/api/status").json()["latest"]["run"] == job["id"]      # a refresh shows the edited pattern
    # The edited pattern can be edited again, and the old page can not.
    again = _post_json(web, "/api/edit", {"run": job["id"], "text": "sleeves 1 cm shorter"})
    assert again.status_code == 202
    stale = _post_json(web, "/api/edit", {"run": rid, "text": "sleeves 1 cm shorter"})
    assert stale.status_code == 400 and "not the latest pattern" in stale.json()["error"]["message"]


def test_req_sew_04_web_edit_rejects_bad_requests_before_any_job(web):
    rid = run_sample(web)["id"]
    assert _post_json(web, "/api/edit", {"run": rid, "text": "   "}).status_code == 400
    assert _post_json(web, "/api/edit", {"run": "nope", "text": "make it longer"}).status_code == 404
    assert _post_json(web, "/api/edit", {"run": rid, "text": 5}).status_code == 400
    assert web.post("/api/edit", content=b"[1]", headers={"Content-Type": "application/json"}).status_code == 400
    too_much = _post_json(web, "/api/edit", {"run": rid, "text": "50 cm longer"})
    job = wait_job(web, too_much.json()["id"])
    assert job["state"] == "error" and "smaller change" in job["error"]["message"]
    gone = _post_json(web, "/api/edit/apply", {"run": rid, "edit": job["id"]})   # a failed proposal can not be applied
    assert gone.status_code == 404
    assert _post_json(web, "/api/edit/apply", {"run": rid, "edit": "nope"}).status_code == 404


def test_req_sew_04_web_apply_after_a_new_run_is_refused(web):
    first = run_sample(web)["id"]
    proposal = wait_job(web, _post_json(web, "/api/edit", {"run": first, "text": "make it 2 cm longer"}).json()["id"])
    run_sample(web)                                                           # a newer pattern
    refused = _post_json(web, "/api/edit/apply", {"run": first, "edit": proposal["id"]})
    assert refused.status_code == 409 and "not the latest pattern" in refused.json()["error"]["message"]


def test_req_api_04_web_edit_keeps_csrf_header_and_basic_auth(tmp_path, sample_root):
    import base64
    secret = "pw-for-the-test"
    auth = {"Authorization": "Basic " + base64.b64encode(f"u:{secret}".encode()).decode()}
    client = make_client(tmp_path, sample_root, password=secret, csrf=False)
    for url in ("/api/edit", "/api/edit/apply"):
        assert client.post(url, json={}).status_code == 401                   # no password
        assert client.post(url, json={}, headers=auth).status_code == 403      # password but no X-Sewtrace header
        assert client.post(url, json={}, headers={**auth, **CSRF}).status_code in (400, 404)  # reaches the route


# ---- provenance ---------------------------------------------------------------------------------

def test_req_geo_04_provenance_of_a_plain_sample_run(finished):
    result = finished["result"]
    for k in MEASUREMENT_KEYS:
        assert result.provenance[k] == "from photo"
    assert result.inference == "model"
    assert [result.provenance[k] for k in ("front_neck_depth", "back_neck_depth")] == ["inferred by Muse"] * 2
    assert result.provenance["sleeve_cap"] == "drafted"                       # drawn straight by code (cap_style)
    stored = json.loads((finished["dir"] / "measurements.json").read_text())
    assert stored["provenance"] == result.provenance
    assert {k: stored[k] for k in MEASUREMENT_KEYS} == result.measurements     # the six keys are unchanged
    assert list(result.measurements) == list(MEASUREMENT_KEYS)
    rows = {r["key"]: r for r in pipeline.inferred_values(result.spec, result.provenance)}
    assert rows["front_neck_depth"]["provenance"] == "inferred by Muse" and rows["front_neck_depth"]["value_mm"] > 0


def test_req_sew_03_provenance_of_a_run_with_a_user_length(finished, tmp_path):
    result = pipeline.run(finished["tee"], tmp_path / "out", _client(tmp_path), overrides={"side": 240.0})
    assert result.provenance["chest_width"] == "your tape"
    assert result.provenance["shoulder_width"] == "from photo" and result.provenance["body_length"] == "from photo"
    (row,) = pipeline.pair_review(result.seam_log)
    assert row["edge"] == "side" and row["source"] == "user" and row["provenance"] == "your tape"
    assert json.loads((tmp_path / "out" / "measurements.json").read_text())["provenance"]["chest_width"] == "your tape"


def test_req_sew_03_provenance_of_an_averaged_pair(finished, tmp_path, monkeypatch):
    real = pipeline.adjust_loop

    def skewed(client, spec):
        back = spec.panel("back")
        longer = set_edge_length(back, "shoulder", edge_length(back, "shoulder") + 2.0)
        return real(client, spec.model_copy(update={"panels": [longer if p.name == "back" else p for p in spec.panels]}))

    monkeypatch.setattr(pipeline, "adjust_loop", skewed)
    result = pipeline.run(finished["tee"], tmp_path / "out", _client(tmp_path))
    assert result.provenance["shoulder_width"] == "averaged front/back" and result.provenance["chest_width"] == "from photo"
    (row,) = pipeline.pair_review(result.seam_log)
    assert row["provenance"] == "averaged front/back"


def test_req_rsn_03_provenance_when_the_fallback_neck_depth_was_used(finished, tmp_path, monkeypatch):
    monkeypatch.setattr(panels, "_mock_answer", lambda draft: {"front_neck_depth_mm": -1})   # fails the schema twice
    result = pipeline.run(finished["tee"], tmp_path / "out", _client(tmp_path))
    assert result.inference == "fallback"
    assert result.provenance["front_neck_depth"] == "drafted" and result.provenance["back_neck_depth"] == "from photo"
    assert result.provenance["sleeve_cap"] == "drafted"                       # drawn straight by code (cap_style)


def test_req_sew_02_provenance_of_the_sleeve_cap_follows_the_last_cap_change():
    base = provenance.initial_provenance("model")
    model_row = {"source": "model", "applied": [{"sleeve_cap": {"crown_run_mm": 1, "underarm_run_mm": 1}}]}
    flat_row = {"source": "fallback", "applied": [{"flat_cap": True}]}
    scaled_row = {"source": "fallback", "applied": [{"sleeve_cap": {"crown_run_mm": 1, "underarm_run_mm": 1}}]}
    skipped = {"source": "model", "applied": []}
    assert provenance.build_provenance([model_row], None, start=base)["sleeve_cap"] == "inferred by Muse"
    assert provenance.build_provenance([model_row, flat_row], None, start=base)["sleeve_cap"] == "drafted"
    assert provenance.build_provenance([scaled_row], None, start=base)["sleeve_cap"] == "drafted"
    assert provenance.build_provenance([skipped], None, start=provenance.initial_provenance("fallback"))["sleeve_cap"] == "from photo"
    assert "sleeve_cap" not in provenance.build_provenance([], "unknown")        # a run from before the labels


def test_req_geo_04_a_run_folder_from_before_the_labels_gets_them_from_its_seam_log(finished, tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    for name in ("spec.json", "seam_log.json"):
        (out / name).write_text((finished["dir"] / name).read_text())
    (out / "measurements.json").write_text(json.dumps(finished["result"].measurements))   # no provenance key
    state = pipeline.load_state(out)
    assert state.provenance["body_length"] == "from photo" and "front_neck_depth" not in state.provenance


def test_req_geo_04_web_run_result_has_labels_on_every_measurement_and_review_row(web):
    result = run_sample(web)["result"]
    assert all(m["provenance"] == "from photo" for m in result["measurements"])
    assert result["provenance"]["front_neck_depth"] == "inferred by Muse"
    assert {v["key"] for v in result["inferred"]} == {"front_neck_depth", "back_neck_depth", "sleeve_cap"}
    page = web.get("/").text
    assert "Co-pilot" in page and "createTextNode" in page and "innerHTML" not in page


def test_req_geo_04_a_scale_edit_labels_the_inferred_values_it_changed():
    row = {"source": "edit", "applied": [{"edit": "overall_scale_pct", "from": 100.0, "to": 105.0, "unit": "%"}]}
    prov = provenance.build_provenance([row], None, start=provenance.initial_provenance("model"))
    for key in (*MEASUREMENT_KEYS, *provenance.INFERRED_KEYS):
        assert prov[key] == "co-pilot edit", key
    old = provenance.build_provenance([row], "unknown")                      # a run from before the labels
    assert "sleeve_cap" not in old and old["body_length"] == "co-pilot edit"


def test_req_sew_03_an_edited_pattern_keeps_the_photos_for_a_rerun(web):
    run = run_sample(web)
    store = web.app.state.store
    first = store.get(run["id"]).dir
    for view in ("front", "back", "sleeve"):                                 # as a real upload leaves them
        (first / f"{view}.jpg").write_bytes(b"photo")
    proposal = wait_job(web, _post_json(web, "/api/edit", {"run": run["id"], "text": "make it 2 cm longer"}).json()["id"])
    job = wait_job(web, _post_json(web, "/api/edit/apply", {"run": run["id"], "edit": proposal["id"]}).json()["id"])
    assert job["state"] == "done", job
    edited = store.get(job["id"]).dir
    assert all((edited / f"{v}.jpg").read_bytes() == b"photo" for v in ("front", "back", "sleeve"))
    assert (edited / "out" / "cutplan.json").is_file() and job["result"]["cutplan"]["summary"]


def test_req_api_04_web_edit_of_an_expired_pattern_is_refused_before_the_call(web):
    rid = run_sample(web)["id"]
    store = web.app.state.store
    store.get(rid).created -= store.retention_s + 1                          # older than the retention time
    res = _post_json(web, "/api/edit", {"run": rid, "text": "make it 2 cm longer"})
    assert res.status_code == 404 and "Run the pattern again" in res.json()["error"]["message"]
