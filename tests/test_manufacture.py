"""Manufacturing outputs: cut plan, cut path, files, bundle, web. Mock only, no key, no network."""

import csv
import dataclasses
import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_web import make_client, run_sample, sample_root  # noqa: F401 - fixture used by name

from sewtrace.api.client import MetaClient
from sewtrace.config import Settings, get_config
from sewtrace.mock import synth
from sewtrace.output import manufacture as mf
from sewtrace.pipeline import run

EXPECTED_FILES = {"cut-path.csv", "toolpath-illustrative.txt", "cutplan.svg", "cutplan.json", "bom.csv",
                  "instructions.md", "bundle.zip"}


@pytest.fixture(scope="module")
def tee_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mf")
    synth.make_tee(synth.TEES[0], tmp / "tee1", seed=0)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp / "cache"))
    result = run(tmp / "tee1", tmp / "out", client)
    return result, tmp / "out"


def test_req_out_03_cut_plan_has_no_overlap_and_respects_the_roll(tee_run):
    """REQ-OUT-03: pieces are cut with seam allowance, nested without overlap inside the roll width."""
    result, _ = tee_run
    cfg = get_config().manufacture
    plan = mf.plan_manufacture(result.spec).plan
    assert plan.overlap_area_mm2 == pytest.approx(0.0, abs=1e-6)
    assert plan.min_gap_mm >= cfg.gap_mm - 1e-6
    for pl in plan.placements:
        x0, y0, x1, y1 = pl.polygon.bounds
        assert x0 >= cfg.margin_mm - 1e-6 and x1 <= cfg.roll_width_mm - cfg.margin_mm + 1e-6
        assert y0 >= cfg.margin_mm - 1e-6 and y1 <= plan.used_length_mm - cfg.margin_mm + 1e-6
    assert 0 < plan.utilisation <= 1
    assert plan.waste_mm2 == pytest.approx(plan.fabric_area_mm2 - plan.piece_area_mm2)


def test_req_out_03_quantities_follow_cut_n(tee_run):
    """front 1, back 1, sleeve 2, neckband 1; every sleeve is a whole piece."""
    result, _ = tee_run
    ids = sorted(p.id for p in mf.pieces_to_cut(result.spec))
    assert ids == ["back", "front", "neckband", "sleeve_1", "sleeve_2"]


def test_grain_line_lies_along_the_roll_length(tee_run):
    """REQ-OUT-04: the grain line lies along the roll length. The neckband is drawn with a horizontal grain line, so it is turned 90 degrees.
    The neckband is drawn with a horizontal grain line, so it is turned 90 degrees; the rest are not."""
    result, _ = tee_run
    turns = {p.id: p.rotation_deg for p in mf.pieces_to_cut(result.spec)}
    assert turns["front"] == 0 and turns["sleeve_1"] == 0 and turns["neckband"] == 90


def test_piece_wider_than_the_roll_is_refused_with_a_clear_message(tee_run):
    """REQ-OUT-03: a cut piece wider than the usable roll width is refused, and the message says what to change."""
    result, _ = tee_run
    cfg = get_config().manufacture.model_copy(update={"roll_width_mm": 300.0})  # the front is about 348 mm wide
    with pytest.raises(mf.PieceTooWide) as err:
        mf.nest(mf.pieces_to_cut(result.spec), cfg)
    text = str(err.value)
    assert "roll_width_mm" in text and ("front" in text or "back" in text)


def test_a_narrow_roll_still_nests_on_more_shelves(tee_run):
    """REQ-OUT-03: a roll that holds one piece per shelf makes a longer cloth, not an overlap."""
    result, _ = tee_run
    wide = mf.plan_manufacture(result.spec).plan
    cfg = get_config().manufacture.model_copy(update={"roll_width_mm": 400.0 + 2 * 15})
    narrow = mf.nest([p for p in mf.pieces_to_cut(result.spec) if p.panel != "neckband"], cfg)
    assert narrow.used_length_mm > wide.used_length_mm
    assert narrow.overlap_area_mm2 == pytest.approx(0.0, abs=1e-6)


def test_cut_paths_are_closed_and_counted(tee_run):
    """REQ-OUT-03: one closed cut path per piece on the cut line; the cut length is the perimeter sum."""
    result, _ = tee_run
    made = mf.plan_manufacture(result.spec)
    assert len(made.path.paths) == 5
    for pp in made.path.paths:
        assert pp.points[0] == pp.points[-1] and len(pp.points) > 4
    perimeter = sum(pl.polygon.length for pl in made.plan.placements)
    assert made.path.cut_length_mm == pytest.approx(perimeter, rel=1e-6)
    cfg = made.cfg
    expect = (made.path.cut_length_mm / cfg.cut_speed_mm_s + made.path.travel_length_mm / cfg.travel_speed_mm_s
              + 5 * cfg.pierce_s)
    assert made.path.duration_s == pytest.approx(expect)


def test_cut_path_is_deterministic(tee_run):
    """REQ-OUT-05: the same pattern gives the same cut paths (no set or dict order)."""
    result, _ = tee_run
    a, b = mf.plan_manufacture(result.spec), mf.plan_manufacture(result.spec)
    assert [p.points for p in a.path.paths] == [p.points for p in b.path.paths]
    assert [p.piece for p in a.path.paths] == [p.piece for p in b.path.paths]


def test_mock_run_writes_the_manufacturing_files(tee_run):
    """REQ-OUT-05: a mock run writes the cut path CSV, the illustrative G-code and the cut plan SVG."""
    result, out = tee_run
    assert EXPECTED_FILES <= {p.name for p in out.iterdir()}
    for key in ("cut_path", "toolpath", "cutplan", "cutplan_json", "bom", "instructions", "bundle"):
        assert result.files[key].is_file() and result.files[key].stat().st_size > 0

    rows = list(csv.DictReader((out / "cut-path.csv").open()))
    assert list(rows[0]) == ["piece", "seq", "move", "x_mm", "y_mm"]
    assert {r["piece"] for r in rows} == {"front", "back", "sleeve_1", "sleeve_2", "neckband", "home"}
    assert {r["move"] for r in rows} == {"travel", "pierce", "cut"}
    assert sum(r["move"] == "pierce" for r in rows) == 5
    assert [int(r["seq"]) for r in rows] == list(range(1, len(rows) + 1))

    gcode = (out / "toolpath-illustrative.txt").read_text()
    assert gcode.splitlines()[0].startswith("; ILLUSTRATIVE ONLY") and "NOT FOR ANY MACHINE" in gcode.splitlines()[1]
    codes = [line.split()[0] for line in gcode.splitlines() if line and not line.startswith(";")]
    assert codes.count("M3") == 5 and codes.count("M5") == 5 and "G21" in codes

    svg = (out / "cutplan.svg").read_text()
    assert 'viewBox="0 0 1500' in svg and "mm" in svg and svg.count("<polygon") == 5


def test_bom_has_fabric_thread_neckband_and_no_notions(tee_run):
    """REQ-OUT-03: the bill of materials from the cut plan and the spec seams."""
    result, out = tee_run
    rows = {r["item"]: r for r in csv.DictReader((out / "bom.csv").open())}
    assert set(rows) == {"fabric", "thread", "neckband rib", "notions"}
    plan = mf.plan_manufacture(result.spec).plan
    assert float(rows["fabric"]["quantity"]) == pytest.approx(plan.used_length_mm / 1000, abs=5e-4)
    assert "1500" in rows["fabric"]["note"]
    seam_sum = 2 * sum(max(s.len_a_mm, s.len_b_mm) for s in result.spec.seams)
    assert float(rows["thread"]["quantity"]) == pytest.approx(seam_sum * 2.5 / 1000, abs=0.05)
    assert "rule of thumb" in rows["thread"]["note"]
    assert float(rows["neckband rib"]["quantity"]) > 0.2
    assert rows["notions"]["note"] == "none"


def test_instructions_list_pieces_and_real_seam_lengths_in_order(tee_run):
    """REQ-OUT-06: the assembly text has one step group per seam kind, with the real seam lengths."""
    result, out = tee_run
    text = (out / "instructions.md").read_text()
    assert "—" not in text and "–" not in text
    for piece in ("front", "back", "sleeve", "neckband"):
        assert f"| {piece} |" in text
    assert "| sleeve | 2 |" in text
    for seam in result.spec.seams:
        assert mf.seam_line(seam) in text
    shoulder = next(s for s in result.spec.seams if "shoulder" in s.a)
    assert f"front.shoulder~back.shoulder {shoulder.len_a_mm:.1f} mm" in text
    order = [text.index(k) for k in ("Sew the shoulder seams", "Sew the short ends of the neckband", "Set each sleeve", "Sew each side seam", "Turn up the hems")]
    assert order == sorted(order)
    assert "notches" in text


def test_bundle_contains_the_run_files(tee_run):
    """REQ-OUT-05: bundle.zip holds the pattern files, the manufacturing files and the debug images."""
    result, out = tee_run
    with zipfile.ZipFile(out / "bundle.zip") as z:
        names = set(z.namelist())
        assert z.testzip() is None
    assert names == ({p.name for p in out.iterdir()} - {"bundle.zip"}) | {
        f"debug/debug_{v}.png" for v in ("front", "back", "sleeve")}
    assert {"pattern.pdf", "pattern.svg", "pattern.dxf", "spec.json", "measurements.json", "seam_log.json",
            "cut-path.csv", "toolpath-illustrative.txt", "cutplan.svg", "bom.csv", "instructions.md"} <= names


def test_req_sew_04_a_failed_run_writes_no_manufacturing_file(tmp_path, monkeypatch):
    """REQ-SEW-04: the seam check fails, so no pattern file and no manufacturing file exist."""
    from sewtrace import pipeline
    from sewtrace.reason.seams import SeamCheckFailed
    synth.make_tee(synth.TEES[0], tmp_path / "tee1", seed=0)
    client = MetaClient(Settings(api_key="", mock=True, cache_dir=tmp_path / "cache"))

    def fail(spec, seam_log):
        raise SeamCheckFailed([], seam_log)

    monkeypatch.setattr(pipeline, "require_seams_ok", fail)
    with pytest.raises(SeamCheckFailed):
        run(tmp_path / "tee1", tmp_path / "out", client)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_a_piece_wider_than_the_roll_stops_the_run_before_any_file(tmp_path):
    """REQ-SEW-04: a refused cut plan leaves no half-written output folder."""
    synth.make_tee(synth.TEES[0], tmp_path / "tee1", seed=0)
    base = Settings(api_key="", mock=True, cache_dir=tmp_path / "cache")
    config = base.config.model_copy(update={"manufacture": base.config.manufacture.model_copy(update={"roll_width_mm": 300.0})})
    client = MetaClient(dataclasses.replace(base, config=config))
    with pytest.raises(mf.PieceTooWide):
        run(tmp_path / "tee1", tmp_path / "out", client)
    assert not (tmp_path / "out").exists()


def test_web_sample_run_lists_the_downloads_and_serves_the_cut_plan(tmp_path, sample_root):  # noqa: F811
    """REQ-SEW-05: the web sample run lists the manufacturing downloads and serves the cut plan JSON."""
    client: TestClient = make_client(tmp_path, sample_root)
    job = run_sample(client)
    assert job["state"] == "done", job
    r, jid = job["result"], job["id"]
    for name in ("cut_path", "toolpath", "cutplan", "bom", "instructions", "bundle"):
        assert name in r["downloads"]
        assert client.get(f"/api/jobs/{jid}/files/{name}").status_code == 200
    assert r["download_files"]["bundle"] == "bundle.zip"
    assert "attachment" in client.get(f"/api/jobs/{jid}/files/bundle").headers["content-disposition"]
    assert client.get(f"/api/jobs/{jid}/files/cutplan").headers["content-type"].startswith("image/svg")
    s = r["cutplan"]["summary"]
    assert s["roll_width_mm"] == 1500 and 0 < s["utilisation"] <= 1 and s["est_duration_s"] > 0
    res = client.get(f"/api/jobs/{jid}/files/{r['cutplan']['paths']}")
    assert res.status_code == 200
    data = res.json()
    assert len(data["paths"]) == 5 and data["paths"][0]["points"][0] == data["paths"][0]["points"][-1]
    assert "Illustrative" in data["label"] and data["cut_speed_mm_s"] == 150
    page = client.get("/").text
    assert "Cut plan" in page and "requestAnimationFrame" in page
    with zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{jid}/files/bundle").content)) as z:
        assert "cutplan.svg" in z.namelist()
