import re
import xml.etree.ElementTree as ET

import cv2
import ezdxf
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from sewtrace.mock import synth
from sewtrace.mock.fixtures import truth_spec
from sewtrace.output import layout as L
from sewtrace.output.pdf import PAGE_PT, PT_PER_MM, cover_inch_square_pt, cover_test_square_pt, render_pdf
from sewtrace.output.vector import render_dxf, render_svg

TEES = (0, 4)


@pytest.fixture(scope="session")
def specs(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("raw_out")
    res = {}
    for i in TEES:
        p = synth.TEES[i]
        synth.make_tee(p, out / p.name, seed=i)
        res[i] = truth_spec(out / p.name)
    return res


def _page_count(data: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])", data))


@pytest.mark.parametrize("i", TEES)
def test_pdf_pages_and_size(specs, tmp_path, i):
    """REQ-OUT-01: pdf pages and size."""
    spec = specs[i]
    path = render_pdf(spec, tmp_path / "p.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    tiles = L.layout(spec).tiles()
    sheet = L.layout(spec)
    assert 0 < len(tiles) < sheet.rows * sheet.cols  # some empty tiles skipped
    assert _page_count(data) == 1 + len(tiles)
    assert PAGE_PT == pytest.approx((210 * 72 / 25.4, 297 * 72 / 25.4))
    boxes = re.findall(rb"/MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
    assert len(boxes) == _page_count(data)
    for w, h in boxes:
        assert float(w) == pytest.approx(595.2756, abs=0.01)
        assert float(h) == pytest.approx(841.8898, abs=0.01)


def test_test_square_exact(specs, tmp_path):
    """REQ-OUT-02: test square exact."""
    side = 100 * 72 / 25.4
    x, y, w, h = cover_test_square_pt()
    assert w == pytest.approx(side, abs=1e-9) and h == pytest.approx(side, abs=1e-9)
    assert cover_inch_square_pt()[2] == pytest.approx(72.0)
    assert PT_PER_MM == 72 / 25.4
    data = render_pdf(specs[0], tmp_path / "p.pdf").read_bytes()
    rects = re.findall(rb"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", data)
    assert any(abs(float(a) - side) < 1e-3 and abs(float(b) - side) < 1e-3 for *_, a, b in rects)


@pytest.mark.parametrize("i", TEES)
def test_cut_line_encloses_stitch_by_allowance(specs, i):
    """REQ-OUT-03: cut line encloses stitch by allowance."""
    for panel in specs[i].panels:
        sa = panel.seam_allowance_mm
        cut = L.cut_polygon(panel)
        stitch = Polygon(panel.outline_mm)
        assert cut.contains(stitch.buffer(-1e-6))
        d_pts = []
        for x, y in panel.outline_mm:
            d = cut.exterior.distance(Point(x, y))
            if panel.on_fold and abs(x) < 1e-6:
                assert d == pytest.approx(0, abs=1e-6)  # cut sits on the fold
            elif not panel.on_fold or x > sa:
                d_pts.append(d)
        assert min(d_pts) == pytest.approx(sa, abs=1e-6)
        # No point of the cut line comes closer than the allowance (except on the fold).
        for x, y in np.asarray(cut.exterior.coords):
            if panel.on_fold and abs(x) < 1e-6:
                continue
            assert stitch.exterior.distance(Point(x, y)) >= sa - 1e-6
        if panel.on_fold:
            assert cut.bounds[0] == pytest.approx(0, abs=1e-9)


def test_notches_cross_cut_line(specs):
    """REQ-OUT-04: notches cross cut line."""
    sheet = L.layout(specs[0])
    arts = {a.name: a for a in sheet.panels}
    for name in ("front", "back", "sleeve"):
        assert arts[name].notches
        ring = Polygon(arts[name].cut).exterior
        for tick in arts[name].notches:
            assert LineString(tick).intersects(ring)


@pytest.mark.parametrize("i", TEES)
def test_layout_no_overlap(specs, i):
    """REQ-OUT-01: layout no overlap."""
    sheet = L.layout(specs[i])
    polys = [Polygon(a.cut) for a in sheet.panels]
    for a in range(len(polys)):
        assert polys[a].bounds[0] >= 0 and polys[a].bounds[1] >= 0
        assert polys[a].bounds[2] <= sheet.width and polys[a].bounds[3] <= sheet.height
        for b in range(a + 1, len(polys)):
            assert polys[a].distance(polys[b]) >= L.GAP_MM - 1e-6


@pytest.mark.parametrize("i", TEES)
def test_dxf_layers_units_and_cut_bbox(specs, tmp_path, i):
    """REQ-OUT-05: dxf layers units and cut bbox."""
    spec = specs[i]
    path = render_dxf(spec, tmp_path / "p.dxf")
    doc = ezdxf.readfile(path)
    assert doc.header["$INSUNITS"] == 4
    names = {layer.dxf.name for layer in doc.layers}
    assert {"CUT", "STITCH", "NOTCH", "TEXT"} <= names
    cuts = [e for e in doc.modelspace().query("LWPOLYLINE") if e.dxf.layer == "CUT"]
    assert len(cuts) == len(spec.panels) and all(e.closed for e in cuts)
    dims = []
    for e in cuts:
        pts = np.array([p[:2] for p in e.get_points()])
        dims.append(pts.max(0) - pts.min(0))
    for panel in spec.panels:
        sa = panel.seam_allowance_mm
        st = np.asarray(panel.outline_mm)
        size = st.max(0) - st.min(0)
        expected = np.array(L.cut_polygon(panel).bounds[2:]) - np.array(L.cut_polygon(panel).bounds[:2])
        # Panel + allowance: fold side gets none; mitred corners may poke out a little.
        lo = size + np.array([sa if panel.on_fold else 2 * sa, 2 * sa])
        assert np.all(expected >= lo - 1e-6) and np.all(expected <= lo + sa)
        assert any(np.all(np.abs(d - expected) < 0.1) for d in dims), panel.name
    band = spec.panel("neckband")
    st = np.asarray(band.outline_mm)
    exact = st.max(0) - st.min(0) + 2 * band.seam_allowance_mm
    assert any(np.all(np.abs(d - exact) < 0.1) for d in dims)


def test_svg_mm_units(specs, tmp_path):
    """REQ-OUT-05: svg mm units."""
    path = render_svg(specs[0], tmp_path / "p.svg")
    root = ET.parse(path).getroot()
    assert root.get("width").endswith("mm") and root.get("height").endswith("mm")
    sheet = L.layout(specs[0])
    assert float(root.get("width")[:-2]) == pytest.approx(sheet.width)
    vb = [float(v) for v in root.get("viewBox").split()]
    assert vb == pytest.approx([0, 0, sheet.width, sheet.height])
    text = path.read_text()
    assert "PLACE ON FOLD" in text and "stroke-dasharray" in text


def test_preview_png(specs, tmp_path):
    """REQ-OUT-04: preview png."""
    path = L.render_preview(specs[0], tmp_path / "p.png", px_per_mm=1.0)
    img = cv2.imread(str(path))
    sheet = L.layout(specs[0])
    assert img.shape[1] == pytest.approx(sheet.width, abs=2)
    assert (img < 100).any()  # some ink


def _page_text(pdf: bytes) -> str:
    """reportlab writes the page streams uncompressed, so strings can be searched."""
    return pdf.decode("latin-1")


@pytest.mark.parametrize("i", TEES)
def test_req_out_06_one_sewing_step_per_seam(specs, tmp_path, i):
    """REQ-OUT-06: every seam in spec.seams is its own step; neckband and hems are steps too."""
    spec = specs[i]
    steps = spec.sewing_order
    assert len(steps) == len(spec.seams) + 2  # + neckband + hems
    for seam in spec.seams:
        words = [w for ref in (seam.a, seam.b) for part in ref.removesuffix(":mirror").split("+")
                 for w in part.split(".")]
        assert sum(all(w in step for w in words) for step in steps) >= 1, seam
    assert "neckband" in steps[1] and "hem" in steps[-1]
    shoulder = next(k for k, s in enumerate(steps) if "shoulder" in s)
    assert shoulder < 1 and steps[-1].startswith("Turn up")
    # In spec.json and on the PDF cover page.
    assert spec.model_dump(mode="json")["sewing_order"] == steps
    text = _page_text(render_pdf(spec, tmp_path / "p.pdf").read_bytes())
    assert "Sewing order" in text and "front shoulder to back shoulder" in text
    assert "neckband" in text


def test_req_out_06_sewing_order_follows_seams(specs):
    spec = specs[0]
    fewer = spec.model_copy(update={"seams": spec.seams[:2]})
    assert len(fewer.sewing_order) == 2 + 2
