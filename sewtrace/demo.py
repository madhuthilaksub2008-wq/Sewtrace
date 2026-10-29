"""Demo bundles (T29, T30): record one run, replay it with no network and no model call.

``record`` runs the pipeline once with the current mode and writes ``demo/<name>/``: the photos,
the run outputs, the mask outlines in mm and px, where each panel sits on its photo, the proof
result and a manifest with the cache keys. The cache files that the run used are copied too, so
the bundle alone also replays the normal pipeline in offline mode (REQ-API-08):

    SEWTRACE_CACHE_DIR=demo/<name>/cache SEWTRACE_OFFLINE=1 sewtrace run demo/<name>

``load_bundle`` reads a bundle for the web demo (``sewtrace/web/demo.html``). It reads files only.
The key, the ledger and absolute paths never enter a bundle.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from sewtrace import pipeline
from sewtrace.api.client import ApiError, MetaClient, SegmentationFailed, Usage, cost_usd
from sewtrace.config import ROOT, get_config
from sewtrace.output.layout import layout, panel_art, render_preview
from sewtrace.eval import is_synthetic
from sewtrace.reason.panels import INFER_ATTEMPTS
from sewtrace.reason.seams import seam_name
from sewtrace.types import MEASUREMENT_KEYS, Panel, PanelSpec, Point
from sewtrace.vision import contour
from sewtrace.vision.marker import Homography, MarkerNotFound, MarkerRejected, photo_homography
from sewtrace.vision.verify import ProofResult, overlay_image, verify_proof

DEMO_DIR = ROOT / "demo"
FORMAT = 1
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
VIEWS = ("front", "back", "sleeve")
PANEL_VIEW = {"front": "front", "back": "back", "sleeve": "sleeve"}  # the neckband has no photo
SERVED_FILES = ("front.jpg", "back.jpg", "sleeve.jpg", "proof.jpg", "proof_overlay.jpg", "preview.png")
PROMPT_TOKEN_ALLOWANCE = 3000  # text of one panel call, an upper bound for the worst-case price
OUTLINE_EPS_PX = 1.0  # display only: simplifies the drawn mask outline
FIT_CAP_MM = 20.0  # a panel point farther than this from the outline is an outlier in the placement fit
FIT_START_DEG = 2.0  # step of the rotation search for the placement fit
FIT_RANGE_DEG = 20.0
EVAL_CSV = ROOT / "out" / "eval" / "results.csv"


class DemoError(RuntimeError):
    """A demo bundle cannot be recorded or played. The message says what to do."""


# ---- bundle file models -------------------------------------------------------------------

class Manifest(BaseModel):
    format: Literal[1]
    name: str
    created: str
    mode: Literal["mock", "real", "offline"]
    garment: str  # folder name of the photos
    reasoning_model: str
    model_calls: int  # calls that missed the cache during the recording
    spent_usd: float  # ledger difference during the recording
    files: list[str]  # every file of the bundle except manifest.json, relative to its folder
    cache_keys: list[str]  # keys of the cache files that the run used
    has_proof: bool
    has_eval: bool


class ViewOutline(BaseModel):
    photo: str
    width_px: int
    height_px: int
    mm_to_px: list[list[float]]  # 3x3, the stored homography
    outline_px: list[Point]
    outline_mm: list[Point]


class Masks(BaseModel):
    views: dict[str, ViewOutline]
    proof_garment: ViewOutline | None = None
    proof_paper: ViewOutline | None = None


class PanelPlacement(BaseModel):
    name: str
    cut: str
    on_fold: bool
    stitch_mm: list[Point]  # panel coordinates (half panel when on fold)
    layout_mm: list[Point]  # the same points on the pattern sheet
    cut_layout_mm: list[Point]  # cut line (stitch line plus seam allowance) on the pattern sheet
    edges: dict[str, tuple[int, int]]
    photo_view: str | None  # None: the panel is drawn, not measured (neckband)
    photo_px: list[Point] | None  # the same points on the photo
    fit_residual_mm: float | None


class Placement(BaseModel):
    sheet_width_mm: float
    sheet_height_mm: float
    panels: list[PanelPlacement]


class ProofShow(BaseModel):
    result: ProofResult
    width_px: int
    height_px: int
    points_px: list[Point]
    neck_points_px: list[Point]
    fold_px: list[Point] | None


class EvalRow(BaseModel):
    garment: str
    run: int
    errors_mm: dict[str, float | None]
    mean_abs_err_mm: float | None
    seams_ok: bool
    error: str = ""
    synthetic: bool | None = None  # None: an eval file from before the flag existed


DataSource = Literal["synthetic", "real", "mixed", "unknown"]


class EvalTable(BaseModel):
    source: str
    data_source: DataSource = "unknown"  # the end card names it: synthetic tees are the best case (SPEC 10.1)
    rows: list[EvalRow]
    mean_abs_err_mm: float | None


class Bundle(BaseModel):
    manifest: Manifest
    spec: PanelSpec
    measurements: dict[str, float]
    seam_log: list[dict]
    masks: Masks
    placement: Placement
    proof: ProofShow | None
    eval: EvalTable | None


# ---- pure geometry --------------------------------------------------------------------------

def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def fit_rigid(points: np.ndarray, target: np.ndarray, start: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Rotation, translation and mean distance (mm) that put ``points`` on the ``target`` outline.

    The photo outline holds the whole garment, the panel holds one half, so a trimmed mean of
    the nearest distances is used and the start is the top of the outline. Returns
    ``(theta_rad, translation_mm, residual_mm)`` for ``p' = R(theta) p + t``."""
    tree = cKDTree(target)

    def cost(v: np.ndarray) -> float:
        d, _ = tree.query(points @ _rot(v[0]).T + v[1:])
        return float(np.mean(np.minimum(d, FIT_CAP_MM)))

    grid = np.radians(np.arange(-FIT_RANGE_DEG, FIT_RANGE_DEG + 0.5, FIT_START_DEG))
    theta0 = float(grid[int(np.argmin([cost(np.array([t, *start])) for t in grid]))])
    x0 = np.array([theta0, *start])
    simplex = [x0, x0 + [0.02, 0, 0], x0 + [0, 5.0, 0], x0 + [0, 0, 5.0]]
    res = minimize(cost, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "initial_simplex": simplex})
    return float(res.x[0]), res.x[1:].copy(), float(res.fun)


def _edge_indices(panel: Panel, name: str) -> list[int]:
    edge = next(e for e in panel.edges if e.name == name)
    n = len(panel.outline_mm)
    if edge.start <= edge.end:
        return list(range(edge.start, edge.end + 1))
    return list(range(edge.start, n)) + list(range(edge.end + 1))


def _fit_points(panel: Panel) -> np.ndarray:
    """Panel points that lie on the outline of the photo: all but the fold edge."""
    pts = np.asarray(panel.outline_mm, dtype=float)
    fold = set(_edge_indices(panel, "fold")) if any(e.name == "fold" for e in panel.edges) else set()
    return pts[[i for i in range(len(pts)) if i not in fold]]


def place_on_photo(panel: Panel, target_mm: np.ndarray, homography: Homography) -> tuple[list[Point], float]:
    """Where the panel lies on its photo: points in photo px, and the fit residual in mm."""
    pts = np.asarray(panel.outline_mm, dtype=float)
    start = np.array([(target_mm[:, 0].min() + target_mm[:, 0].max()) / 2, target_mm[:, 1].min()])
    if panel.name == "sleeve":  # the fold is on the left of the sleeve photo
        start[0] = target_mm[:, 0].min()
    theta, shift, residual = fit_rigid(_fit_points(panel), target_mm, start)
    photo_mm = pts @ _rot(theta).T + shift
    return [(float(x), float(y)) for x, y in homography.mm_to_px(photo_mm)], residual


# ---- record -----------------------------------------------------------------------------------

def check_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise DemoError(f"demo name {name!r} is not allowed. Use letters, digits, '-' and '_' (at most 64).")
    return name


def worst_case_usd(client: MetaClient, with_proof: bool) -> float:
    """The most one recording can cost if nothing is in the cache. The ledger guard uses the same prices."""
    cfg = client.config
    sam = cost_usd(cfg.models.sam, Usage(images=len(VIEWS) + (2 if with_proof else 0)))
    one_call = Usage(input_tokens=len(VIEWS) * cfg.reasoning.est_image_tokens + PROMPT_TOKEN_ALLOWANCE,
                     output_tokens=cfg.reasoning.max_tokens, images=len(VIEWS))
    return sam + (INFER_ATTEMPTS + cfg.seam.max_adjust_calls) * cost_usd(cfg.reasoning_model, one_call)


def has_gps_exif(path: Path | str) -> bool:
    """True when a JPEG has a GPS block (EXIF tag 0x8825). A bundle copies the photos byte for byte,
    so the cache keys stay valid, and a public repository must not publish where a garment was photographed."""
    data = Path(path).read_bytes()[:131072]  # EXIF is in the first segments
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] != 0xDA:
        size = int.from_bytes(data[pos + 2 : pos + 4], "big")
        if data[pos + 1] == 0xE1 and data[pos + 4 : pos + 10] == b"Exif\x00\x00":
            tiff = data[pos + 10 : pos + 2 + size]
            order = "little" if tiff[:2] == b"II" else "big"
            first = int.from_bytes(tiff[4:8], order)
            count = int.from_bytes(tiff[first : first + 2], order)
            return any(int.from_bytes(tiff[first + 2 + 12 * k : first + 4 + 12 * k], order) == 0x8825
                       for k in range(count))
        pos += 2 + size
    return False


def _mode(client: MetaClient) -> Literal["mock", "real", "offline"]:
    s = client.settings
    return "offline" if s.offline else "mock" if s.mock else "real"


def _view_outline(mask: np.ndarray, homography: Homography, photo: str, image: np.ndarray) -> ViewOutline:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DemoError(f"{photo}: the SAM mask is empty.")
    biggest = cv2.approxPolyDP(max(contours, key=cv2.contourArea), OUTLINE_EPS_PX, True)
    px = biggest.reshape(-1, 2).astype(float)
    inverse = np.linalg.inv(homography.H)
    return ViewOutline(
        photo=photo, width_px=int(image.shape[1]), height_px=int(image.shape[0]),
        mm_to_px=(inverse / inverse[2, 2]).tolist(),
        outline_px=[(float(x), float(y)) for x, y in px],
        outline_mm=[(float(x), float(y)) for x, y in homography.px_to_mm(px)],
    )


def _body_target_mm(mask: np.ndarray, homography: Homography) -> np.ndarray:
    outline = contour.smooth_outline(contour.mask_to_outline(mask, homography))
    return contour.remove_sleeves(outline).array()


def _sleeve_target_mm(mask: np.ndarray, homography: Homography) -> np.ndarray:
    return contour.smooth_outline(contour.mask_to_outline(mask, homography)).array()


def _placement(spec: PanelSpec, targets: dict[str, np.ndarray], homographies: dict[str, Homography]) -> Placement:
    sheet = layout(spec)
    panels = []
    for art in sheet.panels:
        panel = spec.panel(art.name)
        offset = art.cut[0] - panel_art(panel).cut[0]  # translate() moved every point by the same amount
        view = PANEL_VIEW.get(panel.name)
        photo_px, residual = (None, None) if view is None else place_on_photo(panel, targets[view], homographies[view])
        panels.append(PanelPlacement(
            name=panel.name, cut=panel.cut, on_fold=panel.on_fold, stitch_mm=panel.outline_mm,
            layout_mm=[(float(x), float(y)) for x, y in np.asarray(panel.outline_mm) + offset],
            cut_layout_mm=[(float(x), float(y)) for x, y in art.cut],
            edges={e.name: (e.start, e.end) for e in panel.edges},
            photo_view=view, photo_px=photo_px, fit_residual_mm=residual))
    return Placement(sheet_width_mm=sheet.width, sheet_height_mm=sheet.height, panels=panels)


def _record_proof(client: MetaClient, proof: Path, spec: PanelSpec, out: Path) -> tuple[ProofShow, ViewOutline, ViewOutline]:
    """Proof result and overlay; every failure names the proof photo."""
    try:
        image = pipeline.load_photo("proof", proof)
        homography = photo_homography(image, f"proof photo {proof.name}")
        result = verify_proof(proof, client, homography, spec=spec)
        prompts = client.config.sam.prompts
        paper = _view_outline(client.segment(proof, prompts.paper), homography, proof.name, image)
        garment = _view_outline(client.segment(proof, prompts.garment), homography, proof.name, image)
    except (pipeline.PhotoMissing, MarkerNotFound, MarkerRejected, SegmentationFailed, ValueError) as err:
        raise DemoError(f"proof photo {proof.name}: {err}. Fix the photo or remove proof.jpg from the folder.") from err
    overlay_image(proof, result, homography, out / "proof_overlay.jpg")

    def px(points: list[Point]) -> list[Point]:
        return [(float(x), float(y)) for x, y in homography.mm_to_px(np.asarray(points).reshape(-1, 2))]

    show = ProofShow(result=result, width_px=int(image.shape[1]), height_px=int(image.shape[0]),
                     points_px=px(result.points_mm), neck_points_px=px(result.neck_points_mm),
                     fold_px=None if result.fold_mm is None else px(list(result.fold_mm)))
    return show, paper, garment


def data_source_of(rows: list[EvalRow]) -> DataSource:
    flags = {r.synthetic for r in rows}
    if flags == {True}:
        return "synthetic"
    if flags == {False}:
        return "real"
    return "mixed" if flags == {True, False} else "unknown"


def read_eval(eval_csv: Path | None) -> EvalTable | None:
    """The last eval table (SPEC 15, demo): out/eval/results.csv, else this tee's truth.json."""
    keys = [f"err_{k}" for k in MEASUREMENT_KEYS]
    if eval_csv is not None and eval_csv.is_file():
        with eval_csv.open(newline="") as f:
            raw = list(csv.DictReader(f))
        rows = [EvalRow(
            garment=r["garment"], run=int(r["run"]),
            errors_mm={k[4:]: (float(r[k]) if r[k] != "" else None) for k in keys},
            mean_abs_err_mm=float(r["mean_abs_err"]) if r["mean_abs_err"] != "" else None,
            seams_ok=r["seams_ok"] == "True", error=r["error"],
            synthetic=None if r.get("synthetic", "") == "" else r["synthetic"] == "True") for r in raw]
        return EvalTable(source=eval_csv.name, rows=rows, data_source=data_source_of(rows),
                         mean_abs_err_mm=_mean([r.mean_abs_err_mm for r in rows]))
    return None


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _truth_row(garment_dir: Path, measurements: dict[str, float], seams_ok: bool) -> EvalTable | None:
    truth_file = garment_dir / "truth.json"
    if not truth_file.is_file():
        return None
    truth = json.loads(truth_file.read_text())["measurements"]
    errors = {k: measurements[k] - truth[k] for k in MEASUREMENT_KEYS}
    mean = sum(abs(v) for v in errors.values()) / len(errors)
    row = EvalRow(garment=garment_dir.name, run=1, errors_mm=errors, mean_abs_err_mm=mean, seams_ok=seams_ok,
                  synthetic=is_synthetic(garment_dir))
    return EvalTable(source="truth.json of this tee", rows=[row], data_source=data_source_of([row]),
                     mean_abs_err_mm=mean)


def record(
    garment_dir: str | Path, name: str | None = None, *, client: MetaClient | None = None,
    demo_dir: Path = DEMO_DIR, yes_spend: bool = False, force: bool = False, eval_csv: Path | None = EVAL_CSV,
    allow_gps: bool = False, allow_synthetic_eval: bool = False,
) -> Path:
    """Run the pipeline once and write ``demo_dir/<name>/``. Returns the bundle folder.

    In REAL mode this makes paid calls through MetaClient (guard, ledger and cache apply), so it
    prints the worst-case cost and refuses without ``yes_spend``. Both checks on the inputs run
    before any spend: a photo with a GPS block (C-07) and, in REAL mode, an eval table that does
    not come from real t-shirts (SPEC 10.1)."""
    garment_dir = Path(garment_dir)
    name = check_name(name or garment_dir.name)
    if not allow_gps:
        _refuse_gps(garment_dir)
    client = client or MetaClient()
    with_proof = (garment_dir / "proof.jpg").is_file()
    mode = _mode(client)
    worst = worst_case_usd(client, with_proof)
    print(f"Recording demo '{name}' from {garment_dir.name} in {mode.upper()} mode.")
    print(f"Worst case if nothing is cached: ${worst:.4f}. Ledger limit ${client.config.api.budget_limit_usd:.2f}.")
    if mode == "real" and not yes_spend:
        raise ApiError("Nothing was called. REAL mode makes paid calls: add --yes-spend to record the demo.")
    table = read_eval(eval_csv)
    if mode == "real" and table is not None:
        table = _real_card_table(table, eval_csv, allow_synthetic_eval)
    target, tmp = demo_dir / name, demo_dir / f".{name}.tmp"
    if target.exists() and not force:
        raise DemoError(f"demo '{name}' already exists. Use --force to replace it, or choose another --name.")
    demo_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)  # a leftover of an earlier failed recording
    tmp.mkdir()
    try:
        _write_bundle(client, garment_dir, name, tmp, mode, table)
        _install(tmp, target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def _install(tmp: Path, target: Path) -> None:
    """Move a finished bundle into place. A file with the same bytes is left alone, so a second
    recording of the same input leaves no change in git."""
    target.mkdir(exist_ok=True)
    # The manifest goes first and returns last: a copy that stops half way leaves a bundle
    # without manifest, which `demo play --check` refuses.
    (target / "manifest.json").unlink(missing_ok=True)
    fresh = {p.relative_to(tmp) for p in tmp.rglob("*") if p.is_file()}
    for old in sorted(target.rglob("*"), reverse=True):
        rel = old.relative_to(target)
        if old.is_file() and rel not in fresh:
            old.unlink()
        elif old.is_dir() and not any(old.iterdir()):
            old.rmdir()
    for rel in sorted(fresh, key=lambda r: r == Path("manifest.json")):  # False sorts first
        src, dst = tmp / rel, target / rel
        if dst.is_file() and dst.read_bytes() == src.read_bytes():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _refuse_gps(garment_dir: Path) -> None:
    """C-07: the photos are copied byte for byte into a folder that is meant to be committed."""
    photos = [Path(path) for _, path in pipeline.photo_set(garment_dir)] + [garment_dir / "proof.jpg"]
    for photo in photos:
        if photo.is_file() and has_gps_exif(photo):  # a missing photo is named later, by the pipeline
            raise DemoError(
                f"{photo.name} has GPS data in its EXIF header, and a bundle copies the photo as it is. "
                "Turn off location tags in the camera app and take the photo again. "
                "Add --allow-gps only if the bundle will never be committed.")


def _real_card_table(table: EvalTable, eval_csv: Path | None, allow_synthetic: bool) -> EvalTable:
    """A real bundle must not end on the synthetic best case (SPEC 10.1)."""
    if table.data_source in ("real", "mixed"):
        return table
    if not allow_synthetic:
        raise DemoError(
            f"{eval_csv} holds only {table.data_source} t-shirts, and this recording is REAL. Nothing was called. "
            "Run the eval on the real t-shirts (data/raw with data/ground_truth/<tee>.json), "
            "or add --allow-synthetic-eval to show the synthetic table, labelled as such.")
    return table.model_copy(update={"data_source": "synthetic"})


def _newest_photo_time(garment_dir: Path) -> str:
    """When the newest input photo was written. Not the clock, so a second recording of the same
    photos gives the same manifest."""
    paths = [Path(path) for _, path in pipeline.photo_set(garment_dir)] + [garment_dir / "proof.jpg"]
    newest = max(p.stat().st_mtime for p in paths if p.is_file())
    return dt.datetime.fromtimestamp(newest, dt.timezone.utc).isoformat(timespec="seconds")


def _write_bundle(
    client: MetaClient, garment_dir: Path, name: str, out: Path, mode: str, csv_table: EvalTable | None,
) -> None:
    spent_before, calls_before = client.ledger_total_usd(), client.calls
    result = pipeline.run(garment_dir, out, client)  # writes pdf, svg, dxf, spec, measurements, seam_log
    photos = pipeline.photo_set(garment_dir)
    prompts = client.config.sam.prompts

    views: dict[str, ViewOutline] = {}
    targets: dict[str, np.ndarray] = {}
    homographies: dict[str, Homography] = {}
    for view, path in photos:  # masks and scales come from the cache: the run just made them
        image = pipeline.load_photo(view, path)
        homographies[view] = pipeline.photo_scale(view, path)
        mask = client.segment(path, prompts.sleeve if view == "sleeve" else prompts.garment)
        views[view] = _view_outline(mask, homographies[view], f"{view}.jpg", image)
        targets[view] = (_sleeve_target_mm if view == "sleeve" else _body_target_mm)(mask, homographies[view])
        shutil.copyfile(path, out / f"{view}.jpg")
    masks = Masks(views=views)

    proof_show = None
    proof_photo = garment_dir / "proof.jpg"
    if proof_photo.is_file():
        proof_show, masks.proof_paper, masks.proof_garment = _record_proof(client, proof_photo, result.spec, out)
        shutil.copyfile(proof_photo, out / "proof.jpg")
        (out / "proof_result.json").write_text(proof_show.model_dump_json(indent=1))

    render_preview(result.spec, out / "preview.png")
    (out / "masks.json").write_text(masks.model_dump_json(indent=1))
    placement = _placement(result.spec, targets, homographies)
    (out / "placement.json").write_text(placement.model_dump_json(indent=1))
    table = csv_table or _truth_row(
        garment_dir, result.measurements, all(s.ok for s in result.spec.seams))
    if table is not None:
        (out / "eval.json").write_text(table.model_dump_json(indent=1))

    cache_out = out / "cache"
    cache_out.mkdir()
    used = sorted(p for p in client.cache_files_used if p.is_file())  # a lookup that missed has no file
    for path in used:
        shutil.copyfile(path, cache_out / path.name)
    keys = sorted({p.name.split(".")[0] for p in used})
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    manifest = Manifest(
        format=FORMAT, name=name, created=_newest_photo_time(garment_dir),
        mode=mode, garment=garment_dir.name, reasoning_model=client.config.reasoning_model.id,
        model_calls=client.calls - calls_before, spent_usd=round(client.ledger_total_usd() - spent_before, 6),
        files=files, cache_keys=keys, has_proof=proof_show is not None, has_eval=table is not None)
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=1))


# ---- play ---------------------------------------------------------------------------------------

def bundle_dir(name: str, demo_dir: Path = DEMO_DIR) -> Path:
    return demo_dir / check_name(name)


def list_bundles(demo_dir: Path = DEMO_DIR) -> list[str]:
    if not demo_dir.is_dir():
        return []
    return sorted(p.name for p in demo_dir.iterdir()
                  if NAME_RE.match(p.name) and (p / "manifest.json").is_file())


def _hint(name: str) -> str:
    return f"Record the demo again: sewtrace demo record <garment_dir> --name {name} --force"


def _read(folder: Path, name: str, file: str, model: type[BaseModel]) -> BaseModel:
    path = folder / file
    if not path.is_file():
        raise DemoError(f"demo '{name}': file {file} is missing. {_hint(name)}")
    try:
        return model.model_validate_json(path.read_text())
    except (ValidationError, ValueError) as err:
        raise DemoError(f"demo '{name}': file {file} cannot be read: {err}. {_hint(name)}") from err


def _read_json(folder: Path, name: str, file: str) -> object:
    path = folder / file
    if not path.is_file():
        raise DemoError(f"demo '{name}': file {file} is missing. {_hint(name)}")
    try:
        return json.loads(path.read_text())
    except ValueError as err:
        raise DemoError(f"demo '{name}': file {file} is not valid JSON: {err}. {_hint(name)}") from err


def load_bundle(name: str, demo_dir: Path = DEMO_DIR) -> Bundle:
    """Read and check a bundle. Needs no network, no key and no cache. Any missing file is named."""
    folder = bundle_dir(name, demo_dir)
    if not folder.is_dir():
        raise DemoError(f"demo '{name}' not found. Record it: sewtrace demo record <garment_dir> --name {name}")
    manifest = _read(folder, name, "manifest.json", Manifest)
    assert isinstance(manifest, Manifest)
    for file in manifest.files:
        if not (folder / file).is_file():
            raise DemoError(f"demo '{name}': file {file} is missing. {_hint(name)}")
    measurements = _read_json(folder, name, "measurements.json")
    seam_log = _read_json(folder, name, "seam_log.json")
    if not isinstance(measurements, dict) or not isinstance(seam_log, list):
        raise DemoError(f"demo '{name}': measurements.json or seam_log.json has the wrong shape. {_hint(name)}")
    proof = _read(folder, name, "proof_result.json", ProofShow) if manifest.has_proof else None
    table = _read(folder, name, "eval.json", EvalTable) if manifest.has_eval else None
    return Bundle(
        manifest=manifest, spec=_read(folder, name, "spec.json", PanelSpec), measurements=measurements,
        seam_log=seam_log, masks=_read(folder, name, "masks.json", Masks),
        placement=_read(folder, name, "placement.json", Placement), proof=proof, eval=table)  # type: ignore[arg-type]


def payload(bundle: Bundle) -> dict:
    """JSON for the demo page: the data the animation draws, and the URLs of the images."""
    m = bundle.manifest
    base = f"/api/demo/{m.name}/files/"
    return {
        "name": m.name, "mode": m.mode, "created": m.created, "garment": m.garment,
        "reasoning_model": m.reasoning_model, "model_calls_recorded": m.model_calls,
        "max_adjust_calls": get_config().seam.max_adjust_calls,
        "urls": {f.split(".")[0]: base + f for f in SERVED_FILES if f in m.files},
        "measurement_keys": list(MEASUREMENT_KEYS),
        "measurements": bundle.measurements,
        "views": {k: v.model_dump() for k, v in bundle.masks.views.items()},
        "placement": bundle.placement.model_dump(),
        "seams": [{"name": seam_name(s), "a": s.a, "b": s.b, "diff_mm": s.diff_mm, "ok": s.ok,
                   "min_diff_mm": s.min_diff_mm, "max_diff_mm": s.max_diff_mm} for s in bundle.spec.seams],
        "seam_log": bundle.seam_log,
        "proof": None if bundle.proof is None else bundle.proof.model_dump(),
        "eval": None if bundle.eval is None else bundle.eval.model_dump(),
    }


def summary(bundle: Bundle) -> str:
    m = bundle.manifest
    proof = "proof overlay" if m.has_proof else "no proof photo"
    return (f"demo '{m.name}': recorded {m.created} in {m.mode.upper()} mode from {m.garment}, "
            f"{len(m.files)} files, {len(m.cache_keys)} cache keys, {proof}, "
            f"{len(bundle.seam_log)} seam checks, {len(bundle.placement.panels)} panels.")
