"""Command line: sewtrace synth | run | eval | verify | serve | day1 | demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sewtrace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write synthetic tee photos")
    s.add_argument("--out", default="data/synth")
    r = sub.add_parser("run", help="photos in a folder -> pattern files")
    r.add_argument("garment_dir")
    r.add_argument("--out", default=None)
    e = sub.add_parser("eval", help="run all garments, compare with ground truth")
    e.add_argument("raw_dir", nargs="?", default="data/synth",
                   help="folder of tees: data/synth (truth.json) or data/raw (needs data/ground_truth/<tee>.json)")
    e.add_argument("--out", default="out/eval")
    e.add_argument("--runs", type=int, default=1)
    v = sub.add_parser("verify", help="proof photo (paper on t-shirt) -> edge gap and overlay")
    v.add_argument("photo")
    v.add_argument("--out", default=None, help="overlay image (default: <photo>.overlay.png)")
    v.add_argument("--spec", default=None,
                   help="spec.json of the cut pattern (default: out/<photo folder name>/spec.json, "
                        "the folder of 'sewtrace run <that folder>'); gives the expected neck gap")
    sv = sub.add_parser("serve", help="start the one-page test UI (see docs/DEPLOY.md)")
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 in a container")
    sv.add_argument("--port", type=int, default=None, help="default: web.port in config.toml")
    d = sub.add_parser("day1", help="Day 1 test: ONE paid SAM call and ONE paid Spark call (REQ-API-09)")
    d.add_argument("photo")
    d.add_argument("--yes-spend", action="store_true",
                   help="required: confirm that the two paid calls are approved")
    dm = sub.add_parser("demo", help="record a demo bundle, or play one with no network (docs/DEMO.md)")
    dsub = dm.add_subparsers(dest="demo_cmd", required=True)
    rec = dsub.add_parser("record", help="run the pipeline once and write demo/<name>/")
    rec.add_argument("garment_dir")
    rec.add_argument("--name", default=None, help="bundle name (default: the folder name)")
    rec.add_argument("--yes-spend", action="store_true", help="required in REAL mode: confirm the paid calls")
    rec.add_argument("--force", action="store_true", help="replace a bundle of the same name")
    rec.add_argument("--eval-csv", default="out/eval/results.csv",
                     help="eval results for the end card (default: out/eval/results.csv, if it exists)")
    rec.add_argument("--demo-dir", default="demo", help="folder of bundles (default: demo)")
    rec.add_argument("--allow-gps", action="store_true",
                     help="record photos that have GPS EXIF data (the bundle must never be committed)")
    rec.add_argument("--allow-synthetic-eval", action="store_true",
                     help="REAL mode: accept an eval table of synthetic tees; the end card says so")
    play = dsub.add_parser("play", help="serve a recorded bundle with no network and no model call")
    play.add_argument("name")
    play.add_argument("--check", action="store_true", help="only check that the bundle is complete")
    play.add_argument("--demo-dir", default="demo", help="folder of bundles (default: demo)")
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=None, help="default: web.port in config.toml")
    args = ap.parse_args(argv)
    from sewtrace.api.client import ApiError
    from sewtrace.demo import DemoError
    from sewtrace.pipeline import ContourFailed, PanelInferenceFailed
    try:
        _dispatch(args)
    except (ApiError, ContourFailed, PanelInferenceFailed, NotImplementedError, DemoError) as err:
        # Budget stop, offline cache miss, timeout, policy stop, rejected request, unwired real
        # endpoint, missing SAM parser, Day 1 refusal: no traceback.
        _fail(f"{args.cmd}: {err}")


def _dispatch(args: argparse.Namespace) -> None:
    if args.cmd == "synth":
        from sewtrace.mock import synth
        synth.main(["--out", args.out])
    elif args.cmd == "run":
        from sewtrace.api.client import SegmentationFailed
        from sewtrace.pipeline import PhotoMissing, run
        from sewtrace.reason.seams import SeamCheckFailed
        from sewtrace.vision.marker import MarkerNotFound, MarkerRejected
        out = args.out or Path("out") / Path(args.garment_dir).name
        try:
            result = run(args.garment_dir, out)
        except (SeamCheckFailed, PhotoMissing, MarkerNotFound, MarkerRejected, SegmentationFailed) as err:
            _fail(err)
        for k, v in result.measurements.items():
            print(f"{k:16s} {v:7.1f} mm")
        print("seams:", "all ok" if all(s.ok for s in result.spec.seams) else "FAILING")
        for row in result.seam_log:
            if row["source"] != "pair_fix":
                continue
            for change in row["applied"]:  # a measured edge was changed by the pair fix (REQ-SEW-03)
                print(f"seam fix: {change['panel']}.{change['edge']} {change['from_mm']:.2f} mm -> "
                      f"{change['to_mm']:.2f} mm (pair {change['pair']})")
        for k, path in result.files.items():
            print(f"{k:12s} {path}")
    elif args.cmd == "serve":
        import logging

        import uvicorn

        from sewtrace.config import get_config
        from sewtrace.web.app import WebConfigError, create_app
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # shows the ledger line
        try:
            app = create_app()
        except WebConfigError as err:
            _fail(err)
        uvicorn.run(app, host=args.host, port=args.port or get_config().web.port)
    elif args.cmd == "day1":
        from sewtrace.day1 import run_day1
        run_day1(args.photo, yes_spend=args.yes_spend)
    elif args.cmd == "demo":
        _demo(args)
    elif args.cmd == "verify":
        _verify(args.photo, args.out, args.spec)
    elif args.cmd == "eval":
        from sewtrace.eval import evaluate, tee_failed
        try:
            rows = evaluate(args.raw_dir, args.out, args.runs)
        except (FileNotFoundError, ValueError) as err:
            _fail(err)
        print((Path(args.out) / "results.md").read_text())
        failed = sum(tee_failed(r) for r in rows)
        if failed:  # exit non-zero so a failing tee cannot pass unnoticed
            _fail(f"eval: {failed} of {len(rows)} runs failed. See the error column above.")


def _demo(args: argparse.Namespace) -> None:
    from sewtrace import demo
    if args.demo_cmd == "record":
        from sewtrace.api.client import SegmentationFailed
        from sewtrace.pipeline import PhotoMissing
        from sewtrace.reason.seams import SeamCheckFailed
        from sewtrace.vision.marker import MarkerNotFound, MarkerRejected
        try:
            folder = demo.record(args.garment_dir, args.name, yes_spend=args.yes_spend, force=args.force,
                                 demo_dir=Path(args.demo_dir), eval_csv=Path(args.eval_csv),
                                 allow_gps=args.allow_gps, allow_synthetic_eval=args.allow_synthetic_eval)
        except (SeamCheckFailed, PhotoMissing, MarkerNotFound, MarkerRejected, SegmentationFailed) as err:
            _fail(err)
        print(f"demo bundle written to {folder}")
        return
    demo_dir = Path(args.demo_dir)
    bundle = demo.load_bundle(args.name, demo_dir)  # a missing file stops here, by name
    print(demo.summary(bundle))
    if args.check:
        return
    import dataclasses

    import uvicorn

    from sewtrace.config import Settings, get_config
    from sewtrace.web.app import WebConfigError, create_app
    # Offline and without a key: the page cannot make a model call, whatever the environment says.
    settings = dataclasses.replace(Settings.load(), api_key="", mock=False, offline=True)
    try:
        app = create_app(settings, demo_root=demo_dir.resolve())
    except WebConfigError as err:
        _fail(err)
    port = args.port or get_config().web.port
    print(f"Open http://{args.host}:{port}/demo?name={args.name}")
    uvicorn.run(app, host=args.host, port=port)


def _fail(message: object) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def _find_spec(photo: Path, spec_arg: str | None) -> Path | None:
    """The spec of the cut pattern: --spec, else the default output folder of ``run`` for the
    photo's folder. None when there is none (the neck gap then has no expected value)."""
    if spec_arg:
        return Path(spec_arg)
    guess = Path("out") / photo.resolve().parent.name / "spec.json"
    return guess if guess.is_file() else None


def _verify(photo: str, out: str | None, spec_arg: str | None = None) -> None:
    """SPEC 9.4 step 5. Masks come from MetaClient (mock or cache by the usual rules)."""
    import cv2

    from sewtrace.api.client import MetaClient, SegmentationFailed
    from sewtrace.types import PanelSpec
    from sewtrace.vision.marker import MarkerNotFound, MarkerRejected, photo_homography
    from sewtrace.vision.verify import overlay_image, verify_proof

    path = Path(photo)
    image = cv2.imread(str(path)) if path.is_file() else None
    if image is None:
        sys.exit(f"error: proof photo {photo} not found or not an image. Check the path.")
    try:
        homography = photo_homography(image, f"proof photo {photo}")
    except (MarkerNotFound, MarkerRejected) as err:
        sys.exit(f"error: {err}")
    spec = None
    spec_path = _find_spec(path, spec_arg)
    if spec_path is not None:
        try:
            spec = PanelSpec.model_validate_json(spec_path.read_text())
        except (OSError, ValueError) as err:  # pydantic ValidationError is a ValueError
            sys.exit(f"error: spec {spec_path} cannot be read: {err}. Run 'sewtrace run' again or fix --spec.")
    try:
        result = verify_proof(path, MetaClient(), homography, spec=spec)
    except (ValueError, SegmentationFailed) as err:
        sys.exit(f"error: {photo}: {err}")
    out_path = overlay_image(path, result, homography, out or path.with_suffix(".overlay.png"))
    print(f"mean gap {result.mean_mm:.2f} mm  max {result.max_mm:.2f} mm  p95 {result.p95_mm:.2f} mm")
    print(f"{100 * result.frac_within_tol:.1f}% of points within {result.tol_mm:g} mm")
    if result.neck_mean_mm is not None:
        print(f"neck (information, not in the mean): {len(result.neck_distances_mm)} points, "
              f"mean gap {result.neck_mean_mm:.1f} mm. The photo shows the back neckline "
              "through the neck opening, so it depends on front depth minus back depth.")
        if result.expected_neck_gap_mm is None:
            print("neck gap expected: unknown (no spec.json found; pass --spec path)")
        else:
            diff = result.neck_gap_difference_mm
            print(f"neck gap expected from the spec: {result.expected_neck_gap_mm:.1f} mm, "
                  f"difference {diff:+.1f} mm")
            if result.neck_gap_flagged:
                print(f"WARNING: the neck gap differs from the spec by more than {result.neck_gap_tol_mm:g} mm. "
                      "Check the neck depths of the cut paper against spec.json.")
    if result.side_offset_mm is not None:
        print(f"The gap is the same along the side ({result.side_offset_mm:.1f} mm). Check that the fold of the "
              "paper is on the centre of the t-shirt. If it is, the paper is cut too wide or too narrow.")
    print(f"overlay {out_path}")


if __name__ == "__main__":
    main()
