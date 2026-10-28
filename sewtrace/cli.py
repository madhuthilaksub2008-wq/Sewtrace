"""Command line: sewtrace synth | run | eval | verify."""

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
    args = ap.parse_args(argv)
    from sewtrace.api.client import ApiError
    from sewtrace.pipeline import ContourFailed
    try:
        _dispatch(args)
    except (ApiError, ContourFailed, NotImplementedError) as err:
        # Budget stop, offline cache miss, timeout, policy stop, unwired real endpoint: no traceback.
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
        for k, path in result.files.items():
            print(f"{k:12s} {path}")
    elif args.cmd == "verify":
        _verify(args.photo, args.out)
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


def _fail(message: object) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def _verify(photo: str, out: str | None) -> None:
    """SPEC 9.4 step 5. Masks come from MetaClient (mock or cache by the usual rules)."""
    import cv2

    from sewtrace.api.client import MetaClient, SegmentationFailed
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
    try:
        result = verify_proof(path, MetaClient(), homography)
    except (ValueError, SegmentationFailed) as err:
        sys.exit(f"error: {photo}: {err}")
    out_path = overlay_image(path, result, homography, out or path.with_suffix(".overlay.png"))
    print(f"mean gap {result.mean_mm:.2f} mm  max {result.max_mm:.2f} mm  p95 {result.p95_mm:.2f} mm")
    print(f"{100 * result.frac_within_tol:.1f}% of points within {result.tol_mm:g} mm")
    if result.neck_mean_mm is not None:
        print(f"neck (information, not in the mean): {len(result.neck_distances_mm)} points, "
              f"mean gap {result.neck_mean_mm:.1f} mm. The photo shows the back neckline "
              "through the neck opening, so this is about front depth minus back depth.")
    print(f"overlay {out_path}")


if __name__ == "__main__":
    main()
