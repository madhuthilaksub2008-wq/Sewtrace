"""Command line: sewtrace synth | run | eval | verify."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sewtrace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write synthetic tee photos")
    s.add_argument("--out", default="data/raw")
    r = sub.add_parser("run", help="photos in a folder -> pattern files")
    r.add_argument("garment_dir")
    r.add_argument("--out", default=None)
    e = sub.add_parser("eval", help="run all garments, compare with ground truth")
    e.add_argument("raw_dir", nargs="?", default="data/raw")
    e.add_argument("--out", default="out/eval")
    e.add_argument("--runs", type=int, default=1)
    v = sub.add_parser("verify", help="proof photo (paper on t-shirt) -> edge gap and overlay")
    v.add_argument("photo")
    v.add_argument("--out", default=None, help="overlay image (default: <photo>.overlay.png)")
    args = ap.parse_args(argv)

    if args.cmd == "synth":
        from sewtrace.mock import synth
        synth.main(["--out", args.out])
    elif args.cmd == "run":
        from sewtrace.pipeline import run
        out = args.out or Path("out") / Path(args.garment_dir).name
        from sewtrace.reason.seams import SeamCheckFailed
        try:
            result = run(args.garment_dir, out)
        except SeamCheckFailed as err:
            print(f"error: {err}", file=sys.stderr)
            sys.exit(1)
        for k, v in result.measurements.items():
            print(f"{k:16s} {v:7.1f} mm")
        print("seams:", "all ok" if all(s.ok for s in result.spec.seams) else "FAILING")
        for k, path in result.files.items():
            print(f"{k:12s} {path}")
    elif args.cmd == "verify":
        _verify(args.photo, args.out)
    elif args.cmd == "eval":
        from sewtrace.eval import evaluate
        rows = evaluate(args.raw_dir, args.out, args.runs)
        print((Path(args.out) / "results.md").read_text())


def _verify(photo: str, out: str | None) -> None:
    """SPEC 9.4 step 5. Masks come from MetaClient (mock or cache by the usual rules)."""
    import cv2

    from sewtrace.api.client import MetaClient
    from sewtrace.vision.marker import MarkerNotFound, photo_homography
    from sewtrace.vision.verify import overlay_image, verify_proof

    path = Path(photo)
    image = cv2.imread(str(path)) if path.is_file() else None
    if image is None:
        sys.exit(f"error: proof photo {photo} not found or not an image. Check the path.")
    try:
        homography = photo_homography(image)
    except MarkerNotFound:
        sys.exit(f"error: marker not found in {photo}. Put the printed marker in view and retake the photo.")
    try:
        result = verify_proof(path, MetaClient(), homography)
    except ValueError as err:
        sys.exit(f"error: {photo}: {err}")
    out_path = overlay_image(path, result, homography, out or path.with_suffix(".overlay.png"))
    print(f"mean gap {result.mean_mm:.2f} mm  max {result.max_mm:.2f} mm  p95 {result.p95_mm:.2f} mm")
    print(f"{100 * result.frac_within_tol:.1f}% of points within {result.tol_mm:g} mm")
    print(f"overlay {out_path}")


if __name__ == "__main__":
    main()
