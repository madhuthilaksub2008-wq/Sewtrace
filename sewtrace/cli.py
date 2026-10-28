"""Command line: sewtrace synth | run | eval."""

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
    elif args.cmd == "eval":
        from sewtrace.eval import evaluate
        rows = evaluate(args.raw_dir, args.out, args.runs)
        print((Path(args.out) / "results.md").read_text())


if __name__ == "__main__":
    main()
