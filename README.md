# Sewtrace

Lay a kid's t-shirt flat, take three photos, and Sewtrace prints a 1:1 sewing
pattern that matches the original within millimetres and is checked to sew
together. Built for the Meta Global AI Developer Hackathon 2026 (Indie track).

```
3 photos + marker
  > marker detection + homography (px to mm)          sewtrace/vision/marker.py
  > SAM 3.1 masks (t-shirt, sleeve)                      sewtrace/api/client.py
  > contours, sleeve removal, symmetry, edges (mm)   sewtrace/vision/contour.py
  > Muse Spark/Wonder: panel spec JSON                 sewtrace/reason/panels.py
  > sewability check loop (max 3 calls)                sewtrace/reason/seams.py
  > tiled A4 PDF, SVG/DXF                              sewtrace/output/
  > verification: paper-on-tee photo, edge gap in mm   sewtrace/vision/verify.py
```

## Status

Work in progress. Until real captures exist, `sewtrace.mock.synth` renders
synthetic tees (known geometry, printed ArUco marker, camera tilt), and the
Meta API calls are mocked. See `docs/CONTRIBUTING.md`.

Read `docs/CAPTURE.md` before you take photos.

Put real captures in `data/raw/<tee>/` (`front.jpg`, `back.jpg`, `sleeve.jpg`). Write the tape
measure values in mm to `data/ground_truth/<tee>.json` (SPEC 10.3). `sewtrace eval data/raw`
stops with an error for a tee that has no such file.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env            # leave META_API_KEY empty to stay mocked
python -m sewtrace.mock.synth   # writes data/synth/tee1..tee5 (each with truth.json)
pytest
```

Real SAM calls need Meta's SAM decoder. Install it:

```bash
pip install -e '.[real]'
```

Start the test page with `sewtrace serve`. To deploy it, read `docs/DEPLOY.md`.

Record a demo with `sewtrace demo record <garment_dir>`. Play it with no network with `sewtrace demo play <name>`. Read `docs/DEMO.md`.

Run the Day 1 test on one photo. It makes two paid calls, so it needs the flag:

```bash
sewtrace day1 <photo> --yes-spend
```
