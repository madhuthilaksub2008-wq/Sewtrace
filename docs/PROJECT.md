# Sewtrace: what we are building

> Lay a kid's t-shirt flat, take three photos, and Sewtrace prints a 1:1 sewing
> pattern that matches the original within millimetres and is checked to sew
> together.

Meta Global AI Developer Hackathon 2026, Indie Developer track, theme *Push the
Frontier* (pending confirmation): turn a physical object into a manufacturable
file. Team: Madhu (dev lead, on camera, submits) and Bala (marker, PDF and
DXF modules; capture, proof, script, video).

## Key dates

| Date | Milestone |
| --- | --- |
| Oct 28 | Day 1 tests pass; credits claimed; tees and marker ready |
| Oct 29 | Chain end to end: photo to panel JSON |
| Oct 30 | Seam check loop + printable PDF |
| Oct 31 | Proof metric on screen (5 tees) |
| Nov 1 | Wow polish, UI, demo script |
| Nov 2 | **Code freeze**, repo public, form texts drafted |
| Nov 3 | Demo video recorded and uploaded |
| Nov 4 | **Submit by noon PT** (hard deadline 6:00 PM PT) |
| Nov 9 | Live finals |

## The problem

Copying a worn favourite in the next size, or digitising a garment, means
drafting a pattern by hand: hours of tracing, measuring and trial fitting. The
curves (armhole, neckline, sleeve cap) are where people get it wrong.

**Users:** home sewists and parents; small tailors digitising garments.

## What it does

1. **Capture.** Three photos on a flat, contrasting surface with a printed
   100 mm ArUco marker in frame: front flat, back flat, one sleeve folded flat
   (fold on the left, cap at the top).
2. **See.** SAM 3.1 segments the garment and its regions (body, sleeve).
3. **Measure.** The marker gives a homography from pixels to mm; masks become
   smooth outlines, halves are mirrored for symmetry, edges are labelled.
4. **Reason.** Muse Spark (or Wonder, after the Day 1 cost test) infers what the
   photos can't show directly: front vs back neck depth, sleeve cap shape,
   grain. It returns a structured panel spec.
5. **Check.** Seams that must join (shoulders, side seams, sleeve cap vs
   armhole) are measured. If they are out of tolerance the model adjusts, up to
   3 calls, then a deterministic code fix.
6. **Output.** A tiled A4 PDF at 1:1 with a 100 mm test square, SVG/DXF cut
   files, and the sewing order.
7. **Verify.** Photograph the printed paper panel laid on the original tee.
   SAM masks both, code measures the edge gap in mm, and an overlay shows green
   (under 3 mm) or red.

**Hero object:** a kid's round neck t-shirt (knit, set-in sleeves).

## Why only Meta

- Without **SAM 3.1** there are no garment outlines, so no measurements and no
  verification.
- Without **Muse Spark/Wonder** there is no panel inference: no hidden geometry
  and no adjust loop.
- The two feed each other twice: SAM drafts, the model reasons, SAM verifies.

| Product | Role | Status |
| --- | --- | --- |
| SAM 3.1 | Garment and region masks; paper-on-tee verification masks | Mocked; real call pending Day 1 test |
| Muse Spark 1.3 | Panel inference and seam adjust loop (default, cheaper) | Mocked; real call pending Day 1 test |
| Muse Wonder | Same job, if the Day 1 A/B shows clearly better output | Pending |
| Muse Image | Optional: preview of the copy in a new fabric print | Cut first |

## Pipeline

```
3 photos + marker
  > marker detection + homography (px to mm)          sewtrace/vision/marker.py
  > SAM 3.1 masks (body, sleeve)                       sewtrace/api/client.py
  > contours, smoothing, symmetry, edges (mm)          sewtrace/vision/contour.py
  > six measurements                                   sewtrace/vision/measure.py
  > Muse Spark/Wonder: panel spec JSON                 sewtrace/reason/panels.py
  > sewability check loop (max 3 calls + code fix)     sewtrace/reason/seams.py
  > tiled A4 PDF, SVG, DXF                             sewtrace/output/
  > verification: paper-on-tee photo, edge gap in mm   sewtrace/vision/verify.py
```

**Stack:** Python, OpenCV, shapely, pydantic, reportlab, svgwrite, ezdxf;
FastAPI + a simple web page for the UI (not built yet).

**Conventions:** all geometry in mm, x right, y down. On-fold panels are stored
as the half panel with the fold on x = 0. Every Meta call goes through
`MetaClient`, which caches by image hash and logs spend.

## Proof: the numbers on screen

| Metric | How measured | Target |
| --- | --- | --- |
| Measurement error | 6 points per tee (chest, body length, shoulder, sleeve length, neck width, armhole depth) vs tape measure, 5 tees | mean under 5 mm |
| Overlay edge gap | paper panel on original tee, mean edge distance | under 3 mm |
| Sewability | joined seam pairs within tolerance | all pass |
| Cost per run | Usage page | under \$0.50 |
| Bonus | a tailor sews one copy, laid next to the original | fits |

## Hard problems (named in the demo)

| Problem | Approach |
| --- | --- |
| Masks into trustworthy millimetres | printed marker, sub-pixel homography, 100 mm test square |
| Geometry the camera can't see | separate back and sleeve photos; the model infers neck depth and cap shape |
| A pattern that actually sews | seam lengths checked in code; the model adjusts until joined seams match |

## Demo (3 minutes)

| Time | On screen |
| --- | --- |
| 0:00 to 0:30 | tee laid flat > outlines snap on > panels lift into a printed pattern. "Three photos. One sewing pattern." |
| 0:30 to 1:00 | parent holding a worn favourite that no longer fits |
| 1:00 to 1:50 | SAM masks > mm > model infers sleeve cap > seam check turns green |
| 1:50 to 2:30 | paper panel on the tee, edges go green; error table for 5 tees |
| 2:30 to 2:50 | (bonus) sewn copy next to original |
| 2:50 to 3:00 | scale line: photo to cut file to automated cutting and sewing |

Live finals backup: a recorded run with cached results for every step.

## Out of scope

- CNC cutting (SVG/DXF export only), robot stitching (scale line only).
- Multiple garment types, live video, user accounts, animated avatar.

**Cut first if slipping:** Muse Image preview, DXF export, tailor-sewn copy, UI
polish. **Keep real:** SAM masks, mm conversion, panel inference, seam check,
overlay proof.

## Credit budget

\$150 of free credits on Madhu's Meta Model API account. There is no hard cap:
paid charges start when credits run out. Develop on Spark, cache every call,
images and JSON only. Checkpoints: over \$40 by Oct 29 or over \$75 by Oct 31
means Spark only plus cached results.

## Where we are (Oct 29)

Built and tested on **synthetic** tees (`sewtrace/mock/synth.py` renders
known geometry with the marker and camera tilt; SAM and the model are mocked):

| Board task | Status |
| --- | --- |
| T14 marker + homography | done, worst error 0.51 mm |
| T15 contours, edges, measurements | done |
| T16 panel inference (prompt, schema, validation, fallback) | done, mocked model |
| T19 pipeline `sewtrace run <dir>` | done |
| T20 seam check loop | done |
| T21 tiled A4 PDF | done |
| T22 SVG + DXF | done |
| T24/T25 proof photo + overlay gap | done on synthetic proof photos |
| T26.1 eval `sewtrace eval` | done: mean error 0.36 mm, all seams pass |
| T8/T9 real SAM and Muse calls | **next**: needs API access (Day 1 tests) |
| T23/T29/T30 web UI, wow sequence, offline mode | done, mock-tested (demo bundles replay offline) |
| Real captures and tape-measure ground truth | waiting on tees and marker |

**Status, Oct 29:** the whole pipeline is built and tested on synthetic tees
(mock mode, no paid call yet): capture check, scale, contours, measurements,
panel inference, seam loop, A4 PDF, SVG and DXF, proof overlay, web page, demo
replay, eval (mean error 0.36 mm, all seams pass). CI runs the tests with no key.

**Not met yet:**

- `REQ-RSN-01`, `REQ-RSN-06`, `REQ-API-09`: need the paid Day 1 run (one real call to each model, token and cost record).
- `REQ-VER-01`: the real proof photo is due on Oct 31.
- `REQ-SEG-03`, `REQ-SEG-04`: partly. No SAM score field has been seen yet, so the score rule is untested on a real answer.

**Values that need your confirmation** (marked in `sewtrace/config.toml`):

- NEEDS USER CONFIRMATION: `[inference]` `front_minus_back_min_mm`, `fallback_front_minus_back_mm`; `[proof]` `neck_gap_tolerance_mm`, `fold_bow_limit_frac`.
- NEEDS REAL PHOTOS: `[contour.joints]` `corner_window_mm`, `pair_tol_hem_mm`, `pair_tol_neck_mm`, `shoulder_plateau_mm`; `[contour.labels]` `corner_window_mm`.

The synthetic numbers are a best case. Real SAM edges, fabric that isn't
perfectly flat and lens distortion will add error; the real proof number comes
from the 5 real tees on Oct 31.

**Open decisions:** confirm the theme; Spark vs Wonder after the Day 1 cost
test; the allowed sleeve cap ease (currently 0 to 12 mm longer than the
armhole).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
python -m sewtrace.cli synth             # 5 synthetic tees in data/synth
python -m sewtrace.cli run data/synth/tee1 # pattern files in out/tee1
python -m sewtrace.cli eval              # out/eval/results.md
pytest
```
