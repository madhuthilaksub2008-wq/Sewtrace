# Contributing

## Rules

- All geometry is in **mm, x right, y down**. On-fold panels are stored as the
  half panel with the fold on `x = 0`. See `sewtrace/types.py`.
- Modules talk only through the types in `sewtrace/types.py` and the
  signatures in the stub files. Change a signature only after agreeing it.
- Every Meta call goes through `sewtrace.api.client.MetaClient`. Never call the
  API directly: the client caches by image hash and logs spend.
- Tests use synthetic tees (`sewtrace.mock.synth.make_tee`) and
  `sewtrace.mock.fixtures.truth_spec`, which gives the exact PanelSpec.
- Run `pytest` before pushing.

## Modules and owners

| Module | Board task | Owner |
| --- | --- | --- |
| `vision/marker.py` marker + homography | T14 | Bala |
| `vision/contour.py`, `vision/measure.py` | T15 | Madhu |
| `reason/panels.py` panel inference | T16 | Madhu |
| `reason/seams.py` seam check loop | T20 | Madhu |
| `output/pdf.py` tiled A4 PDF | T21 | Bala |
| `output/vector.py` SVG + DXF | T22 | Bala |
| `vision/verify.py` overlay proof | T25 | Madhu |

## Mock mode

With no `META_API_KEY` (or `SEWTRACE_MOCK=1`):

- `segment(photo, prompt)` serves `<photo>.masks/<prompt>.png` written by the
  synth, with about 1 px of edge jitter.
- `reason(prompt, images, mock=fn)` returns `fn()`, so each caller supplies its
  own deterministic stand-in answer.
