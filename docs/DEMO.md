# Demo mode

The demo shows a recorded run. It needs no network, no API key and no model call. Use it at the live finals (SPEC C-02, `REQ-API-08`).

**WARNING:** `sewtrace demo record` in REAL mode makes paid calls. It prints the worst-case cost and stops until you add `--yes-spend`. The budget guard and the ledger apply as for any run.

## What a bundle is

A bundle is the folder `demo/<name>/`. It holds:

- the three photos, and `proof.jpg` if the garment folder had one
- the run files: `spec.json`, `measurements.json`, `seam_log.json`, `pattern.pdf`, `pattern.svg`, `pattern.dxf`, `preview.png`
- `masks.json`: the SAM mask outlines in px and in mm, with the homography of each photo
- `placement.json`: where each panel lies on its photo, and where it lies on the pattern sheet
- `proof_result.json` and `proof_overlay.jpg`, if there is a proof photo
- `eval.json`: the error table of the last eval, if there is one
- `cache/`: the cache files that the run used
- `manifest.json`: the mode, the file list and the cache keys

A bundle never holds the API key or the ledger. A test checks this.

## Rule for git

| Bundle | In git? | Why |
| --- | --- | --- |
| Synthetic (`demo/tee1`) | Yes | It is small (about 3.5 MB) and has no private data. |
| Real photos | Yes, only after the checks below | The finals laptop needs it, and the repository is public from Nov 2. |

Before you commit a real bundle:

1. Note: `sewtrace demo record` stops if a photo has GPS data in its EXIF header. It stops before any call and names the photo. Turn off location tags in the camera app and take the photo again. Use `--allow-gps` only for a bundle that you will never commit.
2. Make sure that no person or face is in a photo.
3. Run `du -sh demo/<name>`. Keep the bundle under 25 MB. If it is larger, take the photos at a lower resolution and record again.

The ledger `data/ledger.csv` and the folder `cache/` in the repository root stay out of git. The folder `demo/<name>/cache/` is part of the bundle and goes into git. A test checks this. Copy the bundle to a USB stick as a second backup.

## Record the demo before Nov 3

1. Check the budget.
2. Read the ledger total. Stop if it is near a checkpoint in SPEC 8.3.
3. Make the error table for the end card. The card names its data source.
4. For a synthetic bundle, run the synthetic eval. It costs nothing in mock mode: `python -m sewtrace.cli eval`. The card says "Synthetic t-shirts (best case)".
5. For a real bundle, get approval first. The eval makes paid calls in REAL mode.
6. Run the eval on the real tees with their tape measure files in `data/ground_truth/`: `python -m sewtrace.cli eval data/raw`. The card says "Real t-shirts".
7. Note: a REAL recording stops, before any call, if the table holds only synthetic tees.
8. To show the synthetic table anyway, add `--allow-synthetic-eval`. The card then says that the tees are synthetic.
9. Put the proof photo in the garment folder as `proof.jpg`.
10. Record in mock mode first. This is free: `python -m sewtrace.cli demo record data/synth/tee1`.
11. Open the page: `python -m sewtrace.cli demo play tee1`.
12. Watch the whole demo.
13. For real photos, ask the project owner to approve the cost.
14. Run the command with the worst-case cost shown: `python -m sewtrace.cli demo record data/raw/tee1 --name real1 --yes-spend`.
15. Run it once. If a call fails, read the error before you run it again.
16. Note: the cache keeps every answer that was paid for. A second run does not pay twice.
17. Use `--force` to replace a bundle of the same name. Without it, the command stops.
18. Note: a second recording of the same photos does not change a file that has the same content.
19. Commit the bundle if the checks above pass.

## Play the demo at the finals

1. Turn off the network, or do not connect to it.
2. Run: `SEWTRACE_OFFLINE=1 python -m sewtrace.cli demo play real1`
3. Open the address that the command prints, for example `http://127.0.0.1:8000/demo?name=real1`.
4. Press the Play button, or press the space key.

The command removes the key from the process. The page cannot make a model call. If a file is missing, the command stops and names the file.

### Controls

| Control | Key | Action |
| --- | --- | --- |
| Play and Pause | Space | Start or stop the animation |
| Step | Right arrow | Go to the start of the next step |
| Back | Left arrow | Go to the start of this step or the last step |
| Restart | | Go to the start and play |
| Speed | | 0.5x, 1x, 1.5x or 2x |
| Position bar | | Go to any time |
| Hide controls | H | Remove the control bar, for a screen recording |

Add `?autoplay=1` to the address to start at once. Add `&clean=1` to hide the controls.

### The six steps

1. SAM outline: the mask outline draws on for the front, back and sleeve photos.
2. Measurements: the six values appear in mm.
3. Panels lift: each panel moves from its place on the photo to its place on the pattern sheet.
4. Seam check: each seam pair shows red or green. The page follows `seam_log.json` check by check and shows the number of model calls. When the first check passes, the pairs go from grey to green.
5. Proof: the edge of the paper panel turns green where the edge gap is under the tolerance. The page shows the mean gap.
6. End card: the error table of the last eval, with its data source (synthetic or real t-shirts).

The page shows the whole run in about 60 seconds at 1x speed. Use 0.5x to fit a longer voice-over.

## Replay the pipeline from the bundle

A bundle is also a garment folder, and `cache/` holds every answer that the run used. This replays the full pipeline with no network:

```bash
SEWTRACE_CACHE_DIR=demo/real1/cache SEWTRACE_OFFLINE=1 python -m sewtrace.cli run demo/real1 --out /tmp/replay
```

Add `SEWTRACE_MOCK=1` for a bundle that `manifest.json` lists as `mock`. The run makes zero calls. A cache miss stops the run with a clear error.

## Pre-finals checklist

Do this the day before, and again on the finals laptop.

1. Pull the repository.
2. Check that `demo/<name>/manifest.json` exists.
3. Run `python -m sewtrace.cli demo play <name> --check`.
4. Make sure it prints one line and no error.
5. Turn off Wi-Fi and unplug the network cable.
6. Unset the key: `unset META_API_KEY`.
7. Run `SEWTRACE_OFFLINE=1 python -m sewtrace.cli demo play <name>`.
8. Open the page and play all six steps.
9. Look for a broken image or a blank frame.
10. Run the pipeline replay from the section above.
11. Check that it prints the six measurements.
12. Set the screen to 1920 x 1080.
13. Press H to hide the controls.
14. Check that the text is readable.
15. Keep a screen recording of the full demo as the last backup.
16. Turn the network on again only after the test is done.
