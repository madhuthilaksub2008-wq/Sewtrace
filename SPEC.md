# Sewtrace build specification

Written in ASD-STE100 Simplified Technical English (Issue 9, January 2025).

| Item | Value |
| --- | --- |
| Document | SPEC-001 |
| Revision | 1.0 |
| Date | 2026-10-27 |
| Status | Released for build |
| Owner | Madhu (dev lead) |
| Event | Meta Global AI Developer Hackathon 2026, Indie Developer track |
| Theme | Push the Frontier (manufacturing) |

---

## 0. How to read this specification

- Each sentence has one meaning. Each instruction has one action.
- "Must" identifies a requirement. "Can" identifies a permitted option.
- Each requirement has an identifier, for example `REQ-SEG-01`. Tests, commits and pull requests refer to these identifiers.
- Technical names are permitted. Examples: homography, ArUco marker, SAM 3.1, DXF.
- All lengths are in millimetres (mm). All costs are in US dollars ($).
- A **WARNING** identifies a risk of injury. A **CAUTION** identifies a risk of damage to data, budget or results.
- Words in the glossary (section 16) have one meaning only.

---

## 1. Purpose

Sewtrace makes a 1:1 sewing pattern from three photos of a kid's t-shirt.

The pattern must agree with the original t-shirt within millimetres. The pattern must also sew together correctly.

Sewtrace changes a physical object into a file that a machine can cut. This is the "Push the Frontier" theme: software for manufacturing.

---

## 2. Problem and users

To copy a t-shirt by hand, a person must trace, measure and fit a pattern. This work takes hours. Most errors are on the curves: the armhole, the neckline and the sleeve cap.

| User | Need |
| --- | --- |
| Home sewist | Copy a favourite t-shirt in the next size |
| Parent | Make a new copy of a worn t-shirt that a child likes |
| Small tailor | Make a digital record of a garment |

---

## 3. Scope

### 3.1 In scope

- One garment type: a kid's round-neck t-shirt in knit fabric with set-in sleeves.
- Three photos on a flat surface with a printed 100 mm ArUco marker.
- Garment masks from SAM 3.1.
- Panel inference and seam adjustment from Muse Spark 1.3.
- A seam check in code.
- A tiled A4 PDF at 1:1, an SVG file and a DXF file.
- A proof step: a photo of the paper panel on the original t-shirt, with the edge gap in mm.

### 3.2 Out of scope

- CNC cutting. Sewtrace exports SVG and DXF only.
- Robot sewing. The demo shows this only as a future step.
- Other garment types, live video, user accounts and animated avatars.

### 3.3 Cut order if the schedule slips

Remove these items in this sequence:

1. Muse Image preview.
2. DXF export.
3. The t-shirt sewn by a tailor.
4. UI polish.

Do not remove these items: SAM masks, mm conversion, panel inference, seam check, overlay proof.

---

## 4. Meta models

### 4.1 Model rule

`REQ-MOD-01` The system must use only Meta models through the Meta Model API.

`REQ-MOD-02` Do not use models from other companies. Do not use other model hosts.

Reason: the hackathon gate requires Meta models. "Use of Meta AI tooling" is 25% of the score.

### 4.2 Models and prices

Base URL: `https://api.meta.ai/v1`. The API is compatible with the OpenAI SDK.

| Model | API model ID | Job in Sewtrace | Price | Status |
| --- | --- | --- | --- | --- |
| SAM 3.1 | `sam-3.1` (see note 1) | Masks for the t-shirt, the sleeve and the paper panel | $2.50 per 1,000 images | Required |
| Muse Spark 1.3 | `muse-spark-1.3` | Panel inference and seam adjust loop | $1.25 input, $0.15 cached input, $4.25 output, per 1M tokens | Required, default |
| Muse Spark 1.3 contributor | `muse-spark-1.3-contributor` | Same job, synthetic test data only (see 4.3, item 5) | $0.10 input, $0.20 output, per 1M tokens | Optional |
| Muse Wonder | `muse-wonder-1.0` | Same job as Spark, only if the Day 1 A/B test shows clearly better output | $4.00 input, $0.20 cached input, $20.00 output, per 1M tokens | Optional, preview |
| Muse Image | `muse-image-1.0` | Preview of the copy in a new fabric print | $0.01 per image | Optional, cut first |

Note 1: One source shows the SAM ID as `sam-3-1`. Confirm the ID in the Day 1 test (`REQ-API-09`).

Note 2: These IDs and prices come from public sources on 2026-10-27. Prices can change. The Meta pricing page is the authority.

### 4.3 Model facts that change the design

1. **The SAM 3.1 API accepts text prompts only.** It does not accept box prompts or point prompts. Use short text prompts, for example `t-shirt`, `sleeve`, `paper`.
2. **Muse Spark 1.3 makes many reasoning tokens.** Meta bills reasoning tokens as output tokens. Reasoning tokens also count against `max_tokens`. If `max_tokens` is too low, the JSON output stops before it is complete.
3. **Chat Completions does not keep reasoning between turns.** Each seam-loop call must send the full current state. A fixed prompt prefix also lets the prompt cache work (88% less cost on cached input).
4. **Muse Wonder is a preview model.** Its model card lists visual and spatial reasoning as a known limitation. Do not expect Wonder to correct geometry errors from the vision step.
5. **The contributor endpoint uses your data for training.** Use it only with synthetic photos. Do not send real photos to it.
6. **`reasoning_effort` controls cost.** Permitted values are `minimal`, `low`, `medium`, `high`, `xhigh` and `max`. The value `none` causes an HTTP 400 error.

---

## 5. Constraints

| ID | Constraint |
| --- | --- |
| `C-01` | The credit is $150 on Madhu's Meta Model API account. There is no hard cap. Paid charges start when the credit is used. |
| `C-02` | The credit is for the event window. After the window, the credit can stop. The live finals (Nov 9) must use cached results. |
| `C-03` | The submission deadline is Nov 4, 9:00 PM ET (6:00 PM PT). The team target is Nov 4, 12:00 PM PT. |
| `C-04` | All geometry is in mm. The x axis goes right. The y axis goes down. |
| `C-05` | An on-fold panel is stored as a half panel. The fold is on x = 0. |
| `C-06` | All Meta calls go through `MetaClient`. No other code calls the API. |
| `C-07` | API keys are in environment variables only. Do not put a key in the repository, a log or a cache file. |
| `C-08` | Tests and CI must not call the paid API. They use mocks or the cache. |

---

## 6. System overview

```
3 photos + 100 mm ArUco marker
  │
  ├─ 1. Marker detection + homography (px to mm)       sewtrace/vision/marker.py
  ├─ 2. SAM 3.1 masks (t-shirt, sleeve)                 sewtrace/api/client.py
  ├─ 3. Contours, smoothing, symmetry, edges (mm)       sewtrace/vision/contour.py
  ├─ 4. Six measurements                                sewtrace/vision/measure.py
  ├─ 5. Muse Spark 1.3: panel spec JSON                 sewtrace/reason/panels.py
  ├─ 6. Seam check loop (max 3 model calls + code fix)  sewtrace/reason/seams.py
  ├─ 7. Tiled A4 PDF, SVG, DXF                          sewtrace/output/
  └─ 8. Proof: paper-on-t-shirt photo, edge gap in mm   sewtrace/vision/verify.py
```

The models feed each other two times. SAM makes the first outline. Muse Spark adds the geometry that the camera cannot see. SAM then measures the printed result.

Stack: Python, OpenCV, shapely, pydantic, reportlab, svgwrite, ezdxf. The UI is FastAPI with one web page.

---

## 7. Requirements

Each requirement has an acceptance test. A requirement is complete only when its test passes.

### 7.1 Capture (CAP)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-CAP-01` | The system must accept three photos: front flat, back flat, one sleeve folded flat. | `sewtrace run <dir>` reads the three files by name. A missing file gives a clear error. |
| `REQ-CAP-02` | Each photo must show the full marker. | A photo with a partial marker is rejected. The message names the photo. |
| `REQ-CAP-03` | The sleeve photo must show the fold on the left and the cap at the top. | The capture page shows this rule with a diagram before the user takes the photo. |

### 7.2 Scale (SCL)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-SCL-01` | The system must find the marker corners to sub-pixel accuracy. | Unit test on synthetic photos with camera tilt. |
| `REQ-SCL-02` | The system must calculate a homography from pixels to mm. | The measured marker side is 100 mm ± 1.0 mm. Current synthetic worst case: 0.51 mm. |
| `REQ-SCL-03` | If the marker error is more than 1.0 mm, the system must reject the photo. | The message tells the user to take the photo again from directly above. |

### 7.3 Segmentation (SEG)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-SEG-01` | The system must use SAM 3.1 with text prompts only. | Code review. No box or point prompt is in the request. |
| `REQ-SEG-02` | Prompts: `t-shirt` for front and back, `sleeve` for the sleeve, `paper` and `t-shirt` for the proof photo. | Prompts are in config, not in code. |
| `REQ-SEG-03` | If SAM returns more than one mask, the system must use the mask with the highest score. | Unit test with a mocked response that has two masks. |
| `REQ-SEG-04` | If the best score is less than the config threshold, the system must stop and tell the user. | Unit test with a low-score mock. |

### 7.4 Geometry (GEO)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-GEO-01` | The system must change each mask into a smooth outline in mm. | Synthetic test: outline error is less than 1.0 mm. |
| `REQ-GEO-02` | The system must mirror the halves of front and back to make them symmetrical. | The left and right half differ by 0 mm after the mirror step. |
| `REQ-GEO-03` | The system must label each edge: shoulder, side, armhole, neckline, hem, fold, sleeve cap, underarm, sleeve hem. | Each panel edge has exactly one label. |
| `REQ-GEO-04` | The system must calculate six measurements: chest, body length, shoulder, sleeve length, neck width, armhole depth. | Definitions agree with section 10.2. |

### 7.5 Panel inference (RSN)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-RSN-01` | The system must send the photos, outlines and measurements to Muse Spark 1.3. The model must return a panel spec. | One real call returns a valid panel spec (Day 1). |
| `REQ-RSN-02` | The request must use structured output with the pydantic schema in `sewtrace/reason/panels.py`. | Schema validation passes on every response in the eval. |
| `REQ-RSN-03` | If the response is not valid, the system must try one more time. Then it must use the deterministic fallback. | Unit test with a bad mocked response. |
| `REQ-RSN-04` | The model must infer only what the photos cannot show: front neck depth, back neck depth, sleeve cap shape, grain line. | The prompt lists only these fields as model output. |
| `REQ-RSN-05` | The model must not change a measured value. | Code compares measured edges before and after. A difference of more than 0.5 mm fails the response. |
| `REQ-RSN-06` | `max_tokens` must be large enough for reasoning plus JSON. | No truncated JSON in 20 real calls. |

### 7.6 Seam check (SEW)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-SEW-01` | The system must measure each pair of seams that join. | All pairs in table 7.6.1 are measured. |
| `REQ-SEW-02` | If a pair is out of tolerance, the system must send the failure to the model. The maximum is 3 model calls. | Unit test with a mock that fails two times. |
| `REQ-SEW-03` | After 3 failed calls, the system must apply the deterministic code fix. | Unit test with a mock that always fails. |
| `REQ-SEW-04` | The system must not export a pattern that fails a seam pair. | The output step stops and shows the failed pair. |
| `REQ-SEW-05` | The system must record the number of calls and the result of each check. | The UI shows each pair change from red to green. |

#### 7.6.1 Seam pairs

| Pair | Rule | Tolerance |
| --- | --- | --- |
| Front shoulder and back shoulder | Equal length | ± 1.0 mm (proposed) |
| Front side seam and back side seam | Equal length | ± 1.0 mm (proposed) |
| Sleeve cap and armhole (front + back) | Cap is longer than armhole by the ease | 0 to 12 mm ease (open decision D-03) |
| Sleeve underarm seam (two sides) | Equal length | ± 1.0 mm (proposed) |

The tolerance values are in config. The values in config are the authority.

### 7.7 Output (OUT)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-OUT-01` | The system must make a tiled A4 PDF at 1:1. | Pages join with alignment marks. Overlap is shown. |
| `REQ-OUT-02` | Page 1 must show a 100 mm test square. | The printed square measures 100 mm ± 1 mm with a ruler. |
| `REQ-OUT-03` | Each panel must show the stitch line and the cut line. | The seam allowance is in config. Default: 10 mm. |
| `REQ-OUT-04` | Each panel must show its name, cut count, grain line, fold line and notches. | Visual check on the PDF. |
| `REQ-OUT-05` | The system must export SVG and DXF in mm. | The DXF opens in a CAD viewer at the correct size. |
| `REQ-OUT-06` | The system must give the sewing sequence as text. | The text has one step per seam. |

### 7.8 Proof (VER)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-VER-01` | The system must accept a photo of the paper panel on the original t-shirt with the marker. | Real photo test on Oct 31. |
| `REQ-VER-02` | SAM must mask the paper and the t-shirt. Code must calculate the mean edge gap in mm. | Synthetic test: known gap ± 0.5 mm. |
| `REQ-VER-03` | The overlay must show green where the gap is less than 3 mm. It must show red at all other points. | Visual check. |

### 7.9 API client (API)

| ID | Requirement | Acceptance test |
| --- | --- | --- |
| `REQ-API-01` | All Meta calls must go through `MetaClient`. | A test fails if other code imports the HTTP client. |
| `REQ-API-02` | `MetaClient` must cache each response. The cache key is the SHA-256 of image bytes + model ID + prompt + parameters. | The second identical call makes no network request. |
| `REQ-API-03` | `MetaClient` must log each call: time, model, input tokens, cached tokens, output tokens, images, cost in $. | A ledger file has one line for each call. |
| `REQ-API-04` | `MetaClient` must stop paid calls when the ledger total is more than the limit in config. | Unit test with a low limit. |
| `REQ-API-05` | Model IDs, prices and `reasoning_effort` must be in one config file. | A change of model needs no code change. |
| `REQ-API-06` | On HTTP 429 or 5xx, the client must try again with backoff. The maximum is 3 tries. | Unit test with a mocked 429. |
| `REQ-API-07` | Each call must have a timeout. | Unit test with a slow mock. |
| `REQ-API-08` | Offline mode must replay all steps from the cache with no network. | The demo runs with the network off. |
| `REQ-API-09` | The Day 1 test must make one real call to each model. It must record the model ID, image token count and cost. | The ledger cost agrees with the Usage page within 10%. |

---

## 8. Budget

### 8.1 Cost of one run (estimate)

These numbers are estimates. The token counts are assumptions. The Day 1 test (`REQ-API-09`) replaces them with measured values.

Assumptions: 3 photos at approximately 1,500 tokens each. One panel call: 6,000 input tokens, 15,000 output tokens. One seam call: 4,000 input tokens, 8,000 output tokens. Worst case: 3 seam calls.

| Step | Spark 1.3 (high) | Wonder (high) |
| --- | --- | --- |
| SAM 3.1, 6 images | $0.015 | $0.015 |
| Panel call | $0.07 | $0.32 |
| Seam calls, 3 (worst case) | $0.12 | $0.53 |
| **Total, worst case** | **≈ $0.21** | **≈ $0.87** |
| Target | under $0.50 | under $0.50 |

Result: Spark meets the target. Wonder at `high` effort does not meet the target. If the team uses Wonder, set `reasoning_effort` to `medium` or less, then measure again.

### 8.2 Allocation

| Bucket | Allowance |
| --- | --- |
| Day 1 tests (SAM, Spark, Wonder) | $5 |
| Development on real photos (Spark) | $35 |
| Wonder A/B test, 10 runs | $10 |
| Proof: 5 t-shirts × 3 runs | $5 |
| Demo and video takes | $10 |
| Reserve (do not plan to use) | $85 |
| **Total** | **$150** |

### 8.3 Checkpoints

| Date | If the ledger total is more than | Action |
| --- | --- | --- |
| Oct 29 | $40 | Use Spark only. Use cached results for all repeated runs. |
| Oct 31 | $75 | Use Spark only. Stop the Wonder A/B test. Use cached results for the demo. |
| Any day | $140 | Stop all paid calls. Use offline mode only. |

**CAUTION:** Paid charges start when the credit is used. There is no hard cap on the account. Do not disable the budget guard (`REQ-API-04`).

---

## 9. Procedures

### 9.1 Print the marker

**CAUTION:** If the printer changes the scale, all measurements are wrong. Set the scale to 100%.

1. Open the marker PDF.
2. Set the print scale to 100% ("Actual size").
3. Print the marker on matte paper.
4. Measure the black square with a steel ruler.
5. Make sure that the side is 100 mm ± 0.5 mm.
6. If the side is not correct, do steps 2 to 5 again.
7. Attach the marker to stiff card. Make sure that the card is flat.

### 9.2 Take the photos

1. Put the t-shirt on a flat surface with a colour that contrasts with the t-shirt.
2. Make the t-shirt flat with your hands. Remove all folds that are not part of the garment.
3. Put the marker on the surface, near the t-shirt. Do not put the marker on the t-shirt.
4. Hold the camera directly above the t-shirt.
5. Make sure that the full t-shirt and the full marker are in the frame.
6. Take the front photo.
7. Turn the t-shirt over. Do steps 2 to 5 again.
8. Take the back photo.
9. Fold one sleeve flat along its underarm line. Put the fold on the left and the cap at the top.
10. Do steps 3 to 5 again for the sleeve.
11. Take the sleeve photo.

### 9.3 Print the pattern

1. Open the pattern PDF.
2. Set the print scale to 100%.
3. Print page 1.
4. Measure the test square with a ruler.
5. Make sure that the side is 100 mm ± 1 mm.
6. If the side is correct, print all pages.
7. Join the pages at the alignment marks with tape.

### 9.4 Take the proof photo

1. Cut the paper panel on the stitch line.
2. Put the original t-shirt flat. Do steps 2 to 5 of procedure 9.2.
3. Put the paper panel on the t-shirt. Align the fold of the panel with the centre of the t-shirt.
4. Take the photo from directly above.
5. Run `sewtrace verify <photo>`.

---

## 10. Proof

### 10.1 Metrics

| Metric | How to measure | Target |
| --- | --- | --- |
| Measurement error | 6 points on each t-shirt, Sewtrace against tape measure, 5 t-shirts | Mean less than 5 mm |
| Overlay edge gap | Paper panel on original t-shirt, mean edge distance | Less than 3 mm |
| Sewability | Seam pairs in tolerance | All pass |
| Cost of one run | Ledger, checked against the Usage page | Less than $0.50 |
| Bonus | A tailor sews one copy. Put it next to the original. | It fits |

Current result on synthetic t-shirts: mean error 0.36 mm, all seams pass. Synthetic results are a best case. Real SAM edges, fabric that is not flat and lens distortion add error. The real result comes from 5 real t-shirts on Oct 31.

### 10.2 Measurement definitions

Use the same definitions in the tape measure and in `sewtrace/vision/measure.py`.

| Measurement | Definition |
| --- | --- |
| Chest | Straight across the front, 25 mm below the underarm points |
| Body length | From the high point of the shoulder to the hem, parallel to the centre line |
| Shoulder | From one shoulder seam point to the other, across the back |
| Sleeve length | From the shoulder seam point to the sleeve hem, along the fold |
| Neck width | Straight across the neck opening, from seam to seam |
| Armhole depth | From the shoulder seam point straight down to the underarm point |

### 10.3 Tape measure procedure

1. Put the t-shirt flat. Do steps 1 and 2 of procedure 9.2.
2. Measure each point in table 10.2 with a tape measure.
3. Write each value in mm in `data/ground_truth/<tee>.json`.
4. Measure each point again.
5. If the two values differ by more than 2 mm, measure a third time. Record the mean.

---

## 11. Quality gates

| Gate | Rule |
| --- | --- |
| Unit tests | `pytest` passes before each merge. |
| Requirement links | Each test names the requirement it tests. |
| Synthetic eval | `sewtrace eval` mean error is 0.5 mm or less. |
| Real eval | Mean error on 5 real t-shirts is less than 5 mm. |
| No paid calls in CI | CI runs with no API key. |
| Code review | A second reviewer checks each change before merge. |
| Code freeze | Nov 2. After this date, fix bugs only. |

---

## 12. Schedule

| Date | Milestone | Requirements |
| --- | --- | --- |
| Oct 28 | Day 1 tests pass. Credits claimed. T-shirts and marker ready. | `REQ-API-09` |
| Oct 29 | Full chain: photo to panel JSON on real photos | CAP, SCL, SEG, GEO, RSN |
| Oct 30 | Seam check loop and printable PDF | SEW, OUT |
| Oct 31 | Proof metric on screen, 5 t-shirts | VER, section 10 |
| Nov 1 | Demo sequence, UI, demo script | UI, offline mode |
| Nov 2 | **Code freeze.** Repository public. Form text drafted. | Section 11 |
| Nov 3 | Demo video recorded and uploaded | |
| Nov 4 | **Submit by 12:00 PM PT.** Hard deadline 6:00 PM PT. | |
| Nov 9 | Live finals, offline mode with cache | `REQ-API-08` |

---

## 13. Risks

| Risk | Effect | Control |
| --- | --- | --- |
| SAM edges are less accurate on real fabric | Measurement error is more than 5 mm | Contrasting surface. Smoothing. Report the real number honestly. |
| Fabric is not flat | Lengths are too short | Capture procedure 9.2, step 2. Two photos if the first is bad. |
| Lens distortion | Error at the frame edges | Keep the t-shirt in the centre. Fill most of the frame. |
| The model changes measured values | Pattern does not agree with the t-shirt | `REQ-RSN-05` |
| Truncated JSON from reasoning tokens | Invalid panel spec | `REQ-RSN-06`, `REQ-RSN-03` |
| Cost is more than the plan | Paid charges | `REQ-API-04`, checkpoints in 8.3 |
| API is not available on demo day | No live demo | `REQ-API-08` offline mode |
| Model ID or price changes | Calls fail or cost more | `REQ-API-05`, Day 1 test |

---

## 14. Open decisions

| ID | Decision | Owner | Decide by |
| --- | --- | --- | --- |
| D-01 | Spark or Wonder for panel inference, after the Day 1 A/B test | Madhu | Oct 28 |
| D-02 | Seam tolerance for shoulders, side seams and underarm, and the largest front-to-back difference the code fix may balance (now 4 mm, `seam.measured_pair_max_fix_mm`) | Madhu | Oct 30 |
| D-03 | Sleeve cap ease (now 0 to 12 mm) | Madhu | Oct 30 |
| D-04 | Seam allowance value (now 10 mm) | Madhu | Oct 30 |

---

## 15. Judging criteria and this build

| Criterion | Weight | Evidence in this build |
| --- | --- | --- |
| Impact | 30% | A parent copies a t-shirt in minutes. A tailor makes a digital record. |
| Use of Meta AI tooling | 25% | SAM 3.1 and Muse Spark feed each other two times. Without each model, the chain stops (section 6). |
| Demo | 25% | The edge gap overlay goes green. The error table for 5 t-shirts is on screen. |
| Technical depth | 20% | Sub-pixel homography, seam check loop, budget guard, measured proof. |

---

## 16. Glossary

| Term | Meaning in this specification |
| --- | --- |
| Panel | One piece of the pattern: front, back or sleeve |
| Half panel | The half of a symmetrical panel. The fold is on x = 0. |
| Stitch line | The line where the sewing machine sews |
| Cut line | The stitch line plus the seam allowance |
| Seam allowance | The fabric between the stitch line and the cut line |
| Ease | The quantity by which the sleeve cap is longer than the armhole |
| Seam pair | Two edges that are sewn together |
| Marker | The printed 100 mm ArUco square |
| Homography | The transform from photo pixels to mm on the flat surface |
| Mask | The SAM output that shows which pixels are the object |
| Edge gap | The distance in mm between the paper panel edge and the t-shirt edge |
| Ledger | The file that records the cost of each Meta call |
| Run | One full pass from three photos to pattern files |

---

## 17. Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
python -m sewtrace.cli synth             # 5 synthetic t-shirts in data/raw
python -m sewtrace.cli run data/raw/tee1 # pattern files in out/tee1
python -m sewtrace.cli eval              # out/eval/results.md
pytest
```

---

## 18. Sources (checked 2026-10-27)

- Meta Model API, models and pricing: https://developer.meta.com/ai/products/meta-model-api/
- Meta pricing and rate limits: https://dev.meta.ai/docs/pricing-rate-limits
- SAM 3.1 on the Meta Model API: https://learnopencv.com/sam-3-whats-new/
- Meta SAM cookbook: https://github.com/meta-models/meta-model-cookbook/tree/main/07_sam
- Muse Spark 1.3 pricing: https://www.eesel.ai/blog/muse-spark-1-3-pricing
- Muse Wonder pricing, IDs, limits: https://mer.vin/news/muse-wonder-metas-frontier-agent-model-pricing-and-benchmarks/
- Muse Image API: https://www.orcarouter.ai/blog/muse-image
- ASD-STE100 Issue 9: https://asd-ste100.org/about_STE.html

This specification follows the ASD-STE100 writing rules. It is not certified by ASD. Technical names from the software and sewing subject fields are used as technical nouns.
