"""Where each value of a pattern came from, read off the data a run already has.

The labels are for the user. They answer "can I trust this number?" (photo, average, own tape,
model, default, co-pilot edit). The state starts from the panel inference (did the model answer
or the fallback?) and then each row of the seam log changes it in order, so a later change wins.
Pure functions: no file or API access.
"""

from __future__ import annotations

from sewtrace.types import MEASUREMENT_KEYS

PHOTO = "from photo"
AVERAGED = "averaged front/back"
USER = "your tape"
MODEL = "inferred by Muse"
DRAFTED = "drafted"
EDIT = "co-pilot edit"

# Values that are not one of the six measurements but are inferred or drafted.
INFERRED_KEYS = ("front_neck_depth", "back_neck_depth", "sleeve_cap")
# A seam the front and back photos both measure, and the measurement it sets. The six
# measurements come from one photo each; the pair fix or the user's tape changes the pattern
# edge, so the label follows the edge.
PAIR_MEASUREMENT = {"shoulder": "shoulder_width", "side": "chest_width"}
# What a co-pilot edit changes. A scale changes all six, and the inferred values the run has.
EDIT_KEYS = {"body_length": ("body_length",), "chest_width": ("chest_width",),
             "sleeve_length": ("sleeve_length",), "overall_scale_pct": (*MEASUREMENT_KEYS, *INFERRED_KEYS)}


def initial_provenance(inference: str) -> dict[str, str]:
    """Labels before the seam loop. ``inference`` is "model" when the model answer was accepted and
    "fallback" when the draft with the default front neck depth was used (REQ-RSN-03). Any other
    value (a run from before this field existed) leaves the three inferred values out."""
    prov = {key: PHOTO for key in MEASUREMENT_KEYS}
    if inference == "model":
        prov.update(front_neck_depth=MODEL, back_neck_depth=MODEL, sleeve_cap=MODEL)
    elif inference == "fallback":
        # The front outline follows the back neckline, so its depth is a default, not a measurement.
        prov.update(front_neck_depth=DRAFTED, back_neck_depth=PHOTO, sleeve_cap=PHOTO)
    return prov


def build_provenance(seam_log: list[dict], inference: str | None, start: dict[str, str] | None = None) -> dict[str, str]:
    """Labels after the rows of ``seam_log``. Pass ``start`` (the labels so far) with only the new rows
    to continue an earlier run; otherwise the labels start from ``inference``."""
    prov = dict(start) if start is not None else initial_provenance(inference or "")
    for row in seam_log:
        _apply_row(prov, row)
    return prov


def edge_label(source: str) -> str:
    """Label of a shoulder or side length that the pair fix or the user set ("average" or "user")."""
    return USER if source == "user" else AVERAGED


def _apply_row(prov: dict[str, str], row: dict) -> None:
    source, applied = row.get("source"), row.get("applied") or []
    if source in ("pair_fix", "user"):
        for change in applied:
            key = PAIR_MEASUREMENT.get(change.get("edge", ""))
            if key and prov.get(key) != EDIT:  # after a co-pilot edit, a rebalance keeps the edit's label
                prov[key] = USER if source == "user" else AVERAGED
    elif source in ("model", "fallback") and "sleeve_cap" in prov:
        # A cap change by the model is the model's; the code fix (handles scaled, or the straight
        # top edge of a flat sleeve) is a draft.
        for change in applied:
            if isinstance(change, dict) and (change.get("sleeve_cap") or change.get("flat_cap")):
                prov["sleeve_cap"] = MODEL if source == "model" else DRAFTED
    elif source == "edit":
        for change in applied:
            for key in EDIT_KEYS.get(change.get("edit", ""), ()):
                if key in MEASUREMENT_KEYS or key in prov:  # an old folder's inferred values stay unlabelled
                    prov[key] = EDIT
