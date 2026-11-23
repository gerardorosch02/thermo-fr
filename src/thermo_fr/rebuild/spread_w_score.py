"""Scoring of the test of W under amendment 6 (docs/prereg/spread_w_amendment_6.md, in force 2026-11-23).

The frozen scoring file, forecast/spread_w_test.py (sha256 98fb99817b85...), is not edited. Editing it would change the code
hash of the morning logger, which covers the forecast package. This module takes its place for the one read. It reuses the
frozen functions for everything the amendment leaves alone, and replaces only the checks and the handling of yesterday's
spread:

  unchanged, imported   the three decisions, the gain rule, the Bonferroni level, the block bootstrap, the slices
                        (spread_w_test.decide, paired, daily_error), the window, the read date, the seal
  replaced here         which hours a frame may hold, the floor, the reasons an hour is left out, the figures that involve
                        yesterday's spread, and what is reported about excluded hours

The rule (amendment 6, sections 1 to 7):

  1. An hour is eligible with a realised spread, the German proxies and a forecast from each of H, HW, G, GK, GM, W and C.
     A missing spread of the previous day no longer makes an hour ineligible. The frame holds the eligible hours.
  2. A delivery day with at least one eligible hour is an eligible day.
  3. The test is read if at least 95 days and 2,280 hours are eligible, both. Otherwise the three decisions are recorded as
     "not readable" and the figures are reported marked as not decisive.
  4. The three decisions and every error of a model are computed on the common eligible hours.
  5. Yesterday's spread is scored on the eligible hours where it exists. Every figure that sets a model against it is
     computed on that subset for both sides, and the size of the subset is stated.
  6. No realised price comes from a second source. Nothing here reads one.
  7. The read reports every hour of the window that is not eligible, with its date, hour and reason, every day with no
     eligible hour, and the count of eligible hours without yesterday's spread.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..forecast import spread_eval as se
from ..forecast import spread_w_test as swt

FLOOR_DAYS, FLOOR_HOURS = 95, 2280
COMPARED = ("H", "HW", "G", "GK", "GM", "W", "C")  # scored on the same hours, all of them
BASELINE = "yesterday"  # scored where it exists
COMMON = ("delivery_day", "hour", "dow", "holiday", "de_holiday", "wind_px", "solar_px", "spread", "provenance", *COMPARED)
NO_SPREAD, NO_PROXY = "no realised spread", "no German proxy"  # why an hour with an issued forecast is not eligible
NOT_READABLE = "not readable"


def _reason_ok(reason: pd.Series) -> pd.Series:
    """A reason for leaving out an hour that has an issued forecast: no German proxy, or no realised spread, which may say
    after a colon which country's price is missing."""
    return (reason == NO_PROXY) | (reason == NO_SPREAD) | reason.str.startswith(NO_SPREAD + ":")


def validate(frame: pd.DataFrame) -> None:
    """The seven models are scored on the same hours, each hour once, on days of the registered window. Yesterday's spread may
    be missing. Anything else is refused. The floor is not checked here: a frame under it is reported, marked not decisive."""
    missing = [c for c in (*COMMON, BASELINE) if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing: {missing}")
    holes = frame[list(COMMON)].isna().sum()
    if holes.any():
        raise ValueError(f"missing values, so the models would be scored on different hours: {holes[holes > 0].to_dict()}")
    if not len(frame):
        raise ValueError("the frame holds no hour")
    swt._stamps(frame, "frame")
    twice = frame[frame.duplicated(["delivery_day", "hour"], keep=False)]
    if len(twice) and not (len(twice) == 2 and (twice["delivery_day"].iloc[0], int(twice["hour"].iloc[0])) == swt.REPEATED_HOUR):
        raise ValueError("an hour appears more than once")
    if frame["delivery_day"].min() < pd.Timestamp(swt.FIRST) or frame["delivery_day"].max() > pd.Timestamp(swt.LAST):
        raise ValueError("the frame holds days outside the registered window")
    if not frame["provenance"].isin(swt.PROVENANCE).all() or (frame.groupby("delivery_day")["provenance"].nunique() != 1).any():
        raise ValueError("provenance must be one of saved, rebuilt_french_saved, rebuilt, and the same for every hour of a day")


def floor(frame: pd.DataFrame) -> dict:
    days, hours = int(frame["delivery_day"].nunique()), int(len(frame))
    return {"eligible_days": days, "eligible_hours": hours, "least_days": FLOOR_DAYS, "least_hours": FLOOR_HOURS, "readable": bool(days >= FLOOR_DAYS and hours >= FLOOR_HOURS)}


def _errors(part: pd.DataFrame) -> dict:
    """Hourly error of the seven models on the hours given, and of yesterday's spread on those of them where it exists."""
    if not len(part):
        return {"days": 0, "hours": 0}
    with_baseline = part[part[BASELINE].notna()]
    return {"days": int(part["delivery_day"].nunique()), "hours": int(len(part)), **{m: float((part[m] - part["spread"]).abs().mean()) for m in COMPARED},
            BASELINE: float((with_baseline[BASELINE] - with_baseline["spread"]).abs().mean()) if len(with_baseline) else None, "hours_with_yesterdays_spread": int(len(with_baseline))}


def check_schedule(frame: pd.DataFrame, schedule: pd.DataFrame) -> None:
    """The schedule holds every hour of the 100 registered days, the frame holds exactly its eligible hours, and every hour
    with an issued forecast that is not eligible carries one of the reasons of the amendment."""
    missing = [c for c in swt.SCHEDULE_COLUMNS if c not in schedule.columns]
    if missing:
        raise ValueError(f"schedule columns missing: {missing}")
    if schedule[list(swt.SCHEDULE_COLUMNS)].isna().any().any():
        raise ValueError("the schedule must have no missing value")
    swt._stamps(schedule, "schedule")
    registered = pd.date_range(swt.FIRST, swt.LAST, freq="D")
    if len(schedule) != swt.EXPECTED_HOURS or not np.array_equal(np.sort(schedule["delivery_day"].unique()), registered.to_numpy()):
        raise ValueError(f"the schedule must hold the {swt.EXPECTED_HOURS} hours of the {swt.EXPECTED_DAYS} registered days")
    if schedule["issued"].dtype != bool:
        raise ValueError("`issued` must be true or false")
    if not schedule["provenance"].isin(swt.PROVENANCE).all() or (schedule.groupby("delivery_day")["provenance"].nunique() != 1).any():
        raise ValueError("the schedule's provenance must be one of saved, rebuilt_french_saved, rebuilt, and the same for every hour of a day")
    if not frame.index.isin(schedule.index).all():
        raise ValueError("an eligible hour is not in the schedule")
    if not schedule.loc[frame.index, "issued"].all() or (schedule.loc[frame.index, "provenance"].to_numpy() != frame["provenance"].to_numpy()).any():
        raise ValueError("an eligible hour has no issued forecast, or its provenance differs from the schedule")
    if (schedule.loc[frame.index, "unscored_reason"] != "").any():
        raise ValueError("an eligible hour carries a reason for being left out")
    left_out = schedule[~schedule.index.isin(frame.index)]
    if (left_out["unscored_reason"] == "").any():
        raise ValueError("an hour is left out without a reason")
    unexplained = left_out[left_out["issued"] & ~_reason_ok(left_out["unscored_reason"])]
    if len(unexplained):
        raise ValueError(f"{len(unexplained)} hour(s) with an issued forecast are left out for a reason the amendment does not allow: {sorted(unexplained['unscored_reason'].unique())}")


def availability(frame: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Scheduled, issued and eligible hours by provenance class, beside the error on the eligible hours."""
    check_schedule(frame, schedule)
    eligible = schedule.index.isin(frame.index)
    out = {}
    for name in (*swt.PROVENANCE, "all"):
        rows = np.ones(len(schedule), bool) if name == "all" else (schedule["provenance"] == name).to_numpy()
        part = frame if name == "all" else frame[frame["provenance"] == name]
        out[name] = {"days": int(schedule.loc[rows, "delivery_day"].nunique()), "scheduled_hours": int(rows.sum()), "issued_hours": int((rows & schedule["issued"].to_numpy()).sum()),
                     "eligible_hours": int((rows & eligible).sum()),
                     "left_out_by_reason": {str(k): int(v) for k, v in schedule.loc[rows & ~eligible, "unscored_reason"].value_counts().sort_index().items()},
                     "hours_with_an_input_left_missing": int((rows & (schedule["inputs_missing"].to_numpy() > 0)).sum()),
                     "mae_on_eligible_hours": _errors(part) if len(part) else None, "as_of": swt.AS_OF.get(name, "all classes together")}
    saved_days = schedule.loc[schedule["provenance"] == "saved", "delivery_day"]
    if len(saved_days):
        since = schedule[schedule["delivery_day"] >= saved_days.min()]
        out["since_the_first_saved_day"] = {"first_saved_day": f"{saved_days.min():%Y-%m-%d}", "days": int(since["delivery_day"].nunique()), "days_saved_before_the_auction": int(saved_days.nunique()),
                                            "share_of_days_saved": float(saved_days.nunique() / since["delivery_day"].nunique())}
    else:
        out["since_the_first_saved_day"] = {"first_saved_day": None, "days": 0, "days_saved_before_the_auction": 0, "share_of_days_saved": None}
    return out


def excluded(frame: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Section 7: every hour of the window that is not eligible, by date and reason, and every day with no eligible hour."""
    left_out = schedule[~schedule.index.isin(frame.index)]
    by_day = {}
    for (day, reason), part in left_out.groupby(["delivery_day", "unscored_reason"]):
        by_day.setdefault(f"{day:%Y-%m-%d}", {})[str(reason)] = [t.strftime("%Y-%m-%dT%H:%MZ") for t in part.index.sort_values()]
    whole = sorted(set(schedule["delivery_day"]) - set(frame["delivery_day"]))
    return {"hours_left_out": int(len(left_out)), "by_reason": {str(k): int(v) for k, v in left_out["unscored_reason"].value_counts().sort_index().items()},
            "days_with_no_eligible_hour": {f"{d:%Y-%m-%d}": sorted(by_day[f"{d:%Y-%m-%d}"]) for d in whole}, "hours_by_day_and_reason_utc": by_day,
            "eligible_hours_without_yesterdays_spread": int(frame[BASELINE].isna().sum()),
            "days_with_an_eligible_hour_without_yesterdays_spread": sorted(f"{d:%Y-%m-%d}" for d in frame.loc[frame[BASELINE].isna(), "delivery_day"].unique())}


def report(frame: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Everything the plan says is reported. The decisions are in `decisions`; the rest is never part of them."""
    validate(frame)
    hours = availability(frame, schedule)
    state = floor(frame)
    common = frame[list(COMMON)]  # the decisions never see yesterday's spread
    decisions = []
    for candidate, base, question in swt.DECISIONS:
        d = {"question": question, **swt.decide(common, candidate, base)}
        d["outcome"] = d["rule"]["verdict"] if state["readable"] else NOT_READABLE
        if not state["readable"]:
            d["rule"]["verdict"] = d["second_reading_small_slices_set_aside"]["verdict"] = NOT_READABLE
        decisions.append(d)
    month = frame["delivery_day"].dt.strftime("%Y-%m")
    days = frame.groupby("delivery_day")["provenance"].first()
    daily = frame.groupby("delivery_day")[["spread", *COMPARED]].mean()
    with_baseline = frame[frame[BASELINE].notna()]  # both sides of every figure against yesterday's spread are taken from these hours
    base = with_baseline.groupby("delivery_day")[["spread", BASELINE, *COMPARED]].mean()
    move = base["spread"] - base[BASELINE]
    big = move.abs() > se.MOVE
    against_yesterday = {
        "hours": int(len(with_baseline)), "days": int(with_baseline["delivery_day"].nunique()),
        "mae": {m: float((with_baseline[m] - with_baseline["spread"]).abs().mean()) for m in (BASELINE, *COMPARED)} if len(with_baseline) else None,
        "base_spread_mae": {m: float((base[m] - base["spread"]).abs().mean()) for m in (BASELINE, *COMPARED)} if len(base) else None,
        "direction_right": {m: float((np.sign(base.loc[big, m] - base.loc[big, BASELINE]) == np.sign(move[big])).mean()) if big.any() else None for m in COMPARED},
        "days_with_move_above_5": int(big.sum()),
        "W against yesterday": swt.paired(with_baseline, "W", BASELINE) if with_baseline["delivery_day"].nunique() >= 2 else None,
        "note": "Computed on the eligible hours where the spread of the previous day exists, for both sides."}
    return {"window": [swt.FIRST, swt.LAST], "hours": state["eligible_hours"], "days": state["eligible_days"], "expected": {"days": swt.EXPECTED_DAYS, "hours": swt.EXPECTED_HOURS},
            "floor": state, "status": "read" if state["readable"] else "below the floor: no decision is drawn and every figure here is not decisive",
            "decisions": decisions,
            "mae": _errors(frame), "by_month": {m: _errors(part) for m, part in frame.groupby(month)},
            "provenance": {"days": {k: int((days == k).sum()) for k in swt.PROVENANCE},
                           "errors": {k: _errors(frame[frame["provenance"] == k]) for k in swt.PROVENANCE},
                           "primary_comparison_by_class": {k: swt.paired(common[common["provenance"] == k], "W", "HW") for k in swt.PROVENANCE if (days == k).sum() >= 14},
                           "note": "The classes are different calendar periods: rebuilt days come first, saved days last. A difference between them is not the effect of provenance alone."},
            "availability": hours, "excluded": excluded(frame, schedule),
            "not_attainable": ["C"],
            "base_spread_mae": {m: float((daily[m] - daily["spread"]).abs().mean()) for m in COMPARED},
            "against_yesterdays_spread": against_yesterday,
            "second_source": "No realised price comes from a second source (amendment 6, section 6). No second set of figures is computed (owner, 2026-11-23).",
            "reported_only": {"HW against H": swt.paired(common, "HW", "H"), "GK against G (weekly refits alone)": swt.paired(common, "GK", "G"),
                              "GM against G (weights alone)": swt.paired(common, "GM", "G"), "C against W": swt.paired(common, "C", "W")}}


def load(frame_path: Path, schedule_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=["delivery_day"])
    schedule = pd.read_csv(schedule_path, index_col=0, parse_dates=["delivery_day"], keep_default_na=False)
    frame.index, schedule.index = pd.to_datetime(frame.index, utc=True), pd.to_datetime(schedule.index, utc=True)
    return frame, schedule


def read(frame_path: Path, schedule_path: Path, stop_checks: Path) -> dict:
    """The one read. Refuses before the read date (Paris date of the machine's clock), without the passed stop conditions of
    this test and window, or on forecasts other than the checked ones."""
    if swt._today() < pd.Timestamp(swt.READ_FROM):
        raise SystemExit(f"sealed until {swt.READ_FROM}")
    checks = json.loads(Path(stop_checks).read_text())
    if checks.get("test") != swt.TEST_NAME or checks.get("window") != [swt.FIRST, swt.LAST]:
        raise SystemExit("the stop-condition record is not that of this test and window")
    if not checks.get("all_pass"):
        raise SystemExit("a stop condition failed or was not run: nothing is scored")
    if se.sha256(frame_path) != checks.get("test_predictions_sha256") or se.sha256(schedule_path) != checks.get("schedule_sha256"):
        raise SystemExit("the forecasts on disk are not the ones the stop conditions were run on")
    return report(*load(frame_path, schedule_path))
