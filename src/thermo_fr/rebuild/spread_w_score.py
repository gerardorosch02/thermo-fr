"""Scoring of the test of W under amendment 6 (docs/prereg/spread_w_amendment_6.md, in force 2026-11-23).

The frozen scoring file, forecast/spread_w_test.py (sha256 998d40bf7dcc..., the hash amendment 2 registered; amendment 6
cites the older 98fb9981... of amendment 1 by mistake), is not edited. Editing it would change the code
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

A reason is one or more of six fixed texts joined by "; ". An hour with an issued forecast may carry only the price and
proxy texts. An hour without one carries "no forecast issued" or "weather not as issued two days ahead", and the price or
proxy text as well where that also applies, so a missing price is always named.

read() also carries stop condition 4 into the result, with the words "met under the exception of 2026-11-23", as the
exception requires, and refuses if the frozen scoring file is not the registered one.
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
NO_SPREAD, NO_PROXY = "no realised spread", "no German proxy"
PRICE_REASONS = tuple(f"{NO_SPREAD}: {which}" for which in ("no French price", "no German price", "no French or German price"))
NOT_ISSUED, NOT_STRICT = "no forecast issued", "weather not as issued two days ahead"
ISSUED_REASONS = (*PRICE_REASONS, NO_PROXY)  # why an hour with an issued forecast is not eligible
UNISSUED_REASONS = (NOT_ISSUED, NOT_STRICT)  # why an hour has no issued forecast
SEPARATOR = "; "
NOT_READABLE = "not readable"
FROZEN_SCORING_SHA256 = "998d40bf7dcc8ea3522bdd645abe489112b42c16fdcbcc3e2d481fa8772ddf3b"  # forecast/spread_w_test.py, as registered by amendment 2
EXCEPTION = "met under the exception of 2026-11-23"  # stop condition 4 (docs/prereg/spread_w_stop_condition_4_exception.md)
STOP_4 = "4_stored_forecasts_reproduce"
RECONSTRUCTION_FILE = "comparator_predictions.csv.gz"  # the reconstruction of 2026-11-23 that holds all seven models on the 21,430 hours
RECONSTRUCTION_SHA256 = "6d6a4de11cbb4442b0c4a9569f897144eef5d1a6131faf8658acf5d60b80ffdf"
STOP_4_REQUIRED = ("H", "G", "C", "W")


def stop_4_met(fourth: dict) -> str | None:
    """None if the record of stop condition 4 shows it met under the exception, else what is wrong with it. The read does
    not take `pass` on trust: it looks at what the pass rests on."""
    if not isinstance(fourth, dict) or fourth.get("pass") is not True or fourth.get("basis") != EXCEPTION:
        return "it is not recorded as met under the exception of 2026-11-23"
    if fourth.get("file") != RECONSTRUCTION_FILE or fourth.get("sha256") != RECONSTRUCTION_SHA256:
        return "the stored forecasts are not the reconstruction of 2026-11-23"
    machine = fourth.get("self_test_of_this_workspace") or {}
    if machine.get("pass") is not True or not machine.get("run_at_utc"):
        return "the self-test of the workspace was not run or did not pass"
    stored, compared, largest = fourth.get("stored_hours"), fourth.get("values_compared") or {}, fourth.get("largest_difference") or {}
    for name in STOP_4_REQUIRED:
        if not stored or compared.get(name) != stored:
            return f"{name} was not compared on every stored hour"
        if largest.get(name) is None or not largest[name] <= se.TOLERANCE:
            return f"{name} does not reproduce to {se.TOLERANCE}"
    return None


def reason_text(parts) -> str:
    return SEPARATOR.join(parts)


def _reason_ok(reason: str, issued: bool) -> bool:
    """One or more of the fixed texts, each at most once, at most one price text. With an issued forecast: price and proxy
    texts only. Without one: exactly one of the two texts for that, and the price or proxy text where it also applies."""
    parts = reason.split(SEPARATOR)
    if len(set(parts)) != len(parts) or sum(p in PRICE_REASONS for p in parts) > 1:
        return False
    unissued = [p for p in parts if p in UNISSUED_REASONS]
    return all(p in (*ISSUED_REASONS, *UNISSUED_REASONS) for p in parts) and len(unissued) == (0 if issued else 1)


NUMERIC = ("hour", "dow", "holiday", "de_holiday", "wind_px", "solar_px", "spread", *COMPARED)


def validate(frame: pd.DataFrame, first: str | None = None, last: str | None = None) -> None:
    """The seven models are scored on the same hours, each hour once, on days of the registered window. Yesterday's spread may
    be missing. Anything else is refused. The floor is not checked here: a frame under it is reported, marked not decisive."""
    first, last = first or swt.FIRST, last or swt.LAST
    missing = [c for c in (*COMMON, BASELINE) if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing: {missing}")
    holes = frame[list(COMMON)].isna().sum()
    if holes.any():
        raise ValueError(f"missing values, so the models would be scored on different hours: {holes[holes > 0].to_dict()}")
    if not len(frame):
        raise ValueError("the frame holds no hour")
    for column in (*NUMERIC, BASELINE):
        if not pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"`{column}` is not a number")
    values = frame[list(NUMERIC)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.isinf(frame[BASELINE].to_numpy(dtype=float)).any():
        raise ValueError("a value is infinite")
    swt._stamps(frame, "frame")
    twice = frame[frame.duplicated(["delivery_day", "hour"], keep=False)]
    if len(twice) and not (len(twice) == 2 and (twice["delivery_day"].iloc[0], int(twice["hour"].iloc[0])) == swt.REPEATED_HOUR):
        raise ValueError("an hour appears more than once")
    if frame["delivery_day"].min() < pd.Timestamp(first) or frame["delivery_day"].max() > pd.Timestamp(last):
        raise ValueError("the frame holds days outside the registered window")
    if not frame["provenance"].isin(swt.PROVENANCE).all() or (frame.groupby("delivery_day")["provenance"].nunique() != 1).any():
        raise ValueError("provenance must be one of saved, rebuilt_french_saved, rebuilt, and the same for every hour of a day")


def floor(frame: pd.DataFrame) -> dict:
    days, hours = int(frame["delivery_day"].nunique()), int(len(frame))
    return {"eligible_days": days, "eligible_hours": hours, "least_days": FLOOR_DAYS, "least_hours": FLOOR_HOURS, "readable": bool(days >= FLOOR_DAYS and hours >= FLOOR_HOURS)}


def _errors(part: pd.DataFrame) -> dict:
    """Hourly error of the seven models on the hours given. Yesterday's spread is never in this table: it is scored on its
    own hours, with the models beside it on the same hours, by _against_yesterday."""
    if not len(part):
        return {"days": 0, "hours": 0}
    return {"days": int(part["delivery_day"].nunique()), "hours": int(len(part)), **{m: float((part[m] - part["spread"]).abs().mean()) for m in COMPARED}}


def _against_yesterday(part: pd.DataFrame) -> dict:
    """Hourly error of yesterday's spread and of the seven models, both on the hours of `part` where yesterday's spread exists."""
    sub = part[part[BASELINE].notna()]
    if not len(sub):
        return {"days": 0, "hours": 0}
    return {"days": int(sub["delivery_day"].nunique()), "hours": int(len(sub)), **{m: float((sub[m] - sub["spread"]).abs().mean()) for m in (BASELINE, *COMPARED)}}


def check_schedule(frame: pd.DataFrame, schedule: pd.DataFrame, first: str | None = None, last: str | None = None, expected_hours: int | None = None) -> None:
    """The schedule holds every hour of the 100 registered days, the frame holds exactly its eligible hours, and every hour
    that is not eligible carries a reason of the amendment that fits whether it had an issued forecast."""
    first, last = first or swt.FIRST, last or swt.LAST
    expected_hours = swt.EXPECTED_HOURS if expected_hours is None else expected_hours
    missing = [c for c in swt.SCHEDULE_COLUMNS if c not in schedule.columns]
    if missing:
        raise ValueError(f"schedule columns missing: {missing}")
    if schedule[list(swt.SCHEDULE_COLUMNS)].isna().any().any():
        raise ValueError("the schedule must have no missing value")
    swt._stamps(schedule, "schedule")
    registered = pd.date_range(first, last, freq="D")
    if len(schedule) != expected_hours or not np.array_equal(np.sort(schedule["delivery_day"].unique()), registered.to_numpy()):
        raise ValueError(f"the schedule must hold the {expected_hours} hours of the {len(registered)} registered days")
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
    fits = np.array([_reason_ok(str(reason), bool(issued)) for reason, issued in zip(left_out["unscored_reason"], left_out["issued"])], dtype=bool)
    if not fits.all():
        raise ValueError(f"{int((~fits).sum())} hour(s) are left out for a reason the amendment does not allow: {sorted(left_out.loc[~fits, 'unscored_reason'].unique())}")


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
                     "mae_on_eligible_hours": _errors(part) if len(part) else None, "mae_on_hours_with_yesterdays_spread": _against_yesterday(part) if len(part) else None,
                     "as_of": swt.AS_OF.get(name, "all classes together")}
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
    parts = left_out["unscored_reason"].str.split(SEPARATOR).explode()
    return {"hours_left_out": int(len(left_out)), "by_reason": {str(k): int(v) for k, v in left_out["unscored_reason"].value_counts().sort_index().items()},
            "hours_by_single_reason": {str(k): int(v) for k, v in parts.value_counts().sort_index().items()},
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
        d["decisive"] = state["readable"]
        if not state["readable"]:  # under the floor nothing is decided: the figures stay, the verdicts and the three conditions do not
            for reading in (d["rule"], d["second_reading_small_slices_set_aside"]):
                reading["verdict"], reading["conditions"] = NOT_READABLE, None
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
        "mae": _against_yesterday(frame), "by_month": {m: _against_yesterday(part) for m, part in frame.groupby(month)},
        "by_class": {k: _against_yesterday(frame[frame["provenance"] == k]) for k in swt.PROVENANCE},
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
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=["delivery_day"], float_precision="round_trip")  # the numbers as written, to the last bit
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
    if se.sha256(Path(swt.__file__)) != FROZEN_SCORING_SHA256:
        raise SystemExit("forecast/spread_w_test.py is not the registered file, so the decisions would not be the registered ones")
    fourth = checks.get(STOP_4) or {}
    wrong = stop_4_met(fourth)
    if wrong:
        raise SystemExit(f"stop condition 4 under the exception of 2026-11-23: {wrong}")
    out = report(*load(frame_path, schedule_path))
    # The exception requires this to stand beside the decisions.
    out["stop_condition_4"] = {"statement": f"Stop condition 4 is {EXCEPTION}: the stored forecasts are the reconstructions of that day, not the files of the registered hashes.",
                               **{k: fourth.get(k) for k in ("basis", "file", "sha256", "is_the_reconstruction_of_2026_11_23", "self_test_of_this_workspace", "models", "stored_hours", "hours_compared", "values_compared", "largest_difference",
                                                             "within_tolerance", "tolerance", "required", "reported_not_required")},
                               "reported_models_that_did_not_reproduce": [m for m in fourth.get("reported_not_required") or [] if not (fourth.get("within_tolerance") or {}).get(m)]}
    return out
