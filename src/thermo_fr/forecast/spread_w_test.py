"""Scoring of the test of W (docs/prereg/spread_w_prereg.md with amendment 1). Scoring only: it reads a frame of scored hours.

Window 2026-10-11 to 2027-01-18, one read on or after 2027-01-19. Three decisions, each by the project's gain rule with
Bonferroni for three:
  1. W against HW   primary: do the German inputs add forecast value when both models are trained the same way?
  2. W against H    is W better than the French-only model as first frozen?
  3. W against G    did the redesign help?

The frame holds one row per scored hour with: delivery_day, hour, dow, holiday, de_holiday, wind_px, solar_px, spread,
yesterday, H, HW, G, GK, GM, W, C, and provenance, one of
  saved                 the forecasts of every comparator were written by the spread logger before the auction
  rebuilt_french_saved  rebuilt afterwards, with the French inputs taken from a window saved before 10:05 Paris on the eve
  rebuilt               rebuilt afterwards from the refetch
(docs/prereg/spread_w_refetch_spec.md). C is never attainable at 10:05 and is rebuilt on every day.

report() also takes the schedule: one row per hour of the 100 registered days, scored or not, with delivery_day, provenance,
issued (every comparator has a forecast for the hour) and inputs_missing (how many inputs of G were unavailable for the hour and
stayed missing). With it the result shows, for each provenance class, the scheduled, issued and scored hours beside the error
on the common scored hours, and says whether a later revision could have been used (amendment 2).

The seal is a guard, not a lock: read() refuses before READ_FROM, but report() and decide() are plain functions. The seal
that matters is that the frame of the window is not assembled before the read date.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import spread_eval as se
from .shape import block_bootstrap_mean
from .timing import delivery_days

FIRST, LAST = "2026-10-11", "2027-01-18"
READ_FROM = "2027-01-19"
EXPECTED_DAYS, EXPECTED_HOURS = 100, 2401  # 2026-10-25 has 25 hours
DECISIONS = (("W", "HW", "primary: do the German inputs add forecast value under matched training?"),
             ("W", "H", "is W better than the French-only model as first frozen?"),
             ("W", "G", "did the redesign help?"))
ONE_SIDED = 1 - 0.05 / len(DECISIONS)
MIN_GAIN, MAX_SLICE_LOSS = se.MIN_GAIN, se.MAX_SLICE_LOSS
BLOCK, RESAMPLES, SEED = se.BLOCK, se.RESAMPLES, se.SEED
SMALL_SLICE_DAYS = 10  # second reading only: slices resting on fewer days are set aside
MODELS = ("yesterday", "H", "HW", "G", "GK", "GM", "W", "C")
PROVENANCE = ("saved", "rebuilt_french_saved", "rebuilt")
REQUIRED = ("delivery_day", "hour", "dow", "holiday", "de_holiday", "wind_px", "solar_px", "spread", "provenance", *MODELS)
TEST_NAME = "spread_w"
AS_OF = {  # what each class means for the vintage of the inputs
    "saved": "inputs as fetched that morning; an input that was unavailable stayed missing; no later revision is used",
    "rebuilt_french_saved": "French inputs as saved before 10:05 Paris on the eve; German inputs from the archive replay, where ENTSO-E's forecasts are the latest "
                            "published version and may have been revised after the gate",
    "rebuilt": "every input from the archive replay; ENTSO-E's French load forecast and nuclear output and its German forecasts are the latest published version "
               "and may have been revised after the gate; an input absent from the archive stayed missing",
}
SCHEDULE_COLUMNS = ("delivery_day", "provenance", "issued", "inputs_missing", "unscored_reason")


def availability(frame: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Scheduled, issued and scored hours by provenance class, beside the error on the common scored hours."""
    missing = [c for c in SCHEDULE_COLUMNS if c not in schedule.columns]
    if missing:
        raise ValueError(f"schedule columns missing: {missing}")
    if schedule[list(SCHEDULE_COLUMNS)].isna().any().any():
        raise ValueError("the schedule must have no missing value")
    _stamps(schedule, "schedule")
    registered = pd.date_range(FIRST, LAST, freq="D")
    if len(schedule) != EXPECTED_HOURS or not np.array_equal(np.sort(schedule["delivery_day"].unique()), registered.to_numpy()):
        raise ValueError(f"the schedule must hold the {EXPECTED_HOURS} hours of the {EXPECTED_DAYS} registered days")
    if schedule["issued"].dtype != bool:
        raise ValueError("`issued` must be true or false")
    if not schedule["provenance"].isin(PROVENANCE).all() or (schedule.groupby("delivery_day")["provenance"].nunique() != 1).any():
        raise ValueError("the schedule's provenance must be one of saved, rebuilt_french_saved, rebuilt, and the same for every hour of a day")
    if not frame.index.isin(schedule.index).all():
        raise ValueError("a scored hour is not in the schedule")
    if not schedule.loc[frame.index, "issued"].all() or (schedule.loc[frame.index, "provenance"].to_numpy() != frame["provenance"].to_numpy()).any():
        raise ValueError("a scored hour has no issued forecast, or its provenance differs from the schedule")
    unscored = schedule[~schedule.index.isin(frame.index)]
    if (schedule.loc[frame.index, "unscored_reason"] != "").any():
        raise ValueError("a scored hour carries a reason for not being scored")
    unexplained = unscored[unscored["issued"] & ~unscored["unscored_reason"].isin(UNSCORED_REASONS)]
    if len(unexplained):
        raise ValueError(f"{len(unexplained)} issued hour(s) are left out of scoring without one of the registered reasons: {list(UNSCORED_REASONS)}")
    scored = schedule.index.isin(frame.index)
    out = {}
    for name in (*PROVENANCE, "all"):
        rows = np.ones(len(schedule), bool) if name == "all" else (schedule["provenance"] == name).to_numpy()
        part = frame if name == "all" else frame[frame["provenance"] == name]
        out[name] = {"days": int(schedule.loc[rows, "delivery_day"].nunique()), "scheduled_hours": int(rows.sum()), "issued_hours": int((rows & schedule["issued"].astype(bool).to_numpy()).sum()),
                     "scored_hours": int((rows & scored).sum()),
                     "issued_but_not_scored": {reason: int((rows & ~scored & (schedule["unscored_reason"] == reason).to_numpy()).sum()) for reason in UNSCORED_REASONS},
                     "hours_with_an_input_left_missing": int((rows & (schedule["inputs_missing"].to_numpy() > 0)).sum()),
                     "mae_on_common_scored_hours": {m: float((part[m] - part["spread"]).abs().mean()) for m in MODELS} if len(part) else None,
                     "as_of": AS_OF.get(name, "all classes together")}
    saved_days = schedule.loc[schedule["provenance"] == "saved", "delivery_day"]
    if len(saved_days):
        since = schedule[schedule["delivery_day"] >= saved_days.min()]
        out["since_the_first_saved_day"] = {"first_saved_day": f"{saved_days.min():%Y-%m-%d}", "days": int(since["delivery_day"].nunique()), "days_saved_before_the_auction": int(saved_days.nunique()),
                                            "share_of_days_saved": float(saved_days.nunique() / since["delivery_day"].nunique())}
    else:
        out["since_the_first_saved_day"] = {"first_saved_day": None, "days": 0, "days_saved_before_the_auction": 0, "share_of_days_saved": None}
    return out


REPEATED_HOUR = (pd.Timestamp("2026-10-25"), 2)  # the clocks go back: 02:00 Paris happens twice on this day and on no other in the window
UNSCORED_REASONS = ("no realised spread", "no spread of the previous day at that hour", "no German proxy")  # the scored-hour rule of the first test


def _stamps(table: pd.DataFrame, what: str) -> pd.DatetimeIndex:
    index = table.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f"the {what} must be indexed by time stamps with a time zone")
    if not index.is_unique:
        raise ValueError(f"an hour appears more than once in the {what}")
    if (np.asarray(delivery_days(index)) != table["delivery_day"].to_numpy()).any():
        raise ValueError(f"a delivery day in the {what} is not the Paris day of its time stamp")
    return index


def validate(frame: pd.DataFrame) -> None:
    """Every model is scored on the same hours, each hour once, on the registered days. Anything else is refused."""
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing: {missing}")
    holes = frame[list(REQUIRED)].isna().sum()
    if holes.any():
        raise ValueError(f"missing values, so the models would be scored on different hours: {holes[holes > 0].to_dict()}")
    _stamps(frame, "frame")
    twice = frame[frame.duplicated(["delivery_day", "hour"], keep=False)]
    if len(twice) and not (len(twice) == 2 and (twice["delivery_day"].iloc[0], int(twice["hour"].iloc[0])) == REPEATED_HOUR):
        raise ValueError("an hour appears more than once")
    if frame["delivery_day"].min() < pd.Timestamp(FIRST) or frame["delivery_day"].max() > pd.Timestamp(LAST):
        raise ValueError("the frame holds days outside the registered window")
    days, hours = frame["delivery_day"].nunique(), len(frame)
    if days != EXPECTED_DAYS or not EXPECTED_HOURS - 24 <= hours <= EXPECTED_HOURS:
        raise ValueError(f"{days} days and {hours} hours, expected {EXPECTED_DAYS} days and {EXPECTED_HOURS - 24} to {EXPECTED_HOURS} hours")
    if not frame["provenance"].isin(PROVENANCE).all() or (frame.groupby("delivery_day")["provenance"].nunique() != 1).any():
        raise ValueError("provenance must be one of saved, rebuilt_french_saved, rebuilt, and the same for every hour of a day")


def daily_error(frame: pd.DataFrame, name: str) -> pd.Series:
    return (frame[name] - frame["spread"]).abs().groupby(frame["delivery_day"]).mean().sort_index()


def paired(frame: pd.DataFrame, candidate: str, base: str) -> dict:
    """Candidate minus base by delivery day, with the two-sided 95% interval and the one-sided upper end at ONE_SIDED."""
    diff = (daily_error(frame, candidate) - daily_error(frame, base)).to_numpy()
    lo, hi = block_bootstrap_mean(diff, block=BLOCK, resamples=RESAMPLES, seed=SEED, level=0.95)
    lower, upper = block_bootstrap_mean(diff, block=BLOCK, resamples=RESAMPLES, seed=SEED, level=1 - 2 * (1 - ONE_SIDED))
    return {"days": int(len(diff)), "candidate_mae": float((frame[candidate] - frame["spread"]).abs().mean()), "base_mae": float((frame[base] - frame["spread"]).abs().mean()),
            "mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "one_sided_level": ONE_SIDED, "one_sided_upper": float(upper), "mirrored_lower": float(lower),
            "share_days_better": float((diff < 0).mean())}


def decide(frame: pd.DataFrame, candidate: str, base: str) -> dict:
    """The gain rule, and beside it the same rule with slices of fewer than SMALL_SLICE_DAYS days set aside."""
    p = paired(frame, candidate, base)
    gain = p["base_mae"] - p["candidate_mae"]
    sl = {}
    for name, mask in se.slices(frame).items():
        s = frame[np.asarray(mask)]
        if not len(s):
            sl[name] = {"hours": 0, "days": 0, "difference": None}
            continue
        sl[name] = {"hours": int(len(s)), "days": int(s["delivery_day"].nunique()), "base": float((s[base] - s["spread"]).abs().mean()),
                    "candidate": float((s[candidate] - s["spread"]).abs().mean())}
        sl[name]["difference"] = sl[name]["candidate"] - sl[name]["base"]

    def verdict(kept) -> dict:
        worst = max((v["difference"] for v in kept if v["difference"] is not None), default=float("-inf"))
        conditions = {"gain_at_least_0.25": bool(gain >= MIN_GAIN), "no_slice_worse_by_more_than_1.0": bool(worst <= MAX_SLICE_LOSS), "one_sided_upper_below_zero": bool(p["one_sided_upper"] < 0)}
        return {"conditions": conditions, "worst_slice": None if worst == float("-inf") else float(worst),
                "verdict": "passes" if all(conditions.values()) else "contradicts" if p["mirrored_lower"] > 0 else "fails"}

    return {"candidate": candidate, "base": base, "gain": float(gain), "paired": p, "slices": sl, "rule": verdict(list(sl.values())),
            "second_reading_small_slices_set_aside": verdict([v for v in sl.values() if v["days"] >= SMALL_SLICE_DAYS])}


def report(frame: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Everything the plan says is reported. The decisions are in `decisions`; the rest is never part of them."""
    validate(frame)
    hours = availability(frame, schedule)
    base = frame.groupby("delivery_day")[["spread", *MODELS]].mean()
    move = base["spread"] - base["yesterday"]
    big = move.abs() > se.MOVE
    month = frame["delivery_day"].dt.strftime("%Y-%m")
    days = frame.groupby("delivery_day")["provenance"].first()

    def errors(part) -> dict:
        return {"days": int(part["delivery_day"].nunique()), "hours": int(len(part)), **{m: float((part[m] - part["spread"]).abs().mean()) for m in MODELS}} if len(part) else {"days": 0, "hours": 0}

    return {"window": [FIRST, LAST], "hours": int(len(frame)), "days": int(frame["delivery_day"].nunique()), "expected": {"days": EXPECTED_DAYS, "hours": EXPECTED_HOURS},
            "decisions": [{"question": q, **decide(frame, c, b)} for c, b, q in DECISIONS],
            "mae": errors(frame), "by_month": {m: errors(part) for m, part in frame.groupby(month)},
            "provenance": {"days": {k: int((days == k).sum()) for k in PROVENANCE},
                           "errors": {k: errors(frame[frame["provenance"] == k]) for k in PROVENANCE},
                           "primary_comparison_by_class": {k: paired(frame[frame["provenance"] == k], "W", "HW") for k in PROVENANCE if (days == k).sum() >= 14},
                           "note": "The classes are different calendar periods: rebuilt days come first, saved days last. A difference between them is not the effect of provenance alone."},
            "availability": hours,
            "not_attainable": ["C"],
            "base_spread_mae": {m: float((base[m] - base["spread"]).abs().mean()) for m in MODELS},
            "direction_right": {m: float((np.sign(base.loc[big, m] - base.loc[big, "yesterday"]) == np.sign(move[big])).mean()) if big.any() else None for m in MODELS if m != "yesterday"},
            "days_with_move_above_5": int(big.sum()),
            "reported_only": {"HW against H": paired(frame, "HW", "H"), "GK against G (weekly refits alone)": paired(frame, "GK", "G"), "GM against G (weights alone)": paired(frame, "GM", "G"),
                              "W against yesterday": paired(frame, "W", "yesterday"), "C against W": paired(frame, "C", "W")}}


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()


def read(frame_path: Path, schedule_path: Path, stop_checks: Path) -> dict:
    """The one read. Refuses before READ_FROM (Paris date of the machine's clock), without the passed stop conditions of this
    test and window, or on forecasts other than the checked ones."""
    if _today() < pd.Timestamp(READ_FROM):
        raise SystemExit(f"sealed until {READ_FROM}")
    checks = json.loads(Path(stop_checks).read_text())
    if checks.get("test") != TEST_NAME or checks.get("window") != [FIRST, LAST]:
        raise SystemExit("the stop-condition record is not that of this test and window")
    if not checks.get("all_pass"):
        raise SystemExit("a stop condition failed or was not run: nothing is scored")
    if se.sha256(frame_path) != checks.get("test_predictions_sha256") or se.sha256(schedule_path) != checks.get("schedule_sha256"):
        raise SystemExit("the forecasts on disk are not the ones the stop conditions were run on")
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=["delivery_day"])
    schedule = pd.read_csv(schedule_path, index_col=0, parse_dates=["delivery_day"], keep_default_na=False)
    frame.index, schedule.index = pd.to_datetime(frame.index, utc=True), pd.to_datetime(schedule.index, utc=True)
    return report(frame, schedule)
