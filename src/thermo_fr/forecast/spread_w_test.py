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


def validate(frame: pd.DataFrame) -> None:
    """Every model is scored on the same hours, each hour once, on the registered days. Anything else is refused."""
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing: {missing}")
    holes = frame[list(REQUIRED)].isna().sum()
    if holes.any():
        raise ValueError(f"missing values, so the models would be scored on different hours: {holes[holes > 0].to_dict()}")
    if not frame.index.is_unique or frame.duplicated(["delivery_day", "hour"]).any() and not _repeated_hour_only(frame):
        raise ValueError("an hour appears more than once")
    if frame["delivery_day"].min() < pd.Timestamp(FIRST) or frame["delivery_day"].max() > pd.Timestamp(LAST):
        raise ValueError("the frame holds days outside the registered window")
    days, hours = frame["delivery_day"].nunique(), len(frame)
    if days != EXPECTED_DAYS or not EXPECTED_HOURS - 24 <= hours <= EXPECTED_HOURS:
        raise ValueError(f"{days} days and {hours} hours, expected {EXPECTED_DAYS} days and {EXPECTED_HOURS - 24} to {EXPECTED_HOURS} hours")
    per_day = frame.groupby("delivery_day").size()
    if per_day.max() > 25 or per_day.min() < 1:
        raise ValueError("a day holds more than 25 hours")
    if not frame["provenance"].isin(PROVENANCE).all() or (frame.groupby("delivery_day")["provenance"].nunique() != 1).any():
        raise ValueError("provenance must be one of saved, rebuilt_french_saved, rebuilt, and the same for every hour of a day")


def _repeated_hour_only(frame: pd.DataFrame) -> bool:
    """The only clock hour that may repeat is the extra hour of the 25-hour day, and the time stamps must differ."""
    twice = frame[frame.duplicated(["delivery_day", "hour"], keep=False)]
    return bool(len(twice) == 2 and twice["delivery_day"].nunique() == 1 and twice.index.is_unique and len(frame[frame["delivery_day"] == twice["delivery_day"].iloc[0]]) == 25)


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


def report(frame: pd.DataFrame) -> dict:
    """Everything the plan says is reported. The decisions are in `decisions`; the rest is never part of them."""
    validate(frame)
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
            "not_attainable": ["C"],
            "base_spread_mae": {m: float((base[m] - base["spread"]).abs().mean()) for m in MODELS},
            "direction_right": {m: float((np.sign(base.loc[big, m] - base.loc[big, "yesterday"]) == np.sign(move[big])).mean()) if big.any() else None for m in MODELS if m != "yesterday"},
            "days_with_move_above_5": int(big.sum()),
            "reported_only": {"HW against H": paired(frame, "HW", "H"), "GK against G (weekly refits alone)": paired(frame, "GK", "G"), "GM against G (weights alone)": paired(frame, "GM", "G"),
                              "W against yesterday": paired(frame, "W", "yesterday"), "C against W": paired(frame, "C", "W")}}


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()


def read(frame_path: Path, stop_checks: Path) -> dict:
    """The one read. Refuses before READ_FROM (Paris date of the machine's clock), without the passed stop conditions of this
    test and window, or on forecasts other than the checked ones."""
    if _today() < pd.Timestamp(READ_FROM):
        raise SystemExit(f"sealed until {READ_FROM}")
    checks = json.loads(Path(stop_checks).read_text())
    if checks.get("test") != TEST_NAME or checks.get("window") != [FIRST, LAST]:
        raise SystemExit("the stop-condition record is not that of this test and window")
    if not checks.get("all_pass"):
        raise SystemExit("a stop condition failed or was not run: nothing is scored")
    if se.sha256(frame_path) != checks.get("test_predictions_sha256"):
        raise SystemExit("the forecasts on disk are not the ones the stop conditions were run on")
    frame = pd.read_csv(frame_path, index_col=0, parse_dates=["delivery_day"])
    return report(frame)
