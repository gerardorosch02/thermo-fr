"""Model sprint on the honest_v2 inputs (branch model-sprint; pre-registration in the sprint file, logged in docs/experiments.md).

Candidates, all against honest_v2 trained on all earlier data:
- S1: 0.5 honest_v2 + 0.5 LEAR (forecast/lear.py), every hour;
- S2: honest_v2 and LEAR weighted by inverse 14-day exponentially weighted daily MAE of settled Tuesday-to-Saturday days
  (forecast/combine.py), equal weights until 14 such days exist;
- S3: honest_v2 averaged over training windows of all data, 730 and 365 days ending the day before each month;
- S4: honest_v2 with the asinh target (median/MAD scaling from each fit's training targets).
LEAR alone is a diagnostic. The selection rule and the diagnostics are computed by `selection_report`; nothing here touches
the holdout unless asked to, and `run_sprint` refuses a test window that reaches into it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import FitOptions, evaluate, walk_forward
from .combine import TUESDAY_TO_SATURDAY, combine, daily_mae, inverse_mae_weights
from .features import build_features, feature_timings
from .timing import check_point_in_time
from .lear import LEAR_WINDOWS, lear_walk_forward
from .shape import block_bootstrap_mean

HOLDOUT_START = pd.Timestamp("2026-07-01")
CANDIDATES = ("S1", "S2", "S3", "S4")
BASELINE = "honest_v2"
MIN_GAIN = 0.25
SLICE_LIMIT = 1.0
ONE_SIDED_LEVEL = 1 - 0.05 / 4  # 98.75%, Bonferroni for four candidates
SLICES = {"weekends": "weekends", "holidays": "holidays", "windiest 10%": "windiest_10pct_days", "sunniest 10%": "sunniest_10pct_days",
          "top 5% hours": "top_5pct_price_hours"}
S3_WINDOWS = (None, 730, 365)
S2_HALF_LIFE = 14.0
S2_MIN_DAYS = 14
EXCLUDED_MONTH = "2024-04"  # S3's extra reading: the result without April 2024


def check_window(test_end: str, allow_holdout: bool = False) -> None:
    if pd.Timestamp(test_end) > HOLDOUT_START and not allow_holdout:
        raise ValueError(f"the sprint stops at the holdout: test_end must be on or before {HOLDOUT_START:%Y-%m-%d}")


def gbm_stage(hourly: pd.DataFrame, test_start: str, test_end: str, log=print) -> dict:
    """honest_v2, S3 and S4 predictions (gradient boosting only), after asserting the gate at both deadlines for every row."""
    check_window(test_end)
    table = build_features(hourly, "honest_v2")
    days = table.info["delivery_day"]
    rows = (days >= pd.Timestamp(test_start)) & (days < pd.Timestamp(test_end))
    check_point_in_time(table.X.index[rows], feature_timings("honest_v2"), deadline="gate")
    check_point_in_time(table.X.index[rows], feature_timings("honest_v2"), deadline="premarket")
    out = {"honest_v2": walk_forward(table, test_start, test_end, models=("gbm",), log=log).predictions}
    out["S3"] = walk_forward(table, test_start, test_end, models=("gbm",), log=log, options=FitOptions(train_windows=S3_WINDOWS)).predictions
    out["S4"] = walk_forward(table, test_start, test_end, models=("gbm",), log=log, options=FitOptions(target_transform="asinh")).predictions
    return out


def assemble(base: pd.DataFrame, s3: pd.DataFrame, s4: pd.DataFrame, lear: pd.DataFrame, lear_windows=LEAR_WINDOWS) -> pd.DataFrame:
    """One hourly frame: baseline, LEAR, S1 (equal weights), S2 (online weights), S3, S4."""
    frame = base.rename(columns={"gbm": BASELINE})
    frame["lear"] = lear["lear"].reindex(frame.index)
    for w in lear_windows:
        frame[f"lear_w{w}"] = lear[f"lear_w{w}"].reindex(frame.index)
    frame["S1"] = combine(frame, None, [BASELINE, "lear"])
    errors = daily_mae(frame, [BASELINE, "lear"])
    days = pd.DatetimeIndex(sorted(frame["delivery_day"].unique()))
    weights = inverse_mae_weights(errors, days, half_life=S2_HALF_LIFE, min_days=S2_MIN_DAYS, update_weekdays=TUESDAY_TO_SATURDAY)
    frame["w_lear"] = weights["lear"].reindex(pd.DatetimeIndex(frame["delivery_day"])).to_numpy()
    frame["S2"] = combine(frame, weights, [BASELINE, "lear"])
    frame["S3"] = s3["gbm"].reindex(frame.index)
    frame["S4"] = s4["gbm"].reindex(frame.index)
    frame["lear_missing"] = frame["lear"].isna()
    return frame


def run_sprint(hourly: pd.DataFrame, test_start: str, test_end: str, log=print, n_jobs: int = 1, lear_windows=LEAR_WINDOWS,
               recalibrate: str = "daily", allow_holdout: bool = False) -> pd.DataFrame:
    """Hourly frame with the baseline, LEAR and the four candidates, plus actual, strict and the info columns."""
    check_window(test_end, allow_holdout)
    g = gbm_stage(hourly, test_start, test_end, log=log)
    lear = lear_walk_forward(hourly, test_start, test_end, windows=lear_windows, recalibrate=recalibrate, log=log, n_jobs=n_jobs)
    return assemble(g["honest_v2"], g["S3"], g["S4"], lear, lear_windows)


def _strict(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["strict"].astype(bool)].dropna(subset=["actual"])


def daily_abs_error(frame: pd.DataFrame, column: str) -> pd.Series:
    rows = _strict(frame)
    return (rows[column] - rows["actual"]).abs().groupby(rows["delivery_day"]).mean()


def paired(frame: pd.DataFrame, column: str, baseline: str = BASELINE) -> dict:
    diff = (daily_abs_error(frame, column) - daily_abs_error(frame, baseline)).dropna()
    low95, high95 = block_bootstrap_mean(diff.to_numpy(), block=7, resamples=5000, seed=0, level=0.95)
    _, high_one = block_bootstrap_mean(diff.to_numpy(), block=7, resamples=5000, seed=0, level=1 - 2 * (1 - ONE_SIDED_LEVEL))
    return {"days": int(len(diff)), "mean": round(float(diff.mean()), 4), "ci95": [round(low95, 4), round(high95, 4)],
            "upper_one_sided_98_75": round(high_one, 4), "share_won": round(float((diff < 0).mean()), 3)}


def metrics_for(frame: pd.DataFrame, column: str) -> dict:
    view = frame.assign(gbm=frame[column], linear=frame[column])
    return evaluate(view, strict_only=True)


def lag_diagnostic(frame: pd.DataFrame, column: str, daily_price: pd.Series) -> dict:
    rows = _strict(frame)
    err = (rows[column] - rows["actual"]).groupby(rows["delivery_day"]).mean()
    move = pd.Series(daily_price.reindex(err.index - pd.Timedelta(days=1)).to_numpy()
                     - daily_price.reindex(err.index - pd.Timedelta(days=3)).to_numpy(), index=err.index)
    ok = move.notna() & err.notna()
    e, m = err[ok], move[ok]
    return {"signed": round(float(e.corr(m)), 3), "slope": round(float(np.cov(e, m)[0, 1] / m.var()), 3),
            "absolute": round(float(e.abs().corr(m.abs())), 3)}


def regime_days(daily_price: pd.Series, days) -> set:
    """The 10% of days with the largest |change in the daily mean price, D against D-1|, from actual prices only."""
    change = (daily_price - daily_price.shift(1)).abs().reindex(pd.DatetimeIndex(days)).dropna()
    return set(change[change >= change.quantile(0.9)].index)


def daily_mean_price(hourly: pd.DataFrame) -> pd.Series:
    days = hourly.index.tz_convert("Europe/Paris").normalize().tz_localize(None)
    return hourly["price_eur_mwh"].groupby(days).mean().asfreq("D")


def selection_report(frame: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    """The pre-registered rule for S1 to S4 against honest_v2, and the diagnostics, on the strict rows of the frame."""
    price = daily_mean_price(hourly)
    strict = _strict(frame)
    base_mae = float((strict[BASELINE] - strict["actual"]).abs().mean())
    regime = regime_days(price, strict["delivery_day"].unique())
    out = {"baseline_mae": round(base_mae, 4), "candidates": {}, "diagnostics": {}}
    base_metrics = metrics_for(frame, BASELINE)
    for name in (BASELINE, "lear") + CANDIDATES:
        m = metrics_for(frame, name)
        rows = strict.dropna(subset=[name])
        err = (rows[name] - rows["actual"]).abs()
        no_april = rows["delivery_day"].dt.strftime("%Y-%m") != EXCLUDED_MONTH
        diag = {
            "mae": round(float(err.mean()), 4), "rmse": round(float(np.sqrt(((rows[name] - rows["actual"]) ** 2).mean())), 4),
            "hours": int(len(rows)),
            "peak": m["slices"]["peak"]["gbm"]["mae"], "off_peak": m["slices"]["off_peak"]["gbm"]["mae"],
            "negative_hours": m["slices"]["negative_price_hours"]["gbm"]["mae"],
            "slices": {k: m["slices"][v]["gbm"]["mae"] for k, v in SLICES.items()},
            "by_year": {int(y): round(float(p.mean()), 3) for y, p in err.groupby(rows["delivery_day"].dt.year)},
            "by_month": {str(k): round(float(p.mean()), 3) for k, p in err.groupby(rows["delivery_day"].dt.strftime("%Y-%m"))},
            "mae_without_april_2024": round(float(err[no_april].mean()), 4),
            "regime_change_days": round(float(err[rows["delivery_day"].isin(regime)].mean()), 4),
            "lag": lag_diagnostic(frame, name, price),
        }
        if name != BASELINE:
            diag["paired_vs_baseline"] = paired(frame, name)
        out["diagnostics"][name] = diag
    base_no_april = out["diagnostics"][BASELINE]["mae_without_april_2024"]
    for name in CANDIDATES:
        d = out["diagnostics"][name]
        gain = base_mae - d["mae"]
        worse = {k: round(d["slices"][k] - base_metrics["slices"][v]["gbm"]["mae"], 2) for k, v in SLICES.items()
                 if d["slices"][k] is not None and base_metrics["slices"][v]["gbm"]["mae"] is not None}  # an empty slice cannot be worse
        conditions = {
            "gain_at_least_0.25": gain >= MIN_GAIN,
            "no_slice_worse_than_1.0": all(v <= SLICE_LIMIT for v in worse.values()),
            "one_sided_98.75_upper_below_zero": d["paired_vs_baseline"]["upper_one_sided_98_75"] < 0,
        }
        if name == "S3":
            conditions["gain_without_april_2024_at_least_0.25"] = (base_no_april - d["mae_without_april_2024"]) >= MIN_GAIN
        out["candidates"][name] = {"gain": round(gain, 4), "slice_differences": worse, "conditions": conditions,
                                   "passes": all(v for k, v in conditions.items() if k != "gain_without_april_2024_at_least_0.25"),
                                   "eligible_for_shadow": all(conditions.values())}
    eligible = [n for n in CANDIDATES if out["candidates"][n]["eligible_for_shadow"]]
    out["to_shadow"] = min(eligible, key=lambda n: out["diagnostics"][n]["mae"]) if eligible else None
    out["finding"] = (f"{out['to_shadow']} goes to the shadow" if eligible else
                      "no candidate is distinguishable from honest_v2 on the selection window after correcting for four comparisons")
    both = strict.dropna(subset=["lear"])
    e_base = (both[BASELINE] - both["actual"]).groupby(both["delivery_day"]).mean()
    e_lear = (both["lear"] - both["actual"]).groupby(both["delivery_day"]).mean()
    out["ensemble"] = {"daily_error_correlation_honest_v2_lear": round(float(e_base.corr(e_lear)), 3),
                       "s2_mean_lear_weight_by_month": {str(k): round(float(v), 3) for k, v in
                                                      frame.groupby(frame["delivery_day"].dt.strftime("%Y-%m"))["w_lear"].mean().items()},
                       "lear_missing_hours": int(frame["lear_missing"].sum())}
    return out
