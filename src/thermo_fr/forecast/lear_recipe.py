"""LEAR to the published recipe: the experiment of docs/prereg/lear_recipe_prereg.md (Part 2), run in resumable chunks.

Baseline L0 is LEAR as it is (windows 364, 728, 1092, 1456 days). Candidates: A (windows 56, 84, 1092, 1456), B (all six), C (B
with a rolling per-hour choice of the asinh slope c over the last 56 settled days). Every candidate is an average of per-window
forecasts, so the work is one daily table of forecasts per (slope, window, month), written at once and skipped when present:

    <out>/chunks/<kind>/c<slope>/w<window>/<YYYY-MM>.csv      kind "full" (the L0 regressors) or "ar" (M6, LEAR-AR)

Nothing here reads a delivery day on or after the holdout start: `selection_inputs` cuts the inputs first.

    python -m thermo_fr.forecast.lear_recipe gram-check --out DIR      # the shared-Gram speed-up against the present code
    python -m thermo_fr.forecast.lear_recipe gbm --out DIR             # honest_v2, monthly walk-forward
    python -m thermo_fr.forecast.lear_recipe lear --out DIR --kind full --slopes 1 --shard 0/2
    python -m thermo_fr.forecast.lear_recipe report --out DIR          # reproduction, M1 to M7, the rule, the extras
    python -m thermo_fr.forecast.lear_recipe scramble --out DIR        # the look-ahead test on the new forecast paths
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lear, yardstick
from .backtest import month_starts, walk_forward
from .features import build_features, feature_timings
from .sprint import HOLDOUT_START, SLICES, daily_mean_price, metrics_for
from .timing import NUCLEAR_D1_CUTOFF_HOUR, check_point_in_time, delivery_days

WALK_START = "2024-01-01"  # the walk-forward start of the October sprint
SELECTION_START = pd.Timestamp("2024-02-17")  # first strict delivery day
TEST_END = "2026-07-01"  # exclusive; the holdout starts here
SLOPE_START = "2023-12-23"  # candidate C needs every slope from 56 days before the first strict day
SHORT_WINDOWS = (56, 84)
ALL_WINDOWS = (56, 84, 364, 728, 1092, 1456)
MODEL_WINDOWS = {"L0": lear.LEAR_WINDOWS, "A": (56, 84, 1092, 1456), "B": ALL_WINDOWS}
SLOPES = (0.1, 0.3, 0.5, 0.7, 1.0)
LOOKBACK_DAYS = 56
CANDIDATES = ("A", "B", "C")
BASELINE = "L0"
MIN_GAIN = 0.25
SLICE_LIMIT = 1.0
ONE_SIDED_LEVEL = 1 - 0.05 / 3  # 98.33%, Bonferroni for three candidates
GRAM_TOLERANCE = 1e-6  # EUR/MWh
REPRODUCE = {"honest_v2": ("15.7040", 4), "L0": ("15.36", 2), "S1": ("14.36", 2)}
# The shared Gram matrix is off: on the real inputs the gram-check stage found differences far above the tolerance on some days
# (the LARS path amplifies rounding-level changes of the Gram matrix), so the one speed-up the pre-registration allowed is not used.
FULL_OPTIONS = {"short_rule": lear.SHORT_RULE, "share_gram": False}
GRAM_OPTIONS = {"short_rule": lear.SHORT_RULE, "share_gram": True}  # only for the gram-check stage


def selection_inputs(hourly: pd.DataFrame, end: str = TEST_END) -> pd.DataFrame:
    """The inputs with every delivery day on or after `end` removed (never later than the holdout start)."""
    if pd.Timestamp(end) > HOLDOUT_START:
        raise ValueError(f"the experiment stops at the holdout: end must be on or before {HOLDOUT_START:%Y-%m-%d}")
    hourly = hourly.sort_index()
    return hourly[delivery_days(hourly.index) < pd.Timestamp(end)]


def ar_design(design: pd.DataFrame, blocks: dict) -> tuple[pd.DataFrame, dict]:
    """M6, LEAR-AR: only the four French price-lag blocks and the calendar dummies."""
    keep = {name: blocks[name] for name in lear.PRICE_BLOCKS + ("calendar",)}
    return design[[c for cols in keep.values() for c in cols]], keep


def forecast_days(design: pd.DataFrame, price: pd.DataFrame, blocks: dict, days, window: int, slope: float = 1.0,
                  fit_options: dict | None = None) -> pd.DataFrame:
    """Daily table (index day, columns hours 0 to 23) of one window's forecasts, refitted every day on the days ending D-1."""
    options = {**(FULL_OPTIONS if fit_options is None else fit_options), "slope": float(slope)}
    frames = [lear._fit_predict_window(design, price, blocks, pd.Timestamp(d), window, lear.LEAR_START, [pd.Timestamp(d)], options)
              for d in pd.DatetimeIndex(days)]
    return pd.concat(frames) if frames else pd.DataFrame(columns=lear.HOURS, dtype=float)


def chunk_path(out: Path, kind: str, slope: float, window: int, month) -> Path:
    return Path(out) / "chunks" / kind / f"c{float(slope):g}" / f"w{int(window)}" / f"{month}.csv"


def chunk_days(month: pd.Period, first: str, end: str = TEST_END) -> pd.DatetimeIndex:
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    return days[(days >= pd.Timestamp(first)) & (days < pd.Timestamp(end))]


def job_list(kind: str, slopes, windows, first: str, end: str = TEST_END) -> list[tuple]:
    """(kind, slope, window, month) for every chunk, month by month so that both shards advance through time together."""
    months = pd.period_range(pd.Timestamp(first), pd.Timestamp(end) - pd.Timedelta(days=1), freq="M")
    return [(kind, float(c), int(w), m) for c in slopes for m in months for w in windows]


def run_jobs(hourly: pd.DataFrame, out: Path, jobs: list[tuple], first: str, shard: tuple[int, int] = (0, 1), budget_s: float | None = None,
             log=print) -> int:
    """Run the chunks of this shard that are not on disk yet; stop before a chunk if the time budget is used up. Returns the
    number of chunks still missing (in this shard) when it stops."""
    started = time.time()
    hourly = selection_inputs(hourly)
    design, price, blocks = lear.lear_design(hourly)
    table = build_features(hourly, "honest_v2")
    designs = {"full": (design, blocks), "ar": ar_design(design, blocks)}
    mine = [job for k, job in enumerate(jobs) if k % shard[1] == shard[0]]
    todo = [job for job in mine if not chunk_path(out, *job).exists()]
    log(f"shard {shard[0]}/{shard[1]}: {len(todo)} of {len(mine)} chunks to run", flush=True)
    done = 0
    for kind, slope, window, month in todo:
        if budget_s is not None and time.time() - started > budget_s:
            break
        days = chunk_days(month, first)
        rows = table.info["delivery_day"].isin(days)
        lear.assert_gate(table.X.index[rows], designs[kind][1])  # every input known at the gate and at 10:05, or the run stops
        t0 = time.time()
        X, used = designs[kind]
        daily = forecast_days(X, price, used, days, window, slope)
        path = chunk_path(out, kind, slope, window, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        daily.to_csv(tmp, index_label="delivery_day")
        tmp.replace(path)
        done += 1
        log(f"{kind} c={slope:g} w={window} {month}: {len(days)} days in {time.time() - t0:.0f} s", flush=True)
    left = len(todo) - done
    log(f"shard {shard[0]}/{shard[1]}: {done} chunks written, {left} left, {time.time() - started:.0f} s", flush=True)
    return left


def read_window(out: Path, kind: str, slope: float, window: int, first: str, end: str = TEST_END) -> pd.DataFrame:
    """One window's daily table over [first, end) from its chunks; raises if a chunk is missing."""
    months = pd.period_range(pd.Timestamp(first), pd.Timestamp(end) - pd.Timedelta(days=1), freq="M")
    frames = []
    for month in months:
        path = chunk_path(out, kind, slope, window, month)
        if not path.exists():
            raise FileNotFoundError(f"chunk missing: {path}")
        frame = pd.read_csv(path, index_col=0, parse_dates=[0])
        frame.columns = [int(c) for c in frame.columns]
        frames.append(frame)
    table = pd.concat(frames).sort_index()
    return table[(table.index >= pd.Timestamp(first)) & (table.index < pd.Timestamp(end))]


def window_average(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """The mean over windows, day by day and hour by hour, exactly as lear.lear_forecast_days takes it."""
    return pd.concat(tables).groupby(level=0).mean()


def to_hourly(info: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """The daily table laid on hourly rows by (delivery day, clock hour): both autumn hour-2 rows receive the hour-2 forecast."""
    stacked = daily.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays([pd.DatetimeIndex(info["delivery_day"]), info["hour"].to_numpy()])
    return pd.Series(stacked.reindex(keys).to_numpy(), index=info.index)


def rolling_slope_choice(forecasts: pd.DataFrame, actual: pd.Series, delivery_day: pd.Series, hour: pd.Series,
                         lookback: int = LOOKBACK_DAYS) -> tuple[pd.Series, pd.Series]:
    """Candidate C. `forecasts`: hourly rows, one column per slope c. For delivery day D and clock hour h the slope with the lowest
    mean absolute error over the hourly rows of hour h on delivery days D-lookback to D-1 is used (all settled before the gate);
    ties go to the larger c. Returns (chosen slope, forecast) per row; NaN for a day whose lookback window starts before the
    first day of the frame. The mean is over the scored rows that exist (the spring clock-change day has no hour 2, the autumn
    day two).

    Only rows with delivery day before D enter the choice for D; the row's own actual price is never read for its own choice.
    """
    slopes = sorted((float(c) for c in forecasts.columns), reverse=True)  # the largest first: argmin then breaks ties upward
    columns = {float(c): c for c in forecasts.columns}
    days = pd.DatetimeIndex(delivery_day)
    hours = np.asarray(hour)
    all_days = pd.date_range(days.min(), days.max(), freq="D")
    means = []
    for c in slopes:
        err = (forecasts[columns[c]] - actual).abs()
        keyed = pd.DataFrame({"day": days, "hour": hours, "err": err.to_numpy()}).dropna(subset=["err"])
        grouped = keyed.groupby(["day", "hour"])["err"]
        total = grouped.sum().unstack("hour").reindex(index=all_days, columns=lear.HOURS).fillna(0.0).to_numpy()
        count = grouped.count().unstack("hour").reindex(index=all_days, columns=lear.HOURS).fillna(0.0).to_numpy()
        mean = np.full(total.shape, np.nan)
        if len(all_days) > lookback:
            # row D takes the days D-lookback .. D-1, summed directly (no running sum, so equal errors give exactly equal means)
            window_total = np.lib.stride_tricks.sliding_window_view(total, lookback, axis=0).sum(axis=-1)[:-1]
            window_count = np.lib.stride_tricks.sliding_window_view(count, lookback, axis=0).sum(axis=-1)[:-1]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean[lookback:] = np.where(window_count > 0, window_total / window_count, np.nan)
        means.append(mean)
    stack = np.stack(means, axis=-1)  # day x hour x slope
    defined = np.isfinite(stack).all(axis=-1)
    best = np.where(defined, np.argmin(np.where(defined[..., None], stack, 0.0), axis=-1), -1)
    chosen_table = pd.DataFrame(np.where(best >= 0, np.asarray(slopes)[np.clip(best, 0, None)], np.nan), index=all_days, columns=lear.HOURS)
    keys = pd.MultiIndex.from_arrays([days, hours])
    chosen = pd.Series(chosen_table.stack(future_stack=True).reindex(keys).to_numpy(), index=forecasts.index)
    value = pd.Series(np.nan, index=forecasts.index)
    for c in slopes:
        pick = (chosen == c).to_numpy()
        value[pick] = forecasts[columns[c]].to_numpy()[pick]
    return chosen, value


# --- honest_v2 --------------------------------------------------------------------------------------------------------------

def gbm_honest_v2(hourly: pd.DataFrame, test_start: str = WALK_START, test_end: str = TEST_END, log=print, out: Path | None = None) -> pd.DataFrame:
    """honest_v2 exactly as sprint.gbm_stage runs it (monthly walk-forward, backtest settings, both deadlines asserted), without
    the sprint's S3 and S4 fits. With `out`, each month is written to <out>/gbm/honest_v2_<YYYY-MM>.csv at once and read back if
    present: a month's fit uses only the days before it, so month-by-month calls give the frame of one call."""
    hourly = selection_inputs(hourly, test_end)
    table = build_features(hourly, "honest_v2")
    days = table.info["delivery_day"]
    rows = (days >= pd.Timestamp(test_start)) & (days < pd.Timestamp(test_end))
    check_point_in_time(table.X.index[rows], feature_timings("honest_v2"), deadline="gate")
    check_point_in_time(table.X.index[rows], feature_timings("honest_v2"), deadline="premarket")
    if out is None:
        return walk_forward(table, test_start, test_end, models=("gbm",), log=log).predictions
    folder = Path(out) / "gbm"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for start in month_starts(test_start, test_end):
        path = folder / f"honest_v2_{start:%Y-%m}.csv"
        if not path.exists():
            end = min(start + pd.DateOffset(months=1), pd.Timestamp(test_end))
            part = walk_forward(table, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", models=("gbm",), log=log).predictions
            tmp = path.with_suffix(".tmp")
            part.to_csv(tmp, index_label="timestamp_utc")
            tmp.replace(path)
        frames.append(read_frame(path))
    return pd.concat(frames).sort_index()


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=["delivery_day"])
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


# --- assembling and the rule ------------------------------------------------------------------------------------------------

def assemble(hourly: pd.DataFrame, gbm: pd.DataFrame, out: Path, with_slopes: bool = True, with_ar: bool = True) -> dict:
    """The evaluation frame (hourly rows from the walk-forward start: honest_v2, per-window LEAR, L0, A, B, C, S1, LEAR-AR) and
    the slope frame (hourly rows from SLOPE_START: the six-window average for every slope, and C's choice)."""
    hourly = selection_inputs(hourly)
    info = build_features(hourly, "honest_v2").info
    info = info.assign(actual=hourly["price_eur_mwh"].reindex(info.index))
    wide = info[info["delivery_day"] >= pd.Timestamp(SLOPE_START)]
    windows = {w: read_window(out, "full", 1.0, w, SLOPE_START) for w in ALL_WINDOWS}
    frame = gbm.rename(columns={"gbm": "honest_v2"}).copy()
    for w, table in windows.items():
        frame[f"lear_w{w}"] = to_hourly(frame, table)
    for name, used in MODEL_WINDOWS.items():
        frame[name] = to_hourly(frame, window_average([windows[w] for w in used]))
    frame["S1"] = (frame["honest_v2"] + frame["L0"]) / 2
    result = {"frame": frame}
    if with_ar:
        ar = window_average([read_window(out, "ar", 1.0, w, WALK_START) for w in lear.LEAR_WINDOWS])
        frame["LEAR_AR"] = to_hourly(frame, ar)
    if with_slopes:
        slopes = pd.DataFrame(index=wide.index)
        per_window = {}
        for c in SLOPES:
            tables = {w: (windows[w] if c == 1.0 else read_window(out, "full", c, w, SLOPE_START)) for w in ALL_WINDOWS}
            per_window[c] = tables
            slopes[c] = to_hourly(wide, window_average(list(tables.values())))
        chosen, value = rolling_slope_choice(slopes, wide["actual"], wide["delivery_day"], wide["hour"])
        frame["C"] = value.reindex(frame.index)
        frame["C_slope"] = chosen.reindex(frame.index)
        for c in SLOPES:
            frame[f"B_c{c:g}"] = slopes[c].reindex(frame.index)
        # C's forecast window by window (the chosen slope's forecast of that window), for the per-window MAE
        for w in ALL_WINDOWS:
            by_slope = pd.DataFrame({c: to_hourly(wide, per_window[c][w]) for c in SLOPES})
            picked = pd.Series(np.nan, index=wide.index)
            for c in SLOPES:
                mask = (chosen == c).to_numpy()
                picked[mask] = by_slope[c].to_numpy()[mask]
            frame[f"C_w{w}"] = picked.reindex(frame.index)
        result["slopes"] = slopes.assign(delivery_day=wide["delivery_day"], hour=wide["hour"], actual=wide["actual"], chosen=chosen, C=value)
    return result


def reproduction(frame: pd.DataFrame) -> dict:
    """The three known figures on the strict selection rows; `ok` is False if any does not print as recorded."""
    out, ok = {}, True
    rows = yardstick.strict_rows(frame)
    for name, (printed, digits) in REPRODUCE.items():
        value = yardstick.mae(frame, name)
        same = f"{value:.{digits}f}" == printed
        ok = ok and same
        out[name] = {"mae": value, "expected": printed, "printed": f"{value:.{digits}f}", "reproduced": same}
    out["strict_hours"] = int(len(rows))
    out["strict_days"] = int(rows["delivery_day"].nunique())
    out["first_day"], out["last_day"] = str(rows["delivery_day"].min().date()), str(rows["delivery_day"].max().date())
    out["ok"] = bool(ok and out["strict_hours"] == 20752 and out["strict_days"] == 865)
    return out


def slice_maes(frame: pd.DataFrame, column: str) -> dict:
    """The protocol slices of backtest.evaluate for one column (rounded to 0.01 there), plus negative-price hours."""
    m = metrics_for(frame, column)["slices"]
    out = {label: m[key]["gbm"]["mae"] for label, key in SLICES.items()}
    out["negative-price hours"] = m["negative_price_hours"]["gbm"]["mae"]
    return out


def slice_maes_unrounded(frame: pd.DataFrame, column: str) -> dict:
    """The same slices recomputed without rounding, with the definitions of backtest.evaluate."""
    rows = yardstick.strict_rows(frame)
    err = (rows[column] - rows["actual"]).abs()
    top = rows["actual"] >= rows["actual"].quantile(0.95)
    wind = rows.groupby("delivery_day")["wind_mw"].mean().dropna()
    sun = rows.groupby("delivery_day")["solar_mw"].mean().dropna()
    masks = {"weekends": rows["dow"] >= 5, "holidays": rows["holiday"] == 1,
             "windiest 10%": rows["delivery_day"].isin(wind[wind >= wind.quantile(0.9)].index),
             "sunniest 10%": rows["delivery_day"].isin(sun[sun >= sun.quantile(0.9)].index), "top 5% hours": top,
             "negative-price hours": rows["actual"] < 0}
    return {k: float(err[m].mean()) for k, m in masks.items()} | {f"{k} (hours)": int(m.sum()) for k, m in masks.items()}


def apply_rule(frame: pd.DataFrame, candidates=CANDIDATES, baseline: str = BASELINE) -> dict:
    """The pre-registered rule for each candidate against L0 on the strict rows."""
    base_mae = yardstick.mae(frame, baseline)
    base_slices, base_raw = slice_maes(frame, baseline), slice_maes_unrounded(frame, baseline)
    out = {"baseline": baseline, "baseline_mae": base_mae, "baseline_slices": base_slices, "candidates": {}}
    for name in candidates:
        if name not in frame or frame[name].isna().all():
            continue
        own, raw = slice_maes(frame, name), slice_maes_unrounded(frame, name)
        value = yardstick.mae(frame, name)
        pair = yardstick.paired_daily(frame, name, baseline, one_sided_level=ONE_SIDED_LEVEL)
        worse = {k: round(own[k] - base_slices[k], 2) for k in SLICES}
        worse_raw = {k: raw[k] - base_raw[k] for k in SLICES}
        conditions = {"i_mae_at_least_0.25_below_L0": bool(base_mae - value >= MIN_GAIN),
                      "ii_no_slice_worse_by_more_than_1.0": bool(all(v <= SLICE_LIMIT for v in worse.values())),
                      "iii_one_sided_98.33_upper_below_zero": bool(pair["upper_one_sided"] < 0)}
        out["candidates"][name] = {"mae": value, "gain": base_mae - value, "slices": own, "slice_differences": worse,
                                   "slice_differences_unrounded": worse_raw,
                                   "ii_same_verdict_unrounded": bool(all(v <= SLICE_LIMIT for v in worse_raw.values())) == conditions["ii_no_slice_worse_by_more_than_1.0"],
                                   "paired_vs_L0": pair, "conditions": conditions, "passes": bool(all(conditions.values()))}
    passing = [n for n, c in out["candidates"].items() if c["passes"]]
    out["lear_r"] = min(passing, key=lambda n: out["candidates"][n]["mae"]) if passing else None
    out["finding"] = f"{out['lear_r']} becomes LEAR-R" if passing else "no candidate passes: LEAR stays as it is (null result)"
    return out


def recent_volatility(daily_price: pd.Series, days) -> pd.Series:
    """The standard deviation (sample, ddof 1) of the seven daily mean prices D-7 to D-1."""
    vol = daily_price.rolling(7, min_periods=7).std().shift(1)
    return vol.reindex(pd.DatetimeIndex(days))


def extras(frame: pd.DataFrame, column: str, daily_price: pd.Series, windows: dict | None = None) -> dict:
    """What the pre-registration reports for every candidate whatever the outcome."""
    rows = yardstick.strict_rows(frame)
    err = (rows[column] - rows["actual"]).abs()
    days = pd.DatetimeIndex(sorted(rows["delivery_day"].unique()))
    tercile = yardstick.terciles(recent_volatility(daily_price, days))
    by_row = tercile.reindex(pd.DatetimeIndex(rows["delivery_day"])).to_numpy()
    parts = yardstick.split(frame)
    out = {"mae": float(err.mean()),
           "mae_by_volatility_tercile": {f"tercile_{k}": float(err[by_row == k].mean()) for k in (1, 2, 3)},
           "volatility_tercile_days": {f"tercile_{k}": int((tercile == k).sum()) for k in (1, 2, 3)},
           "mae_before_2025_10_01": yardstick.mae(parts["before"], column), "mae_from_2025_10_01": yardstick.mae(parts["from"], column),
           "mae_negative_price_hours": float(err[rows["actual"] < 0].mean()), "negative_price_hours": int((rows["actual"] < 0).sum()),
           "M1": yardstick.relative_mae(frame, column),
           "M1_before_2025_10_01": yardstick.relative_mae(parts["before"], column),
           "M1_from_2025_10_01": yardstick.relative_mae(parts["from"], column),
           "M2": yardstick.under_reaction(frame, column), "M4": yardstick.shape(frame, column)}
    if windows:
        out["per_window_mae"] = {str(w): yardstick.mae(frame, col) for w, col in windows.items() if col in frame}
    return out


def tidy(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """delivery_day, hour, timestamp (UTC), actual, strict, forecast: one prediction file per model."""
    return pd.DataFrame({"delivery_day": frame["delivery_day"].dt.strftime("%Y-%m-%d"), "hour": frame["hour"].to_numpy(),
                         "timestamp": frame.index.strftime("%Y-%m-%dT%H:%M:%SZ"), "actual": frame["actual"].to_numpy(),
                         "strict": frame["strict"].astype(bool).to_numpy(), "forecast": frame[column].to_numpy()}, index=frame.index)


# --- M7 ---------------------------------------------------------------------------------------------------------------------

def weather_tables(hourly: pd.DataFrame, frame: pd.DataFrame, column: str = "S1") -> dict:
    """Hourly and daily tables for M7 on the days from 2024-05-01 with complete data, with each day's tightness tercile.

    Complete: every hourly row of the day is strict and has the forecast, the actual price, the load forecast, both proxies,
    actual onshore wind and actual solar, and the day has its early D-1 nuclear mean. Actual wind is onshore plus offshore
    (a missing offshore value counts as zero, as the windy-day slice does)."""
    hourly = selection_inputs(hourly)
    X = build_features(hourly, "honest_v2").X
    h = hourly.reindex(frame.index)
    wind_actual = h[["wind_onshore_mw", "wind_offshore_mw"]].sum(axis=1, min_count=1).where(h["wind_onshore_mw"].notna())
    rows = pd.DataFrame({"delivery_day": frame["delivery_day"], "error": frame["actual"] - frame[column],
                         "wind_surprise_gw": (wind_actual - h["wind_proxy_mw"]) / 1000.0,
                         "solar_surprise_gw": (h["solar_mw"] - h["solar_proxy_mw"]) / 1000.0,
                         "load_fc_mw": h["load_fc_mw"], "wind_proxy_mw": h["wind_proxy_mw"], "solar_proxy_mw": h["solar_proxy_mw"],
                         "nuclear_d1_early_mw": X["nuclear_d1_early_mw"].reindex(frame.index), "strict": frame["strict"].astype(bool)})
    rows = rows[rows["delivery_day"] >= yardstick.TIGHTNESS_START]
    needed = ["error", "wind_surprise_gw", "solar_surprise_gw", "load_fc_mw", "wind_proxy_mw", "solar_proxy_mw", "nuclear_d1_early_mw"]
    complete = (rows[needed].notna().all(axis=1) & rows["strict"]).groupby(rows["delivery_day"]).all()
    rows = rows[rows["delivery_day"].isin(complete[complete].index)]
    tight = yardstick.tightness(rows["load_fc_mw"], rows["wind_proxy_mw"], rows["solar_proxy_mw"], rows["nuclear_d1_early_mw"], rows["delivery_day"])
    tercile = yardstick.terciles(tight)
    daily = rows.groupby("delivery_day")[["error", "wind_surprise_gw", "solar_surprise_gw"]].mean()
    daily["tightness_gw"], daily["tercile"] = tight, tercile
    rows = rows.assign(tercile=tercile.reindex(pd.DatetimeIndex(rows["delivery_day"])).to_numpy())
    return {"hourly": rows, "daily": daily, "days_from_start": int(complete.size), "days_complete": int(complete.sum())}


def weather_effect(hourly: pd.DataFrame, frame: pd.DataFrame, column: str = "S1") -> dict:
    t = weather_tables(hourly, frame, column)
    daily, rows = t["daily"], t["hourly"]
    cuts = {"tercile_1_upper_gw": float(daily["tightness_gw"].quantile(1 / 3)), "tercile_2_upper_gw": float(daily["tightness_gw"].quantile(2 / 3))}
    return {"model": column, "days_from_2024_05_01": t["days_from_start"], "days_complete": t["days_complete"], "tightness_cuts": cuts,
            "tightness_mean_by_tercile": {f"tercile_{int(k)}": float(v) for k, v in daily.groupby("tercile")["tightness_gw"].mean().items()},
            "daily": yardstick.weather_effect(daily, daily["tercile"]), "hourly": yardstick.weather_effect(rows, rows["tercile"])}


# --- the report -------------------------------------------------------------------------------------------------------------

def yardstick_report(frame: pd.DataFrame, hourly: pd.DataFrame, models=("honest_v2", "L0", "S1")) -> dict:
    """M1 to M7 for the yardstick models (and LEAR-AR where present)."""
    named = [m for m in models if m in frame] + (["LEAR_AR"] if "LEAR_AR" in frame else [])
    parts = yardstick.split(frame)
    out = {"M1": {m: yardstick.relative_mae(frame, m) for m in named},
           "M2": {m: yardstick.under_reaction(frame, m) for m in named},
           "M3": {m: yardstick.paired_daily(frame, m, "S1") for m in ("honest_v2", "L0")},
           "M4": {m: yardstick.shape(frame, m) for m in named + ["naive_day"]},
           "M5": {m: {k: yardstick.relative_mae(part, m) for k, part in parts.items()} for m in named},
           "M7": weather_effect(hourly, frame, "S1")}
    out["M5_split"] = {k: {"hours": int(len(yardstick.strict_rows(p))), "days": int(yardstick.strict_rows(p)["delivery_day"].nunique())}
                       for k, p in parts.items()}
    if "LEAR_AR" in frame:
        out["M6"] = {"mae": yardstick.mae(frame, "LEAR_AR"), "L0_mae": yardstick.mae(frame, "L0"),
                     "L0_minus_LEAR_AR_paired": yardstick.paired_daily(frame, "L0", "LEAR_AR"),
                     "S1_minus_LEAR_AR_paired": yardstick.paired_daily(frame, "S1", "LEAR_AR")}
    out["M7_cross_check_honest_v2"] = weather_effect(hourly, frame, "honest_v2")
    return out


def full_report(hourly: pd.DataFrame, frame: pd.DataFrame) -> dict:
    hourly = selection_inputs(hourly)
    daily_price = daily_mean_price(hourly)
    report = {"reproduction": reproduction(frame)}
    if not report["reproduction"]["ok"]:
        report["stopped"] = "a known figure did not reproduce: stop condition"
        return report
    report["yardstick"] = yardstick_report(frame, hourly)
    candidates = [c for c in CANDIDATES if c in frame and frame[c].notna().any()]
    report["rule"] = apply_rule(frame, candidates)
    window_columns = {"L0": {w: f"lear_w{w}" for w in MODEL_WINDOWS["L0"]}, "A": {w: f"lear_w{w}" for w in MODEL_WINDOWS["A"]},
                      "B": {w: f"lear_w{w}" for w in ALL_WINDOWS}, "C": {w: f"C_w{w}" for w in ALL_WINDOWS}}
    report["extras"] = {name: extras(frame, name, daily_price, window_columns.get(name)) for name in [BASELINE, *candidates]}
    if "C" in candidates:
        strict = yardstick.strict_rows(frame)
        share = strict["C_slope"].value_counts(normalize=True).sort_index()
        report["extras"]["C"]["slope_share_of_strict_hours"] = {f"{c:g}": float(v) for c, v in share.items()}
        modal = strict.groupby("delivery_day")["C_slope"].agg(lambda s: s.value_counts().sort_index(ascending=False).idxmax())
        report["extras"]["C"]["slope_share_of_days_modal"] = {f"{c:g}": float(v) for c, v in modal.value_counts(normalize=True).sort_index().items()}
        report["extras"]["C"]["fixed_slope_mae_six_windows"] = {f"{c:g}": yardstick.mae(frame, f"B_c{c:g}") for c in SLOPES}
        report["extras"]["C"]["strict_hours_without_choice"] = int(strict["C_slope"].isna().sum())
    if report["rule"]["lear_r"]:
        name = report["rule"]["lear_r"]
        frame["S1_R"] = (frame["honest_v2"] + frame[name]) / 2
        report["S1_R"] = {"lear_r": name, "mae": yardstick.mae(frame, "S1_R"), "S1_mae": yardstick.mae(frame, "S1"),
                          "honest_v2_mae": yardstick.mae(frame, "honest_v2"),
                          "paired_vs_S1": yardstick.paired_daily(frame, "S1_R", "S1"),
                          "paired_vs_honest_v2": yardstick.paired_daily(frame, "S1_R", "honest_v2"),
                          "slices": slice_maes(frame, "S1_R"), "S1_slices": slice_maes(frame, "S1"),
                          "extras": extras(frame, "S1_R", daily_price)}
    return report


# --- checks -----------------------------------------------------------------------------------------------------------------

def sample_days(n: int, seed: int = 0, first=SELECTION_START, end: str = TEST_END) -> pd.DatetimeIndex:
    days = pd.date_range(first, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    return pd.DatetimeIndex(sorted(np.random.default_rng(seed).choice(days, size=n, replace=False)))


def gram_check(hourly: pd.DataFrame, days, log=print) -> dict:
    """L0 for the sampled days by the present code (lear_walk_forward, defaults) and with the shared Gram matrix; the largest
    absolute difference in EUR/MWh over the hourly forecasts (the four-window mean and every window)."""
    hourly = selection_inputs(hourly)
    design, price, blocks = lear.lear_design(hourly)
    worst, per_day, per_window = 0.0, {}, []
    t_old = t_new = 0.0
    for day in pd.DatetimeIndex(days):
        t0 = time.time()
        old = lear.lear_walk_forward(hourly, f"{day:%Y-%m-%d}", f"{day + pd.Timedelta(days=1):%Y-%m-%d}", log=lambda *a, **k: None)
        t1 = time.time()
        tables = {w: forecast_days(design, price, blocks, [day], w, 1.0, GRAM_OPTIONS) for w in lear.LEAR_WINDOWS}
        t2 = time.time()
        t_old, t_new = t_old + t1 - t0, t_new + t2 - t1
        diffs = [float((to_hourly(old, tables[w]) - old[f"lear_w{w}"]).abs().max()) for w in lear.LEAR_WINDOWS]
        diffs.append(float((to_hourly(old, window_average(list(tables.values()))) - old["lear"]).abs().max()))
        per_day[f"{day:%Y-%m-%d}"] = max(diffs)
        per_window.append({"day": f"{day:%Y-%m-%d}", **{f"w{w}": d for w, d in zip(lear.LEAR_WINDOWS, diffs)}, "mean": diffs[-1]})
        worst = max(worst, max(diffs))
        log(f"{day:%Y-%m-%d}: max abs difference {max(diffs):.3e}", flush=True)
    return {"days": list(per_day), "max_abs_difference_by_day": per_day, "by_window": per_window, "max_abs_difference": worst,
            "tolerance": GRAM_TOLERANCE,
            "passes": bool(worst <= GRAM_TOLERANCE), "seconds_present_code": round(t_old, 1), "seconds_shared_gram": round(t_new, 1)}


def scramble_from(hourly: pd.DataFrame, day: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Every input value dated `day` or later that is not known at the gate for `day` replaced with noise.

    Known at the gate for D, and so left alone on D itself: the load forecast, and the weather forecasts and generation proxies
    issued two days ahead (the timing rules load_forecast, weather_issued, wind_proxy, solar_proxy of forecast/timing.py).
    Everything else on D is scrambled, every column of every later day is scrambled (the inputs normally hold nothing beyond
    the day being forecast), and so is the nuclear output of the D-1 hours from 08:00 Paris, which the rule nuclear_d1 excludes.
    The noise is Gaussian, with five times the column's standard deviation, around the column's mean plus that much again.
    """
    out = hourly.copy()
    rng = np.random.default_rng(seed)
    days = delivery_days(out.index)
    local_hour = out.index.tz_convert("Europe/Paris").hour
    known_on_day = {"load_fc_mw", "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2", "wind_proxy_mw", "solar_proxy_mw"}
    known_on_day |= {c for c in out.columns if c.startswith("wind_pt_") or c.startswith("solar_pt_")}
    later, same = np.asarray(days > day), np.asarray(days == day)
    for column in out.columns:
        mask = later | (same if column not in known_on_day else False)
        if column == "nuclear_mw":
            mask = mask | np.asarray((days == day - pd.Timedelta(days=1)) & (local_hour >= NUCLEAR_D1_CUTOFF_HOUR))
        n = int(np.sum(mask))
        if n:
            spread = float(np.nanstd(hourly[column].to_numpy(dtype=float))) or 1.0
            centre = float(np.nanmean(hourly[column].to_numpy(dtype=float)))
            out.loc[mask, column] = centre + 5 * spread + rng.normal(0, 5 * spread, n)
    return out


def scramble_check(hourly: pd.DataFrame, out: Path, days, slopes=SLOPES, windows=ALL_WINDOWS, log=print) -> dict:
    """For each sampled day D: refit every (slope, window) forecast for D from inputs scrambled from D onward and compare with
    the stored chunk; then rerun C's rolling choice with every actual price from D onward scrambled and compare C's forecast."""
    base = selection_inputs(hourly)
    info = build_features(base, "honest_v2").info
    info = info.assign(actual=base["price_eur_mwh"].reindex(info.index))
    wide = info[info["delivery_day"] >= pd.Timestamp(SLOPE_START)]
    stored = {(c, w): read_window(out, "full", c, w, SLOPE_START) for c in slopes for w in windows}
    slope_frame = pd.DataFrame({c: to_hourly(wide, window_average([stored[(c, w)] for w in windows])) for c in slopes})
    chosen0, value0 = rolling_slope_choice(slope_frame, wide["actual"], wide["delivery_day"], wide["hour"])
    result = {"days": {}, "max_abs_difference": 0.0, "paths": {}}
    for k, day in enumerate(pd.DatetimeIndex(days)):
        scrambled = scramble_from(base, day, seed=k)
        design, price, blocks = lear.lear_design(scrambled)
        a, b = scrambled.to_numpy(dtype=float), base.to_numpy(dtype=float)
        changed_inputs = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
        worst = {}
        for c in slopes:
            for w in windows:
                again = forecast_days(design, price, blocks, [day], w, c)
                diff = float((again.loc[day] - stored[(c, w)].loc[day]).abs().max())
                path = ("short window" if w in SHORT_WINDOWS else "long window") + (", slope 1" if c == 1.0 else ", slope below 1")
                worst[path] = max(worst.get(path, 0.0), diff)
        # the rolling choice: actual prices from D onward replaced by noise, the forecasts of later days too
        actual = wide["actual"].copy()
        from_day = (wide["delivery_day"] >= day).to_numpy()
        rng = np.random.default_rng(1000 + k)
        actual[from_day] = rng.normal(500, 300, int(from_day.sum()))
        noisy = slope_frame.copy()
        after = (wide["delivery_day"] > day).to_numpy()
        noisy.loc[after] = rng.normal(500, 300, (int(after.sum()), noisy.shape[1]))
        chosen1, value1 = rolling_slope_choice(noisy, actual, wide["delivery_day"], wide["hour"])
        on_day = (wide["delivery_day"] == day).to_numpy()
        same_choice = bool((chosen1[on_day].to_numpy() == chosen0[on_day].to_numpy()).all())
        worst["rolling slope choice (C)"] = float((value1[on_day] - value0[on_day]).abs().max()) if same_choice else float("inf")
        result["days"][f"{day:%Y-%m-%d}"] = {"scrambled_values": changed_inputs, **worst}
        for path, value in worst.items():
            result["paths"][path] = max(result["paths"].get(path, 0.0), value)
        result["max_abs_difference"] = max(result["max_abs_difference"], max(worst.values()))
        log(f"{day:%Y-%m-%d}: {changed_inputs} input values scrambled, max abs difference {max(worst.values()):.3e}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= 1e-9)
    return result


# --- command line -----------------------------------------------------------------------------------------------------------

def _shard(text: str) -> tuple[int, int]:
    a, b = text.split("/")
    return int(a), int(b)


def main(argv=None) -> None:
    from .inputs import load_inputs

    parser = argparse.ArgumentParser(description="LEAR to the published recipe (docs/prereg/lear_recipe_prereg.md)")
    parser.add_argument("stage", choices=["gram-check", "gbm", "lear", "report", "scramble"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", default="full", choices=["full", "ar"])
    parser.add_argument("--slopes", default="1", help="comma-separated asinh slopes")
    parser.add_argument("--windows", default=None, help="comma-separated windows in days (default: all six, or the four of L0 for ar)")
    parser.add_argument("--shard", default="0/1", type=_shard)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--days", type=int, default=20, help="sample size of gram-check and scramble")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partial", action="store_true", help="report without the slope runs (L0, A and B only)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hourly = load_inputs(Path(args.data))
    if args.stage == "gram-check":
        result = gram_check(hourly, sample_days(args.days, args.seed))
        (out / "gram_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes", "seconds_present_code", "seconds_shared_gram")}))
    elif args.stage == "gbm":
        path = out / "gbm_honest_v2.csv"
        if path.exists():
            print("gbm stage already complete")
            return
        gbm_honest_v2(hourly, log=lambda *a: print(*a, flush=True), out=out).to_csv(path, index_label="timestamp_utc")
    elif args.stage == "lear":
        slopes = [float(c) for c in args.slopes.split(",")]
        first = WALK_START if args.kind == "ar" else SLOPE_START
        windows = [int(w) for w in args.windows.split(",")] if args.windows else (lear.LEAR_WINDOWS if args.kind == "ar" else ALL_WINDOWS)
        left = run_jobs(hourly, out, job_list(args.kind, slopes, windows, first), first, args.shard,
                        None if args.budget_minutes is None else 60 * args.budget_minutes)
        print(f"chunks left in this shard: {left}")
    elif args.stage == "report":
        built = assemble(hourly, read_frame(out / "gbm_honest_v2.csv"), out, with_slopes=not args.partial,
                         with_ar=chunk_path(out, "ar", 1.0, 1456, "2026-06").exists())
        frame = built["frame"]
        report = full_report(hourly, frame)
        name = "results_partial.json" if args.partial else "results.json"
        (out / name).write_text(json.dumps(report, indent=2, default=str))
        frame.to_csv(out / ("frame_partial.csv" if args.partial else "frame.csv"), index_label="timestamp_utc")
        if "slopes" in built:
            built["slopes"].to_csv(out / "slopes.csv", index_label="timestamp_utc")
        pred = out / "predictions"
        pred.mkdir(exist_ok=True)
        skip = {"actual", "strict", "delivery_day", "hour", "dow", "holiday", "is_peak", "weather_point_in_time", "wind_mw", "solar_mw",
                "train_rows", "C_slope"}
        for column in frame.columns:
            if column not in skip:
                tidy(frame, column).to_csv(pred / f"{column}.csv", index=False)
        print(json.dumps(report["reproduction"], indent=2))
        if "rule" in report:
            for cand, c in report["rule"]["candidates"].items():
                print(cand, round(c["mae"], 4), round(c["gain"], 4), c["conditions"], c["passes"])
            print(report["rule"]["finding"])
    elif args.stage == "scramble":
        result = scramble_check(hourly, out, sample_days(args.days, args.seed + 1))
        (out / "scramble_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "paths", "passes")}, indent=2))


if __name__ == "__main__":
    main()
