"""The shape of the day: each hour's price minus the day's base average, forecast, benchmarked and valued with a small battery.

The level forecast (forecast/features.py, the honest_v2 set) is scored on its
hourly error against the auction result. This module scores what a storage
asset cares about: the shape. The shape of a delivery day is each hour's
price minus the day's mean (the base). Three forecasts of it are compared:

- the shape model: gradient boosting on the honest_v2 inputs with the shape
  as the target, retrained at the start of every month (walk_forward_shape);
- the D-1 benchmark: yesterday's shape (price_lag1 minus its daily mean);
- the same-type benchmark: the shape of the most recent earlier day of the
  same type (price_lag_same_type minus its daily mean, see daytypes.py).

Metrics (shape_metrics): the hourly shape MAE; the peak minus off-peak spread
error (peak hours 08:00 to 20:00 Paris); and the hit rate of the cheapest two
and the most expensive two hours, the share of the two forecast hours that
fall among the actual two.

Battery backtest (battery_day, battery_backtest): 1 MW / 2 MWh, 88% round-trip
efficiency, at most one cycle a day, a two-hour charge block before a
two-hour discharge block inside the delivery day. The blocks are chosen on
the forecast price curve (base forecast plus shape forecast) before the gate
and settled at the auction result: value = 0.88 x sum of the two discharge
prices minus the sum of the two charge prices, in EUR for the 2 MWh cycle.
A day is skipped (value 0) when the forecast value of the best blocks is not
positive, that is when the forecast spread does not cover the efficiency
loss. Perfect foresight chooses the blocks on the actual prices; the share
of perfect foresight captured is the ratio of total values.

Everything here works on a frame with one row per hour of a delivery day:
columns delivery_day, hour, actual, and one column per forecast. The same
functions score the backtest tables and the live versions in the store
(live_shape_record), so the dashboards show one definition.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import PEAK_HOURS, FeatureTable
from .models import make_model

ROUND_TRIP = 0.88
BLOCK_HOURS = 2
TOP_N = 2
BENCHMARKS = {"d1": "price_lag1", "same_type": "price_lag_same_type"}


def daily_mean(values: pd.Series, days: pd.Series) -> pd.Series:
    """The mean of `values` over each delivery day, broadcast back to the rows."""
    return values.groupby(days.to_numpy()).transform("mean")


def shape_of(values: pd.Series, days: pd.Series) -> pd.Series:
    return values - daily_mean(values, days)


def benchmark_shapes(table: FeatureTable) -> pd.DataFrame:
    """The two benchmark shapes and their base levels, from the lagged price columns of the feature table."""
    days = table.info["delivery_day"]
    out = pd.DataFrame(index=table.X.index)
    for name, column in BENCHMARKS.items():
        values = table.X[column]
        out[f"shape_{name}"] = shape_of(values, days)
        out[f"base_{name}"] = daily_mean(values, days)
    return out


@dataclass
class ShapeBacktest:
    predictions: pd.DataFrame  # delivery_day, hour, actual, actual_shape, actual_base, shape_model, shape_d1, shape_same_type, base_*, strict
    months: list


def walk_forward_shape(table: FeatureTable, test_start: str, test_end: str, level: pd.Series | None = None, log=print) -> ShapeBacktest:
    """Monthly walk-forward of the shape model on the honest_v2 features; the shape target is the price minus the day's mean.

    `level`, if given, is an out-of-sample level forecast per hour (for example
    the gbm column of the honest_v2 walk-forward); its daily mean is the base
    forecast the battery decision adds to the shape forecast. Without it the
    base forecast is the D-1 mean.
    """
    days = table.info["delivery_day"]
    y_shape = shape_of(table.y, days)
    benchmarks = benchmark_shapes(table)
    frames, months = [], []
    for start in pd.date_range(pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end), freq="MS", inclusive="left"):
        end = start + pd.DateOffset(months=1)
        train = (days < start) & y_shape.notna()
        test = (days >= start) & (days < end)
        if not test.any():
            continue
        model = make_model("gbm")
        model.fit(table.X[train], y_shape[train])
        out = table.info.loc[test, ["delivery_day", "hour", "is_peak", "dow", "holiday"]].copy()
        out["actual"] = table.y[test]
        out["actual_base"] = daily_mean(table.y, days)[test]
        out["actual_shape"] = y_shape[test]
        out["shape_model"] = np.asarray(model.predict(table.X[test]), dtype=float)
        out = out.join(benchmarks[test])
        out["strict"] = table.info.loc[test, "weather_point_in_time"].to_numpy()
        frames.append(out)
        months.append(start.strftime("%Y-%m"))
        log(f"  shape: {start:%Y-%m} trained on {int(train.sum()):,} hours")
    predictions = pd.concat(frames).sort_index()
    if level is not None:
        level = level.reindex(predictions.index)
        predictions["base_model"] = daily_mean(level, predictions["delivery_day"])
    else:
        predictions["base_model"] = predictions["base_d1"]
    return ShapeBacktest(predictions, months)


def _by_day(frame: pd.DataFrame):
    return frame.groupby("delivery_day", sort=True)


def spread_error(frame: pd.DataFrame, column: str) -> float:
    """Mean over days of |forecast spread - actual spread|, spread = peak-hour mean minus off-peak mean of the shape."""
    errors = []
    for _, day in _by_day(frame):
        peak = day["hour"].isin(list(PEAK_HOURS))
        if peak.sum() == 0 or (~peak).sum() == 0:
            continue
        fc = day.loc[peak, column].mean() - day.loc[~peak, column].mean()
        actual = day.loc[peak, "actual_shape"].mean() - day.loc[~peak, "actual_shape"].mean()
        errors.append(abs(fc - actual))
    return float(np.mean(errors)) if errors else float("nan")


def extreme_hit_rate(frame: pd.DataFrame, column: str, cheapest: bool, n: int = TOP_N) -> float:
    """Share of the n forecast cheapest (or dearest) hours that are among the actual n, averaged over days."""
    hits = []
    for _, day in _by_day(frame):
        if len(day) < 2 * n:
            continue
        order_fc = day[column].to_numpy().argsort()
        order_actual = day["actual_shape"].to_numpy().argsort()
        pick_fc = set(order_fc[:n]) if cheapest else set(order_fc[-n:])
        pick_actual = set(order_actual[:n]) if cheapest else set(order_actual[-n:])
        hits.append(len(pick_fc & pick_actual) / n)
    return float(np.mean(hits)) if hits else float("nan")


def shape_metrics(frame: pd.DataFrame, methods=("model", "d1", "same_type")) -> dict:
    """Hourly shape MAE, spread error and extreme-hour hit rates for each method's shape_<method> column."""
    frame = frame.dropna(subset=["actual_shape"])
    out = {"hours": int(len(frame)), "days": int(frame["delivery_day"].nunique())}
    for method in methods:
        column = f"shape_{method}"
        part = frame.dropna(subset=[column])
        out[method] = {
            "shape_mae": round(float((part[column] - part["actual_shape"]).abs().mean()), 2),
            "spread_error": round(spread_error(part, column), 2),
            "cheapest2_hit_rate": round(extreme_hit_rate(part, column, cheapest=True), 3),
            "dearest2_hit_rate": round(extreme_hit_rate(part, column, cheapest=False), 3),
        }
    return out


def best_blocks(prices: np.ndarray, efficiency: float = ROUND_TRIP, block: int = BLOCK_HOURS) -> tuple[int, int, float]:
    """The charge block start, discharge block start and value (per 1 MW over `block` hours) that maximise efficiency x discharge - charge.

    The charge block ends before the discharge block starts. Returns value as
    computed on `prices`; the caller decides whether a non-positive value means
    skipping the day.
    """
    n = len(prices)
    best = (0, block, -np.inf)
    if n < 2 * block:
        return 0, 0, float("nan")
    window = np.convolve(prices, np.ones(block), mode="valid")  # sum of each block of consecutive hours
    for c in range(0, n - 2 * block + 1):
        for d in range(c + block, n - block + 1):
            value = efficiency * window[d] - window[c]
            if value > best[2]:
                best = (c, d, value)
    return best[0], best[1], float(best[2])


def battery_day(forecast: np.ndarray, actual: np.ndarray, efficiency: float = ROUND_TRIP, block: int = BLOCK_HOURS) -> dict:
    """Choose the blocks on the forecast, settle on the actual prices. value_eur is 0 when the forecast value is not positive."""
    c, d, forecast_value = best_blocks(forecast, efficiency, block)
    if not np.isfinite(forecast_value) or forecast_value <= 0:
        return {"charge_start": None, "discharge_start": None, "forecast_value": forecast_value, "value_eur": 0.0, "traded": False}
    value = efficiency * actual[d:d + block].sum() - actual[c:c + block].sum()
    return {"charge_start": int(c), "discharge_start": int(d), "forecast_value": float(forecast_value), "value_eur": float(value), "traded": True}


def battery_backtest(frame: pd.DataFrame, methods=("model", "d1", "same_type"), efficiency: float = ROUND_TRIP) -> dict:
    """Daily battery values for each method (blocks on base_<method> + shape_<method>) and for perfect foresight, with totals.

    Returns {"daily": DataFrame(delivery_day, value_<method>, traded_<method>, value_perfect), "summary": {...}}. EUR per day is
    the mean over every day in the frame, skipped days counting as zero; the share of perfect foresight is the ratio of totals.
    """
    rows = []
    for day, part in _by_day(frame.dropna(subset=["actual"])):
        part = part.sort_index()
        actual = part["actual"].to_numpy()
        row = {"delivery_day": day, "hours": len(part)}
        perfect = battery_day(actual, actual, efficiency)
        row["value_perfect"] = perfect["value_eur"]
        for method in methods:
            forecast = (part[f"base_{method}"] + part[f"shape_{method}"]).to_numpy()
            if np.isnan(forecast).any():
                row[f"value_{method}"], row[f"traded_{method}"] = np.nan, False
                continue
            result = battery_day(forecast, actual, efficiency)
            row[f"value_{method}"], row[f"traded_{method}"] = result["value_eur"], result["traded"]
            row[f"charge_{method}"], row[f"discharge_{method}"] = result["charge_start"], result["discharge_start"]
        rows.append(row)
    daily = pd.DataFrame(rows)
    summary = {"days": int(len(daily)), "perfect_eur_per_day": round(float(daily["value_perfect"].mean()), 2) if len(daily) else None}
    for method in methods:
        values = daily[f"value_{method}"].dropna() if len(daily) else pd.Series(dtype=float)
        total_perfect = float(daily.loc[values.index, "value_perfect"].sum()) if len(values) else 0.0
        summary[method] = {
            "eur_per_day": round(float(values.mean()), 2) if len(values) else None,
            "share_of_perfect": round(float(values.sum() / total_perfect), 3) if total_perfect > 0 else None,
            "days_traded": int(daily[f"traded_{method}"].sum()) if len(daily) else 0,
            "days_skipped": int((~daily[f"traded_{method}"].astype(bool)).sum()) if len(daily) else 0,
            "losing_days": int((values < 0).sum()) if len(values) else 0,
        }
    return {"daily": daily, "summary": summary}


def evaluate_shape(predictions: pd.DataFrame, strict_only: bool = True) -> dict:
    """Shape metrics and battery backtest on the strict rows of a walk-forward table."""
    frame = predictions[predictions["strict"]] if strict_only and "strict" in predictions else predictions
    frame = frame.dropna(subset=["actual"])
    shape = shape_metrics(frame)
    battery = battery_backtest(frame)
    return {"first_day": str(frame["delivery_day"].min().date()) if len(frame) else None,
            "last_day": str(frame["delivery_day"].max().date()) if len(frame) else None,
            "shape": shape, "battery": battery["summary"], "battery_daily": battery["daily"]}


def block_bootstrap_mean(values: np.ndarray, block: int = 7, resamples: int = 5000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval for the mean of a daily series from a moving-block bootstrap (blocks of `block` consecutive days).

    Each resample concatenates ceil(n / block) blocks drawn with replacement from every window of `block` consecutive values,
    truncated to n, and takes its mean; the interval is the central `level` share of those means. Serial dependence within a
    week is kept inside the blocks, which is why a plain i.i.d. bootstrap would be too narrow here.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    per = int(np.ceil(n / block))
    picks = rng.choice(starts, size=(resamples, per), replace=True)
    offsets = np.arange(block)
    index = (picks[:, :, None] + offsets[None, None, :]).reshape(resamples, -1)[:, :n]
    means = values[index].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_comparison(daily: pd.DataFrame, model: str, benchmark: str, block: int = 7, resamples: int = 5000, seed: int = 0) -> dict:
    """Mean daily difference model minus benchmark, its moving-block bootstrap interval, and the share of days the model wins."""
    pair = daily[[model, benchmark]].dropna()
    diff = (pair[model] - pair[benchmark]).to_numpy()
    low, high = block_bootstrap_mean(diff, block=block, resamples=resamples, seed=seed)
    return {"days": int(len(diff)), "mean_difference": round(float(diff.mean()), 2), "ci95_low": round(low, 2), "ci95_high": round(high, 2),
            "share_model_wins": round(float((diff > 0).mean()), 3), "share_ties": round(float((diff == 0).mean()), 3),
            "distinguishable_from_zero": bool(low > 0 or high < 0)}


def daily_shape_mae(frame: pd.DataFrame, methods=("model", "d1", "same_type")) -> pd.DataFrame:
    """One row per delivery day with the hourly shape MAE of each method (lower is better)."""
    frame = frame.dropna(subset=["actual_shape"])
    out = {}
    for method in methods:
        column = f"shape_{method}"
        err = (frame[column] - frame["actual_shape"]).abs()
        out[f"shape_mae_{method}"] = err.groupby(frame["delivery_day"].to_numpy()).mean()
    daily = pd.DataFrame(out)
    daily.index.name = "delivery_day"
    return daily.reset_index()


def paired_report(battery_daily: pd.DataFrame, shape_daily: pd.DataFrame, benchmarks=("d1", "same_type"), **kw) -> dict:
    """Battery value (model minus benchmark, higher is better) and shape MAE (benchmark minus model, so that positive favours the model)."""
    out = {"battery_value": {}, "shape_mae": {}}
    for bench in benchmarks:
        out["battery_value"][bench] = paired_comparison(battery_daily, "value_model", f"value_{bench}", **kw)
        reversed_frame = shape_daily.rename(columns={f"shape_mae_{bench}": "bench", "shape_mae_model": "model"})
        out["shape_mae"][bench] = paired_comparison(reversed_frame, "bench", "model", **kw)  # benchmark minus model: positive = model better
    return out


def live_frame(curve: pd.DataFrame, actual: pd.Series, delivery_day: str) -> pd.DataFrame:
    """A stored forecast version (columns forecast, naive_day) and the day's actual prices as one scoring frame (model and D-1 only)."""
    frame = pd.DataFrame(index=curve.index)
    frame["delivery_day"] = pd.Timestamp(delivery_day)
    frame["hour"] = curve["hour"].astype(int).to_numpy()
    frame["actual"] = actual.reindex(curve.index).to_numpy()
    days = frame["delivery_day"]
    frame["actual_base"] = daily_mean(frame["actual"], days)
    frame["actual_shape"] = frame["actual"] - frame["actual_base"]
    frame["base_model"] = daily_mean(curve["forecast"], days)
    frame["shape_model"] = curve["forecast"] - frame["base_model"]
    frame["base_d1"] = daily_mean(curve["naive_day"], days)
    frame["shape_d1"] = curve["naive_day"] - frame["base_d1"]
    return frame


def live_shape_record(store, feature_set: str, days: int = 90) -> dict:
    """Shape metrics and battery values of the pre-market version of each settled day in the store, for the dashboards."""
    scores = store.headline_scores(feature_set, days=days)
    frames = []
    for _, row in scores.iterrows():
        actual = store.actuals_for(row["delivery_day"])
        if actual.empty:
            continue
        curve = store.forecast_curve(int(row["forecast_id"]))
        frames.append(live_frame(curve, actual, row["delivery_day"]))
    if not frames:
        return {"days": 0}
    frame = pd.concat(frames)
    methods = ("model", "d1")
    metrics = shape_metrics(frame, methods)
    battery = battery_backtest(frame, methods)
    daily = battery["daily"]
    daily_shape = frame.groupby("delivery_day").apply(
        lambda g: pd.Series({"shape_mae_model": float((g["shape_model"] - g["actual_shape"]).abs().mean()),
                             "shape_mae_d1": float((g["shape_d1"] - g["actual_shape"]).abs().mean())}), include_groups=False).reset_index()
    daily = daily.merge(daily_shape, on="delivery_day", how="left")
    daily["delivery_day"] = pd.to_datetime(daily["delivery_day"]).dt.strftime("%Y-%m-%d")
    return {"days": int(metrics["days"]), "shape": metrics, "battery": battery["summary"], "daily": daily}
