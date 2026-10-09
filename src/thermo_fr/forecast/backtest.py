"""Walk-forward backtest with monthly retraining, and the evaluation tables.

For every month of the test window the models are fitted on all rows whose
delivery day is before the first day of that month, then used to forecast
every hour of that month. Nothing from the month itself, or later, enters the
fit. Both benchmarks and both models are evaluated on exactly the same hours.

Two notions of point in time are kept apart. A feature set either passes the
look-ahead check for every row (honest) or not (extended, which carries the
ENTSO-E wind and solar forecasts); that is recorded once per backtest. Rows
whose weather came from the historical-forecast proxy (before the as-issued
archive begins) are still forecast and reported, but flagged as not strict
and kept out of the strict metrics, for both feature sets.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FeatureTable, build_features, feature_timings
from .models import BENCHMARKS, MODELS, benchmark_predictions, fit_predict
from .timing import LookaheadError, check_point_in_time

TOP_SHARE = 0.05
WINDY_SHARE = 0.10  # the windiest days by actual national wind generation (daily mean)
SUNNY_SHARE = 0.10  # the sunniest days by actual national solar generation (daily mean)
DEFAULT_SETS = ("honest", "honest_wind", "honest_base", "extended")


@dataclass
class BacktestResult:
    feature_set: str
    predictions: pd.DataFrame  # actual, benchmarks, models, info columns, strict flag
    months: list[str] = field(default_factory=list)
    point_in_time: bool = True  # whether every feature of the set passes the 12:00 gate


def month_starts(test_start: str, test_end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end), freq="MS", inclusive="left"))


def walk_forward(table: FeatureTable, test_start: str, test_end: str, models=MODELS, log=print) -> BacktestResult:
    """Retrain at the start of every month on all earlier days, forecast that month."""
    days = table.info["delivery_day"]
    frames = []
    months = month_starts(test_start, test_end)
    for start in months:
        end = start + pd.DateOffset(months=1)
        train = days < start
        test = (days >= start) & (days < end)
        if not test.any():
            continue
        X_train, y_train, X_test = table.X[train], table.y[train], table.X[test]
        out = table.info[test].copy()
        out["actual"] = table.y[test]
        out = out.join(benchmark_predictions(X_test))
        for name in models:
            out[name] = fit_predict(name, X_train, y_train, X_test)
        out["strict"] = table.info.loc[test, "weather_point_in_time"].to_numpy()
        out["train_rows"] = int(y_train.notna().sum())
        frames.append(out)
        log(f"  {table.feature_set}: {start:%Y-%m} trained on {int(y_train.notna().sum()):,} hours")
    predictions = pd.concat(frames).sort_index()
    return BacktestResult(
        table.feature_set, predictions, [m.strftime("%Y-%m") for m in months], passes_gate(table, predictions.index)
    )


def passes_gate(table: FeatureTable, index) -> bool:
    """Whether every feature of the set is published before the gate for every forecast row."""
    try:
        check_point_in_time(index, feature_timings(table.feature_set))
        return True
    except LookaheadError:
        return False


def errors(frame: pd.DataFrame, column: str) -> dict:
    diff = frame[column] - frame["actual"]
    diff = diff.dropna()
    if diff.empty:
        return {"mae": None, "rmse": None, "hours": 0}
    return {
        "mae": round(float(diff.abs().mean()), 2),
        "rmse": round(float(np.sqrt((diff**2).mean())), 2),
        "bias": round(float(diff.mean()), 2),
        "hours": int(len(diff)),
    }


def evaluate(predictions: pd.DataFrame, strict_only: bool = True) -> dict:
    """MAE and RMSE overall and by slice, for both benchmarks and both models."""
    frame = predictions[predictions["strict"]] if strict_only else predictions
    frame = frame.dropna(subset=["actual"])
    columns = list(BENCHMARKS) + list(MODELS)
    top_cut = frame["actual"].quantile(1 - TOP_SHARE)
    slices = {
        "all": frame,
        "peak": frame[frame["is_peak"] == 1],
        "off_peak": frame[frame["is_peak"] == 0],
        f"top_{int(TOP_SHARE * 100)}pct_price_hours": frame[frame["actual"] >= top_cut],
        "negative_price_hours": frame[frame["actual"] < 0],
    }
    if "dow" in frame:
        slices["weekends"] = frame[frame["dow"] >= 5]
    if "holiday" in frame:
        slices["holidays"] = frame[frame["holiday"] == 1]
    windy_cut = sunny_cut = None
    if "wind_mw" in frame and frame["wind_mw"].notna().any():
        daily_wind = frame.groupby("delivery_day")["wind_mw"].mean().dropna()
        windy_cut = float(daily_wind.quantile(1 - WINDY_SHARE))
        windy_days = daily_wind[daily_wind >= windy_cut].index
        slices[f"windiest_{int(WINDY_SHARE * 100)}pct_days"] = frame[frame["delivery_day"].isin(windy_days)]
    if "solar_mw" in frame and frame["solar_mw"].notna().any():
        daily_solar = frame.groupby("delivery_day")["solar_mw"].mean().dropna()
        sunny_cut = float(daily_solar.quantile(1 - SUNNY_SHARE))
        sunny_days = daily_solar[daily_solar >= sunny_cut].index
        slices[f"sunniest_{int(SUNNY_SHARE * 100)}pct_days"] = frame[frame["delivery_day"].isin(sunny_days)]
    result = {
        "rows": int(len(frame)),
        "strict_only": strict_only,
        "first_day": str(frame["delivery_day"].min().date()) if len(frame) else None,
        "last_day": str(frame["delivery_day"].max().date()) if len(frame) else None,
        "top_price_cut_eur_mwh": round(float(top_cut), 2) if len(frame) else None,
        "windy_day_cut_mw": round(windy_cut, 1) if windy_cut is not None else None,
        "sunny_day_cut_mw": round(sunny_cut, 1) if sunny_cut is not None else None,
        "slices": {name: {col: errors(part, col) for col in columns} for name, part in slices.items()},
        "by_hour": {},
        "by_month": {},
        "improvement_pct": {},
    }
    for hour, part in frame.groupby("hour"):
        result["by_hour"][int(hour)] = {col: errors(part, col)["mae"] for col in columns}
    for month, part in frame.groupby(frame["delivery_day"].dt.to_period("M")):
        result["by_month"][str(month)] = {col: errors(part, col)["mae"] for col in columns}
    overall = result["slices"]["all"]
    for model in MODELS:
        result["improvement_pct"][model] = {
            bench: round(100.0 * (1 - overall[model]["mae"] / overall[bench]["mae"]), 1)
            for bench in BENCHMARKS
            if overall[bench]["mae"]
        }
    return result


def worst_days(predictions: pd.DataFrame, table: FeatureTable, hourly: pd.DataFrame, model: str = "gbm", n: int = 20) -> pd.DataFrame:
    """The n delivery days with the largest daily MAE for `model`, with context for each."""
    frame = predictions.dropna(subset=["actual"]).copy()
    frame["abs_err"] = (frame[model] - frame["actual"]).abs()
    frame["naive_abs_err"] = (frame["naive_day"] - frame["actual"]).abs()
    X = table.X.loc[frame.index]
    frame["temp_c"] = X["temp_c"]
    frame["wind100_ms"] = X["wind100_ms"]
    frame["load_fc_mw"] = X["load_fc_mw"]
    frame["holiday"] = X["holiday"]
    frame["dow"] = X["dow"]
    renewables = hourly["solar_fc_mw"].add(hourly["wind_onshore_fc_mw"], fill_value=0).add(hourly["wind_offshore_fc_mw"], fill_value=0)
    frame["renewables_fc_mw"] = renewables.reindex(frame.index)

    # temperature anomaly: against the mean of the same calendar month over the whole input table
    climatology = table.X.groupby([table.X.index.month, table.X.index.hour])["temp_c"].mean()
    keys = pd.MultiIndex.from_arrays([frame.index.month, frame.index.hour])
    frame["temp_anomaly_c"] = frame["temp_c"].to_numpy() - climatology.reindex(keys).to_numpy()

    daily = frame.groupby("delivery_day").agg(
        mae=("abs_err", "mean"),
        naive_mae=("naive_abs_err", "mean"),
        bias=("abs_err", lambda s: float((frame.loc[s.index, model] - frame.loc[s.index, "actual"]).mean())),
        mean_price=("actual", "mean"),
        max_price=("actual", "max"),
        min_price=("actual", "min"),
        negative_hours=("actual", lambda s: int((s < 0).sum())),
        prev_day_mean_price=("naive_day", "mean"),
        temp_c=("temp_c", "mean"),
        temp_anomaly_c=("temp_anomaly_c", "mean"),
        wind100_ms=("wind100_ms", "mean"),
        load_fc_mw=("load_fc_mw", "mean"),
        renewables_fc_mw=("renewables_fc_mw", "mean"),
        holiday=("holiday", "max"),
        dow=("dow", "first"),
        strict=("strict", "min"),
    )
    daily["price_jump_vs_prev_day"] = daily["mean_price"] - daily["prev_day_mean_price"]
    return daily.sort_values("mae", ascending=False).head(n)


def summarise_worst(worst: pd.DataFrame, predictions: pd.DataFrame, model: str = "gbm") -> dict:
    """What the worst days have in common, against the whole test window."""
    all_days = predictions.dropna(subset=["actual"]).groupby("delivery_day").agg(
        mean_price=("actual", "mean"), negative_hours=("actual", lambda s: int((s < 0).sum()))
    )
    weekend_or_holiday = ((worst["dow"] >= 5) | (worst["holiday"] > 0)).mean()
    return {
        "days": int(len(worst)),
        "mean_daily_mae": round(float(worst["mae"].mean()), 1),
        "share_weekend_or_holiday": round(float(weekend_or_holiday), 2),
        "share_holiday": round(float((worst["holiday"] > 0).mean()), 2),
        "share_with_negative_hours": round(float((worst["negative_hours"] > 0).mean()), 2),
        "share_with_negative_hours_all_days": round(float((all_days["negative_hours"] > 0).mean()), 2),
        "share_price_spike_days": round(float((worst["max_price"] >= predictions["actual"].quantile(0.99)).mean()), 2),
        "mean_abs_price_jump_vs_prev_day": round(float(worst["price_jump_vs_prev_day"].abs().mean()), 1),
        "mean_abs_price_jump_all_days": round(
            float((all_days["mean_price"] - all_days["mean_price"].shift(1)).abs().mean()), 1
        ),
        "mean_temp_anomaly_c": round(float(worst["temp_anomaly_c"].mean()), 1),
        "share_cold_anomaly_below_minus3": round(float((worst["temp_anomaly_c"] < -3).mean()), 2),
        "mean_wind100_ms": round(float(worst["wind100_ms"].mean()), 1),
        "share_naive_also_bad": round(float((worst["naive_mae"] > worst["mae"]).mean()), 2),
        "months": worst.index.to_period("M").astype(str).value_counts().to_dict(),
    }


def run_backtest(hourly: pd.DataFrame, test_start: str, test_end: str, feature_sets=DEFAULT_SETS, log=print) -> dict:
    """The requested feature sets through the walk-forward, with metrics and worst days for each."""
    results = {}
    for feature_set in feature_sets:
        log(f"Feature set {feature_set}: building features and running the walk-forward ...")
        table = build_features(hourly, feature_set)
        bt = walk_forward(table, test_start, test_end, log=log)
        worst = worst_days(bt.predictions, table, hourly)
        results[feature_set] = {
            "table": table,
            "backtest": bt,
            "metrics_strict": evaluate(bt.predictions, strict_only=True),
            "metrics_all": evaluate(bt.predictions, strict_only=False),
            "worst_days": worst,
            "worst_summary": summarise_worst(worst, bt.predictions),
        }
    return results
