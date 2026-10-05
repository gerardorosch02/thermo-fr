"""Forecast one delivery day with what was known at 12:00 Paris the day before.

The stored inputs table (forecast-fetch) supplies the history. The days around
the requested date are fetched fresh from ENTSO-E (prices up to D-1, the load
forecast for D) and Open-Meteo (the forecasts issued two days before each
hour of D), merged over the stored table, and the honest feature set is built.
The gradient boosting model is fitted on every delivery day before D and the
look-ahead check is run on the rows of D before predicting.
"""

from pathlib import Path

import pandas as pd

from ..data.entsoe_client import EntsoeSource
from ..data.weather_forecast import OpenMeteoForecastSource
from .features import build_features, feature_timings
from .inputs import INPUT_COLUMNS
from .models import fit_predict
from .timing import check_point_in_time

HISTORY_DAYS = 10  # refreshed around the target day so that lags and the day itself are current


def refresh_window(date: str, cache_dir=Path("data/cache"), log=print) -> pd.DataFrame:
    """Inputs for [D - HISTORY_DAYS, D + 1) straight from the sources."""
    day = pd.Timestamp(date).normalize()
    start = (day - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = (day + pd.Timedelta(days=2)).strftime("%Y-%m-%d")  # UTC window covers the whole Paris day
    entsoe = EntsoeSource(cache_dir=Path(cache_dir) / "entsoe")
    log(f"Refreshing ENTSO-E prices and load forecast for {start} to {end} ...")
    price = entsoe.day_ahead_prices(start, end)
    load_fc = entsoe.load_forecast(start, end)
    log("Refreshing Open-Meteo forecasts as issued two days ahead ...")
    weather = OpenMeteoForecastSource(cache_dir=Path(cache_dir) / "open-meteo").fetch(start, end, kinds=("issued",))
    fresh = pd.concat([price, load_fc, weather], axis=1).sort_index()
    return fresh.reindex(columns=INPUT_COLUMNS)


def forecast_day(date: str, inputs: pd.DataFrame, fresh: pd.DataFrame | None = None, model: str = "gbm", log=print) -> pd.DataFrame:
    """Hourly forecast curve for delivery day `date` (Paris), honest feature set only."""
    day = pd.Timestamp(date).normalize()
    merged = inputs.copy()
    if fresh is not None:
        merged = fresh.combine_first(merged).sort_index()
        merged = merged.reindex(columns=INPUT_COLUMNS)
    table = build_features(merged, "honest")
    rows = table.info["delivery_day"] == day
    if not rows.any():
        raise ValueError(f"No input hours found for {date}; the inputs table ends at {merged.index.max()}.")
    check_point_in_time(table.X.index[rows], feature_timings("honest"))
    if not table.info.loc[rows, "weather_point_in_time"].all():
        raise ValueError(f"Weather forecasts as issued are missing for {date}; refusing to use the proxy.")
    if table.X.loc[rows, "load_fc_mw"].isna().all():
        raise ValueError(f"ENTSO-E load forecast for {date} is not available yet.")
    train = table.info["delivery_day"] < day
    log(f"Fitting {model} on {int(table.y[train].notna().sum()):,} hours before {date} ...")
    prediction = fit_predict(model, table.X[train], table.y[train], table.X[rows])
    curve = pd.DataFrame(
        {
            "hour": table.info.loc[rows, "hour"].to_numpy(),
            "forecast": prediction.round(2),
            "naive_day": table.X.loc[rows, "price_lag1"].to_numpy(),
            "actual": table.y[rows].to_numpy(),
            "load_fc_mw": table.X.loc[rows, "load_fc_mw"].to_numpy(),
            "temp_c": table.X.loc[rows, "temp_c"].round(1).to_numpy(),
        },
        index=table.X.index[rows],
    )
    curve.index.name = "timestamp_utc"
    return curve
