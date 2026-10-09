"""Forecast one delivery day with what was known at 12:00 Paris the day before.

The stored inputs table (forecast-fetch) supplies the history. The days around
the requested date are fetched fresh from ENTSO-E (prices up to D-1, the load
forecast for D) and Open-Meteo (the forecasts issued two days before each
hour of D), merged over the stored table, and the feature set is built. The
gradient boosting model is fitted on every delivery day before D. For the
honest set the look-ahead check is run on the rows of D before predicting;
the extended set fails that check by construction (ENTSO-E wind and solar
forecasts may be published after the gate), so it is only built when asked
for explicitly and flagged as such.
"""

from pathlib import Path

import pandas as pd

from ..data.entsoe_client import EntsoeSource
from ..data.solar_points import SolarPointsSource
from ..data.weather_forecast import OpenMeteoForecastSource
from ..data.wind_points import WindPointsSource
from .features import GATED_SETS, build_features, feature_timings
from .gen_proxy import apply_weights, load_weights
from .inputs import INPUT_COLUMNS
from .models import fit_predict, predict_with
from .solar_proxy import SOLAR
from .timing import check_point_in_time
from .wind_proxy import WIND

HISTORY_DAYS = 10  # refreshed around the target day so that lags and the day itself are current
DEFAULT_WEIGHTS_PATH = WIND.default_path


def refresh_window(date: str, cache_dir=Path("data/cache"), log=print, wind_weights=WIND.default_path,
                   solar_weights=SOLAR.default_path) -> pd.DataFrame:
    """Inputs for [D - HISTORY_DAYS, D + 1) straight from the sources (honest inputs only).

    The wind and solar proxies for the window are computed from the stored
    calibration weights (`wind_weights`, `solar_weights`, written by
    refit-model); without a weights file that proxy column stays NaN and a
    warning is logged.
    """
    day = pd.Timestamp(date).normalize()
    start = (day - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = (day + pd.Timedelta(days=2)).strftime("%Y-%m-%d")  # UTC window covers the whole Paris day
    entsoe = EntsoeSource(cache_dir=Path(cache_dir) / "entsoe")
    log(f"Refreshing ENTSO-E prices and load forecast for {start} to {end} ...")
    price = entsoe.day_ahead_prices(start, end)
    load_fc = entsoe.load_forecast(start, end)
    log("Refreshing Open-Meteo forecasts as issued two days ahead ...")
    weather = OpenMeteoForecastSource(cache_dir=Path(cache_dir) / "open-meteo").fetch(start, end, kinds=("issued",))
    log("Refreshing Open-Meteo 100 m wind forecasts at the wind-region points ...")
    points = WindPointsSource(cache_dir=Path(cache_dir) / "open-meteo").fetch_points(start, end)
    log("Refreshing Open-Meteo radiation forecasts at the solar-region points ...")
    solar_points = SolarPointsSource(cache_dir=Path(cache_dir) / "open-meteo").fetch_points(start, end)
    pieces = [price, load_fc, weather, points, solar_points]
    for spec, frame, path in ((WIND, points, wind_weights), (SOLAR, solar_points, solar_weights)):
        weights = load_weights(path)
        if weights is None:
            log(f"Warning: no {spec.name} proxy weights at {path}; the {spec.name} proxy feature will be missing.")
        else:
            pieces.append(apply_weights(frame, weights, spec))
    fresh = pd.concat(pieces, axis=1).sort_index()
    return fresh.reindex(columns=INPUT_COLUMNS)


def merge_inputs(inputs: pd.DataFrame, fresh: pd.DataFrame | None) -> pd.DataFrame:
    merged = inputs.copy()
    if fresh is not None and not fresh.empty:
        merged = fresh.combine_first(merged).sort_index()
    return merged.reindex(columns=INPUT_COLUMNS).astype(float)


def forecast_day(date: str, inputs: pd.DataFrame, fresh: pd.DataFrame | None = None, model: str = "gbm", log=print,
                 feature_set: str = "honest", model_file=None) -> pd.DataFrame:
    """Hourly forecast curve for delivery day `date` (Paris) with one feature set.

    With `model_file` the stored LightGBM model is used instead of refitting, so
    only the days around `date` are needed in `inputs` (the lags reach back a
    week). The returned frame carries `train_hours` and `passes_gate` in attrs.
    """
    day = pd.Timestamp(date).normalize()
    merged = merge_inputs(inputs, fresh)
    table = build_features(merged, feature_set)
    rows = table.info["delivery_day"] == day
    if not rows.any():
        raise ValueError(f"No input hours found for {date}; the inputs table ends at {merged.index.max()}.")
    passes_gate = feature_set in GATED_SETS
    if passes_gate:
        check_point_in_time(table.X.index[rows], feature_timings(feature_set))
    if not table.info.loc[rows, "weather_point_in_time"].all():
        raise ValueError(f"Weather forecasts as issued are missing for {date}; refusing to use the proxy.")
    if table.X.loc[rows, "load_fc_mw"].isna().all():
        raise ValueError(f"ENTSO-E load forecast for {date} is not available yet.")
    if feature_set == "extended" and table.X.loc[rows, "residual_load_fc_mw"].isna().all():
        raise ValueError(f"ENTSO-E wind and solar forecasts for {date} are not available yet.")
    train = table.info["delivery_day"] < day
    train_hours = int(table.y[train].notna().sum())
    if model_file is not None:
        log(f"Predicting {feature_set} with the stored model {model_file} ...")
        prediction = predict_with(model_file, table.X[rows])
        train_hours = 0
    else:
        log(f"Fitting {model} ({feature_set}) on {train_hours:,} hours before {date} ...")
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
    curve.attrs["train_hours"] = train_hours
    curve.attrs["passes_gate"] = passes_gate
    return curve
