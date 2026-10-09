"""Hourly feature table for the day-ahead price forecast, in Paris delivery hours.

Rows are the UTC hours of the input table; the delivery day and hour are the
Paris local calendar day and clock hour, so the spring day has 23 rows and the
autumn day 25 (two rows with hour 2). Two feature sets are built from the same
inputs:

- "honest": only inputs known at 12:00 Paris on D-1. Calendar, the ENTSO-E
  load forecast, the Open-Meteo forecasts issued two days ahead (temperature,
  100 m wind, radiation, which stand in for renewable output), the wind
  generation proxy of wind_proxy.py (hub-height wind forecasts at the wind
  regions, calibrated on actual generation), and lagged prices (D-1, D-2,
  D-7 same hour, plus daily summaries of D-1 and D-7).
- "honest_base": the honest set without the wind proxy, kept so the
  backtest can measure what the proxy adds.
- "extended": the honest set plus the ENTSO-E day-ahead wind and solar
  forecasts and the residual load forecast built from them. ENTSO-E allows
  these until 18:00 on D-1, after the auction, so this set may use late
  information.

Weather columns take the as-issued forecast where the archive has it and the
historical-forecast proxy before that (2021 to early 2024). The `info` table
returned next to the features says, row by row, whether the weather was point
in time, so the backtest can keep proxy rows for training and exclude them
from the strict out-of-sample metrics. FEATURE_TIMINGS maps every feature to
its timing rule for the look-ahead check in timing.py.
"""

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .timing import delivery_days

FEATURE_SETS = ("honest", "extended")
PEAK_HOURS = range(8, 20)  # EPEX peak block, 08:00 to 20:00 local
PRICE_LAGS = (1, 2, 7)

CALENDAR = ["hour", "dow", "month", "holiday", "day_of_year"]
WEATHER = ["temp_c", "wind100_ms", "radiation_wm2"]
LAGS = ["price_lag1", "price_lag2", "price_lag7", "price_lag1_mean", "price_lag1_min", "price_lag1_max", "price_lag7_mean"]
HONEST_BASE = CALENDAR + ["load_fc_mw"] + WEATHER + LAGS
HONEST = HONEST_BASE + ["wind_proxy_mw"]
EXTENDED = HONEST + ["solar_fc_mw", "wind_fc_mw", "residual_load_fc_mw"]
FEATURES = {"honest": HONEST, "honest_base": HONEST_BASE, "extended": EXTENDED}

FEATURE_TIMINGS = {
    **{c: "calendar" for c in CALENDAR},
    "load_fc_mw": "load_forecast",
    **{c: "weather_issued" for c in WEATHER},
    "wind_proxy_mw": "wind_proxy",
    "price_lag1": "price_lag1",
    "price_lag1_mean": "price_lag1",
    "price_lag1_min": "price_lag1",
    "price_lag1_max": "price_lag1",
    "price_lag2": "price_lag2",
    "price_lag7": "price_lag7",
    "price_lag7_mean": "price_lag7",
    "solar_fc_mw": "wind_solar_forecast",
    "wind_fc_mw": "wind_solar_forecast",
    "residual_load_fc_mw": "wind_solar_forecast",
}


def feature_timings(feature_set: str) -> dict[str, str]:
    return {f: FEATURE_TIMINGS[f] for f in FEATURES[feature_set]}


@dataclass
class FeatureTable:
    X: pd.DataFrame  # features, indexed by valid hour UTC
    y: pd.Series  # price, same index (NaN where unknown)
    info: pd.DataFrame  # delivery_day, hour, is_peak, weather_point_in_time
    feature_set: str


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
    days = delivery_days(index)
    years = range(days.year.min(), days.year.max() + 2)
    french = holidays.France(years=years)
    frame = pd.DataFrame(index=index)
    frame["delivery_day"] = days
    frame["hour"] = local.hour
    frame["dow"] = local.dayofweek
    frame["month"] = local.month
    frame["day_of_year"] = local.dayofyear
    frame["holiday"] = np.array([d.date() in french for d in days], dtype=int)
    frame["is_peak"] = (local.hour.isin(list(PEAK_HOURS)) & (local.dayofweek < 5)).astype(int)
    return frame


def price_lags(price: pd.Series, calendar: pd.DataFrame) -> pd.DataFrame:
    """Same-hour lags by local (day, hour), with daily summaries of D-1 and D-7.

    The autumn repeated hour is averaged; a lookup that lands on the missing
    spring hour falls back to the value 24 hours earlier in UTC.
    """
    keyed = pd.DataFrame({"day": calendar["delivery_day"].values, "hour": calendar["hour"].values, "price": price.values})
    table = keyed.groupby(["day", "hour"])["price"].mean()
    daily = keyed.groupby("day")["price"].agg(["mean", "min", "max"])
    out = pd.DataFrame(index=price.index)
    for lag in PRICE_LAGS:
        lookup = pd.MultiIndex.from_arrays([keyed["day"] - pd.Timedelta(days=lag), keyed["hour"]])
        values = table.reindex(lookup).to_numpy()
        fallback = price.shift(24 * lag).to_numpy()
        out[f"price_lag{lag}"] = np.where(np.isnan(values), fallback, values)
    for lag in (1, 7):
        stats = daily.reindex(keyed["day"] - pd.Timedelta(days=lag))
        out[f"price_lag{lag}_mean"] = stats["mean"].to_numpy()
        if lag == 1:
            out["price_lag1_min"] = stats["min"].to_numpy()
            out["price_lag1_max"] = stats["max"].to_numpy()
    return out


def build_features(hourly: pd.DataFrame, feature_set: str = "honest") -> FeatureTable:
    """Feature table for one feature set from the inputs table of forecast/inputs.py."""
    if feature_set not in FEATURES:
        raise ValueError(f"feature_set must be one of {tuple(FEATURES)}")
    hourly = hourly.sort_index()
    calendar = calendar_frame(hourly.index)
    X = calendar[CALENDAR].copy()
    X["load_fc_mw"] = hourly["load_fc_mw"]

    issued = hourly[["temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"]]
    proxy = hourly[["temp_proxy_c", "wind100_proxy_ms", "radiation_proxy_wm2"]]
    point_in_time = issued.notna().all(axis=1)
    for name, fc, px in zip(WEATHER, issued.columns, proxy.columns):
        X[name] = hourly[fc].where(point_in_time, hourly[px])

    X = X.join(price_lags(hourly["price_eur_mwh"], calendar))
    X["wind_proxy_mw"] = hourly["wind_proxy_mw"] if "wind_proxy_mw" in hourly else np.nan

    if feature_set == "extended":
        wind = hourly["wind_onshore_fc_mw"].add(hourly["wind_offshore_fc_mw"].fillna(0.0))
        X["solar_fc_mw"] = hourly["solar_fc_mw"]
        X["wind_fc_mw"] = wind
        X["residual_load_fc_mw"] = hourly["load_fc_mw"] - hourly["solar_fc_mw"] - wind

    info = calendar[["delivery_day", "hour", "is_peak"]].copy()
    info["weather_point_in_time"] = point_in_time.to_numpy()
    # actual wind generation, never a feature: it defines the windy-day slice of the backtest
    actual_wind = [c for c in ("wind_onshore_mw", "wind_offshore_mw") if c in hourly]
    info["wind_mw"] = hourly[actual_wind].sum(axis=1, min_count=1) if actual_wind else np.nan
    X = X[FEATURES[feature_set]].astype(float)
    return FeatureTable(X=X, y=hourly["price_eur_mwh"].rename("price_eur_mwh"), info=info, feature_set=feature_set)
