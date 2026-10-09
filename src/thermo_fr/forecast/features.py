"""Hourly feature table for the day-ahead price forecast, in Paris delivery hours.

Rows are the UTC hours of the input table; the delivery day and hour are the
Paris local calendar day and clock hour, so the spring day has 23 rows and the
autumn day 25 (two rows with hour 2). Several feature sets are built from the
same inputs:

- "honest": only inputs known at 12:00 Paris on D-1. Calendar (hour, weekday,
  month, day of year, public holiday, and the day-type structure of
  daytypes.py: working day / Saturday / Sunday-or-holiday, bridge days, the
  eve and the day after a holiday, the Christmas to New Year break), the
  ENTSO-E load forecast, the Open-Meteo forecasts issued two days ahead
  (temperature, 100 m wind, radiation), the wind and solar generation
  proxies (hub-height wind and radiation forecasts at the generation
  regions, calibrated on actual generation), and lagged prices: D-1, D-2 and
  D-7 same hour, daily summaries of D-1 and D-7, and the same hour of the
  most recent earlier day of the same type (the previous Friday for a
  Monday, the previous Saturday for a Saturday, the previous Sunday or
  holiday for a holiday).
- "honest_wind": the honest set as it was before the solar proxy and the
  calendar structure were added (wind proxy included). The published model
  of early October 2026 used it.
- "honest_base": the set without either proxy and without the calendar
  structure, the original honest set.
- "honest_solar" and "honest_calendar": honest_wind plus only the solar
  proxy, or only the calendar structure and the same-type lag, so the
  backtest can attribute the change to each addition.
- "extended": the honest set plus the ENTSO-E day-ahead wind and solar
  forecasts and the residual load forecast built from them. ENTSO-E allows
  these until 18:00 on D-1, after the auction, so this set may use late
  information.

Weather columns take the as-issued forecast where the archive has it and the
historical-forecast proxy before that (2021 to early 2024). The `info` table
returned next to the features says, row by row, whether the weather was point
in time, so the backtest can keep proxy rows for training and exclude them
from the strict out-of-sample metrics. It also carries the actual wind and
solar generation (never features) for the windy-day and sunny-day slices.
FEATURE_TIMINGS maps every feature to its timing rule for the look-ahead
check in timing.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .daytypes import day_table, same_type_day
from .timing import delivery_days

PEAK_HOURS = range(8, 20)  # EPEX peak block, 08:00 to 20:00 local
PRICE_LAGS = (1, 2, 7)

CALENDAR = ["hour", "dow", "month", "holiday", "day_of_year"]
CALENDAR_STRUCTURE = ["day_type", "bridge_day", "pre_holiday", "post_holiday", "year_end_break", "same_type_lag_days"]
WEATHER = ["temp_c", "wind100_ms", "radiation_wm2"]
LAGS = ["price_lag1", "price_lag2", "price_lag7", "price_lag1_mean", "price_lag1_min", "price_lag1_max", "price_lag7_mean"]
SAME_TYPE_LAGS = ["price_lag_same_type", "price_lag_same_type_mean"]
HONEST_BASE = CALENDAR + ["load_fc_mw"] + WEATHER + LAGS
HONEST_WIND = HONEST_BASE + ["wind_proxy_mw"]
HONEST_SOLAR = HONEST_WIND + ["solar_proxy_mw"]
HONEST_CALENDAR = HONEST_WIND + CALENDAR_STRUCTURE + SAME_TYPE_LAGS
HONEST = HONEST_WIND + ["solar_proxy_mw"] + CALENDAR_STRUCTURE + SAME_TYPE_LAGS
EXTENDED = HONEST + ["solar_fc_mw", "wind_fc_mw", "residual_load_fc_mw"]
FEATURES = {
    "honest": HONEST,
    "honest_wind": HONEST_WIND,
    "honest_base": HONEST_BASE,
    "honest_solar": HONEST_SOLAR,
    "honest_calendar": HONEST_CALENDAR,
    "extended": EXTENDED,
}
FEATURE_SETS = tuple(FEATURES)
GATED_SETS = tuple(s for s in FEATURES if s != "extended")  # the sets whose every feature passes the 12:00 gate

FEATURE_TIMINGS = {
    **{c: "calendar" for c in CALENDAR + CALENDAR_STRUCTURE},
    "load_fc_mw": "load_forecast",
    **{c: "weather_issued" for c in WEATHER},
    "wind_proxy_mw": "wind_proxy",
    "solar_proxy_mw": "solar_proxy",
    "price_lag1": "price_lag1",
    "price_lag1_mean": "price_lag1",
    "price_lag1_min": "price_lag1",
    "price_lag1_max": "price_lag1",
    "price_lag2": "price_lag2",
    "price_lag7": "price_lag7",
    "price_lag7_mean": "price_lag7",
    "price_lag_same_type": "price_lag_same_type",
    "price_lag_same_type_mean": "price_lag_same_type",
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
    info: pd.DataFrame  # delivery_day, hour, dow, holiday, is_peak, weather_point_in_time, wind_mw, solar_mw
    feature_set: str


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
    days = delivery_days(index)
    table = day_table(days)
    by_day = table.reindex(days)
    frame = pd.DataFrame(index=index)
    frame["delivery_day"] = days
    frame["hour"] = local.hour
    frame["month"] = local.month
    frame["day_of_year"] = local.dayofyear
    for column in table.columns:
        frame[column] = by_day[column].to_numpy()
    comparable = same_type_day(pd.DatetimeIndex(table.index))
    lookup = pd.Series(comparable, index=table.index).reindex(days)
    frame["same_type_day"] = lookup.to_numpy()
    frame["same_type_lag_days"] = (days - pd.DatetimeIndex(lookup)).days.to_numpy()
    frame["is_peak"] = (local.hour.isin(list(PEAK_HOURS)) & (local.dayofweek < 5)).astype(int)
    return frame


def price_lags(price: pd.Series, calendar: pd.DataFrame) -> pd.DataFrame:
    """Same-hour lags by local (day, hour), daily summaries of D-1 and D-7, and the same-type-day lag.

    The autumn repeated hour is averaged; a lookup that lands on the missing
    spring hour falls back to the value 24 hours earlier in UTC (for the fixed
    lags) or stays missing (for the same-type lag).
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
    same_day = pd.DatetimeIndex(calendar["same_type_day"].values)
    lookup = pd.MultiIndex.from_arrays([same_day, keyed["hour"]])
    out["price_lag_same_type"] = table.reindex(lookup).to_numpy()
    out["price_lag_same_type_mean"] = daily["mean"].reindex(same_day).to_numpy()
    return out


def build_features(hourly: pd.DataFrame, feature_set: str = "honest") -> FeatureTable:
    """Feature table for one feature set from the inputs table of forecast/inputs.py."""
    if feature_set not in FEATURES:
        raise ValueError(f"feature_set must be one of {tuple(FEATURES)}")
    hourly = hourly.sort_index()
    calendar = calendar_frame(hourly.index)
    X = calendar[CALENDAR + CALENDAR_STRUCTURE].copy()
    X["load_fc_mw"] = hourly["load_fc_mw"]

    issued = hourly[["temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"]]
    proxy = hourly[["temp_proxy_c", "wind100_proxy_ms", "radiation_proxy_wm2"]]
    point_in_time = issued.notna().all(axis=1)
    for name, fc, px in zip(WEATHER, issued.columns, proxy.columns):
        X[name] = hourly[fc].where(point_in_time, hourly[px])

    X = X.join(price_lags(hourly["price_eur_mwh"], calendar))
    for column in ("wind_proxy_mw", "solar_proxy_mw"):
        X[column] = hourly[column] if column in hourly else np.nan

    if feature_set == "extended":
        wind = hourly["wind_onshore_fc_mw"].add(hourly["wind_offshore_fc_mw"].fillna(0.0))
        X["solar_fc_mw"] = hourly["solar_fc_mw"]
        X["wind_fc_mw"] = wind
        X["residual_load_fc_mw"] = hourly["load_fc_mw"] - hourly["solar_fc_mw"] - wind

    info = calendar[["delivery_day", "hour", "dow", "holiday", "is_peak"]].copy()
    info["weather_point_in_time"] = point_in_time.to_numpy()
    # actual generation, never a feature: it defines the windy-day and sunny-day slices of the backtest
    actual_wind = [c for c in ("wind_onshore_mw", "wind_offshore_mw") if c in hourly]
    info["wind_mw"] = hourly[actual_wind].sum(axis=1, min_count=1) if actual_wind else np.nan
    info["solar_mw"] = hourly["solar_mw"] if "solar_mw" in hourly else np.nan
    X = X[FEATURES[feature_set]].astype(float)
    return FeatureTable(X=X, y=hourly["price_eur_mwh"].rename("price_eur_mwh"), info=info, feature_set=feature_set)
