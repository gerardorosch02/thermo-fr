"""Turn hourly series into a daily modelling table without daylight-saving bugs.

Everything is stored in UTC. We only convert to Paris local time at the point
where we group hours into days, so the March day has 23 hours and the October
day has 25, and nothing gets silently shifted or duplicated.
"""

import holidays
import numpy as np
import pandas as pd

from ..config import LOCAL_TZ


def to_hourly_utc(series: pd.Series) -> pd.Series:
    """Convert a tz-aware series to a clean hourly UTC series (mean of sub-hourly values)."""
    index = pd.DatetimeIndex(series.index)
    if index.tz is None:
        raise ValueError("Timestamps must be timezone-aware; refusing to guess the zone.")
    s = series.copy()
    s.index = index.tz_convert("UTC")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.resample("1h").mean()


def daily_frame(hourly: pd.DataFrame, tz: str = LOCAL_TZ, min_hours: int = 22) -> pd.DataFrame:
    """Average hourly UTC data into local calendar days and add calendar features.

    Days with fewer than `min_hours` load observations are dropped. The default of
    22 keeps the 23-hour spring DST day while removing days with real gaps.
    """
    if hourly.index.tz is None:
        raise ValueError("Hourly frame must have a tz-aware index.")
    local = hourly.tz_convert(tz)
    grouped = local.groupby(local.index.date)
    daily = grouped.mean()
    daily["hours"] = grouped["load_mw"].count()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily = daily[daily["hours"] >= min_hours]
    return add_calendar(daily)


def add_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    idx = daily.index
    years = range(idx.year.min(), idx.year.max() + 1)
    french_holidays = holidays.France(years=years)
    out = daily.copy()
    out["dow"] = idx.dayofweek
    out["holiday"] = [d.date() in french_holidays for d in idx]
    out["month_key"] = idx.to_period("M").astype(str)  # e.g. 2024-01, absorbs gas and policy shifts
    out["moy"] = idx.month  # month of year, used for out-of-sample checks
    return out


def hdd(temperature, threshold: float) -> np.ndarray:
    """Heating degrees: how far the day's temperature sits below the threshold."""
    return np.maximum(threshold - np.asarray(temperature, dtype=float), 0.0)
