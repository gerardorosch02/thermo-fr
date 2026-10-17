"""Clear-sky timing of solar series: the ENTSO-E relabelling rule (amendment 2) and the clear-sky labelling check.

Clear-sky irradiance. Global horizontal irradiance under a clear sky by the Haurwitz model, GHI = 1098 cos(z)
exp(-0.057 / cos(z)) W/m2, with the solar zenith angle z from the NOAA approximations of declination and the equation of
time. Each hour is integrated over its own interval on start-of-hour labels: the value at T is the mean over [T, T+1),
sampled at ten six-minute midpoints. The national clear-sky curve is the plain mean over the 21 solar points of
data/solar_points.py. Only the timing of the curve matters here, never its level.

Daily centroid. For a non-negative hourly series x on start-of-hour labels, the centroid of UTC day d is
sum(h * x_h) / sum(x_h) over the hours h of d (h = 0..23). A series one hour late has its centroid one hour later.

ENTSO-E relabelling rule (final_plan.md, amendment 2). For each UTC day, the offset is the centroid of actual solar
output minus the clear-sky centroid, both on start-of-hour labels. The offsets are averaged over a centred 7-day window
(at least 4 days with data). The reference is the median offset over the whole history. Days whose smoothed offset exceeds
that median by more than 0.5 h are relabelled by one hour: the value published for T+1 is taken as the output of T. They
are not excluded. Amendment 1 of fix_prereg.md applies the same flagged days to wind onshore, wind offshore and nuclear
(relabel_actuals), which come from the same ENTSO-E A75 documents and show the same one-hour lag. The rule is a repair
of a historical publication fault, run once over the history: its centred window and whole-history median are not causal. Live refits use the trailing 365 days, which no longer reach the flagged period.

Clear-sky labelling check. On clear days (the 10% of days with the highest ratio of point radiation to clear-sky
radiation), the mean centroid of the radiation points minus the mean centroid of actual solar. It passes when the
difference is at most 0.25 h in absolute value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RELABEL_THRESHOLD_H = 0.5
SMOOTH_DAYS = 7
SMOOTH_MIN_DAYS = 4
CHECK_TOLERANCE_H = 0.25
CLEAR_DAY_QUANTILE = 0.90
MIN_DAY_HOURS = 20
SAMPLES_PER_HOUR = 10


def cos_zenith(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Cosine of the solar zenith angle at UTC times (NOAA approximations)."""
    t = pd.DatetimeIndex(times).tz_convert("UTC")
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60 + t.second / 3600).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    true_solar_minutes = hour * 60 + eot + 4 * lon
    hour_angle = np.radians(true_solar_minutes / 4 - 180)
    phi = np.radians(lat)
    return np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle)


def clear_sky_ghi(cos_z: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance in W/m2."""
    c = np.asarray(cos_z, dtype=float)
    return np.where(c > 0, 1098.0 * c * np.exp(-0.057 / np.maximum(c, 1e-3)), 0.0)


def clear_sky_hourly(index: pd.DatetimeIndex, lat: float, lon: float) -> pd.Series:
    """Clear-sky GHI averaged over [T, T+1) for each start-of-hour label T."""
    index = pd.DatetimeIndex(index).tz_convert("UTC")
    offsets = (np.arange(SAMPLES_PER_HOUR) + 0.5) / SAMPLES_PER_HOUR
    total = np.zeros(len(index))
    for f in offsets:
        total += clear_sky_ghi(cos_zenith(index + pd.Timedelta(hours=float(f)), lat, lon))
    return pd.Series(total / SAMPLES_PER_HOUR, index=index, name="clear_sky_wm2")


def national_clear_sky(index: pd.DatetimeIndex, points=None) -> pd.Series:
    """The plain mean of the clear-sky curves of the solar points (start-of-hour labels)."""
    if points is None:
        from ..data.solar_points import SOLAR_POINTS as points
    curves = [clear_sky_hourly(index, p.lat, p.lon) for p in points]
    return (sum(curves) / len(curves)).rename("clear_sky_wm2")


def daily_centroid(series: pd.Series, min_hours: int = MIN_DAY_HOURS) -> pd.Series:
    """Centroid (UTC hours) of each UTC day of a non-negative hourly series; days with fewer than `min_hours` values or no
    positive value are dropped."""
    x = series.astype(float).clip(lower=0)
    day = x.index.tz_convert("UTC").normalize()
    hour = x.index.tz_convert("UTC").hour.to_numpy(dtype=float)
    frame = pd.DataFrame({"x": x.to_numpy(), "hx": x.to_numpy() * hour, "n": x.notna().to_numpy()}, index=day)
    g = frame.groupby(level=0)
    total, weighted, count = g["x"].sum(), g["hx"].sum(), g["n"].sum()
    keep = (count >= min_hours) & (total > 0)
    out = (weighted / total)[keep]
    out.index = pd.DatetimeIndex(out.index)
    return out


def entsoe_solar_offsets(actual: pd.Series, clear: pd.Series | None = None) -> pd.DataFrame:
    """Per UTC day: offset (actual centroid minus clear-sky centroid), its centred 7-day mean, and the excess over the
    whole-history median of the daily offsets."""
    clear = national_clear_sky(actual.index) if clear is None else clear
    offset = (daily_centroid(actual) - daily_centroid(clear.reindex(actual.index))).dropna()
    days = pd.date_range(offset.index.min(), offset.index.max(), freq="D")
    daily = offset.reindex(days)
    smooth = daily.rolling(SMOOTH_DAYS, center=True, min_periods=SMOOTH_MIN_DAYS).mean()
    median = float(offset.median())
    frame = pd.DataFrame({"offset_h": daily, "smooth_h": smooth, "excess_h": smooth - median})
    frame.attrs["median_offset_h"] = median
    return frame


def flagged_days(offsets: pd.DataFrame, threshold: float = RELABEL_THRESHOLD_H) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(offsets.index[(offsets["excess_h"] > threshold).to_numpy()])


def _utc_days(days) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(days)
    return days.tz_localize("UTC") if days.tz is None else days.tz_convert("UTC")


def relabel_days(series: pd.Series, days, block_end_value: float = np.nan) -> pd.Series:
    """On the given UTC days (published one hour late), the value labelled h takes the value originally labelled h+1, so
    each label holds its own period. The last hour of each block of consecutive flagged days has no value of its own period
    inside the block (the next label is an unflagged hour, which holds its own period) and is set to `block_end_value`:
    missing, or 0 for solar, where it is night. The block's original first value (a duplicate of the hour before the
    block) drops out."""
    out = series.astype(float).copy()
    days = _utc_days(days)
    if len(days) == 0:
        return out
    index = pd.DatetimeIndex(out.index)
    day_of = index.tz_convert("UTC").normalize()
    rows = day_of.isin(days)
    later_index = index[rows] + pd.Timedelta(hours=1)
    later = series.reindex(later_index).to_numpy(dtype=float)
    later_flagged = later_index.tz_convert("UTC").normalize().isin(days)
    later[~later_flagged] = block_end_value  # the next label lies outside the block
    out[rows] = later
    return out


ENTSOE_ACTUAL_COLUMNS = {"solar_mw": 0.0, "wind_onshore_mw": np.nan, "wind_offshore_mw": np.nan, "nuclear_mw": np.nan}


def relabel_actuals(table: pd.DataFrame, days, columns=None) -> tuple[pd.DataFrame, dict]:
    """Amendment 1 of fix_prereg.md: every ENTSO-E actual generation series from the A75 documents (solar, wind onshore,
    wind offshore, nuclear) relabelled on one flagged-day set, the one the solar rule flags. Returns the table and, per
    column, the hours on flagged days that are missing after the relabel."""
    out = table.copy()
    days = _utc_days(days)
    on_days = pd.DatetimeIndex(out.index).tz_convert("UTC").normalize().isin(days)
    missing = {}
    for column, end_value in (columns or ENTSOE_ACTUAL_COLUMNS).items():
        if column not in out:
            continue
        out[column] = relabel_days(out[column], days, end_value)
        missing[column] = int(out.loc[on_days, column].isna().sum())
    return out, missing


def relabel_entsoe_solar(actual: pd.Series, clear: pd.Series | None = None) -> tuple[pd.Series, pd.DatetimeIndex, pd.DataFrame]:
    """Amendment 2: the relabelled actual solar series, the flagged days and the daily offsets."""
    offsets = entsoe_solar_offsets(actual, clear)
    days = flagged_days(offsets)
    return relabel_days(actual, days, ENTSOE_ACTUAL_COLUMNS["solar_mw"]), days, offsets


def day_ranges(days: pd.DatetimeIndex) -> list[tuple[str, str, int]]:
    """Consecutive runs of days as (first, last, count)."""
    days = pd.DatetimeIndex(sorted(days))
    out = []
    if len(days) == 0:
        return out
    start = prev = days[0]
    for d in days[1:]:
        if d - prev != pd.Timedelta(days=1):
            out.append((start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d"), int((prev - start).days) + 1))
            start = d
        prev = d
    out.append((start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d"), int((prev - start).days) + 1))
    return out


def clear_days(radiation: pd.Series, clear: pd.Series, quantile: float = CLEAR_DAY_QUANTILE) -> pd.DatetimeIndex:
    """UTC days whose ratio of radiation to clear-sky radiation is in the top (1 - quantile) of the days with data."""
    both = pd.DataFrame({"r": radiation, "c": clear}).dropna()
    day = both.index.tz_convert("UTC").normalize()
    sums = both.groupby(day).sum()
    sums = sums[(both.groupby(day).size() >= MIN_DAY_HOURS) & (sums["c"] > 0)]
    ratio = sums["r"] / sums["c"]
    return pd.DatetimeIndex(ratio.index[ratio >= ratio.quantile(quantile)])


def clear_sky_label_check(radiation: pd.Series, actual: pd.Series, clear: pd.Series | None = None,
                          tolerance: float = CHECK_TOLERANCE_H) -> dict:
    """Mean centroid of radiation minus mean centroid of actual solar on clear days; passes when within `tolerance` hours.
    Also reports each series against the clear-sky centroid on start-of-hour labels."""
    clear = national_clear_sky(radiation.index) if clear is None else clear.reindex(radiation.index)
    days = clear_days(radiation, clear)
    c_rad, c_act, c_clear = daily_centroid(radiation), daily_centroid(actual), daily_centroid(clear)
    common = days.intersection(c_rad.index).intersection(c_act.index).intersection(c_clear.index)
    if len(common) == 0:
        return {"passed": False, "clear_days": 0, "reason": "no clear day with both series"}
    gap = float((c_rad[common] - c_act[common]).mean())
    return {"passed": bool(abs(gap) <= tolerance), "tolerance_h": tolerance, "clear_days": int(len(common)),
            "radiation_minus_actual_h": round(gap, 3),
            "radiation_minus_clear_sky_h": round(float((c_rad[common] - c_clear[common]).mean()), 3),
            "actual_minus_clear_sky_h": round(float((c_act[common] - c_clear[common]).mean()), 3)}
