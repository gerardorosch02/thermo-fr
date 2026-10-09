"""A pre-gate proxy for French wind generation, calibrated on actual generation history.

The ENTSO-E day-ahead wind forecast may be published after the 12:00 Paris
auction, so the honest feature set cannot use it. This module builds a
substitute from information that is available: hub-height wind speed
forecasts issued two days ahead at the points of data/wind_points.py, turned
into expected generation in MW.

Method
- Each point's speed goes through a generic turbine power curve: nothing
  below the cut-in speed (3 m/s), a cubic ramp to rated output at 12 m/s,
  flat to the cut-out speed (25 m/s), nothing above. The result is a
  capacity factor between 0 and 1.
- National generation is modelled as a non-negative weighted sum of the
  points' capacity factors, fitted by non-negative least squares against the
  ENTSO-E actual wind generation (onshore plus offshore). Each weight is the
  megawatts that point stands for, so the sum of the weights is an effective
  installed capacity; the trailing-year fit follows the growth of the fleet.
- For the backtest the weights are refitted at the start of every month on
  the trailing 365 days of actual generation ending two days before the
  month, and applied to that month. Actual generation per production type is
  published within an hour of the operating period (Regulation (EU)
  543/2013, Article 16(1)(a)), so the last day used (two days before the
  first delivery day of the month) is public before the gate of that first
  day. timing.py encodes this as the "wind_proxy" rule.
- For the live pipeline the monthly refit saves the latest weights to
  published/model/wind_proxy.json and morning-run applies them to the fresh
  point forecasts.

The proxy is only as good as a 48-hour wind forecast and a static power
curve; curtailment, icing, outages and the lag of the trailing-year
capacity are all errors it carries. Its value is measured in the backtest
(docs/forecast.md), not assumed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.wind_points import HUB_HEIGHT_M, POINT_COLUMNS
from .timing import delivery_days

CUT_IN_MS, RATED_MS, CUT_OUT_MS = 3.0, 12.0, 25.0
WINDOW_DAYS = 365
MIN_TRAIN_DAYS = 60
LAG_DAYS = 2  # the last day of actual generation used before a month is first(month) - LAG_DAYS
DEFAULT_WEIGHTS_PATH = Path("published/model/wind_proxy.json")


def power_curve(speed) -> np.ndarray:
    """Capacity factor in [0, 1] for a wind speed in m/s (NaN stays NaN)."""
    v = np.asarray(speed, dtype=float)
    ramp = np.clip((v - CUT_IN_MS) / (RATED_MS - CUT_IN_MS), 0.0, 1.0) ** 3
    out = np.where(v >= CUT_OUT_MS, 0.0, ramp)
    return np.where(np.isnan(v), np.nan, out)


def point_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in POINT_COLUMNS if c in frame.columns]


def calibrate(points: pd.DataFrame, actual: pd.Series) -> dict:
    """Non-negative least squares of actual generation on the points' capacity factors.

    Returns the weights (MW per point) and fit statistics. Rows with any
    missing point or missing actual are dropped.
    """
    from scipy.optimize import nnls

    columns = point_columns(points)
    if not columns:
        raise ValueError("No wind point columns to calibrate on.")
    both = points[columns].join(actual.rename("actual"), how="inner").dropna()
    if both.empty:
        raise ValueError("No overlapping hours between the point forecasts and actual wind generation.")
    A = power_curve(both[columns].to_numpy())
    b = both["actual"].to_numpy(dtype=float)
    weights, _ = nnls(A, b)
    fitted = A @ weights
    resid = fitted - b
    ss_tot = float(((b - b.mean()) ** 2).sum())
    return {
        "hub_height_m": HUB_HEIGHT_M,
        "power_curve": {"cut_in_ms": CUT_IN_MS, "rated_ms": RATED_MS, "cut_out_ms": CUT_OUT_MS},
        "weights_mw": {c: round(float(w), 1) for c, w in zip(columns, weights)},
        "fit": {
            "from": str(both.index.min()),
            "to": str(both.index.max()),
            "hours": int(len(both)),
            "capacity_mw": round(float(weights.sum()), 1),
            "mae_mw": round(float(np.abs(resid).mean()), 1),
            "bias_mw": round(float(resid.mean()), 1),
            "r2": round(float(1.0 - (resid**2).sum() / ss_tot), 4) if ss_tot > 0 else None,
            "mean_actual_mw": round(float(b.mean()), 1),
        },
    }


def apply_weights(points: pd.DataFrame, weights: dict) -> pd.Series:
    """Expected generation in MW. Points missing in a row are rescaled out; a row with no point is NaN."""
    w = weights["weights_mw"]
    columns = [c for c in w if c in points.columns]
    if not columns:
        raise ValueError("None of the calibrated wind points are in the table.")
    factors = power_curve(points[columns].to_numpy())
    vec = np.array([w[c] for c in columns], dtype=float)
    present = ~np.isnan(factors)
    contribution = np.where(present, factors, 0.0) @ vec
    available = present @ vec
    total = float(sum(w.values()))
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(available > 0, contribution * total / available, np.nan)
    return pd.Series(scaled, index=points.index, name="wind_proxy_mw")


def month_starts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(days.to_period("M").to_timestamp())))


def rolling_proxy(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS,
                  lag_days: int = LAG_DAYS, log=lambda *_: None) -> tuple[pd.Series, list[dict]]:
    """Point-in-time proxy: weights refitted each month on the trailing window ending before the month.

    Returns the hourly series (NaN where no calibration was possible) and
    the list of monthly fits.
    """
    columns = point_columns(points)
    proxy = pd.Series(np.nan, index=points.index, name="wind_proxy_mw")
    fits: list[dict] = []
    if not columns or not points[columns].notna().all(axis=1).any():
        return proxy, fits
    days = delivery_days(points.index)
    actual = actual.reindex(points.index)
    usable = (points[columns].notna().all(axis=1) & actual.notna()).to_numpy()
    have_points = points[columns].notna().any(axis=1).to_numpy()
    for month in month_starts(days[have_points]):
        last_day = month - pd.Timedelta(days=lag_days)  # inclusive
        train = usable & (days <= last_day) & (days > last_day - pd.Timedelta(days=window_days))
        train_days = days[train].nunique()
        if train_days < min_days:
            continue
        weights = calibrate(points[train], actual[train])
        rows = (days >= month) & (days < month + pd.DateOffset(months=1))
        proxy[rows] = apply_weights(points[rows], weights).to_numpy()
        fits.append({"month": month.strftime("%Y-%m"), "train_days": int(train_days), "last_train_day": str(last_day.date()), **weights["fit"]})
        log(f"  wind proxy {month:%Y-%m}: fitted on {train_days} days to {last_day.date()}, capacity {weights['fit']['capacity_mw']:,.0f} MW, "
            f"MAE {weights['fit']['mae_mw']:,.0f} MW, R2 {weights['fit']['r2']}")
    return proxy, fits


def latest_weights(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS) -> dict | None:
    """Weights on the trailing window up to the last hour with both points and actual generation."""
    columns = point_columns(points)
    if not columns:
        return None
    actual = actual.reindex(points.index)
    usable = (points[columns].notna().all(axis=1) & actual.notna()).to_numpy()
    if not usable.any():
        return None
    end = points.index[usable].max()
    window = usable & (points.index > end - pd.Timedelta(days=window_days))
    if delivery_days(points.index[window]).nunique() < min_days:
        return None
    return calibrate(points[window], actual[window])


def save_weights(weights: dict, path=DEFAULT_WEIGHTS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))
    return path


def load_weights(path=DEFAULT_WEIGHTS_PATH) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
