"""Pre-gate generation proxies: a weighted sum of forecast points, calibrated on actual generation.

The wind proxy (wind_proxy.py) and the solar proxy (solar_proxy.py) share
one method, so it lives here once:

- Each forecast point (a hub-height wind speed, a shortwave radiation) goes
  through a transform that turns it into a capacity factor between 0 and 1
  (a turbine power curve, the ratio to a reference irradiance).
- National generation is modelled as a non-negative weighted sum of the
  points' capacity factors, fitted by non-negative least squares against the
  ENTSO-E actual generation of that type. Each weight is the megawatts a
  point stands for, so the sum of the weights is an effective installed
  capacity and a trailing-window fit follows the growth of the fleet.
- For the backtest the weights are refitted at the start of every month on
  the trailing window of actual generation ending two days before the month,
  and applied to that month. Actual generation per production type is
  published within an hour of the operating period (Regulation (EU)
  543/2013, Article 16(1)(a)), so the last day used is public before the
  gate of the first delivery day of the month. timing.py encodes this.
- For the live pipeline the monthly refit saves the latest weights to a JSON
  file under published/model/ and morning-run applies them to the fresh
  point forecasts.

A ProxySpec names the proxy, its output column, the point columns of the
inputs table, the transform and what to record about it in the weights file.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .timing import delivery_days

WINDOW_DAYS = 365
MIN_TRAIN_DAYS = 60
LAG_DAYS = 2  # the last day of actual generation used before a month is first(month) - LAG_DAYS


@dataclass(frozen=True)
class ProxySpec:
    name: str  # "wind" or "solar"
    output: str  # column of the proxy in the inputs table, e.g. wind_proxy_mw
    columns: tuple  # point columns of the inputs table
    transform: Callable  # point values -> capacity factor in [0, 1], NaN preserved
    default_path: Path  # where refit-model writes the weights
    window_days: int = WINDOW_DAYS
    min_days: int = MIN_TRAIN_DAYS
    meta: dict = field(default_factory=dict)  # recorded in the weights file (hub height, power curve, reference irradiance)


def point_columns(frame: pd.DataFrame, spec: ProxySpec) -> list[str]:
    return [c for c in spec.columns if c in frame.columns]


def calibrate(points: pd.DataFrame, actual: pd.Series, spec: ProxySpec) -> dict:
    """Non-negative least squares of actual generation on the points' capacity factors.

    Returns the weights (MW per point) and fit statistics. Rows with any
    missing point or missing actual are dropped.
    """
    from scipy.optimize import nnls

    columns = point_columns(points, spec)
    if not columns:
        raise ValueError(f"No {spec.name} point columns to calibrate on.")
    both = points[columns].join(actual.rename("actual"), how="inner").dropna()
    if both.empty:
        raise ValueError(f"No overlapping hours between the point forecasts and actual {spec.name} generation.")
    A = spec.transform(both[columns].to_numpy())
    b = both["actual"].to_numpy(dtype=float)
    weights, _ = nnls(A, b)
    fitted = A @ weights
    resid = fitted - b
    ss_tot = float(((b - b.mean()) ** 2).sum())
    return {
        "proxy": spec.name,
        **spec.meta,
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


def apply_weights(points: pd.DataFrame, weights: dict, spec: ProxySpec) -> pd.Series:
    """Expected generation in MW. Points missing in a row are rescaled out; a row with no point is NaN."""
    if weights.get("proxy", spec.name) != spec.name:
        raise ValueError(f"These weights belong to the {weights['proxy']} proxy, not the {spec.name} proxy.")
    w = weights["weights_mw"]
    columns = [c for c in w if c in points.columns]
    if not columns:
        raise ValueError(f"None of the calibrated {spec.name} points are in the table.")
    factors = spec.transform(points[columns].to_numpy())
    vec = np.array([w[c] for c in columns], dtype=float)
    present = ~np.isnan(factors)
    contribution = np.where(present, factors, 0.0) @ vec
    available = present @ vec
    total = float(sum(w.values()))
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(available > 0, contribution * total / available, np.nan)
    return pd.Series(scaled, index=points.index, name=spec.output)


def month_starts(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(days.to_period("M").to_timestamp())))


def rolling_proxy(points: pd.DataFrame, actual: pd.Series, spec: ProxySpec, window_days: int | None = None, min_days: int | None = None,
                  lag_days: int = LAG_DAYS, log=lambda *_: None) -> tuple[pd.Series, list[dict]]:
    """Point-in-time proxy: weights refitted each month on the trailing window ending before the month.

    Returns the hourly series (NaN where no calibration was possible) and
    the list of monthly fits.
    """
    window_days = spec.window_days if window_days is None else window_days
    min_days = spec.min_days if min_days is None else min_days
    columns = point_columns(points, spec)
    proxy = pd.Series(np.nan, index=points.index, name=spec.output)
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
        weights = calibrate(points[train], actual[train], spec)
        rows = (days >= month) & (days < month + pd.DateOffset(months=1))
        proxy[rows] = apply_weights(points[rows], weights, spec).to_numpy()
        fits.append({"month": month.strftime("%Y-%m"), "train_days": int(train_days), "last_train_day": str(last_day.date()), **weights["fit"]})
        log(f"  {spec.name} proxy {month:%Y-%m}: fitted on {train_days} days to {last_day.date()}, capacity {weights['fit']['capacity_mw']:,.0f} MW, "
            f"MAE {weights['fit']['mae_mw']:,.0f} MW, R2 {weights['fit']['r2']}")
    return proxy, fits


def latest_weights(points: pd.DataFrame, actual: pd.Series, spec: ProxySpec, window_days: int | None = None,
                   min_days: int | None = None) -> dict | None:
    """Weights on the trailing window up to the last hour with both points and actual generation."""
    window_days = spec.window_days if window_days is None else window_days
    min_days = spec.min_days if min_days is None else min_days
    columns = point_columns(points, spec)
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
    return calibrate(points[window], actual[window], spec)


def save_weights(weights: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))
    return path


def load_weights(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
