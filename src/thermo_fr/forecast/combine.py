"""Forecast combination (branch model-sprint): a simple average and online weights from settled days only.

Settlement and timing. The forecast for delivery day D is issued at 10:05 Paris on D-1. The auction result of day d is
published at 13:00 Paris on d-1, so at D's issue time every day up to and including D-1 is settled (SETTLE_LAG_DAYS = 1).
Weights for D therefore use the components' errors on days d <= D-1 only, and never any outcome of D or later. Only
delivery days whose weekday is in `update_weekdays` update the weights (default Tuesday to Saturday, the days the live
pre-market runs exist for), in the backtest as live.

- `inverse_mae_weights`: w_m proportional to 1 / EWMA_m, the exponentially weighted mean daily MAE of component m over the
  eligible settled days, half-life in calendar days; equal weights until `min_days` eligible days exist.
- `boa_weights`: Bernstein online aggregation (Wintenberger 2017) on the linearised absolute loss of the combined forecast,
  with a fixed learning rate; written from the published update rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SETTLE_LAG_DAYS = 1
TUESDAY_TO_SATURDAY = (1, 2, 3, 4, 5)  # pandas dayofweek: Monday 0


def daily_mae(frame: pd.DataFrame, columns, actual: str = "actual", day: str = "delivery_day") -> pd.DataFrame:
    """Daily MAE of each forecast column over the hours with an actual price (strict or not: weights use every settled day)."""
    rows = frame.dropna(subset=[actual])
    err = rows[list(columns)].sub(rows[actual], axis=0).abs()
    return err.groupby(rows[day]).mean()


def _eligible(errors: pd.DataFrame, D: pd.Timestamp, update_weekdays) -> pd.DataFrame:
    last = D - pd.Timedelta(days=SETTLE_LAG_DAYS)
    past = errors[(errors.index <= last) & errors.index.dayofweek.isin(update_weekdays)]
    return past.dropna(how="any")


def inverse_mae_weights(errors: pd.DataFrame, days, half_life: float = 14.0, min_days: int = 14,
                        update_weekdays=TUESDAY_TO_SATURDAY) -> pd.DataFrame:
    """Weights (rows: delivery days D, columns: components) from the daily errors of settled eligible days before D."""
    columns = list(errors.columns)
    out = []
    for D in pd.DatetimeIndex(days):
        past = _eligible(errors, D, update_weekdays)
        if len(past) < min_days:
            w = np.full(len(columns), 1.0 / len(columns))
        else:
            age = (D - past.index).days.to_numpy()
            decay = 0.5 ** (age / half_life)
            ewma = (past.to_numpy() * decay[:, None]).sum(axis=0) / decay.sum()
            inv = 1.0 / np.maximum(ewma, 1e-9)
            w = inv / inv.sum()
        out.append(w)
    return pd.DataFrame(out, index=pd.DatetimeIndex(days), columns=columns)


def boa_weights(frame: pd.DataFrame, columns, days, eta: float = 0.05, update_weekdays=TUESDAY_TO_SATURDAY,
                actual: str = "actual", day: str = "delivery_day") -> pd.DataFrame:
    """Bernstein online aggregation with the linearised absolute loss.

    State starts uniform. Weights for D are the state after the updates of every eligible settled day d <= D-1 in date order.
    The update for day d: g_h = sign(combined_h - actual_h); l_m = mean_h g_h (f_mh - combined_h);
    w_m <- w_m exp(-eta l_m (1 + eta l_m)), then normalised. The combined forecast of d is formed with the weights used on d.
    """
    columns = list(columns)
    rows = frame.dropna(subset=[actual])
    by_day = {d: part for d, part in rows.groupby(day)}
    days = pd.DatetimeIndex(days)
    w = np.full(len(columns), 1.0 / len(columns))
    weights = {}
    updates = sorted(d for d in by_day if pd.Timestamp(d).dayofweek in update_weekdays)
    pending = list(updates)
    for D in sorted(days):
        last = D - pd.Timedelta(days=SETTLE_LAG_DAYS)
        while pending and pending[0] <= last:
            d = pending.pop(0)
            part = by_day[d]
            f = part[columns].to_numpy()
            ok = np.isfinite(f).all(axis=1)
            if ok.any():
                use = weights.get(pd.Timestamp(d), w)  # the weights the combination actually used on d
                combined = f[ok] @ use
                g = np.sign(combined - part[actual].to_numpy()[ok])
                loss = (g[:, None] * (f[ok] - combined[:, None])).mean(axis=0)
                w = w * np.exp(-eta * loss * (1 + eta * loss))
                w = w / w.sum()
        weights[pd.Timestamp(D)] = w.copy()
    return pd.DataFrame([weights[pd.Timestamp(D)] for D in days], index=days, columns=columns)


def combine(frame: pd.DataFrame, weights: pd.DataFrame | None, columns, day: str = "delivery_day") -> pd.Series:
    """Hourly combination: sum_m w[D, m] f_m, renormalised over the components present in each row (a missing component falls
    back to the others). weights None means equal weights."""
    columns = list(columns)
    f = frame[columns].to_numpy(dtype=float)
    if weights is None:
        w = np.full_like(f, 1.0 / len(columns))
    else:
        w = weights.reindex(pd.DatetimeIndex(frame[day]))[columns].to_numpy(dtype=float)
    present = np.isfinite(f)
    w = np.where(present, w, 0.0)
    total = w.sum(axis=1)
    value = np.where(present, f, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (value * w).sum(axis=1) / total
    out[total == 0] = np.nan
    return pd.Series(out, index=frame.index)
