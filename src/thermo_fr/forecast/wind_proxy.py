"""A pre-gate proxy for French wind generation, calibrated on actual generation history.

The ENTSO-E day-ahead wind forecast may be published after the 12:00 Paris
auction, so the honest feature set cannot use it. This module builds a
substitute from information that is available: hub-height wind speed
forecasts issued two days ahead at the points of data/wind_points.py, turned
into expected generation in MW.

Each point's speed goes through a generic turbine power curve: nothing below
the cut-in speed (3 m/s), a cubic ramp to rated output at 12 m/s, flat to the
cut-out speed (25 m/s), nothing above. The result is a capacity factor
between 0 and 1. The weighting, the monthly point-in-time refits and the
weights file are the shared method of gen_proxy.py (one weight in MW per
point, non-negative least squares against the ENTSO-E actual onshore plus
offshore generation, trailing 365 days ending two days before the month).

The proxy is only as good as a 48-hour wind forecast and a static power
curve; curtailment, icing, outages and the lag of the trailing-year
capacity are all errors it carries. Its value is measured in the backtest
(docs/forecast.md), not assumed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.wind_points import HUB_HEIGHT_M, POINT_COLUMNS
from . import gen_proxy
from .gen_proxy import LAG_DAYS, MIN_TRAIN_DAYS, WINDOW_DAYS, ProxySpec, load_weights, save_weights  # noqa: F401  re-exported

CUT_IN_MS, RATED_MS, CUT_OUT_MS = 3.0, 12.0, 25.0
DEFAULT_WEIGHTS_PATH = Path("published/model/wind_proxy.json")


def power_curve(speed) -> np.ndarray:
    """Capacity factor in [0, 1] for a wind speed in m/s (NaN stays NaN)."""
    v = np.asarray(speed, dtype=float)
    ramp = np.clip((v - CUT_IN_MS) / (RATED_MS - CUT_IN_MS), 0.0, 1.0) ** 3
    out = np.where(v >= CUT_OUT_MS, 0.0, ramp)
    return np.where(np.isnan(v), np.nan, out)


WIND = ProxySpec(
    name="wind",
    output="wind_proxy_mw",
    columns=tuple(POINT_COLUMNS),
    transform=power_curve,
    default_path=DEFAULT_WEIGHTS_PATH,
    meta={"hub_height_m": HUB_HEIGHT_M, "power_curve": {"cut_in_ms": CUT_IN_MS, "rated_ms": RATED_MS, "cut_out_ms": CUT_OUT_MS}},
)


def actual_wind(hourly: pd.DataFrame) -> pd.Series:
    """Actual onshore plus offshore generation from the inputs table (NaN where neither is reported)."""
    columns = [c for c in ("wind_onshore_mw", "wind_offshore_mw") if c in hourly]
    if not columns:
        return pd.Series(np.nan, index=hourly.index, name="wind_mw")
    return hourly[columns].sum(axis=1, min_count=1).rename("wind_mw")


def point_columns(frame: pd.DataFrame) -> list[str]:
    return gen_proxy.point_columns(frame, WIND)


def calibrate(points: pd.DataFrame, actual: pd.Series) -> dict:
    return gen_proxy.calibrate(points, actual, WIND)


def apply_weights(points: pd.DataFrame, weights: dict) -> pd.Series:
    return gen_proxy.apply_weights(points, weights, WIND)


def rolling_proxy(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS,
                  lag_days: int = LAG_DAYS, log=lambda *_: None) -> tuple[pd.Series, list[dict]]:
    return gen_proxy.rolling_proxy(points, actual, WIND, window_days=window_days, min_days=min_days, lag_days=lag_days, log=log)


def latest_weights(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS) -> dict | None:
    return gen_proxy.latest_weights(points, actual, WIND, window_days=window_days, min_days=min_days)
