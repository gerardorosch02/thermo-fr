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

Curve option (branch solar-align, wind_prereg.md): WIND_CURVE = "cubic" is the curve above and the default. "W" is the
smoothed curve of step4_spec.md: the cubic curve averaged over a Gaussian spread of speeds v(1 + 0.15 z), z on 31 evenly
spaced nodes in [-3, 3] with normalised Gaussian weights, and each hour's capacity factor the mean of c(v_T) and c(v_T+1),
which turns instant speeds into a mean over [T, T+1). W_LAST_HOUR says what the last Paris hour of each day uses:
"next" takes the next hour's speed as for every other hour (step 4's definition), "own" uses c(v_T) alone, so that every
hour of a delivery day uses only speeds valid within that day (the fallback of wind_prereg.md, for when the saved morning
windows cannot be shown to hold the speed one hour after the day). The NNLS weighting and monthly refits are unchanged.

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


CUBIC, SMOOTHED = "cubic", "W"
WIND_CURVE = CUBIC  # the setting; "W" only after its pre-registered test passes
W_LAST_HOUR = "own"  # the prereg's fallback: the saved morning windows could not be shown to hold the speed after D (2026-10-18)
W_SIGMA = 0.15
W_NODES = np.linspace(-3.0, 3.0, 31)
W_WEIGHTS = np.exp(-W_NODES ** 2 / 2) / np.exp(-W_NODES ** 2 / 2).sum()


def smoothed_curve(speed) -> np.ndarray:
    """The cubic curve averaged over speeds v(1 + 0.15 z), z on 31 Gaussian nodes (NaN stays NaN)."""
    v = np.asarray(speed, dtype=float)
    return sum(w * power_curve(v * (1.0 + W_SIGMA * z)) for z, w in zip(W_NODES, W_WEIGHTS))


def last_paris_hour(index: pd.DatetimeIndex) -> np.ndarray:
    """True for the last hour of each Paris day (the hour whose end is 00:00 Paris)."""
    end = (pd.DatetimeIndex(index) + pd.Timedelta(hours=1)).tz_convert("Europe/Paris")
    return np.asarray((end.hour == 0) & (end.minute == 0))


def capacity_factors(points: pd.DataFrame, curve: str | None = None, last_hour: str | None = None) -> pd.DataFrame:
    """Each point's capacity factor per hour under the chosen curve."""
    curve = WIND_CURVE if curve is None else curve
    if curve == CUBIC:
        return pd.DataFrame(power_curve(points.to_numpy()), index=points.index, columns=points.columns)
    if curve != SMOOTHED:
        raise ValueError(f"unknown wind curve {curve!r}")
    last_hour = W_LAST_HOUR if last_hour is None else last_hour
    c = pd.DataFrame(smoothed_curve(points.to_numpy()), index=points.index, columns=points.columns)
    following = pd.DataFrame(c.reindex(c.index + pd.Timedelta(hours=1)).to_numpy(), index=c.index, columns=c.columns)
    out = (c + following) / 2
    if last_hour == "own":
        last = last_paris_hour(c.index)
        out[last] = c[last]
    elif last_hour != "next":
        raise ValueError(f"W_LAST_HOUR must be 'own' or 'next', not {last_hour!r}")
    return out


def _factor_identity(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.where(np.isnan(v), np.nan, np.clip(v, 0.0, 1.0))


def spec_for(curve: str | None = None, last_hour: str | None = None) -> ProxySpec:
    """The proxy spec: WIND for the cubic curve; for W, one whose transform takes capacity factors already computed."""
    curve = WIND_CURVE if curve is None else curve
    if curve == CUBIC:
        return WIND
    return ProxySpec(name="wind", output="wind_proxy_mw", columns=tuple(POINT_COLUMNS), transform=_factor_identity,
                     default_path=DEFAULT_WEIGHTS_PATH,
                     meta={"hub_height_m": HUB_HEIGHT_M, "curve": SMOOTHED, "sigma": W_SIGMA, "nodes": len(W_NODES),
                           "last_hour": W_LAST_HOUR if last_hour is None else last_hour,
                           "power_curve": {"cut_in_ms": CUT_IN_MS, "rated_ms": RATED_MS, "cut_out_ms": CUT_OUT_MS}})


def _inputs(points: pd.DataFrame, curve, last_hour) -> pd.DataFrame:
    curve = WIND_CURVE if curve is None else curve
    if curve == CUBIC:
        return points
    columns = [c for c in POINT_COLUMNS if c in points.columns]
    return capacity_factors(points[columns], curve, last_hour)


def actual_wind(hourly: pd.DataFrame) -> pd.Series:
    """Actual onshore plus offshore generation from the inputs table (NaN where neither is reported)."""
    columns = [c for c in ("wind_onshore_mw", "wind_offshore_mw") if c in hourly]
    if not columns:
        return pd.Series(np.nan, index=hourly.index, name="wind_mw")
    return hourly[columns].sum(axis=1, min_count=1).rename("wind_mw")


def point_columns(frame: pd.DataFrame) -> list[str]:
    return gen_proxy.point_columns(frame, WIND)


def calibrate(points: pd.DataFrame, actual: pd.Series, curve: str | None = None, last_hour: str | None = None) -> dict:
    return gen_proxy.calibrate(_inputs(points, curve, last_hour), actual, spec_for(curve, last_hour))


def apply_weights(points: pd.DataFrame, weights: dict, curve: str | None = None, last_hour: str | None = None) -> pd.Series:
    """Expected generation; a weights file written for W says so ("curve") and is applied with W whatever the setting."""
    if curve is None and weights.get("curve") is not None:
        curve, last_hour = weights["curve"], weights.get("last_hour", last_hour)
    return gen_proxy.apply_weights(_inputs(points, curve, last_hour), weights, spec_for(curve, last_hour))


def rolling_proxy(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS,
                  lag_days: int = LAG_DAYS, log=lambda *_: None, curve: str | None = None,
                  last_hour: str | None = None) -> tuple[pd.Series, list[dict]]:
    return gen_proxy.rolling_proxy(_inputs(points, curve, last_hour), actual, spec_for(curve, last_hour), window_days=window_days,
                                   min_days=min_days, lag_days=lag_days, log=log)


def latest_weights(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS,
                   curve: str | None = None, last_hour: str | None = None) -> dict | None:
    return gen_proxy.latest_weights(_inputs(points, curve, last_hour), actual, spec_for(curve, last_hour), window_days=window_days,
                                    min_days=min_days)
