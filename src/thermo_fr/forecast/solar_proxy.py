"""A pre-gate proxy for French solar generation, calibrated on actual generation history.

Built like the wind proxy (gen_proxy.py): shortwave radiation forecasts issued
two days ahead at the points of data/solar_points.py, each turned into a
capacity factor as the ratio of the forecast global horizontal irradiance to
a reference of 1,000 W/m2 (the standard test condition of a photovoltaic
module), then weighted in MW by non-negative least squares against the
ENTSO-E actual solar generation (A75, psrType B16).

The transform is linear, so a weight is the megawatts of panels a point
stands for at reference irradiance, and the sum of the weights is an
effective capacity. Two things the transform ignores are left to the
calibration: panels are tilted, so their plane-of-array irradiance exceeds
the horizontal one in winter more than in summer, and module efficiency
falls with cell temperature. Both make the ratio of actual generation to
horizontal irradiance drift through the year, and the French fleet grows by
about a fifth a year, which argued for a shorter calibration window than
the wind proxy's. Checked on the proxy's own error against actual
generation (2024-05 to 2026-09, docs/forecast.md), windows from 60 to 365
days were within 3 percent of each other in MAE, with the 365-day window
lowest, so the solar proxy keeps the wind proxy's window and minimum.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.solar_points import POINT_COLUMNS
from . import gen_proxy
from .gen_proxy import LAG_DAYS, ProxySpec, load_weights, save_weights  # noqa: F401  re-exported

REFERENCE_WM2 = 1000.0
WINDOW_DAYS = gen_proxy.WINDOW_DAYS
MIN_TRAIN_DAYS = gen_proxy.MIN_TRAIN_DAYS
DEFAULT_WEIGHTS_PATH = Path("published/model/solar_proxy.json")


def pv_factor(radiation) -> np.ndarray:
    """Capacity factor for a global horizontal irradiance in W/m2: the ratio to 1,000 W/m2, floored at 0 (NaN stays NaN)."""
    g = np.asarray(radiation, dtype=float)
    return np.where(np.isnan(g), np.nan, np.clip(g / REFERENCE_WM2, 0.0, None))


SOLAR = ProxySpec(
    name="solar",
    output="solar_proxy_mw",
    columns=tuple(POINT_COLUMNS),
    transform=pv_factor,
    default_path=DEFAULT_WEIGHTS_PATH,
    window_days=WINDOW_DAYS,
    min_days=MIN_TRAIN_DAYS,
    meta={"reference_wm2": REFERENCE_WM2},
)


def actual_solar(hourly: pd.DataFrame) -> pd.Series:
    """Actual solar generation from the inputs table (NaN where not reported)."""
    if "solar_mw" not in hourly:
        return pd.Series(np.nan, index=hourly.index, name="solar_mw")
    return hourly["solar_mw"].rename("solar_mw")


def calibrate(points: pd.DataFrame, actual: pd.Series) -> dict:
    return gen_proxy.calibrate(points, actual, SOLAR)


def apply_weights(points: pd.DataFrame, weights: dict) -> pd.Series:
    return gen_proxy.apply_weights(points, weights, SOLAR)


def rolling_proxy(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS,
                  lag_days: int = LAG_DAYS, log=lambda *_: None) -> tuple[pd.Series, list[dict]]:
    return gen_proxy.rolling_proxy(points, actual, SOLAR, window_days=window_days, min_days=min_days, lag_days=lag_days, log=log)


def latest_weights(points: pd.DataFrame, actual: pd.Series, window_days: int = WINDOW_DAYS, min_days: int = MIN_TRAIN_DAYS) -> dict | None:
    return gen_proxy.latest_weights(points, actual, SOLAR, window_days=window_days, min_days=min_days)
