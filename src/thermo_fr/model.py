"""Piecewise-linear model of how temperature drives daily load (or price).

The idea is simple. Above some threshold temperature, heating is off and the
weather barely moves demand. Below it, every degree colder switches on more
electric heating. We search for the threshold that fits best, and the slope
below it is the thermosensitivity, in MW per degree (or EUR/MWh per degree
when the target is price).

Calendar effects (weekday, public holidays) and a fixed effect per month are
included, so the slope is identified from day-to-day weather swings within a
month, not from the difference between summer and winter. That keeps changes
in gas prices or demand trends from leaking into the estimate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import THRESHOLD_GRID
from .data.dataset import hdd


@dataclass
class OLSResult:
    coef: pd.Series
    se: pd.Series
    resid: np.ndarray
    r2: float
    n: int


def ols(y: np.ndarray, X: pd.DataFrame) -> OLSResult:
    """Ordinary least squares with classical standard errors."""
    Xv = X.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    resid = y - Xv @ beta
    n, k = Xv.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.pinv(Xv.T @ Xv)
    centred = y - y.mean()
    r2 = 1.0 - (resid @ resid) / (centred @ centred)
    return OLSResult(
        coef=pd.Series(beta, index=X.columns),
        se=pd.Series(np.sqrt(np.diag(cov)), index=X.columns),
        resid=resid,
        r2=float(r2),
        n=n,
    )


def design(daily: pd.DataFrame, threshold: float, fixed_effects: str = "month_key") -> pd.DataFrame:
    """Regressors: heating degrees, weekday dummies, holiday flag, and fixed effects.

    The full set of fixed-effect dummies plays the role of the intercept.
    """
    X = pd.DataFrame(index=daily.index)
    X["hdd"] = hdd(daily["temperature"], threshold)
    X = X.join(pd.get_dummies(daily["dow"], prefix="dow", drop_first=True, dtype=float))
    X["holiday"] = daily["holiday"].astype(float)
    X = X.join(pd.get_dummies(daily[fixed_effects], prefix="fe", dtype=float))
    return X


@dataclass
class ThermoFit:
    target: str
    threshold: float
    gradient: float  # change in target per degree colder, below the threshold
    gradient_se: float
    r2: float
    n: int


class ThermoModel:
    def __init__(self, target: str = "load_mw", grid=THRESHOLD_GRID, fixed_effects: str = "month_key"):
        self.target = target
        self.grid = grid
        self.fixed_effects = fixed_effects
        self.fit_: ThermoFit | None = None
        self._result: OLSResult | None = None
        self._columns = None

    def _thresholds(self):
        start, stop, step = self.grid
        return np.arange(start, stop + step / 2, step)

    def fit(self, daily: pd.DataFrame, threshold: float | None = None) -> "ThermoModel":
        """Fit the model. If `threshold` is given, use it instead of searching for one."""
        data = daily.dropna(subset=[self.target, "temperature"])
        y = data[self.target].to_numpy(dtype=float)
        candidates = [threshold] if threshold is not None else self._thresholds()

        best = None
        for t in candidates:
            X = design(data, float(t), self.fixed_effects)
            res = ols(y, X)
            sse = float(res.resid @ res.resid)
            if best is None or sse < best[0]:
                best = (sse, float(t), res, X.columns)

        _, t, res, columns = best
        self._result, self._columns = res, columns
        self.fit_ = ThermoFit(
            target=self.target,
            threshold=t,
            gradient=float(res.coef["hdd"]),
            gradient_se=float(res.se["hdd"]),
            r2=res.r2,
            n=res.n,
        )
        return self

    def _check_fitted(self):
        if self.fit_ is None:
            raise RuntimeError("Call fit() first.")

    def predict(self, daily: pd.DataFrame) -> np.ndarray:
        """Predict the target. Fixed-effect levels unseen in training contribute zero."""
        self._check_fitted()
        X = design(daily, self.fit_.threshold, self.fixed_effects)
        X = X.reindex(columns=self._columns, fill_value=0.0)
        return X.to_numpy(dtype=float) @ self._result.coef.to_numpy()

    def adjusted(self, daily: pd.DataFrame) -> pd.Series:
        """Target with calendar and fixed effects removed, for plotting against temperature."""
        self._check_fitted()
        data = daily.dropna(subset=[self.target, "temperature"])
        X = design(data, self.fit_.threshold, self.fixed_effects).reindex(
            columns=self._columns, fill_value=0.0
        )
        others = X.drop(columns="hdd").to_numpy(dtype=float) @ self._result.coef.drop("hdd").to_numpy()
        return data[self.target] - others + others.mean()

    def curve(self, temperatures: np.ndarray, daily: pd.DataFrame) -> np.ndarray:
        """Fitted temperature response on the same scale as `adjusted`."""
        self._check_fitted()
        data = daily.dropna(subset=[self.target, "temperature"])
        X = design(data, self.fit_.threshold, self.fixed_effects).reindex(
            columns=self._columns, fill_value=0.0
        )
        others = X.drop(columns="hdd").to_numpy(dtype=float) @ self._result.coef.drop("hdd").to_numpy()
        return others.mean() + self.fit_.gradient * hdd(temperatures, self.fit_.threshold)
