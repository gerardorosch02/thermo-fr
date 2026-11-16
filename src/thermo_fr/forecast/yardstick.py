"""The yardstick: measurements M1 to M7 of docs/prereg/lear_recipe_prereg.md (Part 1). Measurement only, no decision.

Every function takes an hourly frame of the walk-forward kind (one row per delivery hour, with delivery_day, hour, actual,
strict, naive_day = the same hour of D-1, naive_week = the same hour of D-7, and one column per forecast) and scores the
strict rows with an actual price, like the sprint's selection report. Nothing here fits a model.

- M1 `relative_mae`: MAE divided by the MAE of three naive forecasts (D-1, D-7 and the benchmark naive of Lago et al. 2021:
  D-1 for delivery days Tuesday to Friday, D-7 for Saturday, Sunday and Monday).
- M2 `under_reaction`: ordinary least squares, with an intercept, of (actual - D-1) on (forecast - D-1), on hourly rows and on
  daily means. A slope above one means the forecast moves less than the price does.
- M3 `paired_daily`: paired daily MAE difference with the project's moving-block bootstrap.
- M4 `shape`: the per-day Spearman rank correlation of forecast and actual hourly prices, and the spread capture.
- M5 `split`: any of the above before and from a date.
- M7 `tightness`, `surprises`, `weather_effect`: the regression of the forecast error on the wind and solar surprise, by
  tercile of the day's tightness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .shape import block_bootstrap_mean

NAIVES = ("naive_day", "naive_week", "naive_lago")
LAGO_D1_WEEKDAYS = (1, 2, 3, 4)  # pandas dayofweek: Tuesday to Friday take D-1; Saturday, Sunday and Monday take D-7
SPREAD_HOURS = 4
QUARTER_HOUR_AUCTION = pd.Timestamp("2025-10-01")
TIGHTNESS_START = pd.Timestamp("2024-05-01")


def strict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["strict"].astype(bool)].dropna(subset=["actual"])


def lago_naive(frame: pd.DataFrame) -> pd.Series:
    """The benchmark naive of Lago et al.: the same hour of D-1 for Tuesday to Friday, of D-7 for Saturday, Sunday and Monday."""
    weekday = pd.DatetimeIndex(frame["delivery_day"]).dayofweek
    return frame["naive_day"].where(weekday.isin(LAGO_D1_WEEKDAYS), frame["naive_week"])


def with_naives(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(naive_lago=lago_naive(frame))


def mae(frame: pd.DataFrame, column: str) -> float:
    rows = strict_rows(frame)
    return float((rows[column] - rows["actual"]).abs().mean())


def relative_mae(frame: pd.DataFrame, column: str, naives=NAIVES) -> dict:
    """M1. MAE of `column` and its ratio to each naive's MAE, all on the strict rows where the forecast and every naive exist."""
    rows = strict_rows(with_naives(frame)).dropna(subset=[column, *naives])
    own = float((rows[column] - rows["actual"]).abs().mean()) if len(rows) else float("nan")
    out = {"hours": int(len(rows)), "mae": own}
    for name in naives:
        base = float((rows[name] - rows["actual"]).abs().mean()) if len(rows) else float("nan")
        out[f"mae_{name}"] = base
        out[f"rmae_{name}"] = own / base if base else float("nan")
    return out


def ols_slope(x, y) -> dict:
    """Ordinary least squares of y on x with an intercept: slope, intercept, correlation and the number of points."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.var(x) == 0:
        return {"n": int(len(x)), "slope": float("nan"), "intercept": float("nan"), "correlation": float("nan")}
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    return {"n": int(len(x)), "slope": slope, "intercept": float(y.mean() - slope * x.mean()),
            "correlation": float(np.corrcoef(x, y)[0, 1])}


def under_reaction(frame: pd.DataFrame, column: str) -> dict:
    """M2. (actual - D-1) regressed on (forecast - D-1): hourly rows, and daily means of the same strict rows."""
    rows = strict_rows(frame).dropna(subset=[column, "naive_day"])
    hourly = ols_slope(rows[column] - rows["naive_day"], rows["actual"] - rows["naive_day"])
    daily = rows.groupby("delivery_day")[[column, "actual", "naive_day"]].mean()
    return {"hourly": hourly, "daily": ols_slope(daily[column] - daily["naive_day"], daily["actual"] - daily["naive_day"])}


def daily_abs_error(frame: pd.DataFrame, column: str) -> pd.Series:
    rows = strict_rows(frame)
    return (rows[column] - rows["actual"]).abs().groupby(rows["delivery_day"]).mean()


def paired_daily(frame: pd.DataFrame, column: str, baseline: str, one_sided_level: float | None = None, block: int = 7,
                 resamples: int = 5000, seed: int = 0) -> dict:
    """M3 and the decision rule's interval. Mean of (daily MAE of column - daily MAE of baseline), its two-sided 95%
    moving-block bootstrap interval and, if asked, the upper end of the one-sided interval at `one_sided_level`."""
    diff = (daily_abs_error(frame, column) - daily_abs_error(frame, baseline)).dropna().to_numpy()
    low, high = block_bootstrap_mean(diff, block=block, resamples=resamples, seed=seed, level=0.95)
    out = {"days": int(len(diff)), "mean": float(diff.mean()) if len(diff) else float("nan"), "ci95": [low, high],
           "share_days_better": float((diff < 0).mean()) if len(diff) else float("nan")}
    if one_sided_level is not None:
        _, upper = block_bootstrap_mean(diff, block=block, resamples=resamples, seed=seed, level=1 - 2 * (1 - one_sided_level))
        out["one_sided_level"] = one_sided_level
        out["upper_one_sided"] = upper
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def spearman(a, b) -> float:
    """Spearman rank correlation with average ranks for ties; NaN if either side is constant."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def spread_of(order_by: np.ndarray, actual: np.ndarray, k: int = SPREAD_HOURS) -> float:
    """Mean actual price in the k highest hours of `order_by` minus the mean in its k lowest hours (ties: the earlier row)."""
    high = np.argsort(-order_by, kind="stable")[:k]
    low = np.argsort(order_by, kind="stable")[:k]
    return float(actual[high].mean() - actual[low].mean())


def shape(frame: pd.DataFrame, column: str, k: int = SPREAD_HOURS) -> dict:
    """M4. Per strict day (rows in time order): the Spearman correlation of forecast and actual, and the spread capture, the sum
    over days of the spread earned by the forecast's k highest and k lowest hours divided by the sum of the best possible."""
    rows = strict_rows(frame).dropna(subset=[column])
    rhos, earned, best = [], 0.0, 0.0
    for _, part in rows.groupby("delivery_day", sort=True):
        f, a = part[column].to_numpy(dtype=float), part["actual"].to_numpy(dtype=float)
        if len(f) < 2 * k:
            continue
        rhos.append(spearman(f, a))
        earned += spread_of(f, a, k)
        best += spread_of(a, a, k)
    rhos = np.asarray(rhos, dtype=float)
    return {"days": int(len(rhos)), "days_with_rank_correlation": int(np.isfinite(rhos).sum()),
            "spearman_mean": float(np.nanmean(rhos)) if len(rhos) else float("nan"),
            "spearman_median": float(np.nanmedian(rhos)) if len(rhos) else float("nan"),
            "spread_capture": earned / best if best else float("nan")}


def split(frame: pd.DataFrame, at=QUARTER_HOUR_AUCTION) -> dict:
    """M5. The frame before `at` and from `at` (delivery days)."""
    at = pd.Timestamp(at)
    return {"before": frame[frame["delivery_day"] < at], "from": frame[frame["delivery_day"] >= at]}


# --- M7 ---------------------------------------------------------------------------------------------------------------------

def tightness(load_fc_mw: pd.Series, wind_proxy_mw: pd.Series, solar_proxy_mw: pd.Series, nuclear_d1_early_mw: pd.Series,
              delivery_day) -> pd.Series:
    """Tightness of each delivery day in GW: the maximum over its hours of (load forecast - wind proxy - solar proxy), minus the
    mean D-1 nuclear output over the hours ending by 08:00 Paris. NaN for a day with any hour of any input missing."""
    residual = load_fc_mw - wind_proxy_mw - solar_proxy_mw
    days = pd.DatetimeIndex(delivery_day)
    complete = (residual.notna() & nuclear_d1_early_mw.notna()).groupby(days).all()
    peak = residual.groupby(days).max()
    nuclear = nuclear_d1_early_mw.groupby(days).first()
    return ((peak - nuclear) / 1000.0).where(complete)


def terciles(values: pd.Series) -> pd.Series:
    """1, 2 or 3 by the terciles of the non-missing values (cut points at the 1/3 and 2/3 quantiles, upper bounds inclusive)."""
    v = values.dropna()
    low, high = v.quantile(1 / 3), v.quantile(2 / 3)
    out = pd.Series(np.where(v <= low, 1, np.where(v <= high, 2, 3)), index=v.index)
    return out.reindex(values.index)


def regress(y, X) -> dict:
    """Least squares of y on the columns of X with an intercept: coefficients, intercept, R squared, points."""
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) <= X.shape[1] + 1:
        return {"n": int(len(y)), "coefficients": [float("nan")] * X.shape[1], "intercept": float("nan"), "r2": float("nan")}
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    total = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(y)), "coefficients": [float(b) for b in beta[1:]], "intercept": float(beta[0]),
            "r2": float(1 - (resid**2).sum() / total) if total > 0 else float("nan")}


def weather_effect(table: pd.DataFrame, group: pd.Series | None = None) -> dict:
    """M7 regressions. `table` has error (actual - forecast, EUR/MWh), wind_surprise_gw and solar_surprise_gw (actual - proxy);
    `group` (same index) gives each row's tercile. Returns the pooled fit and one fit per group, coefficients in EUR/MWh per GW."""
    def fit(part):
        r = regress(part["error"], part[["wind_surprise_gw", "solar_surprise_gw"]])
        return {"n": r["n"], "wind_eur_per_gw": r["coefficients"][0], "solar_eur_per_gw": r["coefficients"][1],
                "intercept": r["intercept"], "r2": r["r2"]}

    out = {"pooled": fit(table)}
    if group is not None:
        for key in sorted(pd.unique(group.dropna())):
            out[f"tercile_{int(key)}"] = fit(table[group == key])
    return out
