"""LEAR: a per-hour lasso on variance-stabilised 24-hour vectors (branch model-sprint).

The design follows the LEAR model described by Lago, De Ridder and De Schutter (2021, "Forecasting day-ahead electricity
prices: a review of state-of-the-art algorithms, best practices and an open-access benchmark"): one linear model per hour
of the day, inputs as full 24-hour vectors of lagged prices and day-ahead exogenous forecasts, an asinh transform after
median/MAD scaling, the lasso penalty chosen by an information criterion on the training rows, recalibration every day and
an average over several calibration windows. This is an independent implementation written from the published method; no
code is taken from the authors' epftoolbox library (whose licence statements disagree between its README and repository
metadata), so none of its licence terms apply here.

Adapted to this project's information gate (forecast/timing.py), every input is known by 10:05 Paris on D-1:
- French prices of D-1, D-2, D-3 and D-7 (published 13:00 Paris on the day before each);
- for D: the ENTSO-E load forecast and the honest weather columns temp_c, wind100_ms, radiation_wm2 (as issued two days
  ahead where the archive has them, the proxy before, exactly as features.py; rows with proxy weather are not strict);
- the load forecast of D-1 and of D-7, as published for those days;
- the D-1 day-ahead prices of CH and DE-LU;
- the mean D-1 nuclear output over the hours ending by 08:00 Paris (one value);
- weekday dummies (7), day-type dummies (3), bridge day, eve of a holiday, day after a holiday.
About 300 regressors. The wind and solar proxies and the residual load are left out: they do not exist before 2024-05 and a
lasso cannot take missing values.

Clock changes: rows are Paris clock hours 0 to 23. On the autumn day the two hour-2 rows are averaged in every vector and
both hourly rows receive the hour-2 forecast; on the spring day hour 2 does not exist, so it is filled with the mean of
hours 1 and 3 inside vectors and is not forecast.

Scaling: for each fit, the target and the four price-lag blocks share the target's training median and MAD (x 1.4826);
every other numeric block has its own, all from the training window's rows only; dummies are not scaled.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .daytypes import day_table
from .features import (FEATURE_TIMINGS, LEAR_BLOCK_TIMINGS, LEAR_COLUMNS, LEAR_NH_BLOCK_TIMINGS, LEAR_NH_BLOCKS, LEAR_NH_SOURCE,
                       build_features, lear_columns, lear_nh_columns)
from .timing import check_point_in_time
from .transforms import MAD_SCALE, asinh_forward, asinh_inverse, robust_scale  # noqa: F401

LEAR_WINDOWS = (364, 728, 1092, 1456)  # calibration windows in days, averaged
LEAR_START = pd.Timestamp("2021-01-01")  # windows are truncated here
PRICE_LAGS = (1, 2, 3, 7)
HOURS = list(range(24))

BLOCK_TIMINGS = {**LEAR_BLOCK_TIMINGS, **LEAR_NH_BLOCK_TIMINGS}  # block name -> timing rule (registered column by column in features.FEATURE_TIMINGS)
PRICE_BLOCKS = ("price_d1", "price_d2", "price_d3", "price_d7")
VECTOR_BLOCKS = PRICE_BLOCKS + ("load_d0", "load_d1", "load_d7", "temp_d0", "wind_d0", "rad_d0", "ch_d1", "de_lu_d1")


def day_hour_table(values: pd.Series, days: pd.DatetimeIndex, hours: np.ndarray) -> pd.DataFrame:
    """Rows: Paris delivery days; columns: clock hours 0 to 23. The repeated autumn hour is averaged; the missing spring hour is
    the mean of its neighbours."""
    keyed = pd.DataFrame({"day": days, "hour": hours, "value": np.asarray(values, dtype=float)})
    table = keyed.groupby(["day", "hour"])["value"].mean().unstack("hour").reindex(columns=HOURS)
    spring = table[2].isna() & table[1].notna() & table[3].notna()
    table.loc[spring, 2] = (table.loc[spring, 1] + table.loc[spring, 3]) / 2
    return table


def lear_design(hourly: pd.DataFrame, nh: pd.DataFrame | None = None, variant: str = "base") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(X, Y, blocks): X one row per delivery day with every regressor, Y the 24 prices of the day, blocks the column names of
    each block. Built only from columns whose timing rules are in BLOCK_TIMINGS.

    variant "n1", "n2" or "n3" (branch nuclear-hydro) appends the blocks of features.LEAR_NH_BLOCKS from the `nh` frame: the 24-hour
    vector of nuclear availability for D (clock hours, as the other vectors) and scalars for the change and the hydro values,
    each with its own scaling. The base design is unchanged."""
    table = build_features(hourly, "honest_v2")
    info, X = table.info, table.X
    days = pd.DatetimeIndex(info["delivery_day"])
    hours = info["hour"].to_numpy()
    all_days = pd.date_range(days.min(), days.max(), freq="D")

    def daily(series):
        return day_hour_table(series, days, hours).reindex(all_days)

    price = daily(hourly["price_eur_mwh"].reindex(X.index))
    load = daily(X["load_fc_mw"])
    tables = {
        "temp_d0": daily(X["temp_c"]), "wind_d0": daily(X["wind100_ms"]), "rad_d0": daily(X["radiation_wm2"]),
        "load_d0": load,
        "ch_d1": daily(hourly["price_ch_eur_mwh"].reindex(X.index)) if "price_ch_eur_mwh" in hourly else None,
        "de_lu_d1": daily(hourly["price_de_lu_eur_mwh"].reindex(X.index)) if "price_de_lu_eur_mwh" in hourly else None,
    }

    def shifted(frame, lag):
        out = frame.reindex(all_days - pd.Timedelta(days=lag))
        out.index = all_days
        return out

    parts, blocks = [], {}

    def add(name, frame):
        frame = frame if frame is not None else pd.DataFrame(np.nan, index=all_days, columns=HOURS)
        cols = [f"{name}_h{h:02d}" for h in HOURS]
        parts.append(pd.DataFrame(frame.to_numpy(), index=all_days, columns=cols))
        blocks[name] = cols

    for lag in PRICE_LAGS:
        add(f"price_d{lag}", shifted(price, lag))
    add("load_d0", load)
    add("load_d1", shifted(load, 1))
    add("load_d7", shifted(load, 7))
    for name in ("temp_d0", "wind_d0", "rad_d0"):
        add(name, tables[name])
    add("ch_d1", shifted(tables["ch_d1"], 1) if tables["ch_d1"] is not None else None)
    add("de_lu_d1", shifted(tables["de_lu_d1"], 1) if tables["de_lu_d1"] is not None else None)
    nuclear = X["nuclear_d1_early_mw"].groupby(days).first().reindex(all_days)
    parts.append(pd.DataFrame({"nuclear_d1_early": nuclear.to_numpy()}, index=all_days))
    blocks["nuclear_d1_early"] = ["nuclear_d1_early"]
    cal = day_table(all_days)
    dummies = {f"dow_{k}": (cal["dow"] == k).astype(float) for k in range(7)}
    dummies.update({f"day_type_{k}": (cal["day_type"] == k).astype(float) for k in range(3)})
    dummies.update({c: cal[c].astype(float) for c in ("bridge_day", "pre_holiday", "post_holiday")})
    parts.append(pd.DataFrame(dummies, index=all_days))
    blocks["calendar"] = list(dummies)
    extra = LEAR_NH_BLOCKS[variant]
    if extra:
        if nh is None:
            raise ValueError(f"LEAR variant {variant} needs the nuclear-hydro inputs")
        frame = nh.reindex(X.index)
        for name in extra:
            if name == "nuc_avail_d0":
                add(name, daily(frame["nuc_avail_mw"]))
            else:
                value = frame[LEAR_NH_SOURCE[name]].groupby(days).first().reindex(all_days)
                parts.append(pd.DataFrame({name: value.to_numpy()}, index=all_days))
                blocks[name] = lear_nh_columns(name)
    design = pd.concat(parts, axis=1)
    assert list(design.columns) == lear_columns(variant)
    return design, price, blocks


def lear_timings(blocks: dict) -> dict:
    """Every LEAR column with its timing rule: the block's rule, which must equal the column's entry in FEATURE_TIMINGS."""
    out = {}
    for name, cols in blocks.items():
        for col in cols:
            if FEATURE_TIMINGS.get(col) != BLOCK_TIMINGS[name]:
                raise KeyError(f"LEAR column {col} has no matching timing rule in FEATURE_TIMINGS")
            out[col] = BLOCK_TIMINGS[name]
    return out


def assert_gate(index: pd.DatetimeIndex, blocks: dict) -> None:
    """Every LEAR input is published before the 12:00 gate and the 10:05 pre-market issue time, for every forecast row; raises
    LookaheadError otherwise (the run stops)."""
    timings = lear_timings(blocks)
    check_point_in_time(index, timings, deadline="gate")
    check_point_in_time(index, timings, deadline="premarket")


def fit_scalers(X: pd.DataFrame, Y: pd.DataFrame, blocks: dict) -> dict:
    """Scaling parameters from the training rows passed in (and only those)."""
    target = robust_scale(Y.to_numpy())
    scalers = {"target": target}
    for name, cols in blocks.items():
        if name == "calendar":
            continue
        scalers[name] = target if name in PRICE_BLOCKS else robust_scale(X[cols].to_numpy())
    return scalers


def transform(X: pd.DataFrame, blocks: dict, scalers: dict) -> np.ndarray:
    out = X.copy()
    for name, cols in blocks.items():
        if name == "calendar":
            continue
        m, s = scalers[name]
        out[cols] = asinh_forward(out[cols].to_numpy(), m, s)
    return out.to_numpy(dtype=float)


def fit_hour(Xt: np.ndarray, yt: np.ndarray, max_iter: int = 2500):
    """LassoLarsIC (AIC) picks the penalty on the training rows; a Lasso with that penalty is the model."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso, LassoLarsIC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        alpha = LassoLarsIC(criterion="aic", max_iter=max_iter).fit(Xt, yt).alpha_
        model = Lasso(alpha=max(alpha, 1e-12), max_iter=max_iter, precompute=True).fit(Xt, yt)  # Gram precomputed: same solution, faster
    return model


def fit_window(X: pd.DataFrame, Y: pd.DataFrame, blocks: dict, max_iter: int = 2500):
    """Scalers and 24 hourly models from the given training days (rows with any missing value are dropped)."""
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X[ok], Y[ok]
    if len(X) <= X.shape[1] + 1:
        return None
    scalers = fit_scalers(X, Y, blocks)
    Xt = transform(X, blocks, scalers)
    m, s = scalers["target"]
    models = [fit_hour(Xt, asinh_forward(Y[h].to_numpy(), m, s), max_iter) for h in HOURS]
    return {"scalers": scalers, "models": models, "rows": int(len(X))}


def predict_window(fitted, X: pd.DataFrame, blocks: dict) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=X.index, columns=HOURS)
    if fitted is None:
        return out
    ok = X.notna().all(axis=1)
    if not ok.any():
        return out
    Xt = transform(X[ok], blocks, fitted["scalers"])
    m, s = fitted["scalers"]["target"]
    for h, model in zip(HOURS, fitted["models"]):
        out.loc[ok, h] = asinh_inverse(model.predict(Xt), m, s)
    return out


def window_days(day: pd.Timestamp, length: int | None, start: pd.Timestamp = LEAR_START) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Training days for a forecast of `day`: the `length` days ending on D-1 (the last day whose price is published before the
    issue time), never before `start`. length None means every day from `start`."""
    last = day - pd.Timedelta(days=1)
    first = start if length is None else max(start, last - pd.Timedelta(days=length - 1))
    return first, last


def _fit_predict_window(design, price, blocks, anchor, w, start, group, coef_blocks=None):
    first, last = window_days(anchor, w, start)
    train = (design.index >= first) & (design.index <= last)
    fitted = fit_window(design[train], price[train], blocks)
    pred = predict_window(fitted, design.loc[group], blocks)
    if coef_blocks is None:
        return pred
    return pred, coefficient_rows(fitted, design.columns, blocks, coef_blocks, anchor, w)


def coefficient_rows(fitted, columns, blocks: dict, coef_blocks, day, window) -> list:
    """Per hourly model: the coefficient on the own-hour nuclear availability, the sum over that 24-hour block, and the coefficient
    of each scalar block (on the scaled asinh inputs; the transforms are increasing, so the signs carry over)."""
    if fitted is None:
        return []
    position = {c: i for i, c in enumerate(columns)}
    rows = []
    for h, model in zip(HOURS, fitted["models"]):
        coef = model.coef_
        row = {"day": day, "window": window, "hour": h}
        for name in coef_blocks:
            if name not in blocks:
                continue
            if name == "nuc_avail_d0":
                idx = [position[c] for c in blocks[name]]
                row["nuc_own_hour"] = float(coef[idx[h]])
                row["nuc_block_sum"] = float(coef[idx].sum())
                row["nuc_block_nonzero"] = int((coef[idx] != 0).sum())
            else:
                row[name] = float(coef[position[blocks[name][0]]])
        rows.append(row)
    return rows


def lear_forecast_days(design: pd.DataFrame, price: pd.DataFrame, blocks: dict, days, windows=LEAR_WINDOWS, recalibrate: str = "daily",
                       start: pd.Timestamp = LEAR_START, log=print, n_jobs: int = 1, coef_sink: list | None = None,
                       coef_blocks=None) -> pd.DataFrame:
    """Daily table (index day, MultiIndex columns (window, hour) plus ('mean', hour)) of LEAR forecasts for `days`.

    recalibrate="daily": every day refits on its own windows (ending D-1). "monthly": one fit per month on windows ending on
    the day before the month starts, used for every day of the month.
    """
    days = pd.DatetimeIndex(days)
    frames = []
    groups = [[d] for d in days] if recalibrate == "daily" else [list(g) for _, g in pd.Series(days, index=days).groupby(days.to_period("M"))]
    for group in groups:
        anchor = pd.Timestamp(group[0]) if recalibrate == "daily" else pd.Timestamp(group[0]).to_period("M").to_timestamp()
        if coef_sink is not None:
            pairs = [_fit_predict_window(design, price, blocks, anchor, w, start, group, coef_blocks) for w in windows]
            preds = [p for p, _ in pairs]
            for _, rows in pairs:
                coef_sink.extend(rows)
        elif n_jobs == 1:
            preds = [_fit_predict_window(design, price, blocks, anchor, w, start, group) for w in windows]
        else:
            from joblib import Parallel, delayed

            preds = Parallel(n_jobs=n_jobs)(delayed(_fit_predict_window)(design, price, blocks, anchor, w, start, group) for w in windows)
        per_window = dict(zip(windows, preds))
        stacked = pd.concat(per_window, axis=1)
        mean = pd.concat([per_window[w] for w in windows]).groupby(level=0).mean()
        mean.columns = pd.MultiIndex.from_product([["mean"], HOURS])
        frames.append(pd.concat([stacked, mean], axis=1))
        if recalibrate != "daily" or pd.Timestamp(group[0]).day == 1:
            log(f"  lear: {pd.Timestamp(group[0]):%Y-%m} recalibrated ({recalibrate})")
    return pd.concat(frames).sort_index()


def lear_walk_forward(hourly: pd.DataFrame, test_start: str, test_end: str, windows=LEAR_WINDOWS, recalibrate: str = "daily",
                      start: pd.Timestamp = LEAR_START, log=print, n_jobs: int = 1, nh: pd.DataFrame | None = None,
                      variant: str = "base", coef_sink: list | None = None) -> pd.DataFrame:
    """LEAR forecasts on the hourly rows of the test window [test_start, test_end), with the honest_v2 info columns, the actual
    price and the strict flag, so they align row for row with the gradient boosting walk-forward. `variant` and `nh` add the
    nuclear-hydro blocks (lear_design); `coef_sink`, a list, collects their coefficients per window and hourly model."""
    hourly = hourly.sort_index()
    design, price, blocks = lear_design(hourly, nh, variant)
    table = build_features(hourly, "honest_v2")
    info = table.info
    rows = (info["delivery_day"] >= pd.Timestamp(test_start)) & (info["delivery_day"] < pd.Timestamp(test_end))
    assert_gate(table.X.index[rows], blocks)
    days = pd.DatetimeIndex(sorted(info.loc[rows, "delivery_day"].unique()))
    daily = lear_forecast_days(design, price, blocks, days, windows, recalibrate, start, log, n_jobs, coef_sink=coef_sink,
                               coef_blocks=LEAR_NH_BLOCKS[variant] if coef_sink is not None else None)
    out = info[rows].copy()
    out["actual"] = table.y[rows]
    keys = list(zip(out["delivery_day"], out["hour"]))
    mean = daily["mean"]
    out["lear"] = [mean.at[d, h] if d in mean.index else np.nan for d, h in keys]
    for w in windows:
        out[f"lear_w{w}"] = [daily[w].at[d, h] if d in daily.index else np.nan for d, h in keys]
    out["strict"] = info.loc[rows, "weather_point_in_time"].to_numpy()
    return out
