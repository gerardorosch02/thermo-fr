"""Probabilistic forecasts next to the point forecast: price quantiles with a conformal interval, and event probabilities.

Everything here uses the honest_v2 inputs only and the monthly walk-forward of
the point model (retrain at the start of every month on all earlier days).
Branch prob-forecasts; the definitions and acceptance rules were written into
docs/experiments.md before anything ran.

Quantiles. Three LightGBM models with the quantile objective at alpha 0.10,
0.50 and 0.90. Crossing quantiles are sorted per row. The 10-90 interval is
then recalibrated conformally: for the rows of a day, the conformity scores
s = max(q10 - y, y - q90) of the out-of-sample rows of the previous
CONFORMAL_WINDOW_DAYS days (all of them known at the issue time, since a
day's actual is published the day before the next issue) give the margin
Q = the ceil((n + 1)(1 - alpha)) / n quantile with alpha = 0.20, and the
interval becomes [q10 - Q, q90 + Q]. Days without enough past scores keep
the raw interval. Benchmarks for the interval: (a) the point forecast plus
the trailing-year empirical quantiles of its out-of-sample error by hour of
day; (b) the D-1 same-hour price plus the trailing-year empirical quantiles
of the D-1 error by hour of day. Both use only errors of days before the
delivery day.

Events. Negative price: y < 0. Spike: y above the 95th percentile of the
hourly prices of the trailing 365 days known at the issue time (days up to
D-1); that threshold is also an input of the classifier. Two LightGBM
classifiers. Benchmarks: (a) climatology, the event frequency for that hour
of day and calendar month over the trailing year; (b) the frequency of the
event at that hour over the last 7 days. Metrics: Brier score, Brier skill
score against each benchmark, log loss, a 10-bin reliability table and the
number of events.

Pinball loss at level a for forecast q and outcome y is a (y - q) if y >= q,
else (1 - a)(q - y); the mean over the three levels is the headline.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FeatureTable
from .models import GBM_PARAMS

QUANTILES = (0.10, 0.50, 0.90)
ALPHA = 0.20  # the 10-90 interval misses 20% of outcomes when calibrated
CONFORMAL_WINDOW_DAYS = 90
TRAILING_DAYS = 365
SPIKE_PERCENTILE = 0.95
LAST_DAYS = 7
CLASSIFIER_PARAMS = {**GBM_PARAMS, "n_estimators": 400, "objective": "binary"}
EVENTS = ("negative", "spike")
SLICES = ("weekends", "holidays", "windiest_10pct_days", "sunniest_10pct_days", "top_5pct_price_hours")
COVERAGE_RANGE = (0.75, 0.85)


def quantile_model(alpha: float, params: dict | None = None):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**{**GBM_PARAMS, **(params or {}), "objective": "quantile", "alpha": alpha})


def event_model(params: dict | None = None):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**{**CLASSIFIER_PARAMS, **(params or {})})


def sort_quantiles(frame: pd.DataFrame, columns=("q10", "q50", "q90")) -> pd.DataFrame:
    """Crossing quantiles are fixed by sorting each row."""
    values = np.sort(frame[list(columns)].to_numpy(dtype=float), axis=1)
    out = frame.copy()
    for i, c in enumerate(columns):
        out[c] = values[:, i]
    return out


def spike_thresholds(y: pd.Series, days: pd.Series, trailing_days: int = TRAILING_DAYS, percentile: float = SPIKE_PERCENTILE) -> pd.Series:
    """For every row, the `percentile` of the hourly prices of the trailing_days delivery days before the row's day (days up to D-1)."""
    frame = pd.DataFrame({"y": y.to_numpy(), "day": pd.DatetimeIndex(days).normalize()})
    unique_days = np.sort(frame["day"].unique())
    by_day = frame.dropna().groupby("day")["y"].apply(np.asarray)
    thresholds = {}
    for day in unique_days:
        start = day - pd.Timedelta(days=trailing_days)
        window = by_day[(by_day.index >= start) & (by_day.index < day)]
        thresholds[day] = float(np.quantile(np.concatenate(window.to_numpy()), percentile)) if len(window) >= 30 else np.nan
    return pd.Series([thresholds[d] for d in frame["day"]], index=y.index, name="spike_threshold")


def event_labels(y: pd.Series, threshold: pd.Series) -> pd.DataFrame:
    labels = pd.DataFrame(index=y.index)
    labels["negative"] = (y < 0).astype(float).where(y.notna())
    labels["spike"] = (y > threshold).astype(float).where(y.notna() & threshold.notna())
    return labels


@dataclass
class ProbBacktest:
    predictions: pd.DataFrame
    months: list = field(default_factory=list)


def walk_forward_prob(table: FeatureTable, test_start: str, test_end: str, log=print, params: dict | None = None,
                      classifier_params: dict | None = None) -> ProbBacktest:
    """Monthly walk-forward of the point model, the three quantile models and the two event classifiers on the same features."""
    from .models import make_model

    days = table.info["delivery_day"]
    X = table.X.copy()
    X["spike_threshold"] = spike_thresholds(table.y, days)
    labels = event_labels(table.y, X["spike_threshold"])
    frames, months = [], []
    for start in pd.date_range(pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end), freq="MS", inclusive="left"):
        end = start + pd.DateOffset(months=1)
        train = (days < start) & table.y.notna()
        test = (days >= start) & (days < end)
        if not test.any():
            continue
        X_train, X_test = X[train], X[test]
        out = table.info.loc[test, ["delivery_day", "hour", "is_peak", "dow", "holiday", "wind_mw", "solar_mw"]].copy()
        out["actual"] = table.y[test]
        out["naive_day"] = table.X.loc[test, "price_lag1"]
        out["spike_threshold"] = X_test["spike_threshold"]
        point = make_model("gbm", params)
        point.fit(X_train.drop(columns=["spike_threshold"]), table.y[train])
        out["point"] = np.asarray(point.predict(X_test.drop(columns=["spike_threshold"])), dtype=float)
        for alpha, name in zip(QUANTILES, ("q10", "q50", "q90")):
            model = quantile_model(alpha, params)
            model.fit(X_train.drop(columns=["spike_threshold"]), table.y[train])
            out[name] = np.asarray(model.predict(X_test.drop(columns=["spike_threshold"])), dtype=float)
        for event in EVENTS:
            keep = train & labels[event].notna()
            y_train = labels.loc[keep, event]
            if y_train.nunique() < 2:
                out[f"p_{event}"] = float(y_train.mean()) if len(y_train) else np.nan
                continue
            model = event_model(classifier_params)
            model.fit(X[keep], y_train.astype(int))
            out[f"p_{event}"] = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
        out["strict"] = table.info.loc[test, "weather_point_in_time"].to_numpy()
        frames.append(out)
        months.append(start.strftime("%Y-%m"))
        log(f"  prob: {start:%Y-%m} trained on {int(train.sum()):,} hours")
    predictions = sort_quantiles(pd.concat(frames).sort_index())
    predictions = predictions.join(labels.rename(columns={"negative": "is_negative", "spike": "is_spike"}), how="left")
    return ProbBacktest(predictions, months)


# ---------------------------------------------------------------------------- trailing windows (no future data)

def _by_day(frame: pd.DataFrame):
    return frame.groupby(pd.DatetimeIndex(frame["delivery_day"]).normalize(), sort=True)


def conformal_interval(pred: pd.DataFrame, window_days: int = CONFORMAL_WINDOW_DAYS, alpha: float = ALPHA, min_days: int = 20) -> pd.DataFrame:
    """[q10 - Q, q90 + Q] with Q from the conformity scores of the previous `window_days` days' out-of-sample rows."""
    out = pred.copy()
    scores = np.maximum(out["q10"] - out["actual"], out["actual"] - out["q90"])
    day_index = pd.DatetimeIndex(out["delivery_day"]).normalize()
    daily_scores = {day: s.dropna().to_numpy() for day, s in scores.groupby(day_index)}
    margins = {}
    for day in np.sort(day_index.unique()):
        start = day - pd.Timedelta(days=window_days)
        past = [daily_scores[d] for d in daily_scores if start <= d < day and len(daily_scores[d])]
        if len(past) < min_days:
            margins[day] = 0.0
            continue
        values = np.concatenate(past)
        n = len(values)
        rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        margins[day] = float(np.sort(values)[rank - 1])
    margin = np.array([margins[d] for d in day_index])
    out["conformal_margin"] = margin
    out["lo"] = out["q10"] - margin
    out["hi"] = out["q90"] + margin
    return out


def trailing_error_quantiles(pred: pd.DataFrame, column: str, trailing_days: int = TRAILING_DAYS, min_days: int = 30) -> pd.DataFrame:
    """Benchmark interval: `column` plus the 10th/50th/90th percentiles of its past errors (actual minus column) by hour of day."""
    out = pred.copy()
    error = (out["actual"] - out[column])
    day_index = pd.DatetimeIndex(out["delivery_day"]).normalize()
    hours = out["hour"].to_numpy()
    lo = np.full(len(out), np.nan)
    mid = np.full(len(out), np.nan)
    hi = np.full(len(out), np.nan)
    frame = pd.DataFrame({"day": day_index, "hour": hours, "err": error.to_numpy()}).dropna()
    for day in np.sort(day_index.unique()):
        rows = np.where(day_index == day)[0]
        past = frame[(frame["day"] >= day - pd.Timedelta(days=trailing_days)) & (frame["day"] < day)]
        if past["day"].nunique() < min_days:
            continue
        q = past.groupby("hour")["err"].quantile([0.10, 0.50, 0.90]).unstack()
        for r in rows:
            h = hours[r]
            if h in q.index:
                lo[r], mid[r], hi[r] = q.loc[h, 0.10], q.loc[h, 0.50], q.loc[h, 0.90]
    out[f"{column}_lo"] = out[column] + lo
    out[f"{column}_mid"] = out[column] + mid
    out[f"{column}_hi"] = out[column] + hi
    return out


def event_benchmarks(pred: pd.DataFrame, event: str, trailing_days: int = TRAILING_DAYS, last_days: int = LAST_DAYS) -> pd.DataFrame:
    """Climatology (hour of day and calendar month over the trailing year) and the last-7-day frequency at that hour, from past labels only."""
    out = pred.copy()
    label = out[f"is_{event}"]
    day_index = pd.DatetimeIndex(out["delivery_day"]).normalize()
    frame = pd.DataFrame({"day": day_index, "hour": out["hour"].to_numpy(), "month": day_index.month, "y": label.to_numpy()}).dropna()
    clim = np.full(len(out), np.nan)
    recent = np.full(len(out), np.nan)
    hours, months = out["hour"].to_numpy(), day_index.month
    for day in np.sort(day_index.unique()):
        rows = np.where(day_index == day)[0]
        past = frame[(frame["day"] >= day - pd.Timedelta(days=trailing_days)) & (frame["day"] < day)]
        last = frame[(frame["day"] >= day - pd.Timedelta(days=last_days)) & (frame["day"] < day)]
        if past.empty:
            continue
        by_hm = past.groupby(["hour", "month"])["y"].mean()
        by_h = past.groupby("hour")["y"].mean()
        by_h_last = last.groupby("hour")["y"].mean() if not last.empty else pd.Series(dtype=float)
        for r in rows:
            key = (hours[r], months[r])
            clim[r] = by_hm.get(key, by_h.get(hours[r], np.nan))
            recent[r] = by_h_last.get(hours[r], np.nan)
    out[f"clim_{event}"] = clim
    out[f"last7_{event}"] = recent
    return out


# ---------------------------------------------------------------------------- metrics

def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    diff = y - q
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))


def interval_metrics(frame: pd.DataFrame, lo: str, mid: str, hi: str) -> dict:
    f = frame.dropna(subset=["actual", lo, mid, hi])
    y = f["actual"].to_numpy()
    covered = (y >= f[lo].to_numpy()) & (y <= f[hi].to_numpy())
    return {"hours": int(len(f)),
            "pinball_q10": round(pinball(y, f[lo].to_numpy(), 0.10), 3), "pinball_q50": round(pinball(y, f[mid].to_numpy(), 0.50), 3),
            "pinball_q90": round(pinball(y, f[hi].to_numpy(), 0.90), 3),
            "pinball_mean": round(float(np.mean([pinball(y, f[lo].to_numpy(), 0.10), pinball(y, f[mid].to_numpy(), 0.50), pinball(y, f[hi].to_numpy(), 0.90)])), 3),
            "coverage_10_90": round(float(covered.mean()), 4), "mean_width": round(float((f[hi] - f[lo]).mean()), 2)}


def slice_masks(frame: pd.DataFrame) -> dict:
    masks = {"all": np.ones(len(frame), dtype=bool)}
    if "dow" in frame:
        masks["weekends"] = (frame["dow"] >= 5).to_numpy()
    if "holiday" in frame:
        masks["holidays"] = (frame["holiday"] == 1).to_numpy()
    days = pd.DatetimeIndex(frame["delivery_day"])
    for column, name in (("wind_mw", "windiest_10pct_days"), ("solar_mw", "sunniest_10pct_days")):
        if column in frame and frame[column].notna().any():
            daily = frame.groupby(days)[column].mean().dropna()
            cut = daily.quantile(0.9)
            masks[name] = days.isin(daily[daily >= cut].index)
    top = frame["actual"].quantile(0.95)
    masks["top_5pct_price_hours"] = (frame["actual"] >= top).to_numpy()
    return masks


def evaluate_intervals(pred: pd.DataFrame, strict_only: bool = True) -> dict:
    frame = pred[pred["strict"]] if strict_only and "strict" in pred else pred
    frame = frame.dropna(subset=["actual"])
    methods = {"model": ("lo", "q50", "hi"), "model_raw": ("q10", "q50", "q90"), "bench_point": ("point_lo", "point_mid", "point_hi"),
               "bench_d1": ("naive_day_lo", "naive_day_mid", "naive_day_hi")}
    out = {"first_day": str(frame["delivery_day"].min().date()), "last_day": str(frame["delivery_day"].max().date()), "by_slice": {}}
    masks = slice_masks(frame)
    for name, mask in masks.items():
        part = frame[mask]
        out["by_slice"][name] = {m: interval_metrics(part, *cols) for m, cols in methods.items()}
    out["overall"] = out["by_slice"]["all"]
    overall = out["overall"]
    out["accepted"] = bool(overall["model"]["pinball_mean"] < overall["bench_point"]["pinball_mean"]
                           and overall["model"]["pinball_mean"] < overall["bench_d1"]["pinball_mean"]
                           and COVERAGE_RANGE[0] <= overall["model"]["coverage_10_90"] <= COVERAGE_RANGE[1])
    return out


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()), "mean_forecast": round(float(p[mask].mean()), 4) if mask.any() else np.nan,
                     "observed_frequency": round(float(y[mask].mean()), 4) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def evaluate_events(pred: pd.DataFrame, strict_only: bool = True) -> dict:
    frame = pred[pred["strict"]] if strict_only and "strict" in pred else pred
    out = {}
    for event in EVENTS:
        f = frame.dropna(subset=[f"is_{event}", f"p_{event}", f"clim_{event}", f"last7_{event}"])
        y = f[f"is_{event}"].to_numpy(dtype=float)
        p = f[f"p_{event}"].to_numpy(dtype=float)
        result = {"hours": int(len(f)), "events": int(y.sum()), "base_rate": round(float(y.mean()), 4) if len(y) else None}
        if len(y) == 0:
            out[event] = result
            continue
        b_model = brier(y, p)
        scores = {"model": b_model}
        for bench in ("clim", "last7"):
            b = brier(y, f[f"{bench}_{event}"].to_numpy(dtype=float))
            scores[bench] = b
            result[f"bss_vs_{bench}"] = round(1 - b_model / b, 4) if b > 0 else None
        result["brier"] = {k: round(v, 5) for k, v in scores.items()}
        result["log_loss"] = {"model": round(log_loss(y, p), 4), "clim": round(log_loss(y, f[f"clim_{event}"].to_numpy(dtype=float)), 4),
                              "last7": round(log_loss(y, f[f"last7_{event}"].to_numpy(dtype=float)), 4)}
        result["reliability"] = reliability_table(y, p)
        result["accepted"] = bool((result.get("bss_vs_clim") or 0) > 0 and (result.get("bss_vs_last7") or 0) > 0)
        out[event] = result
    return out


def prepare(pred: pd.DataFrame) -> pd.DataFrame:
    """Conformal interval, benchmark intervals and event benchmarks, all from past rows only."""
    out = conformal_interval(pred)
    out = trailing_error_quantiles(out, "point")
    out = trailing_error_quantiles(out, "naive_day")
    for event in EVENTS:
        out = event_benchmarks(out, event)
    return out


# ---------------------------------------------------------------------------- production: model files, live prediction, live calibration

import json  # noqa: E402
from pathlib import Path  # noqa: E402

PROB_PARTS = ("q10", "q50", "q90", "negative", "spike")
REFIT_PROB_PARAMS = {"n_estimators": 300, "num_leaves": 31}  # the same small models as the published point model
PROB_COLUMNS = ["q10", "q50", "q90", "lo", "hi", "p_negative", "p_spike", "spike_threshold", "conformal_margin"]


def prob_paths(model_dir, feature_set: str) -> dict:
    out = {part: Path(model_dir) / f"{feature_set}_{part}.txt" for part in PROB_PARTS}
    out["meta"] = Path(model_dir) / f"{feature_set}_probabilistic.json"
    return out


def fit_probabilistic(table: FeatureTable, feature_set: str, out_dir, holdout_days: int = 30, params: dict | None = None, now=None,
                      log=print) -> dict:
    """Fit the three quantile models and the two classifiers on every known row, write them next to the point model, with metadata.

    The initial conformal margin comes from a holdout fit: models fitted on the rows up to `holdout_days` before the last known
    day, scored on the days after it. The live margin (live_margin) replaces it once enough settled days exist in the store.
    """
    params = {**REFIT_PROB_PARAMS, **(params or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    days = table.info["delivery_day"]
    X = table.X.copy()
    X["spike_threshold"] = spike_thresholds(table.y, days)
    labels = event_labels(table.y, X["spike_threshold"])
    known = table.y.notna()
    last_day = days[known].max()
    cut = last_day - pd.Timedelta(days=holdout_days)
    train, test = known & (days <= cut), known & (days > cut)
    margin, holdout = 0.0, {}
    if test.sum() >= 24 * 10:
        q = {}
        for alpha, name in zip(QUANTILES, ("q10", "q50", "q90")):
            m = quantile_model(alpha, params)
            m.fit(X.loc[train].drop(columns=["spike_threshold"]), table.y[train])
            q[name] = np.asarray(m.predict(X.loc[test].drop(columns=["spike_threshold"])), dtype=float)
        frame = sort_quantiles(pd.DataFrame(q, index=X.index[test]))
        y = table.y[test].to_numpy()
        scores = np.maximum(frame["q10"].to_numpy() - y, y - frame["q90"].to_numpy())
        n = len(scores)
        rank = min(int(np.ceil((n + 1) * (1 - ALPHA))), n)
        margin = float(np.sort(scores)[rank - 1])
        covered_raw = float(np.mean((y >= frame["q10"]) & (y <= frame["q90"])))
        covered = float(np.mean((y >= frame["q10"] - margin) & (y <= frame["q90"] + margin)))
        holdout = {"from": str((cut + pd.Timedelta(days=1)).date()), "to": str(last_day.date()), "hours": int(n), "raw_coverage_10_90": round(covered_raw, 4),
                   "coverage_10_90_with_margin": round(covered, 4), "pinball_mean": round(float(np.mean([pinball(y, frame[c].to_numpy(), a) for c, a in zip(("q10", "q50", "q90"), QUANTILES)])), 3)}
        log(f"Probabilistic holdout {holdout['from']} to {holdout['to']}: raw coverage {100 * covered_raw:.1f}%, margin {margin:.2f}, "
            f"coverage with margin {100 * covered:.1f}%")
    paths = prob_paths(out, feature_set)
    for alpha, name in zip(QUANTILES, ("q10", "q50", "q90")):
        m = quantile_model(alpha, params)
        m.fit(X.loc[known].drop(columns=["spike_threshold"]), table.y[known])
        m.booster_.save_model(str(paths[name]))
    events = {}
    for event in EVENTS:
        keep = known & labels[event].notna()
        y_event = labels.loc[keep, event].astype(int)
        events[event] = {"events": int(y_event.sum()), "hours": int(len(y_event)), "base_rate": round(float(y_event.mean()), 4)}
        m = event_model({**params, "objective": "binary"})
        m.fit(X[keep], y_event)
        m.booster_.save_model(str(paths[event]))
    fit_time = pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")
    meta = {"feature_set": feature_set, "parts": {k: v.name for k, v in paths.items() if k != "meta"}, "fitted_at_utc": fit_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_from": str(days[known].min().date()), "train_to": str(last_day.date()), "train_hours": int(known.sum()),
            "features": list(table.X.columns), "event_features": list(X.columns), "quantiles": list(QUANTILES), "alpha": ALPHA,
            "conformal_margin": round(margin, 3), "conformal_window_days": CONFORMAL_WINDOW_DAYS, "spike_percentile": SPIKE_PERCENTILE,
            "trailing_days": TRAILING_DAYS, "params": {k: v for k, v in params.items() if k != "verbose"}, "holdout": holdout, "events": events,
            "note": "Quantile forecasts with a conformal 10-90 interval and event probabilities (negative price, spike above the trailing-year "
                    "95th percentile); see docs/experiments.md. Without these files the forecast carries no band."}
    paths["meta"].write_text(json.dumps(meta, indent=2))
    return meta


def load_probabilistic(model_dir, feature_set: str):
    """The boosters and metadata, or None when any file is missing (the forecast then carries no band)."""
    from lightgbm import Booster

    paths = prob_paths(model_dir, feature_set)
    if not all(p.exists() for p in paths.values()):
        return None
    models = {part: Booster(model_str=paths[part].read_text(encoding="utf-8").replace("\r\n", "\n")) for part in PROB_PARTS}
    models["meta"] = json.loads(paths["meta"].read_text())
    return models


def predict_probabilistic(models: dict, X_day: pd.DataFrame, spike_threshold: float, margin: float | None = None) -> pd.DataFrame:
    """Quantiles (sorted), the conformal interval with `margin` (default the metadata's), and the two event probabilities, per hour."""
    margin = float(models["meta"]["conformal_margin"]) if margin is None else float(margin)
    features = models["meta"]["features"]
    event_features = models["meta"]["event_features"]
    X = X_day.copy()
    X["spike_threshold"] = spike_threshold
    out = pd.DataFrame(index=X_day.index)
    for part in ("q10", "q50", "q90"):
        out[part] = np.asarray(models[part].predict(X[features]), dtype=float)
    out = sort_quantiles(out)
    out["lo"] = out["q10"] - margin
    out["hi"] = out["q90"] + margin
    for event in EVENTS:
        out[f"p_{event}"] = np.clip(np.asarray(models[event].predict(X[event_features]), dtype=float), 0.0, 1.0)
    out["spike_threshold"] = spike_threshold
    out["conformal_margin"] = margin
    return out.round(4)


def spike_threshold_for_day(hourly: pd.DataFrame, day, trailing_days: int = TRAILING_DAYS, percentile: float = SPIKE_PERCENTILE) -> float:
    """The trailing-year 95th percentile of the known hourly prices before `day`, from the inputs table."""
    from .timing import delivery_days

    prices = hourly["price_eur_mwh"].dropna()
    days = pd.Series(delivery_days(prices.index), index=prices.index)
    day = pd.Timestamp(day).normalize()
    window = prices[(days >= day - pd.Timedelta(days=trailing_days)) & (days < day)]
    return float(np.quantile(window.to_numpy(), percentile)) if len(window) >= 24 * 30 else float("nan")


def live_margin(settled: pd.DataFrame, min_days: int = 20, alpha: float = ALPHA) -> float | None:
    """The conformal margin from stored raw quantiles and actuals of the last settled days (columns q10, q90, actual, delivery_day)."""
    frame = settled.dropna(subset=["q10", "q90", "actual"])
    if frame.empty or frame["delivery_day"].nunique() < min_days:
        return None
    scores = np.maximum(frame["q10"] - frame["actual"], frame["actual"] - frame["q90"]).to_numpy()
    n = len(scores)
    rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[rank - 1])


def live_calibration(settled: pd.DataFrame) -> dict:
    """Aggregates only: coverage and width of the stored 10-90 band, Brier scores and reliability of the event probabilities on settled days."""
    frame = settled.dropna(subset=["actual"])
    out = {"days": int(frame["delivery_day"].nunique()) if len(frame) else 0, "hours": int(len(frame))}
    if frame.empty:
        return out
    band = frame.dropna(subset=["lo", "hi"])
    if len(band):
        y = band["actual"].to_numpy()
        out["coverage_10_90"] = round(float(np.mean((y >= band["lo"]) & (y <= band["hi"]))), 4)
        out["mean_width"] = round(float((band["hi"] - band["lo"]).mean()), 2)
        out["pinball_mean"] = round(float(np.mean([pinball(y, band[c].to_numpy(), a) for c, a in zip(("q10", "q50", "q90"), QUANTILES)])), 3)
    for event, label in (("negative", frame["actual"] < 0), ("spike", frame["actual"] > frame["spike_threshold"])):
        part = frame.dropna(subset=[f"p_{event}"])
        if part.empty:
            continue
        yv = label.loc[part.index].to_numpy(dtype=float)
        pv = part[f"p_{event}"].to_numpy(dtype=float)
        out[event] = {"events": int(yv.sum()), "hours": int(len(yv)), "brier": round(brier(yv, pv), 5),
                      "brier_base_rate": round(brier(yv, np.full(len(yv), yv.mean())), 5) if len(yv) else None,
                      "log_loss": round(log_loss(yv, pv), 4), "reliability": reliability_table(yv, pv).to_dict(orient="records")}
    return out


def backtest_record(ready: pd.DataFrame, label: str) -> dict:
    """The aggregates of a backtest, for published/probabilistic_backtest.json (no hourly data)."""
    intervals = evaluate_intervals(ready)
    events = evaluate_events(ready)
    return {"label": label, "first_day": intervals["first_day"], "last_day": intervals["last_day"], "intervals_accepted": intervals["accepted"],
            "intervals": {s: {m: v for m, v in part.items()} for s, part in intervals["by_slice"].items()},
            "events": {e: {k: (v.to_dict(orient="records") if hasattr(v, "to_dict") else v) for k, v in r.items()} for e, r in events.items()},
            "written_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
