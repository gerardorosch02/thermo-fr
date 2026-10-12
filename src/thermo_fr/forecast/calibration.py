"""Monotone (isotonic) recalibration of the negative-price probability, fitted walk-forward on out-of-sample predictions.

Pre-registered in docs/experiments.md (2026-10-12, branch prob-recalibration). For each calendar month the calibration is fitted only on
out-of-sample walk-forward predictions and outcomes of earlier months (expanding window, at least `min_history_months` months) and applied
to that month. Fitting rows are every out-of-sample row with an outcome, strict or not; scoring rows are the strict rows, raw and calibrated
on identical rows. The acceptance rule and the metrics (Brier, skill against both reference frequencies, mean absolute calibration error over
the 10-bin reliability table) are in `evaluate_recalibration`; the yearly diagnostic in `yearly_diagnostic`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .probabilistic import brier, low_end_bins, reliability_table

MIN_HISTORY_MONTHS = 12
PAIRS_DECIMALS = 4  # the out-of-sample pairs are kept aggregated by the raw probability rounded to this: (p, hours n, negative hours k)


def fit_isotonic(p, y) -> IsotonicRegression:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return model


def breakpoints(model: IsotonicRegression) -> dict:
    """The fitted step function as breakpoints, enough to apply it anywhere without scikit-learn."""
    return {"x": [round(float(v), 6) for v in model.X_thresholds_], "y": [round(float(v), 6) for v in model.y_thresholds_]}


def apply_breakpoints(bp: dict, p) -> np.ndarray:
    """Linear interpolation between the breakpoints, clipped at the ends (what IsotonicRegression.predict does)."""
    x, y = np.asarray(bp["x"], dtype=float), np.asarray(bp["y"], dtype=float)
    values = np.asarray(p, dtype=float)
    if len(x) == 0:
        return np.full(values.shape, np.nan)
    return np.clip(np.interp(values, x, y), 0.0, 1.0)


def walk_forward_calibration(raw: pd.DataFrame, months, column: str = "p_negative", target: str = "is_negative",
                             min_history_months: int = MIN_HISTORY_MONTHS, log=lambda *_: None) -> tuple[pd.Series, dict]:
    """Calibrated probabilities for the rows of `months` (pandas Periods), each month fitted on all earlier months' rows of `raw`.

    Returns the calibrated series (NaN where a month had less than `min_history_months` months of history) and, per month, the number of
    fitting rows and months.
    """
    frame = raw.dropna(subset=[column, target]).copy()
    frame["month"] = frame["delivery_day"].dt.to_period("M")
    out = pd.Series(np.nan, index=raw.index, name=f"{column}_cal")
    fits = {}
    for month in sorted(set(months)):
        history = frame[frame["month"] < month]
        used_months = int(history["month"].nunique())
        if used_months < min_history_months:
            fits[str(month)] = {"rows": int(len(history)), "months": used_months, "fitted": False}
            log(f"{month}: {used_months} months of history, fewer than {min_history_months}; not calibrated")
            continue
        model = fit_isotonic(history[column], history[target])
        target_rows = frame.index[frame["month"] == month]
        out.loc[target_rows] = model.predict(frame.loc[target_rows, column].to_numpy(dtype=float))
        fits[str(month)] = {"rows": int(len(history)), "months": used_months, "fitted": True, "events": int(history[target].sum())}
        log(f"{month}: fitted on {len(history):,} rows of {used_months} months ({int(history[target].sum())} negative hours)")
    return out, fits


def mace(y, p, bins: int = 10) -> dict:
    """Mean absolute calibration error over the non-empty bins of the 10-bin reliability table: unweighted (primary) and hours-weighted."""
    table = reliability_table(np.asarray(y, dtype=float), np.asarray(p, dtype=float), bins).dropna(subset=["mean_forecast"])
    gap = (table["mean_forecast"] - table["observed_frequency"]).abs()
    return {"unweighted": round(float(gap.mean()), 5), "weighted": round(float((gap * table["n"]).sum() / table["n"].sum()), 5), "bins_used": int(len(table))}


def evaluate_recalibration(frame: pd.DataFrame, raw_column: str = "p_negative", cal_column: str = "p_negative_cal", target: str = "is_negative",
                           strict_only: bool = True) -> dict:
    """Raw against calibrated on identical scoring rows: Brier, skill against both references, MACE; the pre-registered acceptance."""
    f = frame[frame["strict"]] if strict_only and "strict" in frame else frame
    f = f.dropna(subset=[raw_column, cal_column, target, f"clim_negative", "last7_negative"])
    y = f[target].to_numpy(dtype=float)
    result = {"hours": int(len(f)), "events": int(y.sum()), "first_day": str(f["delivery_day"].min().date()) if len(f) else None,
              "last_day": str(f["delivery_day"].max().date()) if len(f) else None}
    if len(f) == 0:
        return {**result, "accepted": False}
    refs = {name: brier(y, f[f"{name}_negative"].to_numpy(dtype=float)) for name in ("clim", "last7")}
    for label, column in (("raw", raw_column), ("calibrated", cal_column)):
        p = f[column].to_numpy(dtype=float)
        b = brier(y, p)
        result[label] = {"brier": round(b, 6), "bss_vs_clim": round(1 - b / refs["clim"], 4) if refs["clim"] > 0 else None,
                         "bss_vs_last7": round(1 - b / refs["last7"], 4) if refs["last7"] > 0 else None, "mace": mace(y, p),
                         "reliability": reliability_table(y, p).to_dict(orient="records"), "low_end_bins": low_end_bins(y, p),
                         "mean_forecast": round(float(p.mean()), 5)}
    result["reference_brier"] = {k: round(v, 6) for k, v in refs.items()}
    result["base_rate"] = round(float(y.mean()), 5)
    cal, raw = result["calibrated"], result["raw"]
    result["rule"] = ("calibrated Brier below raw, Brier skill above zero against climatology and against the last 7 days, unweighted MACE below raw;"
                      " strict rows of the window")
    result["accepted"] = bool(cal["brier"] < raw["brier"] and (cal["bss_vs_clim"] or 0) > 0 and (cal["bss_vs_last7"] or 0) > 0
                              and cal["mace"]["unweighted"] < raw["mace"]["unweighted"])
    return result


# ---------------------------------------------------------------------------- production: the pairs archive and the published calibration


def calibration_paths(model_dir, feature_set: str) -> dict:
    return {"meta": Path(model_dir) / f"{feature_set}_calibration.json", "pairs": Path(model_dir) / f"{feature_set}_calibration_pairs.csv"}


def aggregate_pairs(p, y, decimals: int = PAIRS_DECIMALS) -> pd.DataFrame:
    """(p, n, k): hours and negative hours per raw probability rounded to `decimals`; an isotonic fit on these weights equals one on the rows."""
    frame = pd.DataFrame({"p": np.round(np.asarray(p, dtype=float), decimals), "y": np.asarray(y, dtype=float)}).dropna()
    out = frame.groupby("p").agg(n=("y", "size"), k=("y", "sum")).reset_index()
    out["k"] = out["k"].astype(int)
    return out


def merge_pairs(*tables: pd.DataFrame) -> pd.DataFrame:
    frames = [t for t in tables if t is not None and len(t)]
    if not frames:
        return pd.DataFrame(columns=["p", "n", "k"])
    out = pd.concat(frames).groupby("p", as_index=False).agg(n=("n", "sum"), k=("k", "sum")).sort_values("p").reset_index(drop=True)
    out["n"], out["k"] = out["n"].astype(int), out["k"].astype(int)
    return out


def fit_from_pairs(pairs: pd.DataFrame) -> IsotonicRegression:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(pairs["p"].to_numpy(dtype=float), (pairs["k"] / pairs["n"]).to_numpy(dtype=float), sample_weight=pairs["n"].to_numpy(dtype=float))
    return model


def live_pairs_from_published(published_dir, feature_set: str, after_day: str | None, premarket_only: bool = True) -> tuple[pd.DataFrame, str | None]:
    """Settled live versions as pairs: each day's headline version from probabilistic.csv (its raw probability) against actuals.csv.

    Only delivery days after `after_day` are taken, so a day enters the archive once. Returns (pairs, last delivery day used).
    """
    from .market import headline_version

    src = Path(published_dir)
    prob_path, act_path = src / "probabilistic.csv", src / "actuals.csv"
    if not (prob_path.exists() and act_path.exists() and prob_path.stat().st_size and act_path.stat().st_size):
        return pd.DataFrame(columns=["p", "n", "k"]), None
    prob = pd.read_csv(prob_path)
    actuals = pd.read_csv(act_path)
    if prob.empty or actuals.empty:
        return pd.DataFrame(columns=["p", "n", "k"]), None
    prob = prob[prob["feature_set"] == feature_set]
    raw_column = "p_negative_raw" if "p_negative_raw" in prob else "p_negative"  # files written before the calibration carry the raw value as p_negative
    if after_day:
        prob = prob[prob["delivery_day"] > after_day]
    if prob.empty:
        return pd.DataFrame(columns=["p", "n", "k"]), None
    versions = prob[["delivery_day", "issued_at_utc"]].drop_duplicates()
    heads = headline_version(versions)
    if premarket_only:
        heads = heads[heads["premarket"]]
    rows = prob.merge(heads[["delivery_day", "issued_at_utc"]], on=["delivery_day", "issued_at_utc"], how="inner")
    rows = rows.merge(actuals[["timestamp_utc", "price"]], on="timestamp_utc", how="inner").dropna(subset=[raw_column, "price"])
    if rows.empty:
        return pd.DataFrame(columns=["p", "n", "k"]), None
    return aggregate_pairs(rows[raw_column], (rows["price"] < 0).astype(float)), str(rows["delivery_day"].max())


def build_calibration(model_dir, feature_set: str = "honest_v2", archives=(), published_dir=None, now=None, log=print) -> dict | None:
    """Fit the live calibration on every out-of-sample pair before now and write it next to the models.

    Pairs come from the existing pairs file (if any), the walk-forward archives given (raw parquet files with p_negative and is_negative;
    the laptop's one-off build), and the settled live headline versions in `published_dir` after the archive's last day (the monthly
    refit). Nothing is fitted on in-sample rows. Returns the metadata, or None when there are no pairs at all.
    """
    paths = calibration_paths(model_dir, feature_set)
    meta_old = json.loads(paths["meta"].read_text()) if paths["meta"].exists() else {}
    pairs = pd.read_csv(paths["pairs"]) if paths["pairs"].exists() and paths["pairs"].stat().st_size else pd.DataFrame(columns=["p", "n", "k"])
    last_day = meta_old.get("last_day")
    sources = list(meta_old.get("sources", []))
    for archive in archives:
        raw = pd.read_parquet(archive)
        raw = raw.dropna(subset=["p_negative", "is_negative"])
        pairs = merge_pairs(pairs, aggregate_pairs(raw["p_negative"], raw["is_negative"]))
        archive_last = str(raw["delivery_day"].max().date())
        last_day = max(last_day or "", archive_last)
        sources.append(f"walk-forward archive {Path(archive).name}: {len(raw):,} hours to {archive_last}")
        log(f"calibration pairs: {len(raw):,} out-of-sample hours from {Path(archive).name} (to {archive_last})")
    if published_dir is not None:
        live, live_last = live_pairs_from_published(published_dir, feature_set, last_day)
        if len(live):
            pairs = merge_pairs(pairs, live)
            sources.append(f"settled live pre-market versions {last_day} to {live_last}: {int(live['n'].sum()):,} hours")
            log(f"calibration pairs: {int(live['n'].sum()):,} settled live hours after {last_day} (to {live_last})")
            last_day = live_last
    if pairs.empty:
        log("calibration: no out-of-sample pairs; nothing written")
        return None
    model = fit_from_pairs(pairs)
    fitted = (pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"feature_set": feature_set, "event": "negative", "method": "isotonic regression (increasing, clipped to [0, 1]) of the outcome on the raw "
            "probability, fitted on out-of-sample pairs only: walk-forward backtest predictions and settled live pre-market versions; refitted "
            "monthly with the models; applied to the raw probability at issue time", "breakpoints": breakpoints(model), "fitted_at_utc": fitted,
            "pairs_hours": int(pairs["n"].sum()), "pairs_events": int(pairs["k"].sum()), "distinct_raw_values": int(len(pairs)), "last_day": last_day,
            "sources": sources, "pre_registration": "docs/experiments.md, 2026-10-12, branch prob-recalibration"}
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    pairs.to_csv(paths["pairs"], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2))
    log(f"calibration fitted on {meta['pairs_hours']:,} hours ({meta['pairs_events']} negative) through {last_day}: {len(meta['breakpoints']['x'])} breakpoints")
    return meta


def load_calibration(model_dir, feature_set: str = "honest_v2") -> dict | None:
    path = calibration_paths(model_dir, feature_set)["meta"]
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
    except ValueError:
        return None
    return meta if meta.get("breakpoints", {}).get("x") else None


def apply_calibration(meta: dict, p) -> np.ndarray:
    return apply_breakpoints(meta["breakpoints"], p)


def public_calibration(meta: dict | None) -> dict:
    """What status.json carries about the live calibration (no breakpoints)."""
    if not meta:
        return {"applied": False, "note": "no calibration file: the published negative-price probability is the raw classifier output"}
    return {"applied": True, **{k: meta.get(k) for k in ("fitted_at_utc", "pairs_hours", "pairs_events", "last_day", "method", "pre_registration")}}


def yearly_diagnostic(prices: pd.Series, raw: pd.DataFrame | None, column: str = "p_negative", target: str = "is_negative") -> pd.DataFrame:
    """By calendar year (Paris time): share of negative hours in the auction prices, and the mean raw forecast where predictions exist."""
    local = prices.dropna()
    local.index = local.index.tz_convert("Europe/Paris")
    by_year = local.groupby(local.index.year).agg(hours="size", negative_hours=lambda s: int((s < 0).sum()))
    by_year["share_negative"] = (by_year["negative_hours"] / by_year["hours"]).round(5)
    out = by_year
    if raw is not None and len(raw):
        pred = raw.dropna(subset=[column, target]).copy()
        pred["year"] = pred["delivery_day"].dt.year
        agg = pred.groupby("year").agg(predicted_hours=(column, "size"), mean_raw_forecast=(column, "mean"), share_negative_predicted_rows=(target, "mean"))
        out = by_year.join(agg.round(5), how="left")
    return out.reset_index().rename(columns={"index": "year", "timestamp_utc": "year"})
