"""Monthly refit: fetch the full history, fit the honest model, save the model file and its metadata.

The GitHub Actions workflow runs this on the first weekday of each month. It
needs the ENTSO-E token (full history fetch) and writes two files under
published/model/:

    honest.txt    the LightGBM model in its native text format
    honest.json   training period, fit date, rows, feature list, a recent check and the frozen-holdout result from the log
    wind_proxy.json   weights of the wind generation proxy, fitted on the trailing year
    solar_proxy.json  weights of the solar generation proxy, fitted on the trailing months

The daily morning runs then predict with that file and only fetch the ten
days of inputs around the next delivery day; no history table has to live in
the repository. Metadata metrics come from a fit on all but the last
`holdout_days` days, scored on those days against the same-hour-D-1
benchmark; the saved model is then refitted on everything.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, build_features
from .inputs import fetch_inputs, load_inputs
from . import solar_proxy, wind_proxy
from .gen_proxy import latest_weights, save_weights
from .models import GBM_PARAMS, fit_predict, make_model, save_model

PROXIES = ((wind_proxy.WIND, wind_proxy.actual_wind), (solar_proxy.SOLAR, solar_proxy.actual_solar))

DEFAULT_MODEL_DIR = Path("published/model")
# Overrides applied to GBM_PARAMS for the published model. 300 trees with 31 leaves gave a
# holdout MAE within 0.2 EUR/MWh of the backtest settings (800 trees, 63 leaves) on September
# 2026 at a fifth of the file size (0.85 MB against 4.45 MB), which matters for a monthly commit.
REFIT_PARAMS: dict = {"n_estimators": 300, "num_leaves": 31}
# The frozen holdout of docs/experiments.md (2026-07-01 onward, evaluated once per accepted change, strict rows, monthly walk-forward with the
# backtest settings). Copied here by hand so that the model metadata can carry it; the recent check below is a different, weaker thing.
FROZEN_HOLDOUT = {
    "honest_v2": {"window": "2026-07-01 to 2026-10-10", "strict_hours": 2448, "mae": 25.81, "naive_mae": 29.37, "crash_2026_10_01_to_10_10_mae": 36.00,
                  "source": "docs/experiments.md, holdout row of 2026-10-10", "method": "monthly walk-forward, backtest settings, not this file"},
    "honest": {"window": "2026-07-01 to 2026-10-10", "strict_hours": 2448, "mae": 26.23, "naive_mae": 29.37, "crash_2026_10_01_to_10_10_mae": 33.76,
               "source": "docs/experiments.md, holdout row of 2026-10-10", "method": "monthly walk-forward, backtest settings, not this file"},
}


def refit_model(hourly: pd.DataFrame, out_dir=DEFAULT_MODEL_DIR, feature_set: str = "honest", holdout_days: int = 30,
                now=None, log=print, params: dict | None = None, save_proxies: bool = True) -> dict:
    """Fit on `hourly` (the inputs table) and write the model and metadata. Returns the metadata.

    `params` overrides GBM_PARAMS for the published model (for example fewer
    trees to keep the file small); the metadata records what was used.
    """
    params = {**GBM_PARAMS, **(REFIT_PARAMS if params is None else params)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = build_features(hourly, feature_set)
    known = table.y.notna()
    days = table.info["delivery_day"]
    last_day = days[known].max()
    cut = last_day - pd.Timedelta(days=holdout_days)
    train, test = known & (days <= cut), known & (days > cut)
    metrics = {}
    if test.sum() >= 24:
        log(f"Recent check: fit on {int(train.sum()):,} hours, scoring {int(test.sum()):,} hours after {cut.date()} (these days join the final fit) ...")
        pred = fit_predict("gbm", table.X[train], table.y[train], table.X[test], params=params)
        actual = table.y[test].to_numpy()
        naive = table.X.loc[test, "price_lag1"].to_numpy()
        ok = ~np.isnan(naive)
        metrics = {
            "from": str((cut + pd.Timedelta(days=1)).date()),
            "to": str(last_day.date()),
            "hours": int(test.sum()),
            "excluded_from_stored_fit": False,
            "note": "a check on the last days before the fit, scored by a model fitted without them; the stored file was then fitted on all "
                    "rows including these days, so this is not an out-of-sample result for the stored file and not the frozen holdout of "
                    "docs/experiments.md",
            "mae": round(float(np.mean(np.abs(pred - actual))), 2),
            "rmse": round(float(np.sqrt(np.mean((pred - actual) ** 2))), 2),
            "naive_mae": round(float(np.mean(np.abs(naive[ok] - actual[ok]))), 2),
            "naive_rmse": round(float(np.sqrt(np.mean((naive[ok] - actual[ok]) ** 2))), 2),
        }
    log(f"Final fit on {int(known.sum()):,} hours ...")
    model = make_model("gbm", params)
    model.fit(table.X[known], table.y[known])
    model_path = save_model(model, out / f"{feature_set}.txt")
    fit_time = pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")
    meta = {
        "feature_set": feature_set,
        "model": "lightgbm",
        "model_file": model_path.name,
        "fitted_at_utc": fit_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_from": str(days[known].min().date()),
        "train_to": str(last_day.date()),
        "train_hours": int(known.sum()),
        "features": list(FEATURES[feature_set]),
        "params": {k: v for k, v in params.items() if k != "verbose"},
        "recent_check": metrics,
        "frozen_holdout": FROZEN_HOLDOUT.get(feature_set),
        "note": f"Feature set {feature_set}: every input is published before 12:00 Paris on the day before delivery"
                + (" and before the pre-market issue time, 10:05 Paris on D-1." if feature_set != "honest" else "."),
    }
    for spec, actual_of in (PROXIES if save_proxies else ()):
        weights = latest_weights(hourly, actual_of(hourly), spec)
        name = f"{spec.name}_proxy"
        if weights is not None:
            save_weights(weights, out / f"{name}.json")
            meta[name] = {"file": f"{name}.json", **weights["fit"]}
            log(f"{spec.name.capitalize()} proxy weights refitted on {weights['fit']['hours']:,} hours to {weights['fit']['to']}: "
                f"capacity {weights['fit']['capacity_mw']:,.0f} MW, MAE {weights['fit']['mae_mw']:,.0f} MW, R2 {weights['fit']['r2']}")
        else:
            log(f"{spec.name.capitalize()} proxy weights not refitted: no point forecasts with actual generation in the inputs table.")
    (out / f"{feature_set}.json").write_text(json.dumps(meta, indent=2))
    return meta


def refit_sets(hourly: pd.DataFrame, out_dir=DEFAULT_MODEL_DIR, feature_sets=("honest",), log=print, probabilistic: bool = True) -> dict:
    """One model file per feature set; the proxy weights, and the probabilistic models, are written with the first set only."""
    from .probabilistic import fit_probabilistic

    metas = {}
    for i, feature_set in enumerate(feature_sets):
        metas[feature_set] = refit_model(hourly, out_dir=out_dir, feature_set=feature_set, log=log, save_proxies=(i == 0))
        if i == 0 and probabilistic:
            log(f"Fitting the quantile and event models for {feature_set} ...")
            metas[feature_set]["probabilistic"] = fit_probabilistic(build_features(hourly, feature_set), feature_set, out_dir, log=log)
    return metas


def fetch_and_refit(start: str, end: str, out_dir=DEFAULT_MODEL_DIR, cache_dir=Path("data/cache"), csv_dir=Path("data/csv"),
                    inputs_path=Path("data/forecast/inputs.csv"), log=print, feature_sets=("honest",), probabilistic: bool = True) -> dict:
    """Fetch the full inputs history for [start, end) (ENTSO-E key needed), save it locally, refit. Returns {feature_set: metadata}."""
    hourly, sources, comparison = fetch_inputs(start, end, cache_dir=cache_dir, csv_dir=csv_dir, log=log)
    inputs_path = Path(inputs_path)
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(inputs_path, index_label="timestamp_utc")
    return refit_sets(hourly, out_dir=out_dir, feature_sets=feature_sets, log=log, probabilistic=probabilistic)


def refit_from_file(inputs_path=Path("data/forecast/inputs.csv"), out_dir=DEFAULT_MODEL_DIR, log=print, feature_sets=("honest",),
                    probabilistic: bool = True) -> dict:
    return refit_sets(load_inputs(inputs_path), out_dir=out_dir, feature_sets=feature_sets, log=log, probabilistic=probabilistic)
