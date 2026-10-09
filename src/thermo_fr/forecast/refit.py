"""Monthly refit: fetch the full history, fit the honest model, save the model file and its metadata.

The GitHub Actions workflow runs this on the first weekday of each month. It
needs the ENTSO-E token (full history fetch) and writes two files under
published/model/:

    honest.txt    the LightGBM model in its native text format
    honest.json   training period, fit date, rows, feature list, holdout metrics
    wind_proxy.json  weights of the wind generation proxy, fitted on the trailing year

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
from .models import GBM_PARAMS, fit_predict, make_model, save_model
from .wind_proxy import latest_weights, save_weights

DEFAULT_MODEL_DIR = Path("published/model")
# Overrides applied to GBM_PARAMS for the published model. 300 trees with 31 leaves gave a
# holdout MAE within 0.2 EUR/MWh of the backtest settings (800 trees, 63 leaves) on September
# 2026 at a fifth of the file size (0.85 MB against 4.45 MB), which matters for a monthly commit.
REFIT_PARAMS: dict = {"n_estimators": 300, "num_leaves": 31}


def refit_model(hourly: pd.DataFrame, out_dir=DEFAULT_MODEL_DIR, feature_set: str = "honest", holdout_days: int = 30,
                now=None, log=print, params: dict | None = None) -> dict:
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
        log(f"Holdout fit on {int(train.sum()):,} hours, scoring {int(test.sum()):,} hours after {cut.date()} ...")
        pred = fit_predict("gbm", table.X[train], table.y[train], table.X[test], params=params)
        actual = table.y[test].to_numpy()
        naive = table.X.loc[test, "price_lag1"].to_numpy()
        ok = ~np.isnan(naive)
        metrics = {
            "holdout_from": str((cut + pd.Timedelta(days=1)).date()),
            "holdout_to": str(last_day.date()),
            "holdout_hours": int(test.sum()),
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
        "holdout": metrics,
        "note": "Honest feature set: every input is published before 12:00 Paris on the day before delivery.",
    }
    actual_wind = hourly[[c for c in ("wind_onshore_mw", "wind_offshore_mw") if c in hourly]].sum(axis=1, min_count=1)
    weights = latest_weights(hourly, actual_wind)
    if weights is not None:
        save_weights(weights, out / "wind_proxy.json")
        meta["wind_proxy"] = {"file": "wind_proxy.json", **weights["fit"]}
        log(f"Wind proxy weights refitted on {weights['fit']['hours']:,} hours to {weights['fit']['to']}: "
            f"capacity {weights['fit']['capacity_mw']:,.0f} MW, MAE {weights['fit']['mae_mw']:,.0f} MW, R2 {weights['fit']['r2']}")
    else:
        log("Wind proxy weights not refitted: no point forecasts with actual generation in the inputs table.")
    (out / f"{feature_set}.json").write_text(json.dumps(meta, indent=2))
    return meta


def fetch_and_refit(start: str, end: str, out_dir=DEFAULT_MODEL_DIR, cache_dir=Path("data/cache"), csv_dir=Path("data/csv"),
                    inputs_path=Path("data/forecast/inputs.csv"), log=print) -> dict:
    """Fetch the full inputs history for [start, end) (ENTSO-E key needed), save it locally, refit."""
    hourly, sources, comparison = fetch_inputs(start, end, cache_dir=cache_dir, csv_dir=csv_dir, log=log)
    inputs_path = Path(inputs_path)
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(inputs_path, index_label="timestamp_utc")
    return refit_model(hourly, out_dir=out_dir, log=log)


def refit_from_file(inputs_path=Path("data/forecast/inputs.csv"), out_dir=DEFAULT_MODEL_DIR, log=print) -> dict:
    return refit_model(load_inputs(inputs_path), out_dir=out_dir, log=log)
