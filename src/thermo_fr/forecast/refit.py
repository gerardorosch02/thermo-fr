"""Monthly refit: fetch the full history, fit the honest model, save the model file and its metadata.

The GitHub Actions workflow runs this on the first weekday of each month. It
needs the ENTSO-E token (full history fetch) and writes two files under
published/model/:

    honest.txt    the LightGBM model in its native text format
    honest.json   training period, fit date, rows, feature list, holdout metrics

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

DEFAULT_MODEL_DIR = Path("published/model")


def refit_model(hourly: pd.DataFrame, out_dir=DEFAULT_MODEL_DIR, feature_set: str = "honest", holdout_days: int = 30,
                now=None, log=print) -> dict:
    """Fit on `hourly` (the inputs table) and write the model and metadata. Returns the metadata."""
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
        pred = fit_predict("gbm", table.X[train], table.y[train], table.X[test])
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
    model = make_model("gbm")
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
        "params": {k: v for k, v in GBM_PARAMS.items() if k != "verbose"},
        "holdout": metrics,
        "note": "Honest feature set: every input is published before 12:00 Paris on the day before delivery.",
    }
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
