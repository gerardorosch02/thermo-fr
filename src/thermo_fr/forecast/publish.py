"""The public dataset under published/, and the round trip that lets GitHub Actions keep it current.

GitHub Actions has no persistent disk, so the published folder is both the
output and the state: a workflow run imports it into a fresh SQLite store,
runs morning-run or settle, and exports again. The files are plain CSV so
git stores small deltas when rows are appended.

Files written by `export_published`:

    forecasts.csv      forecast versions of the last `days` days (honest_v2, the default, and honest, the fallback, told apart by
                       the feature_set column; the model column names the file that produced each version): delivery day,
                       issue time, hour, forecast, the same-hour-previous-day baseline and
                       `premarket` (1 when issued before the 11:15 Paris market window of D-1)
    actuals.csv        actual day-ahead prices for the same window
    scores.csv         daily MAE and RMSE of each version against actuals and the naive baseline,
                       with the same `premarket` flag
    tomorrow.csv       the headline forecast of the default model for the next delivery day, hourly: the last
                       version issued before the market window, else the latest
    error_band.csv     backtest error percentiles by hour (the dashboard's shaded band until the quantile band is made the default)
    probabilistic.csv  per version and hour of the published sets: q10, q50, q90, the conformal 10-90 band (lo, hi), the probabilities of a
                       negative price and of a spike, and the spike threshold; tomorrow_probabilistic.csv the headline version's rows
    probabilistic_backtest.json  aggregates of the probabilistic backtest (docs/experiments.md), written by `thermo-fr prob-backtest`
    model/honest_v2.txt   the default LightGBM model, refitted monthly by the workflow (see refit.py)
    model/honest_v2.json  its training period, fit date, features and holdout metrics
    model/honest.txt      the fallback model (used when a v2 input is missing), refitted with it; honest.json its metadata
    status.json        when the dataset was written, the last run, the attributions and, under
                       "market", aggregates of the forecast against EEX traded prices (market.py):
                       days scored, hit rate, mean P&L per MWh, model and market error. Never a
                       traded price, and nothing below MIN_PUBLIC_DAYS scored days.

Only the honest_v2 and honest feature sets are published, and at most `days` days of actual
prices. The inputs history, the data status, the timing log and the traded
prices stay local.
"""

import json
from pathlib import Path

import pandas as pd

from .market import DEFAULT_PATH as MARKET_PATH
from .probabilistic import live_calibration
from .shape import live_shape_record
from .market import evaluate as evaluate_market
from .market import headline_version, load_market, premarket_flag, public_summary
from .store import Store, iso, utc_now

PUBLIC_FEATURE_SET = "honest_v2"  # the default model's set (published/model/honest_v2.txt); honest.txt is the fallback
PUBLIC_FEATURE_SETS = ("honest_v2", "honest")  # both are exported, with a feature_set column; the public app shows the default
FALLBACK_FEATURE_SET = "honest"
DEFAULT_DIR = Path("published")

ATTRIBUTIONS = [
    "Day-ahead prices, load forecast and wind and solar forecasts: ENTSO-E Transparency Platform, "
    "https://transparency.entsoe.eu, reused with the source cited as required by its terms.",
    "French load history used in the thermosensitivity work: RTE eCO2mix via ODRE, "
    "https://opendata.reseaux-energies.fr, Licence Ouverte v2.0 (Etalab).",
    "Weather forecasts: Open-Meteo.com, https://open-meteo.com, CC BY 4.0.",
]


SHAPE_BACKTEST_FILE = "shape_battery_backtest.json"  # written by `thermo-fr shape-backtest`, aggregates only


PROB_BACKTEST_FILE = "probabilistic_backtest.json"  # written by `thermo-fr prob-backtest`, aggregates only


def probabilistic_status(store: Store, out_dir) -> dict:
    """Live calibration of the published set's probabilistic forecasts (aggregates only) plus a pointer to the backtest aggregates."""
    out = Path(out_dir)
    live = live_calibration(store.probabilistic_settled(PUBLIC_FEATURE_SET))
    meta_path = out / "model" / f"{PUBLIC_FEATURE_SET}_probabilistic.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"live": live, "backtest_file": PROB_BACKTEST_FILE if (out / PROB_BACKTEST_FILE).exists() else None,
            "model": {k: meta.get(k) for k in ("fitted_at_utc", "train_from", "train_to", "conformal_margin", "holdout", "events")} if meta else {},
            "definitions": {"band": "10th to 90th percentile forecasts of each hour's price from quantile models on the same inputs, widened by a "
                                    "conformal margin from the last settled days so that about 80% of outcomes fall inside",
                            "negative": "probability that the hour's price is below zero",
                            "spike": "probability that the hour's price is above the 95th percentile of the trailing year's hourly prices known "
                                     "at the issue time"}}


def shape_status(store: Store, out_dir) -> dict:
    """Write shape_battery.json (backtest aggregates plus the live record of the published set) and return the live aggregates for status."""
    out = Path(out_dir)
    backtest_path = out / SHAPE_BACKTEST_FILE
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}
    live = live_shape_record(store, PUBLIC_FEATURE_SET)
    daily = live.pop("daily", None)
    payload = {"backtest": backtest, "live": {**live, "daily": daily.to_dict(orient="records") if daily is not None else []},
               "definitions": {"shape": "each hour's price minus the day's mean (base)",
                               "battery": "1 MW / 2 MWh, 88% round-trip, one cycle a day, a 2-hour charge block before a 2-hour discharge block "
                                          "chosen on the forecast before the gate, settled at the auction result; skipped when the forecast "
                                          "spread does not cover the efficiency loss",
                               "benchmarks": "d1: yesterday's shape; same_type: the shape of the most recent earlier day of the same type"}}
    (out / "shape_battery.json").write_text(json.dumps(payload, indent=2, default=str))
    return {k: v for k, v in live.items() if k in ("days", "shape", "battery")}


def market_status(store: Store, market_path=MARKET_PATH) -> dict:
    """The public aggregates of the forecast against traded prices (see market.public_summary)."""
    rows = load_market(market_path)
    if rows.empty:
        return {"scored_days": 0, "note": "No traded prices recorded yet."}
    return public_summary(evaluate_market(store, rows, feature_set=PUBLIC_FEATURE_SET))


def export_published(store: Store, out_dir=DEFAULT_DIR, days: int = 90, now=None, last_run: dict | None = None,
                     market_path=MARKET_PATH) -> dict:
    """Write the public files. Returns a summary with row counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = utc_now(now).tz_convert("Europe/Paris").normalize().tz_localize(None)
    cutoff = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    sets_sql = ", ".join("?" for _ in PUBLIC_FEATURE_SETS)
    versions = pd.read_sql_query(
        "SELECT forecast_id, delivery_day, issued_at_utc, kind, train_hours, feature_set, model FROM forecasts"
        f" WHERE feature_set IN ({sets_sql}) AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    values = pd.read_sql_query(
        "SELECT v.forecast_id, v.timestamp_utc, v.hour, v.forecast, v.naive_day FROM forecast_values v"
        f" JOIN forecasts f ON f.forecast_id = v.forecast_id WHERE f.feature_set IN ({sets_sql}) AND f.delivery_day >= ?",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    versions["premarket"] = premarket_flag(versions["delivery_day"], versions["issued_at_utc"]).astype(int) if len(versions) else []
    forecasts = versions.merge(values, on="forecast_id", how="inner").drop(columns=["forecast_id"])
    forecasts = forecasts.sort_values(["delivery_day", "issued_at_utc", "timestamp_utc"])
    forecasts.to_csv(out / "forecasts.csv", index=False)

    actuals = pd.read_sql_query(
        "SELECT delivery_day, timestamp_utc, hour, price FROM actuals WHERE delivery_day >= ? ORDER BY timestamp_utc",
        store.conn, params=(cutoff,),
    )
    actuals.to_csv(out / "actuals.csv", index=False)

    scores = pd.read_sql_query(
        "SELECT delivery_day, issued_at_utc, feature_set, hours, mae, rmse, naive_mae, naive_rmse, mae_below_baseline, settled_at_utc"
        f" FROM scores WHERE feature_set IN ({sets_sql}) AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    scores["premarket"] = premarket_flag(scores["delivery_day"], scores["issued_at_utc"]).astype(int) if len(scores) else []
    model_of = versions[["delivery_day", "issued_at_utc", "feature_set", "model"]].drop_duplicates()
    scores = scores.merge(model_of, on=["delivery_day", "issued_at_utc", "feature_set"], how="left") if len(scores) else scores.assign(model=[])
    scores.to_csv(out / "scores.csv", index=False)

    tomorrow_day = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = forecasts[forecasts["delivery_day"] >= today.strftime("%Y-%m-%d")]
    if not tomorrow.empty:
        latest_day = tomorrow["delivery_day"].max()
        day_versions = versions[versions["delivery_day"] == latest_day]
        preferred = day_versions[day_versions["feature_set"] == PUBLIC_FEATURE_SET]  # the default model's versions, else whatever exists
        day_versions = preferred if len(preferred) else day_versions
        headline = headline_version(day_versions).iloc[0]
        latest = tomorrow[(tomorrow["delivery_day"] == latest_day) & (tomorrow["issued_at_utc"] == headline["issued_at_utc"])
                          & (tomorrow["feature_set"] == headline["feature_set"])]
        latest.to_csv(out / "tomorrow.csv", index=False)
    else:
        pd.DataFrame(columns=forecasts.columns).to_csv(out / "tomorrow.csv", index=False)

    prob = pd.read_sql_query(
        "SELECT f.delivery_day, f.issued_at_utc, f.feature_set, p.timestamp_utc, p.q10, p.q50, p.q90, p.lo, p.hi, p.p_negative, p.p_spike,"
        " p.spike_threshold FROM forecast_prob p JOIN forecasts f ON f.forecast_id = p.forecast_id"
        f" WHERE f.feature_set IN ({sets_sql}) AND f.delivery_day >= ? ORDER BY f.delivery_day, f.issued_at_utc, p.timestamp_utc",
        store.conn, params=(*PUBLIC_FEATURE_SETS, cutoff),
    )
    prob.to_csv(out / "probabilistic.csv", index=False)
    if not tomorrow.empty and len(prob):
        head_prob = prob[(prob["delivery_day"] == latest_day) & (prob["issued_at_utc"] == headline["issued_at_utc"])
                         & (prob["feature_set"] == headline["feature_set"])]
        head_prob.to_csv(out / "tomorrow_probabilistic.csv", index=False)
    else:
        pd.DataFrame(columns=prob.columns).to_csv(out / "tomorrow_probabilistic.csv", index=False)

    band = pd.DataFrame()
    for band_set in (PUBLIC_FEATURE_SET, FALLBACK_FEATURE_SET):  # the default model's band, else the fallback model's
        band = pd.read_sql_query(
            "SELECT feature_set, hour, n, mae, p10, p25, p50, p75, p90 FROM error_band WHERE feature_set = ? ORDER BY hour",
            store.conn, params=(band_set,),
        )
        if len(band):
            break
    band.to_csv(out / "error_band.csv", index=False)

    fallback_meta = {}
    fallback_path = out / "model" / f"{FALLBACK_FEATURE_SET}.json"
    if fallback_path.exists():
        fallback_meta = json.loads(fallback_path.read_text())
    model_meta = {}
    meta_path = out / "model" / f"{PUBLIC_FEATURE_SET}.json"
    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text())

    status = {
        "written_at_utc": iso(utc_now(now)),
        "feature_set": PUBLIC_FEATURE_SET,
        "published_feature_sets": list(PUBLIC_FEATURE_SETS),
        "default_model_file": f"model/{PUBLIC_FEATURE_SET}.txt",
        "fallback_model_file": f"model/{FALLBACK_FEATURE_SET}.txt",
        "model_note": "Every forecast version in forecasts.csv and scores.csv carries the model that produced it in the model column: "
                      "gbm:<file> for a stored model, with the suffix :fallback when the default model's inputs were missing and the "
                      "fallback model was used, and gbm for a model fitted live.",
        "window_days": days,
        "from_day": cutoff,
        "tomorrow": tomorrow_day,
        "forecast_versions": int(len(versions)),
        "actual_days": int(actuals["delivery_day"].nunique()) if len(actuals) else 0,
        "scored_versions": int(len(scores)),
        "model": {k: model_meta.get(k) for k in ("feature_set", "model_file", "fitted_at_utc", "train_from", "train_to", "train_hours",
                                                 "holdout", "features")} if model_meta else {},
        "fallback_model": {k: fallback_meta.get(k) for k in ("feature_set", "model_file", "fitted_at_utc", "train_from", "train_to",
                                                             "train_hours", "holdout")} if fallback_meta else {},
        "last_run": last_run or {},
        "market": market_status(store, market_path),
        "shape_battery": shape_status(store, out),
        "probabilistic": probabilistic_status(store, out),
        "attributions": ATTRIBUTIONS,
        "note": "Forecasts use only information available at 12:00 Paris time on the day before delivery. "
                "Forecast error is measured against a naive same-hour-previous-day baseline (the spot auction result of the "
                "previous day); trading value is measured against EEX French day-ahead futures traded before the auction, "
                "as aggregates under \"market\" once enough days are recorded. Traded prices are not republished. "
                "Historical errors are shown as context, not as a probability forecast.",
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


def import_published(store: Store, in_dir=DEFAULT_DIR) -> dict:
    """Load the published files into a (usually empty) store."""
    src = Path(in_dir)
    summary = {"forecast_versions": 0, "actual_rows": 0, "scores": 0, "error_band_rows": 0, "probabilistic_versions": 0}
    forecasts_path = src / "forecasts.csv"
    if forecasts_path.exists() and forecasts_path.stat().st_size > 0:
        forecasts = pd.read_csv(forecasts_path)
        if not forecasts.empty:
            forecasts["timestamp_utc"] = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
            if "feature_set" not in forecasts:  # files written when only one set was published
                forecasts["feature_set"] = FALLBACK_FEATURE_SET
            if "model" not in forecasts:
                forecasts["model"] = "gbm"
            forecasts["model"] = forecasts["model"].fillna("gbm")
            keys = ["delivery_day", "issued_at_utc", "kind", "train_hours", "feature_set", "model"]
            for (day, issued, kind, train_hours, feature_set, model), group in forecasts.groupby(keys, sort=True):
                if not store.forecast_versions(day, feature_set).query("issued_at_utc == @issued").empty:
                    continue
                curve = group.set_index("timestamp_utc")[["hour", "forecast", "naive_day"]].sort_index()
                store.save_forecast(None, day, feature_set, pd.Timestamp(issued), str(model), str(kind), int(train_hours), True, curve)
                summary["forecast_versions"] += 1
    actuals_path = src / "actuals.csv"
    if actuals_path.exists() and actuals_path.stat().st_size > 0:
        actuals = pd.read_csv(actuals_path)
        if not actuals.empty:
            index = pd.to_datetime(actuals["timestamp_utc"], utc=True)
            prices = pd.Series(actuals["price"].to_numpy(), index=index)
            summary["actual_rows"] = store.save_actuals(prices, actuals["delivery_day"].reset_index(drop=True))
    scores_path = src / "scores.csv"
    if scores_path.exists() and scores_path.stat().st_size > 0:
        scores = pd.read_csv(scores_path).rename(columns={"model_won": "mae_below_baseline"})  # files written before the rename
        if "feature_set" not in scores:
            scores["feature_set"] = FALLBACK_FEATURE_SET
        versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc, feature_set FROM forecasts", store.conn)
        merged = scores.merge(versions, on=["delivery_day", "issued_at_utc", "feature_set"], how="inner")
        for _, r in merged.iterrows():
            store.save_score(int(r["forecast_id"]), r["delivery_day"], r["feature_set"], r["issued_at_utc"], {
                "hours": r["hours"], "mae": r["mae"], "rmse": r["rmse"], "naive_mae": r["naive_mae"], "naive_rmse": r["naive_rmse"],
                "mae_below_baseline": None if pd.isna(r["mae_below_baseline"]) else bool(r["mae_below_baseline"]),
            }, now=pd.Timestamp(r["settled_at_utc"]) if isinstance(r["settled_at_utc"], str) else None)
            summary["scores"] += 1
    prob_path = src / "probabilistic.csv"
    if prob_path.exists() and prob_path.stat().st_size > 0:
        prob = pd.read_csv(prob_path)
        if not prob.empty:
            versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc, feature_set FROM forecasts", store.conn)
            merged = prob.merge(versions, on=["delivery_day", "issued_at_utc", "feature_set"], how="inner")
            for fid, group in merged.groupby("forecast_id"):
                if not store.probabilistic_curve(int(fid)).empty:
                    continue
                frame = group.set_index(pd.to_datetime(group["timestamp_utc"], utc=True))[["q10", "q50", "q90", "lo", "hi", "p_negative", "p_spike", "spike_threshold"]]
                frame["conformal_margin"] = (frame["q10"] - frame["lo"]).round(4)
                store.save_probabilistic(int(fid), frame)
                summary["probabilistic_versions"] += 1
    band_path = src / "error_band.csv"
    if band_path.exists() and band_path.stat().st_size > 0:
        band = pd.read_csv(band_path)
        if not band.empty:
            store.save_error_band(band, source=str(band_path))
            summary["error_band_rows"] = int(len(band))
    return summary

