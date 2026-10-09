"""The public dataset under published/, and the round trip that lets GitHub Actions keep it current.

GitHub Actions has no persistent disk, so the published folder is both the
output and the state: a workflow run imports it into a fresh SQLite store,
runs morning-run or settle, and exports again. The files are plain CSV so
git stores small deltas when rows are appended.

Files written by `export_published`:

    forecasts.csv      honest forecast versions of the last `days` days: delivery day,
                       issue time, hour, forecast, the same-hour-previous-day baseline and
                       `premarket` (1 when issued before the 11:15 Paris market window of D-1)
    actuals.csv        actual day-ahead prices for the same window
    scores.csv         daily MAE and RMSE of each version against actuals and the naive baseline,
                       with the same `premarket` flag
    tomorrow.csv       the headline honest forecast for the next delivery day, hourly: the last
                       version issued before the market window, else the latest
    error_band.csv     backtest error percentiles by hour (the dashboard's shaded band)
    model/honest.txt   the LightGBM model refitted monthly by the workflow (see refit.py)
    model/honest.json  its training period, fit date and holdout metrics
    status.json        when the dataset was written, the last run, the attributions and, under
                       "market", aggregates of the forecast against EEX traded prices (market.py):
                       days scored, hit rate, mean P&L per MWh, model and market error. Never a
                       traded price, and nothing below MIN_PUBLIC_DAYS scored days.

Only the honest feature set is published, and at most `days` days of actual
prices. The inputs history, the data status, the timing log and the traded
prices stay local.
"""

import json
from pathlib import Path

import pandas as pd

from .market import DEFAULT_PATH as MARKET_PATH
from .market import evaluate as evaluate_market
from .market import headline_version, load_market, premarket_flag, public_summary
from .store import Store, iso, utc_now

PUBLIC_FEATURE_SET = "honest"
DEFAULT_DIR = Path("published")

ATTRIBUTIONS = [
    "Day-ahead prices, load forecast and wind and solar forecasts: ENTSO-E Transparency Platform, "
    "https://transparency.entsoe.eu, reused with the source cited as required by its terms.",
    "French load history used in the thermosensitivity work: RTE eCO2mix via ODRE, "
    "https://opendata.reseaux-energies.fr, Licence Ouverte v2.0 (Etalab).",
    "Weather forecasts: Open-Meteo.com, https://open-meteo.com, CC BY 4.0.",
]


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

    versions = pd.read_sql_query(
        "SELECT forecast_id, delivery_day, issued_at_utc, kind, train_hours FROM forecasts"
        " WHERE feature_set = ? AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(PUBLIC_FEATURE_SET, cutoff),
    )
    values = pd.read_sql_query(
        "SELECT v.forecast_id, v.timestamp_utc, v.hour, v.forecast, v.naive_day FROM forecast_values v"
        " JOIN forecasts f ON f.forecast_id = v.forecast_id WHERE f.feature_set = ? AND f.delivery_day >= ?",
        store.conn, params=(PUBLIC_FEATURE_SET, cutoff),
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
        "SELECT delivery_day, issued_at_utc, hours, mae, rmse, naive_mae, naive_rmse, mae_below_baseline, settled_at_utc FROM scores"
        " WHERE feature_set = ? AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(PUBLIC_FEATURE_SET, cutoff),
    )
    scores["premarket"] = premarket_flag(scores["delivery_day"], scores["issued_at_utc"]).astype(int) if len(scores) else []
    scores.to_csv(out / "scores.csv", index=False)

    tomorrow_day = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = forecasts[forecasts["delivery_day"] >= today.strftime("%Y-%m-%d")]
    if not tomorrow.empty:
        latest_day = tomorrow["delivery_day"].max()
        day_versions = versions[versions["delivery_day"] == latest_day]
        headline = headline_version(day_versions).iloc[0]
        latest = tomorrow[(tomorrow["delivery_day"] == latest_day) & (tomorrow["issued_at_utc"] == headline["issued_at_utc"])]
        latest.to_csv(out / "tomorrow.csv", index=False)
    else:
        pd.DataFrame(columns=forecasts.columns).to_csv(out / "tomorrow.csv", index=False)

    band = pd.read_sql_query(
        "SELECT feature_set, hour, n, mae, p10, p25, p50, p75, p90 FROM error_band WHERE feature_set = ? ORDER BY hour",
        store.conn, params=(PUBLIC_FEATURE_SET,),
    )
    band.to_csv(out / "error_band.csv", index=False)

    model_meta = {}
    meta_path = out / "model" / f"{PUBLIC_FEATURE_SET}.json"
    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text())

    status = {
        "written_at_utc": iso(utc_now(now)),
        "feature_set": PUBLIC_FEATURE_SET,
        "window_days": days,
        "from_day": cutoff,
        "tomorrow": tomorrow_day,
        "forecast_versions": int(len(versions)),
        "actual_days": int(actuals["delivery_day"].nunique()) if len(actuals) else 0,
        "scored_versions": int(len(scores)),
        "model": {k: model_meta.get(k) for k in ("fitted_at_utc", "train_from", "train_to", "train_hours", "holdout")} if model_meta else {},
        "last_run": last_run or {},
        "market": market_status(store, market_path),
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
    summary = {"forecast_versions": 0, "actual_rows": 0, "scores": 0, "error_band_rows": 0}
    forecasts_path = src / "forecasts.csv"
    if forecasts_path.exists() and forecasts_path.stat().st_size > 0:
        forecasts = pd.read_csv(forecasts_path)
        if not forecasts.empty:
            forecasts["timestamp_utc"] = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
            keys = ["delivery_day", "issued_at_utc", "kind", "train_hours"]
            for (day, issued, kind, train_hours), group in forecasts.groupby(keys, sort=True):
                if not store.forecast_versions(day, PUBLIC_FEATURE_SET).query("issued_at_utc == @issued").empty:
                    continue
                curve = group.set_index("timestamp_utc")[["hour", "forecast", "naive_day"]].sort_index()
                store.save_forecast(None, day, PUBLIC_FEATURE_SET, pd.Timestamp(issued), "gbm", str(kind), int(train_hours), True, curve)
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
        versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc FROM forecasts WHERE feature_set = ?",
                                     store.conn, params=(PUBLIC_FEATURE_SET,))
        merged = scores.merge(versions, on=["delivery_day", "issued_at_utc"], how="inner")
        for _, r in merged.iterrows():
            store.save_score(int(r["forecast_id"]), r["delivery_day"], PUBLIC_FEATURE_SET, r["issued_at_utc"], {
                "hours": r["hours"], "mae": r["mae"], "rmse": r["rmse"], "naive_mae": r["naive_mae"], "naive_rmse": r["naive_rmse"],
                "mae_below_baseline": None if pd.isna(r["mae_below_baseline"]) else bool(r["mae_below_baseline"]),
            }, now=pd.Timestamp(r["settled_at_utc"]) if isinstance(r["settled_at_utc"], str) else None)
            summary["scores"] += 1
    band_path = src / "error_band.csv"
    if band_path.exists() and band_path.stat().st_size > 0:
        band = pd.read_csv(band_path)
        if not band.empty:
            store.save_error_band(band, source=str(band_path))
            summary["error_band_rows"] = int(len(band))
    return summary

