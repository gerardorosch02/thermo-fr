"""The public dataset under published/, and the round trip that lets GitHub Actions keep it current.

GitHub Actions has no persistent disk, so the published folder is both the
output and the state: a workflow run imports it into a fresh SQLite store,
runs morning-run or settle, and exports again. The files are plain CSV so
git stores small deltas when rows are appended.

Files written by `export_published`:

    forecasts.csv      honest forecast versions of the last `days` days: delivery day,
                       issue time, hour, forecast and the same-hour-previous-day benchmark
    actuals.csv        actual day-ahead prices for the same window
    scores.csv         daily MAE and RMSE of each version against actuals and benchmark
    tomorrow.csv       the latest honest forecast for the next delivery day, hourly
    error_band.csv     backtest error percentiles by hour (the dashboard's shaded band)
    history.csv        the hourly inputs table the model is trained on (prices, load and
                       renewables forecasts, weather), needed to refit in the workflow
    status.json        when the dataset was written, the last run and the attributions

Only the honest feature set is published. The data status and timing log stay
in the local database.
"""

import json
from pathlib import Path

import pandas as pd

from .inputs import INPUT_COLUMNS, load_inputs
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


def export_published(store: Store, out_dir=DEFAULT_DIR, inputs_path=Path("data/forecast/inputs.csv"), days: int = 90,
                     now=None, last_run: dict | None = None) -> dict:
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
    forecasts = versions.merge(values, on="forecast_id", how="inner").drop(columns=["forecast_id"])
    forecasts = forecasts.sort_values(["delivery_day", "issued_at_utc", "timestamp_utc"])
    forecasts.to_csv(out / "forecasts.csv", index=False)

    actuals = pd.read_sql_query(
        "SELECT delivery_day, timestamp_utc, hour, price FROM actuals WHERE delivery_day >= ? ORDER BY timestamp_utc",
        store.conn, params=(cutoff,),
    )
    actuals.to_csv(out / "actuals.csv", index=False)

    scores = pd.read_sql_query(
        "SELECT delivery_day, issued_at_utc, hours, mae, rmse, naive_mae, naive_rmse, model_won, settled_at_utc FROM scores"
        " WHERE feature_set = ? AND delivery_day >= ? ORDER BY delivery_day, issued_at_utc",
        store.conn, params=(PUBLIC_FEATURE_SET, cutoff),
    )
    scores.to_csv(out / "scores.csv", index=False)

    tomorrow_day = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = forecasts[forecasts["delivery_day"] >= today.strftime("%Y-%m-%d")]
    if not tomorrow.empty:
        latest_day = tomorrow["delivery_day"].max()
        latest = tomorrow[tomorrow["delivery_day"] == latest_day]
        latest = latest[latest["issued_at_utc"] == latest["issued_at_utc"].max()]
        latest.to_csv(out / "tomorrow.csv", index=False)
    else:
        pd.DataFrame(columns=forecasts.columns).to_csv(out / "tomorrow.csv", index=False)

    band = pd.read_sql_query(
        "SELECT feature_set, hour, n, mae, p10, p25, p50, p75, p90 FROM error_band WHERE feature_set = ? ORDER BY hour",
        store.conn, params=(PUBLIC_FEATURE_SET,),
    )
    band.to_csv(out / "error_band.csv", index=False)

    history_rows = 0
    if Path(inputs_path).exists():
        history = load_inputs(inputs_path).reindex(columns=INPUT_COLUMNS)
        history.round(3).to_csv(out / "history.csv", index_label="timestamp_utc")
        history_rows = int(len(history))

    status = {
        "written_at_utc": iso(utc_now(now)),
        "feature_set": PUBLIC_FEATURE_SET,
        "window_days": days,
        "from_day": cutoff,
        "tomorrow": tomorrow_day,
        "forecast_versions": int(len(versions)),
        "actual_days": int(actuals["delivery_day"].nunique()) if len(actuals) else 0,
        "scored_versions": int(len(scores)),
        "history_rows": history_rows,
        "last_run": last_run or {},
        "attributions": ATTRIBUTIONS,
        "note": "Forecasts use only information available at 12:00 Paris time on the day before delivery. "
                "Historical errors are shown as context, not as a probability forecast.",
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


def import_published(store: Store, in_dir=DEFAULT_DIR, inputs_path=Path("data/forecast/inputs.csv")) -> dict:
    """Load the published files into a (usually empty) store and restore the inputs history."""
    src = Path(in_dir)
    summary = {"forecast_versions": 0, "actual_rows": 0, "scores": 0, "history_rows": 0, "error_band_rows": 0}
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
        scores = pd.read_csv(scores_path)
        versions = pd.read_sql_query("SELECT forecast_id, delivery_day, issued_at_utc FROM forecasts WHERE feature_set = ?",
                                     store.conn, params=(PUBLIC_FEATURE_SET,))
        merged = scores.merge(versions, on=["delivery_day", "issued_at_utc"], how="inner")
        for _, r in merged.iterrows():
            store.save_score(int(r["forecast_id"]), r["delivery_day"], PUBLIC_FEATURE_SET, r["issued_at_utc"], {
                "hours": r["hours"], "mae": r["mae"], "rmse": r["rmse"], "naive_mae": r["naive_mae"], "naive_rmse": r["naive_rmse"],
                "model_won": None if pd.isna(r["model_won"]) else bool(r["model_won"]),
            }, now=pd.Timestamp(r["settled_at_utc"]) if isinstance(r["settled_at_utc"], str) else None)
            summary["scores"] += 1
    band_path = src / "error_band.csv"
    if band_path.exists() and band_path.stat().st_size > 0:
        band = pd.read_csv(band_path)
        if not band.empty:
            store.save_error_band(band, source=str(band_path))
            summary["error_band_rows"] = int(len(band))
    history_path = src / "history.csv"
    if history_path.exists() and inputs_path is not None:
        inputs_path = Path(inputs_path)
        if not inputs_path.exists():
            inputs_path.parent.mkdir(parents=True, exist_ok=True)
            history = load_inputs(history_path)
            history.to_csv(inputs_path, index_label="timestamp_utc")
            summary["history_rows"] = int(len(history))
    return summary


def extend_history(inputs_path=Path("data/forecast/inputs.csv"), store: Store | None = None, delivery_day: str | None = None) -> int:
    """Append the latest stored inputs and actual prices for a delivery day to the history table.

    The morning run fetches the days around the target day but writes them to
    the store only; this folds the honest inputs of `delivery_day` (and any actual
    prices the store has) into the CSV history so the next refit sees them.
    """
    if store is None or delivery_day is None or not Path(inputs_path).exists():
        return 0
    history = load_inputs(inputs_path).reindex(columns=INPUT_COLUMNS)
    fresh = store.latest_input_values(delivery_day)
    added = 0
    if not fresh.empty:
        fresh = fresh.reindex(columns=[c for c in INPUT_COLUMNS if c in fresh.columns])
        history = fresh.combine_first(history)
        added += int(len(fresh))
    actual_rows = pd.read_sql_query("SELECT timestamp_utc, price FROM actuals", store.conn)
    if not actual_rows.empty:
        prices = pd.Series(actual_rows["price"].to_numpy(), index=pd.to_datetime(actual_rows["timestamp_utc"], utc=True), name="price_eur_mwh")
        history["price_eur_mwh"] = history["price_eur_mwh"].combine_first(prices) if "price_eur_mwh" in history else prices
        missing = prices.index.difference(history.index)
        if len(missing):
            history = history.reindex(history.index.union(missing))
            history.loc[missing, "price_eur_mwh"] = prices.loc[missing]
            added += int(len(missing))
    history = history.reindex(columns=INPUT_COLUMNS).sort_index()
    history.to_csv(inputs_path, index_label="timestamp_utc")
    return added
