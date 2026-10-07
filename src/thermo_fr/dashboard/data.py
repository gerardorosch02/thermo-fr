"""Read-only queries behind the dashboard. Nothing here calls an API.

Every function opens the SQLite store, reads, and closes it, returning plain
DataFrames and dicts that Streamlit can cache. The app keys its cache on the
database file's modification time, so a job run shows up within a minute.
"""

from pathlib import Path

import pandas as pd

from ..config import LOCAL_TZ
from ..forecast.jobs import next_delivery_day
from ..forecast.store import Store

INPUT_LABELS = {
    "load_fc_mw": "Load forecast (MW)",
    "wind_fc_mw": "Wind forecast (MW)",
    "solar_fc_mw": "Solar forecast (MW)",
    "residual_fc_mw": "Residual load (MW)",
    "temp_fc_c": "Temperature (C)",
}


def db_version(path) -> float:
    path = Path(path)
    return path.stat().st_mtime if path.exists() else 0.0


def today_and_tomorrow(now=None) -> tuple[str, str]:
    tomorrow = next_delivery_day(now)
    today = (pd.Timestamp(tomorrow) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    return today, tomorrow


def _with_store(path, fn):
    store = Store(Path(path))
    try:
        return fn(store)
    finally:
        store.close()


def forecast_panel(path, delivery_day: str, feature_set: str, actual_day: str) -> dict:
    """Latest forecast version for a day, its benchmark, the actual of another day by hour, and the band."""

    def read(store: Store):
        meta, curve = store.latest_forecast(delivery_day, feature_set)
        versions = store.forecast_versions(delivery_day, feature_set)
        band = store.error_band(feature_set)
        actual = store.actuals_for(actual_day)
        own_actual = store.actuals_for(delivery_day)
        return {"meta": meta, "curve": curve, "versions": versions, "band": band, "actual_other": actual, "actual_own": own_actual}

    out = _with_store(path, read)
    curve = out["curve"]
    if curve.empty:
        return out
    table = curve.reset_index()
    table["hour"] = table["hour"].astype(int)
    by_hour = table.groupby("hour").agg(forecast=("forecast", "mean"), naive_day=("naive_day", "mean")).reset_index()
    for name, series in (("actual_other", out["actual_other"]), ("actual_own", out["actual_own"])):
        if len(series):
            hours = series.index.tz_convert(LOCAL_TZ).hour
            by_hour = by_hour.merge(series.groupby(hours).mean().rename(name).reset_index().rename(columns={"index": "hour"}),
                                    on="hour", how="left")
        else:
            by_hour[name] = float("nan")
    if not out["band"].empty:
        band = out["band"][["p10", "p25", "p50", "p75", "p90", "mae", "n"]]
        by_hour = by_hour.merge(band, left_on="hour", right_index=True, how="left")
        for q in ("p10", "p25", "p75", "p90"):
            by_hour[f"band_{q}"] = by_hour["forecast"] + by_hour[q]
    out["by_hour"] = by_hour
    return out


def daily_inputs(frame: pd.DataFrame) -> dict:
    """Daily means of the stored hourly inputs, with wind summed and residual load derived."""
    if frame.empty:
        return {}
    wind = frame.get("wind_onshore_fc_mw", pd.Series(dtype=float)).fillna(0) + frame.get("wind_offshore_fc_mw", pd.Series(dtype=float)).fillna(0)
    if "wind_onshore_fc_mw" in frame and frame["wind_onshore_fc_mw"].isna().all():
        wind = pd.Series(float("nan"), index=frame.index)
    out = pd.DataFrame(index=frame.index)
    out["load_fc_mw"] = frame.get("load_fc_mw")
    out["wind_fc_mw"] = wind
    out["solar_fc_mw"] = frame.get("solar_fc_mw")
    out["residual_fc_mw"] = out["load_fc_mw"] - out["wind_fc_mw"] - out["solar_fc_mw"]
    out["temp_fc_c"] = frame.get("temp_fc_c")
    return {"hourly": out, "means": out.mean(numeric_only=True).to_dict()}


def inputs_panel(path, delivery_day: str, previous_day: str) -> dict:
    def read(store: Store):
        return {"tomorrow": daily_inputs(store.latest_input_values(delivery_day)),
                "today": daily_inputs(store.latest_input_values(previous_day))}

    return _with_store(path, read)


def performance_panel(path, feature_set: str, days: int = 30) -> dict:
    def read(store: Store):
        scores = store.latest_scores(feature_set, days=days)
        hourly = []
        for _, row in scores.iterrows():
            curve = store.forecast_curve(int(row["forecast_id"]))
            actual = store.actuals_for(row["delivery_day"])
            joined = curve.join(actual.rename("actual"), how="inner")
            joined["delivery_day"] = row["delivery_day"]
            hourly.append(joined)
        hourly_frame = pd.concat(hourly) if hourly else pd.DataFrame()
        return {"scores": scores, "hourly": hourly_frame}

    out = _with_store(path, read)
    scores = out["scores"]
    if not scores.empty:
        scores = scores.sort_values("delivery_day").copy()
        scores["rolling_mae"] = scores["mae"].rolling(7, min_periods=1).mean()
        scores["rolling_naive_mae"] = scores["naive_mae"].rolling(7, min_periods=1).mean()
        out["scores"] = scores
        out["share_won"] = float(scores["model_won"].fillna(0).mean())
        out["mae"] = float(scores["mae"].mean())
        out["naive_mae"] = float(scores["naive_mae"].mean())
    return out


def status_panel(path, delivery_day: str) -> dict:
    def read(store: Store):
        return {"latest": store.latest_status(delivery_day), "history": store.status_for(delivery_day),
                "timing": store.timing_log(), "summary": store.timing_summary(), "runs": store.runs(limit=12)}

    out = _with_store(path, read)
    latest = out["latest"]
    timing = out["timing"]
    if not latest.empty:
        table = latest[["status", "hours", "revision", "checked_at_utc", "message"]].copy()
        day_log = timing[timing["delivery_day"] == delivery_day].set_index("item") if not timing.empty else pd.DataFrame()
        table["first_seen_paris"] = day_log["first_seen_paris"] if not day_log.empty else None
        table["minutes_before_gate"] = day_log["minutes_before_gate"] if not day_log.empty else None
        table["changes"] = day_log["changes"] if not day_log.empty else None
        table["flag"] = [flag_for(r) for _, r in table.iterrows()]
        out["table"] = table
    return out


def flag_for(row) -> str:
    if row["status"] == "error":
        return "source error"
    if row["status"] == "absent":
        return "missing"
    minutes = row.get("minutes_before_gate")
    if minutes is not None and not pd.isna(minutes) and minutes < 0:
        return "late (after the gate)"
    if row.get("hours") is not None and row["hours"] < 23:
        return "incomplete"
    return "ok"
