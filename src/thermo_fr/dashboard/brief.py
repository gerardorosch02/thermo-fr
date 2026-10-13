"""The morning brief's read layer for the local dashboard: everything from the SQLite store and the local market file.

The brief is for the next delivery day. Fair value and the negative-price hours come from the headline version (the pre-market one
when it exists); "what changed" and "why fair value moved" come from the drivers stored with the versions (brief.driver_table). A
headline version issued before drivers were stored (before 2026-10-13) has none; the brief then uses the latest version of the day
that has them and says so. The previous day's fundamentals fall back to its stored inputs when its versions carry no drivers.

The early EEX trade line reads the local snapshot file (the latest base trade before the window, from the 10:20 London snapshot) and,
without one, the local window file; both are git-ignored and the line exists in this module only: nothing of it is exported.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from ..forecast import brief as b
from ..forecast.market import DEFAULT_PATH as MARKET_PATH
from ..forecast.market import EARLY_PATH, MIN_PUBLIC_DAYS, evaluate, load_early, load_market
from ..forecast.store import Store

INPUT_FALLBACK = {"load_fc_mw": "load_fc_mw", "temp_fc_c": "temp_c", "wind_proxy_mw": "wind_proxy_mw", "solar_proxy_mw": "solar_proxy_mw"}


def _with_store(path, fn):
    store = Store(Path(path))
    try:
        return fn(store)
    finally:
        store.close()


def drivers_for_day(store: Store, delivery_day: str, feature_set: str) -> tuple[pd.DataFrame | None, dict | None, str]:
    """(drivers, version meta, note): the headline version's drivers, else the latest version of the day that has any."""
    meta, _, _ = store.headline_forecast(delivery_day, feature_set)
    if meta is None:
        return None, None, "no version"
    drivers = store.drivers(int(meta["forecast_id"]))
    if not drivers.empty:
        return drivers, meta, "headline"
    others = store.versions_with_drivers(delivery_day, feature_set)
    if others.empty:
        return None, meta, "no drivers stored for this day's versions"
    other = others.iloc[-1].to_dict()
    issued = pd.Timestamp(other["issued_at_utc"]).tz_convert(LOCAL_TZ).strftime("%H:%M")
    return store.drivers(int(other["forecast_id"])), other, f"from the version issued {issued} Paris (the headline version carries no drivers)"


def inputs_as_drivers(store: Store, delivery_day: str) -> pd.DataFrame | None:
    """The previous day's stored inputs as a value-only driver table (load, temperature, wind and solar proxies), for the fallback."""
    frame = store.latest_input_values(delivery_day)
    if frame.empty:
        return None
    means = frame.mean(numeric_only=True)
    rows = [{"feature": feature, "value": float(means[column]), "contribution": float("nan")}
            for column, feature in INPUT_FALLBACK.items() if column in means and not pd.isna(means[column])]
    return pd.DataFrame(rows, columns=b.DRIVER_COLUMNS) if rows else None


def hourly_probabilities(store: Store, forecast_id: int) -> pd.DataFrame:
    prob = store.probabilistic_curve(forecast_id)
    if prob.empty:
        return pd.DataFrame(columns=["hour", "p_negative", "p_negative_cal"])
    prob = prob.copy()
    prob["hour"] = prob.index.tz_convert(LOCAL_TZ).hour
    return prob.groupby("hour").agg(p_negative=("p_negative", "mean"), p_negative_cal=("p_negative_cal", "mean")).reset_index()


def brief_panel(path, delivery_day: str, previous_day: str, feature_set: str = "honest_v2", market_path=MARKET_PATH,
                min_windows: int = MIN_PUBLIC_DAYS, early_path=EARLY_PATH) -> dict:
    """Everything the brief page shows, as plain values."""

    def read(store: Store):
        meta, curve, _ = store.headline_forecast(delivery_day, feature_set)
        out = {"delivery_day": delivery_day, "previous_day": previous_day, "feature_set": feature_set, "meta": meta, "curve": curve}
        if meta is not None:
            out["fair"] = b.fair_value(curve.reset_index())
            out["negative_hours"] = b.negative_hours(hourly_probabilities(store, int(meta["forecast_id"])))
            out["issued_paris"] = pd.Timestamp(meta["issued_at_utc"]).tz_convert(LOCAL_TZ)
        current, current_meta, current_note = drivers_for_day(store, delivery_day, feature_set)
        previous, previous_meta, previous_note = drivers_for_day(store, previous_day, feature_set)
        previous_source = "the previous day's pre-market version" if previous_note == "headline" else previous_note
        if previous is None:
            fallback = inputs_as_drivers(store, previous_day)
            if fallback is not None:
                previous, previous_source = fallback, "the inputs stored for the previous day (its versions carry no drivers)"
        out.update(current_drivers=current, previous_drivers=previous, current_note=current_note, previous_source=previous_source,
                   current_meta=current_meta, previous_meta=previous_meta)
        out["fundamentals"] = b.fundamentals_table(current, previous)
        complete_previous = previous if previous is not None and previous["contribution"].notna().any() else None
        out["waterfall"] = b.waterfall(complete_previous, current)
        out["summary"] = b.summary_line(out["waterfall"], current, complete_previous, (b.day_name(previous_day), b.day_name(delivery_day)))
        rows = load_market(market_path)
        summary = evaluate(store, rows, feature_set=feature_set)["summary"] if not rows.empty else {}
        out["record"] = b.record_strip(summary, min_windows)
        out["record"]["versions"] = summary.get("versions", "local versions")
        out["early_trade"] = b.early_trade(rows, delivery_day, out.get("fair", {}).get("base", float("nan")), summary, load_early(early_path))
        return out

    return _with_store(path, read)


def waterfall_frame(result: dict) -> pd.DataFrame:
    """Rows for a plotly waterfall: yesterday's fair value, one step per group, today's fair value."""
    rows = [{"label": "yesterday", "value": result["start"], "measure": "absolute"}]
    rows += [{"label": s["group"], "value": s["change"], "measure": "relative"} for s in result["steps"]]
    rows.append({"label": "today", "value": result["end"], "measure": "total"})
    return pd.DataFrame(rows)


def format_fundamentals(table: pd.DataFrame) -> pd.DataFrame:
    """The "What changed" table as text: value and change for the FUNDAMENTALS rows, units in the quantity column.

    The neighbour rows are not in it: they are shown as the two-line block of brief.neighbour_block under the table.
    """
    def fmt(value, unit, signed=False):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        digits = 1 if unit in ("GW", "C") else 0
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"

    kept = table[table["quantity"].isin([label for label, *_ in b.FUNDAMENTALS])].reset_index(drop=True)
    shown = pd.DataFrame({
        "Quantity": kept["quantity"] + " (" + kept["unit"] + ")",
        "Tomorrow": [fmt(v, u) for v, u in zip(kept["tomorrow"], kept["unit"])],
        "Previous day": [fmt(v, u) for v, u in zip(kept["previous"], kept["unit"])],
        "Change": [fmt(v, u, signed=True) for v, u in zip(kept["change"], kept["unit"])],
    })
    return shown
