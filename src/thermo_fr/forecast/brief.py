"""The morning brief: fair value, what changed in the fundamentals, and why the fair value moved.

Written for a power trader and shown as the first page of both dashboards. Everything here is arithmetic on
what the morning run already stores; nothing calls an API.

Why the fair value moved comes from LightGBM's own contributions (`Booster.predict(pred_contrib=True)`): for
every hour the prediction is exactly the sum of one contribution per feature plus the model's expected value,
so the daily mean prediction (the base fair value) is the sum of the daily mean contributions. The morning
run stores those daily means per feature next to each version (`forecast_drivers`), together with the daily
mean of the feature itself, and the brief splits the change between yesterday's pre-market version and
today's into feature groups (GROUPS below). The group changes sum to the change in the base prediction by
construction; when the model was refitted in between, the two expected values differ and that difference is
shown as its own step, "model refit". Nothing from EEX is stored in the drivers or published.
"""

import numpy as np
import pandas as pd

from .features import CALENDAR, CALENDAR_STRUCTURE, LAGS, NEIGHBOURS, NUCLEAR, PEAK_HOURS, SAME_TYPE_LAGS

EXPECTED_VALUE = "expected_value"  # the contribution column LightGBM appends: the model's average prediction
AVERAGE_GROUP = "model average"
REFIT_STEP = "model refit"
NEGATIVE_THRESHOLD = 0.05  # an hour is listed as a negative-price hour from this calibrated probability
MIN_STEP = 0.5  # groups moving less than this (EUR/MWh) are not named in the one-line summary

GROUPS = {
    "recent French prices": list(LAGS) + list(SAME_TYPE_LAGS),
    "neighbour prices": list(NEIGHBOURS),
    "wind": ["wind100_ms", "wind_proxy_mw", "wind_fc_mw"],
    "solar": ["radiation_wm2", "solar_proxy_mw", "solar_fc_mw"],
    "load and temperature": ["load_fc_mw", "temp_c", "residual_v2_mw", "residual_load_fc_mw"],
    "nuclear": list(NUCLEAR),
    "calendar": list(CALENDAR) + list(CALENDAR_STRUCTURE),
    AVERAGE_GROUP: [EXPECTED_VALUE],
}
GROUP_ORDER = list(GROUPS)
_GROUP_OF = {feature: group for group, features in GROUPS.items() for feature in features}

# the "What changed" table: label, unit, feature whose daily mean is shown, divisor; the neighbour rows add the FR spread
FUNDAMENTALS = [
    ("Load forecast", "GW", "load_fc_mw", 1000.0),
    ("Temperature", "C", "temp_c", 1.0),
    ("Wind proxy", "GW", "wind_proxy_mw", 1000.0),
    ("Solar proxy", "GW", "solar_proxy_mw", 1000.0),
    ("Latest nuclear output", "GW", "nuclear_d1_early_mw", 1000.0),
    ("Residual load", "GW", "residual_v2_mw", 1000.0),
]
NEIGHBOUR_ROWS = [("DE", "de_lu"), ("BE", "be"), ("ES", "es"), ("IT-North", "it_north"), ("CH", "ch")]
DRIVER_COLUMNS = ["feature", "value", "contribution"]


def group_of(feature: str) -> str:
    return _GROUP_OF.get(feature, "other")


def driver_table(features: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """Per feature, the daily mean of the feature and of its contribution over the delivery day's hours, plus the expected value row.

    `features` are the model's feature rows of the day (one per hour), `contributions` the matching
    pred_contrib frame (the features' columns plus `expected_value`). The sum of the contribution column is
    the daily mean prediction.
    """
    rows = []
    for column in contributions.columns:
        value = float(features[column].mean()) if column in features else float("nan")
        rows.append({"feature": column, "value": value, "contribution": float(contributions[column].mean())})
    return pd.DataFrame(rows, columns=DRIVER_COLUMNS)


def base_of(drivers: pd.DataFrame | None) -> float:
    """The daily mean prediction implied by a driver table (the sum of its contributions), NaN without one."""
    if drivers is None or drivers.empty:
        return float("nan")
    return float(pd.to_numeric(drivers["contribution"], errors="coerce").sum())


def grouped(drivers: pd.DataFrame | None) -> pd.Series:
    """Contribution per group, every group of GROUP_ORDER present (zero when the table has no feature of it), "other" last when needed."""
    out = pd.Series(0.0, index=GROUP_ORDER, dtype=float)
    if drivers is None or drivers.empty:
        return out
    frame = drivers.copy()
    frame["group"] = frame["group"] if "group" in frame else frame["feature"].map(group_of)
    sums = pd.to_numeric(frame["contribution"], errors="coerce").groupby(frame["group"]).sum()
    for group, total in sums.items():
        out[group] = float(total)  # a group outside GROUP_ORDER ("other") is appended
    return out


def fair_value(curve: pd.DataFrame) -> dict:
    """Base, peak (08:00 to 20:00 Paris) and off-peak means of an hourly forecast curve with `hour` and `forecast` columns."""
    if curve is None or curve.empty:
        return {"base": float("nan"), "peak": float("nan"), "offpeak": float("nan"), "hours": 0}
    forecast = pd.to_numeric(curve["forecast"], errors="coerce")
    peak = curve["hour"].astype(int).isin(list(PEAK_HOURS)).to_numpy()
    return {"base": float(forecast.mean()), "peak": float(forecast[peak].mean()) if peak.any() else float("nan"),
            "offpeak": float(forecast[~peak].mean()) if (~peak).any() else float("nan"), "hours": int(len(curve))}


def negative_hours(by_hour: pd.DataFrame | None, threshold: float = NEGATIVE_THRESHOLD) -> list[dict]:
    """The hours whose probability of a negative price is at or above the threshold: [{hour, p}], by hour.

    Uses the calibrated probability (`p_negative_cal`) where the frame has it, else `p_negative`.
    """
    if by_hour is None or by_hour.empty:
        return []
    column = "p_negative_cal" if "p_negative_cal" in by_hour and by_hour["p_negative_cal"].notna().any() else "p_negative"
    if column not in by_hour:
        return []
    values = pd.to_numeric(by_hour[column], errors="coerce")
    picked = by_hour.loc[values >= threshold, ["hour"]].assign(p=values[values >= threshold].to_numpy())
    return [{"hour": int(r["hour"]), "p": float(r["p"])} for _, r in picked.sort_values("hour").iterrows()]


def waterfall(previous: pd.DataFrame | None, current: pd.DataFrame | None) -> dict:
    """From yesterday's base fair value to today's, one step per feature group, in EUR/MWh.

    Returns {"start", "end", "change", "steps": [{"group", "change"}], "complete"}; `complete` is False when
    either driver table is missing, in which case the steps are empty. The expected value's change, non-zero only
    after a refit between the two versions, is the step "model refit"; the steps always sum to end minus start.
    """
    if previous is None or previous.empty or current is None or current.empty:
        return {"start": base_of(previous), "end": base_of(current), "change": float("nan"), "steps": [], "complete": False}
    before, after = grouped(previous), grouped(current)
    groups = [g for g in GROUP_ORDER if g != AVERAGE_GROUP] + [g for g in after.index.union(before.index) if g not in GROUP_ORDER]
    steps = [{"group": g, "change": float(after.get(g, 0.0) - before.get(g, 0.0))} for g in groups]
    refit = float(after.get(AVERAGE_GROUP, 0.0) - before.get(AVERAGE_GROUP, 0.0))
    if abs(refit) > 1e-9:
        steps.append({"group": REFIT_STEP, "change": refit})
    start, end = base_of(previous), base_of(current)
    return {"start": start, "end": end, "change": end - start, "steps": steps, "complete": True}


def feature_value(drivers: pd.DataFrame | None, feature: str) -> float:
    if drivers is None or drivers.empty:
        return float("nan")
    rows = drivers.loc[drivers["feature"] == feature, "value"]
    return float(pd.to_numeric(rows, errors="coerce").iloc[0]) if len(rows) else float("nan")


def fundamentals_table(current: pd.DataFrame | None, previous: pd.DataFrame | None) -> pd.DataFrame:
    """What changed: tomorrow's fundamentals against the previous delivery day's, each with value and change.

    Columns: quantity, unit, tomorrow, previous, change, fr_spread (the French minus neighbour D-1 spread for
    the neighbour rows, tomorrow's value). `previous` may be a table of feature values only (its `contribution`
    column is not read), so the local dashboard can fall back to the stored inputs of the previous day.
    """
    rows = []
    for label, unit, feature, divisor in FUNDAMENTALS:
        now, then = feature_value(current, feature) / divisor, feature_value(previous, feature) / divisor
        rows.append({"quantity": label, "unit": unit, "tomorrow": now, "previous": then, "change": now - then, "fr_spread": float("nan")})
    for label, zone in NEIGHBOUR_ROWS:
        now, then = feature_value(current, f"price_{zone}_lag1"), feature_value(previous, f"price_{zone}_lag1")
        rows.append({"quantity": f"{label} D-1 base", "unit": "EUR/MWh", "tomorrow": now, "previous": then, "change": now - then,
                     "fr_spread": feature_value(current, f"spread_{zone}_lag1")})
    return pd.DataFrame(rows, columns=["quantity", "unit", "tomorrow", "previous", "change", "fr_spread"])


def _delta(current, previous, feature: str, divisor: float = 1.0) -> float:
    return (feature_value(current, feature) - feature_value(previous, feature)) / divisor


def step_phrase(group: str, current: pd.DataFrame | None, previous: pd.DataFrame | None, day_names: tuple[str, str] | None = None) -> str:
    """The plain-English cause behind one group's step: the fundamental and how much it moved, where one stands for the group."""
    def with_change(prefix: str, feature: str, unit: str, divisor: float = 1.0, digits: int = 1) -> str:
        change = _delta(current, previous, feature, divisor)
        return prefix if np.isnan(change) else f"{prefix} {change:+.{digits}f} {unit}".rstrip()

    if group == "wind":
        return with_change("wind", "wind_proxy_mw", "GW", 1000.0)
    if group == "solar":
        return with_change("solar", "solar_proxy_mw", "GW", 1000.0)
    if group == "nuclear":
        return with_change("nuclear", "nuclear_d1_early_mw", "GW", 1000.0)
    if group == "load and temperature":
        load, temp = _delta(current, previous, "load_fc_mw", 1000.0), _delta(current, previous, "temp_c")
        if not np.isnan(load) and (np.isnan(temp) or abs(load) >= 0.3):
            return f"load {load:+.1f} GW"
        if not np.isnan(temp):
            return f"temperature {temp:+.1f} C"
        return "load and temperature"
    if group == "recent French prices":
        return with_change("French D-1 base", "price_lag1_mean", "", 1.0, 0)
    if group == "neighbour prices":
        return with_change("neighbour D-1 prices, DE", "price_de_lu_lag1", "", 1.0, 0)
    if group == "calendar":
        return f"calendar ({day_names[0]} to {day_names[1]})" if day_names else "calendar"
    return group


def summary_line(result: dict, current: pd.DataFrame | None, previous: pd.DataFrame | None, day_names: tuple[str, str] | None = None,
                 min_step: float = MIN_STEP) -> str:
    """One line from the two largest steps, e.g. "Fair value down 14: wind +5.8 GW (-11), French D-1 base -9 (-4)"."""
    if not result.get("complete"):
        return ""
    change = result["change"]
    head = "Fair value unchanged" if abs(change) < min_step else f"Fair value {'up' if change > 0 else 'down'} {abs(change):.0f}"
    top = sorted((s for s in result["steps"] if abs(s["change"]) >= min_step), key=lambda s: -abs(s["change"]))[:2]
    if not top:
        return head + "."
    parts = [f"{step_phrase(s['group'], current, previous, day_names)} ({s['change']:+.0f})" for s in top]
    return f"{head}: " + ", ".join(parts) + "."


def record_strip(summary: dict | None, min_windows: int = 20) -> dict:
    """The live record against EEX next to the fair value: the window count always, the aggregates once `min_windows` exist.

    `summary` is market.evaluate()["summary"] or the "market" section of status.json (keys scored_windows or scored_days,
    hit_rate, mean_pnl_per_mwh, model_mae, market_mae).
    """
    summary = summary or {}
    windows = int(summary.get("scored_windows", summary.get("scored_days", 0)) or 0)
    out = {"windows": windows, "min_windows": min_windows, "show": windows >= min_windows and summary.get("hit_rate") is not None}
    if out["show"]:
        out.update(hit_rate=float(summary["hit_rate"]), mean_pnl_per_mwh=float(summary["mean_pnl_per_mwh"]),
                   model_mae=float(summary["model_mae"]), market_mae=float(summary["market_mae"]))
        out["text"] = (f"Live record against EEX, {windows} windows: hit rate {100 * out['hit_rate']:.0f}%, mean P&L {out['mean_pnl_per_mwh']:+.2f} EUR/MWh "
                       f"per window, model error {out['model_mae']:.1f} against the market's {out['market_mae']:.1f} EUR/MWh.")
    else:
        out["text"] = f"Live record against EEX: {windows} of {min_windows} windows scored; the hit rate, P&L and errors appear at {min_windows}."
    return out


def early_trade(market_rows: pd.DataFrame | None, delivery_day: str, fair_base: float, record: dict | None) -> dict | None:
    """Local dashboard only: the gap between the base fair value and the latest EEX base trade of the morning for the delivery day.

    The latest row is the one with the latest window end; its close is the latest traded price. Labelled with the model's live hit
    rate (`record` is market.evaluate()["summary"]). Returns None without a row. Never published: the caller is the local app.
    """
    if market_rows is None or market_rows.empty or np.isnan(fair_base):
        return None
    rows = market_rows[(market_rows["delivery_date"] == delivery_day) & (market_rows["product"] == "base")]
    if rows.empty:
        return None
    last = rows.sort_values("window_end").iloc[-1]
    price = float(last["close"])
    gap = fair_base - price
    record = record or {}
    windows = int(record.get("scored_days", 0) or 0)
    label = (f"the model's live hit rate is {100 * float(record['hit_rate']):.0f}% over {windows} windows" if windows and record.get("hit_rate") is not None
             else "no scored window yet, so no live hit rate")
    return {"price": price, "window": f"{last['window_start']}-{last['window_end']}", "trade_date": str(last.get("trade_date", "")),
            "gap": gap, "direction": "above" if gap > 0 else "below" if gap < 0 else "at",
            "text": f"Fair value {abs(gap):.1f} EUR/MWh {'above' if gap > 0 else 'below' if gap < 0 else 'at'} the latest EEX base trade "
                    f"({last['window_start']}-{last['window_end']} Paris, close); {label}."}


def day_name(day: str) -> str:
    return pd.Timestamp(day).strftime("%a")
