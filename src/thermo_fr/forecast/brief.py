"""The morning brief: the model forecast, what changed in the fundamentals, and why the model forecast moved.

Written for a power trader and shown as the first page of both dashboards. Everything here is arithmetic on
what the morning run already stores; nothing calls an API.

Why the model forecast moved comes from LightGBM's own contributions (`Booster.predict(pred_contrib=True)`): for
every hour the prediction is exactly the sum of one contribution per feature plus the model's expected value,
so the daily mean prediction (the base model forecast) is the sum of the daily mean contributions. The morning
run stores those daily means per feature next to each version (`forecast_drivers`), together with the daily
mean of the feature itself, and the brief splits the change between yesterday's pre-market version and
today's into feature groups (GROUPS below). The group changes sum to the change in the base prediction by
construction; when the model was refitted in between, the two expected values differ and that difference is
shown as its own step, "model refit". Nothing from EEX is stored in the drivers or published.
"""

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .features import CALENDAR, CALENDAR_STRUCTURE, LAGS, NEIGHBOURS, NUCLEAR, PEAK_HOURS, SAME_TYPE_LAGS

EXPECTED_VALUE = "expected_value"  # the contribution column LightGBM appends: the model's average prediction
AVERAGE_GROUP = "model average"
REFIT_STEP = "model refit"
FRENCH_GROUP = "recent French prices (model following the market)"  # the price lags: the model carrying the last traded level forward
LEGACY_GROUPS = {"recent French prices": FRENCH_GROUP}  # the group's name in drivers.csv files exported before 2026-10-14
NEGATIVE_THRESHOLD = 0.05  # an hour is listed as a negative-price hour from this calibrated probability
MIN_STEP = 0.5  # groups moving less than this (EUR/MWh) are not named in the one-line summary

GROUPS = {
    FRENCH_GROUP: list(LAGS) + list(SAME_TYPE_LAGS),
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
    ("French base, previous day", "EUR/MWh", "price_lag1_mean", 1.0),  # first: the spreads below are measured against it
    ("Load forecast", "GW", "load_fc_mw", 1000.0),
    ("Temperature", "C", "temp_c", 1.0),
    ("Wind estimate", "GW", "wind_proxy_mw", 1000.0),
    ("Solar estimate", "GW", "solar_proxy_mw", 1000.0),
    ("Latest nuclear output", "GW", "nuclear_d1_early_mw", 1000.0),
    ("Load minus wind, solar and nuclear", "GW", "residual_v2_mw", 1000.0),
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
    frame["group"] = (frame["group"].replace(LEGACY_GROUPS) if "group" in frame else frame["feature"].map(group_of))
    sums = pd.to_numeric(frame["contribution"], errors="coerce").groupby(frame["group"]).sum()
    for group, total in sums.items():
        out[group] = float(total)  # a group outside GROUP_ORDER ("other") is appended
    return out


def model_forecast(curve: pd.DataFrame) -> dict:
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
    """From yesterday's base model forecast to today's, one step per feature group, in EUR/MWh.

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


def neighbour_block(table: pd.DataFrame) -> pd.DataFrame:
    """The five neighbours as a compact two-line block: one column per zone, a "D-1 base" row (value, change against the previous day
    in brackets) and an "FR spread" row (French minus neighbour, tomorrow). Text cells; a missing value reads n/a, a missing change is left out."""
    rows = table.set_index("quantity")
    prices, spreads = {}, {}
    for label, _ in NEIGHBOUR_ROWS:
        row = rows.loc[f"{label} D-1 base"]
        value = "n/a" if np.isnan(row["tomorrow"]) else f"{row['tomorrow']:.0f}"
        prices[label] = value + ("" if np.isnan(row["change"]) else f" ({row['change']:+.0f})")
        spreads[label] = "n/a" if np.isnan(row["fr_spread"]) else f"{row['fr_spread']:+.0f}"
    return pd.DataFrame([prices, spreads], index=["D-1 base", "FR spread"], columns=[label for label, _ in NEIGHBOUR_ROWS])


def _delta(current, previous, feature: str, divisor: float = 1.0) -> float:
    return (feature_value(current, feature) - feature_value(previous, feature)) / divisor


def step_phrase(group: str, current: pd.DataFrame | None, previous: pd.DataFrame | None, day_names: tuple[str, str] | None = None) -> str:
    """The plain-English cause behind one group's step, e.g. "wind 5.8 GW higher" or "French base price 9 EUR/MWh lower"."""
    def moved(name: str, feature: str, unit: str, divisor: float = 1.0, digits: int = 1) -> str:
        change = _delta(current, previous, feature, divisor)
        if np.isnan(change):
            return name
        if round(abs(change), digits) == 0:
            return f"{name} unchanged"
        return f"{name} {abs(change):.{digits}f} {unit} {'higher' if change > 0 else 'lower'}"

    if group == "wind":
        return moved("wind", "wind_proxy_mw", "GW", 1000.0)
    if group == "solar":
        return moved("solar", "solar_proxy_mw", "GW", 1000.0)
    if group == "nuclear":
        return moved("nuclear output", "nuclear_d1_early_mw", "GW", 1000.0)
    if group == "load and temperature":
        load, temp = _delta(current, previous, "load_fc_mw", 1000.0), _delta(current, previous, "temp_c")
        if not np.isnan(load) and (np.isnan(temp) or abs(load) >= 0.3):
            return moved("load", "load_fc_mw", "GW", 1000.0)
        if not np.isnan(temp):
            return moved("temperature", "temp_c", "C")
        return "load and temperature"
    if group == FRENCH_GROUP:
        return moved("French base price", "price_lag1_mean", "EUR/MWh", 1.0, 0)
    if group == "neighbour prices":
        return moved("German base price", "price_de_lu_lag1", "EUR/MWh", 1.0, 0)
    if group == "calendar":
        return f"calendar, {day_names[0]} to {day_names[1]}" if day_names else "calendar"
    return group


def summary_line(result: dict, current: pd.DataFrame | None, previous: pd.DataFrame | None, day_names: tuple[str, str] | None = None,
                 min_step: float = MIN_STEP) -> str:
    """One line from the two largest steps, e.g. "Model forecast down 14.0 EUR/MWh, mostly from wind (-11.0) and recent French prices
    (-4.0)." The figures are the steps of the chart, to one decimal. step_phrase gives the fundamental behind each group.
    """
    if not result.get("complete"):
        return ""
    change = result["change"]
    head = ("Model forecast unchanged" if abs(change) < min_step
            else f"Model forecast {'up' if change > 0 else 'down'} {abs(change):.1f} EUR/MWh")
    top = sorted((s for s in result["steps"] if abs(s["change"]) >= min_step), key=lambda s: -abs(s["change"]))[:2]
    if not top:
        return head + "."
    parts = [f"{s['group'].split(' (')[0]} ({s['change']:+.1f})" for s in top]
    return f"{head}, mostly from " + " and ".join(parts) + "."


def record_strip(summary: dict | None, min_windows: int = 20) -> dict:
    """The live record against EEX next to the model forecast: the window count always, the aggregates once `min_windows` exist.

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


BEFORE_WINDOW_LABEL = "Latest EEX base trade before the window ({time} Paris)"
WINDOW_CLOSE_LABEL = "EEX window close, after the forecast was issued"


def _hit_rate_phrase(record: dict | None) -> str:
    record = record or {}
    windows = int(record.get("scored_days", 0) or 0)
    if windows and record.get("hit_rate") is not None:
        return f"the model's live hit rate is {100 * float(record['hit_rate']):.0f}% over {windows} windows"
    return "no scored window yet, so no live hit rate"


def early_trade(market_rows: pd.DataFrame | None, delivery_day: str, forecast_base: float, record: dict | None,
                early_rows: pd.DataFrame | None = None) -> dict | None:
    """Local dashboard only: the gap between the base model forecast and the latest EEX base trade for the delivery day.

    Two sources, in order. `early_rows` is the morning snapshot file (market.load_early): its latest trade before the window is one a
    trader could have seen after the forecast was issued and before the window opened, so the gap is shown as the model forecast against that
    trade, labelled BEFORE_WINDOW_LABEL with the trade time. Without a snapshot, `market_rows` (the window file) gives the close of the
    latest window, which came after the forecast was issued: labelled WINDOW_CLOSE_LABEL and worded as a check on the forecast, not a
    gap to trade. Both carry the model's live hit rate (`record` is market.evaluate()["summary"]). Returns None without any row.
    Never published: the caller is the local app.
    """
    if np.isnan(forecast_base):
        return None
    hit = _hit_rate_phrase(record)
    if early_rows is not None and not early_rows.empty:
        rows = early_rows[(early_rows["delivery_date"] == delivery_day) & (early_rows["product"] == "base")]
        if not rows.empty:
            last = rows.sort_values("trade_time_utc").iloc[-1]
            price, time = float(last["price"]), str(last["trade_time_paris"])
            gap = forecast_base - price
            direction = "above" if gap > 0 else "below" if gap < 0 else "at"
            snapshot = pd.Timestamp(last["snapshot_at_utc"]).tz_convert(LOCAL_TZ).strftime("%H:%M")
            count = last.get("trades_before_window")
            traded = "" if _blank_count(count) else f", {int(count)} trades before the window"
            return {"kind": "before_window", "label": BEFORE_WINDOW_LABEL.format(time=time), "price": price, "trade_time_paris": time,
                    "trade_date": str(last.get("trade_date", "")), "gap": gap, "direction": direction,
                    "text": f"Model forecast {abs(gap):.1f} EUR/MWh {direction} that trade (tape read at {snapshot} Paris{traded}); {hit}."}
    if market_rows is None or market_rows.empty:
        return None
    rows = market_rows[(market_rows["delivery_date"] == delivery_day) & (market_rows["product"] == "base")]
    if rows.empty:
        return None
    last = rows.sort_values("window_end").iloc[-1]
    price = float(last["close"])
    gap = forecast_base - price
    close_side = "below" if gap > 0 else "above" if gap < 0 else "at"  # where the close sits relative to the model forecast
    window = f"{last['window_start']}-{last['window_end']}"
    return {"kind": "window_close", "label": WINDOW_CLOSE_LABEL, "price": price, "window": window, "trade_date": str(last.get("trade_date", "")),
            "gap": gap, "direction": "above" if gap > 0 else "below" if gap < 0 else "at",
            "text": f"The {window} Paris window closed {abs(gap):.1f} EUR/MWh {close_side} the model forecast. It traded after the forecast was issued, "
                    f"so this checks the forecast and is not a gap to trade; {hit}."}


def _blank_count(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def day_name(day: str) -> str:
    return pd.Timestamp(day).strftime("%a")
