"""How event probabilities are displayed (the public app carries the same functions; a test keeps the two equal)."""

import pandas as pd

# ---------------------------------------------------------------------------- display of event probabilities (the published values keep full precision)
SUMMARY_THRESHOLD = 0.05  # hours above this count in the one-line summary
OWN_FIGURE_FROM = 0.10  # from here the model's own figure is shown; below, the bin's historical rate


def low_end_bins(backtest: dict, window: str = "selection") -> list[dict]:
    """The low-end reliability bins written by `thermo-fr prob-low-end` (selection window by default), or [] when the file has none."""
    record = (backtest or {}).get(window) or {}
    rows = (record.get("events", {}).get("negative", {}) or {}).get("low_end_bins") or []
    return [dict(r) for r in rows if r.get("observed_rate") is not None]


def rate_text(rate: float) -> str:
    """One decimal below 10%, whole percents from there."""
    return f"{100 * rate:.1f}%" if rate < 0.10 else f"{100 * rate:.0f}%"


def bin_for(p: float, bins: list[dict]) -> dict | None:
    for row in bins:
        if row["lower"] <= p < row["upper"]:
            return row
    return None


def format_probability(p, bins: list[dict] | None = None) -> str:
    """An hour below 10% shows its bin's observed rate on the selection window ('about 2.0% historically'); from 10% the model's own figure."""
    if p is None or pd.isna(p):
        return "n/a"
    p = float(p)
    if p >= OWN_FIGURE_FROM:
        return rate_text(p)
    row = bin_for(p, bins or [])
    if row is None:
        return "low" if p < 0.01 else rate_text(p)  # no table published: nothing to map to
    return f"about {rate_text(row['observed_rate'])} historically"


def negative_summary(hourly: pd.DataFrame, bins: list[dict] | None = None, column: str = "p_negative") -> dict:
    """One line about tomorrow: the highest hour with its label, and the hours above 5% when there are any."""
    if hourly is None or hourly.empty or column not in hourly or hourly[column].isna().all():
        return {"text": "", "max_p": None, "max_hour": None, "hours_above_5pct": 0, "label": ""}
    values = hourly[[c for c in ("hour", column) if c in hourly]].dropna(subset=[column])
    top = values.loc[values[column].idxmax()]
    max_p, max_hour = float(top[column]), int(top["hour"]) if "hour" in values else None
    above = int((values[column] > SUMMARY_THRESHOLD).sum())
    label = format_probability(max_p, bins)
    where = f"Highest hour {max_hour:02d}:00 Paris" if max_hour is not None else "Highest hour"
    text = f"{where}, {label}" + (f"; {above} hour(s) above 5%" if above else "") + "."
    return {"text": text, "max_p": max_p, "max_hour": max_hour, "hours_above_5pct": above, "label": label}


def probability_axis_top(max_p) -> float:
    """Top of the probability axis: the larger of 10% and 1.2 times the highest probability, never above 100%."""
    value = 0.0 if max_p is None or pd.isna(max_p) else float(max_p)
    return min(1.0, max(0.10, 1.2 * value))


def show_half_line(axis_top: float) -> bool:
    """The light reference line at 50% is drawn only when the axis reaches it."""
    return axis_top >= 0.5


def low_end_sentence(backtest: dict) -> str:
    """The low-end bins with the selection and holdout rates side by side, so a reader can see whether the mapping held up."""
    selection = low_end_bins(backtest, "selection")
    if not selection:
        return ""
    holdout = {r["bin"]: r for r in low_end_bins(backtest, "holdout")}
    parts = []
    for r in selection:
        h = holdout.get(r["bin"])
        held = (f"holdout {rate_text(h['observed_rate'])} of {h['hours']:,} hours" if h else "holdout: no hours")
        parts.append(f"{r['bin']}: the model gave {rate_text(r['mean_forecast'])} on average, {rate_text(r['observed_rate'])} of {r['hours']:,} hours "
                     f"went negative ({held})")
    note = ((backtest or {}).get("selection") or {}).get("events", {}).get("negative", {}).get("low_end_merge_note")
    merge = f" {note[0].upper() + note[1:]}." if note and note != "no bin merged" else ""
    return ("Low-end reliability of the forecasts below 10%, selection window with the holdout alongside: " + "; ".join(parts) + "." + merge
            + " Hours below 10% are therefore labelled with their bin's selection-window rate (\"about X% historically\") rather than the model's "
            "own figure; the published CSV keeps the full values.")
