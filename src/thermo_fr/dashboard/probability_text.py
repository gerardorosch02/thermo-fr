"""How event probabilities are displayed (the public app carries the same functions; a test keeps the two equal)."""

import pandas as pd

# ---------------------------------------------------------------------------- display of event probabilities (the published values keep full precision)
SUMMARY_THRESHOLD = 0.05  # hours above this count in the one-line summary
FALLBACK_BIN = {"upper": 0.01, "observed": None, "mean_forecast": None, "source": None}  # without a backtest file: under 1% is "low"


def lowest_bin(backtest: dict) -> dict:
    """The lowest bin of the negative-price reliability table, from the holdout window (else the selection window), of probabilistic_backtest.json.

    Returns upper (the bin's upper edge as a probability), observed (the share of its hours that went negative), mean_forecast and source;
    FALLBACK_BIN when the file carries no table. Nothing here is hard-coded: the label follows the table.
    """
    for source in ("holdout", "selection", *sorted(k for k in (backtest or {}) if k not in ("holdout", "selection"))):
        record = (backtest or {}).get(source) or {}
        table = (record.get("events", {}).get("negative", {}) or {}).get("reliability") or []
        if table:
            first = table[0]
            upper = float(str(first["bin"]).split("-")[-1])
            return {"upper": upper, "observed": float(first["observed_frequency"]), "mean_forecast": float(first["mean_forecast"]), "source": source}
    return dict(FALLBACK_BIN)


def bin_label(info: dict) -> str:
    """'low (about 2% historically)' from the bin's observed rate, rounded to the nearest percent; 'low' when no table is available."""
    if info is None or info.get("observed") is None:
        return "low"
    return f"low (about {round(100 * info['observed']):d}% historically)"


def format_probability(p, info: dict | None = None) -> str:
    """Display form of a probability: the bin label for hours in the lowest reliability bin, one decimal under 10%, else whole percents."""
    if p is None or pd.isna(p):
        return "n/a"
    info = info or FALLBACK_BIN
    p = float(p)
    if p < info["upper"]:
        return bin_label(info)
    return f"{100 * p:.1f}%" if p < 0.10 else f"{100 * p:.0f}%"


def unlikely_sentence(info: dict | None) -> str:
    info = info or FALLBACK_BIN
    if info.get("observed") is None:
        return f"Negative prices unlikely tomorrow (every hour below {100 * info['upper']:.0f}%)."
    return (f"Negative prices unlikely tomorrow (every hour in the lowest band, where about {round(100 * info['observed']):d}% of hours "
            "historically went negative).")


def negative_summary(hourly: pd.DataFrame, info: dict | None = None, column: str = "p_negative") -> dict:
    """One line about tomorrow: the highest hourly probability and its hour, and the hours above 5%; or the unlikely sentence."""
    info = info or FALLBACK_BIN
    if hourly is None or hourly.empty or column not in hourly or hourly[column].isna().all():
        return {"text": "", "max_p": None, "max_hour": None, "hours_above_5pct": 0, "all_in_lowest_band": False}
    values = hourly[[c for c in ("hour", column) if c in hourly]].dropna(subset=[column])
    top = values.loc[values[column].idxmax()]
    max_p, max_hour = float(top[column]), int(top["hour"]) if "hour" in values else None
    above = int((values[column] > SUMMARY_THRESHOLD).sum())
    all_lowest = bool((values[column] < info["upper"]).all())
    if all_lowest:
        text = unlikely_sentence(info)
    else:
        where = f" at {max_hour:02d}:00 Paris" if max_hour is not None else ""
        text = f"Highest hourly probability of a negative price tomorrow: {format_probability(max_p, info)}{where}; {above} hour(s) above 5%."
    return {"text": text, "max_p": max_p, "max_hour": max_hour, "hours_above_5pct": above, "all_in_lowest_band": all_lowest}


def probability_axis_top(max_p) -> float:
    """Top of the probability axis: the larger of 10% and 1.2 times the highest probability, never above 100%."""
    value = 0.0 if max_p is None or pd.isna(max_p) else float(max_p)
    return min(1.0, max(0.10, 1.2 * value))


def show_half_line(axis_top: float) -> bool:
    """The light reference line at 50% is drawn only when the axis reaches it."""
    return axis_top >= 0.5


def lowest_bin_sentence(backtest: dict) -> str:
    """Why hours in the lowest bin carry the bin's label instead of the model's figure: that bin is overconfident."""
    info = lowest_bin(backtest)
    if info.get("observed") is None:
        return ""
    others = []
    for source in ("holdout", "selection"):
        if source == info["source"]:
            continue
        other = lowest_bin({source: (backtest or {}).get(source) or {}})
        if other.get("observed") is not None:
            others.append(f"{source}: {100 * other['mean_forecast']:.1f}% against {100 * other['observed']:.1f}%")
    return (f"In the lowest reliability bin (forecasts below {100 * info['upper']:.0f}%) the model is overconfident: on the {info['source']} "
            f"window it gave {100 * info['mean_forecast']:.1f}% on average where {100 * info['observed']:.1f}% of those hours went negative"
            + (f" ({'; '.join(others)})" if others else "") + f". Hours in that bin are therefore labelled \"{bin_label(info)}\" rather than with "
            "the model's own figure; the published CSV keeps the full values.")
