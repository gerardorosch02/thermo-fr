"""How event probabilities are displayed (the public app carries the same functions; a test keeps the two equal)."""

import pandas as pd

# ---------------------------------------------------------------------------- display of event probabilities (the published values keep full precision)
BELOW_1PCT = "below 1%"
SUMMARY_THRESHOLD = 0.05  # hours above this count in the one-line summary
VERY_UNLIKELY = "Negative prices very unlikely tomorrow (every hour below 1%)."


def format_probability(p) -> str:
    """Display form of a probability: 'below 1%' under 1% (the lowest reliability bin is overconfident), one decimal under 10%, else whole."""
    if p is None or pd.isna(p):
        return "n/a"
    p = float(p)
    if p < 0.01:
        return BELOW_1PCT
    return f"{100 * p:.1f}%" if p < 0.10 else f"{100 * p:.0f}%"


def negative_summary(hourly: pd.DataFrame, column: str = "p_negative") -> dict:
    """One line about tomorrow: the highest hourly probability and its hour, and the hours above 5%; or the very-unlikely sentence."""
    if hourly is None or hourly.empty or column not in hourly or hourly[column].isna().all():
        return {"text": "", "max_p": None, "max_hour": None, "hours_above_5pct": 0, "all_below_1pct": False}
    values = hourly[[c for c in ("hour", column) if c in hourly]].dropna(subset=[column])
    top = values.loc[values[column].idxmax()]
    max_p, max_hour = float(top[column]), int(top["hour"]) if "hour" in values else None
    above = int((values[column] > SUMMARY_THRESHOLD).sum())
    all_below = bool((values[column] < 0.01).all())
    if all_below:
        text = VERY_UNLIKELY
    else:
        where = f" at {max_hour:02d}:00 Paris" if max_hour is not None else ""
        text = f"Highest hourly probability of a negative price tomorrow: {format_probability(max_p)}{where}; {above} hour(s) above 5%."
    return {"text": text, "max_p": max_p, "max_hour": max_hour, "hours_above_5pct": above, "all_below_1pct": all_below}


def probability_axis_top(max_p) -> float:
    """Top of the probability axis: the larger of 10% and 1.2 times the highest probability, never above 100%."""
    value = 0.0 if max_p is None or pd.isna(max_p) else float(max_p)
    return min(1.0, max(0.10, 1.2 * value))


def show_half_line(axis_top: float) -> bool:
    """The light reference line at 50% is drawn only when the axis reaches it."""
    return axis_top >= 0.5


def lowest_bin_sentence(backtest: dict) -> str:
    """Why probabilities below 1% are shown as 'below 1%': the lowest reliability bin of the negative-price backtest is overconfident."""
    parts = []
    for label, record in (backtest or {}).items():
        table = (record.get("events", {}).get("negative", {}) or {}).get("reliability") or []
        if table:
            first = table[0]
            parts.append(f"{label}: forecast about {100 * float(first['mean_forecast']):.1f}%, observed {100 * float(first['observed_frequency']):.1f}%")
    if not parts:
        return ""
    return ("The lowest reliability bin is overconfident: where the model gave well under 1%, negative prices still occurred about 2% of the "
            "time (" + "; ".join(parts) + "). Probabilities below 1% are therefore shown as \"below 1%\", never as 0.0%; the published CSV keeps "
            "the full values.")
