"""How event probabilities are displayed (the public app carries the same functions; a test keeps the two equal)."""

import pandas as pd

# ---------------------------------------------------------------------------- display of event probabilities (the published values keep full precision)
SUMMARY_THRESHOLD = 0.05  # hours above this count in the one-line summary


def format_probability(p) -> str:
    """The probability itself: one decimal below 10%, whole percents from there, and '<0.1%' under 0.05%."""
    if p is None or pd.isna(p):
        return "n/a"
    p = float(p)
    if p < 0.0005:
        return "<0.1%"
    return f"{100 * p:.1f}%" if p < 0.10 else f"{100 * p:.0f}%"


def negative_summary(hourly: pd.DataFrame, column: str = "p_negative") -> dict:
    """One line about tomorrow: the highest hour with its probability, and the hours above 5% when there are any."""
    if hourly is None or hourly.empty or column not in hourly or hourly[column].isna().all():
        return {"text": "", "max_p": None, "max_hour": None, "hours_above_5pct": 0, "label": ""}
    values = hourly[[c for c in ("hour", column) if c in hourly]].dropna(subset=[column])
    top = values.loc[values[column].idxmax()]
    max_p, max_hour = float(top[column]), int(top["hour"]) if "hour" in values else None
    above = int((values[column] > SUMMARY_THRESHOLD).sum())
    label = format_probability(max_p)
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


def calibration_sentence(calibration: dict | None, backtest: dict) -> str:
    """How the shown probability was calibrated, with the pre-registered selection and holdout results side by side."""
    if not calibration or not calibration.get("applied"):
        return ("The probability shown is the raw classifier output: no calibration file is published yet. Its low-end reliability is in the "
                "backtest aggregates.")
    parts = []
    for label in ("selection", "holdout"):
        r = ((backtest or {}).get(label) or {}).get("recalibration")
        if not r or "raw" not in r or "calibrated" not in r:
            continue
        parts.append(f"{label} window ({r.get('first_day')} to {r.get('last_day')}, {r.get('events')} negative hours): Brier {r['raw']['brier']:.4f} raw to "
                     f"{r['calibrated']['brier']:.4f} calibrated, calibration error {r['raw']['mace']['unweighted']:.3f} to {r['calibrated']['mace']['unweighted']:.3f}, "
                     f"skill against climatology {r['raw']['bss_vs_clim']:+.2f} to {r['calibrated']['bss_vs_clim']:+.2f}")
    fitted = f" fitted on {calibration['pairs_hours']:,} out-of-sample hours ({calibration['pairs_events']} negative) through {calibration['last_day']}," if calibration.get("pairs_hours") else ""
    return (f"The probability shown is calibrated: an isotonic (monotone) map from the classifier's raw output to the observed frequency,{fitted} "
            "refitted monthly with the models and applied at issue time; the raw output stays in the local store and in the CSV's p_negative_raw "
            "column." + (" Pre-registered walk-forward results (docs/experiments.md, 2026-10-12): " + "; ".join(parts) + "." if parts else ""))
