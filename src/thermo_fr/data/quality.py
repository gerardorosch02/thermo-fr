"""Data quality checks on the fetched hourly table. Gaps are reported, never filled."""

import pandas as pd

from ..config import PLAUSIBLE_RANGES


def quality_report(hourly: pd.DataFrame, start: str, end: str) -> dict:
    """Missing-hour share and implausible-value counts for every series in `hourly`."""
    expected = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    aligned = hourly.reindex(hourly.index.union(expected))
    report = {"expected_hours": int(len(expected)), "series": {}, "flags": []}
    for column in hourly.columns:
        s = aligned[column]
        present = int(s.notna().sum())
        missing = len(expected) - int(s.reindex(expected).notna().sum())
        entry = {
            "present_hours": present,
            "missing_hours": int(missing),
            "missing_share": round(missing / len(expected), 4) if len(expected) else 0.0,
            "min": None if present == 0 else round(float(s.min()), 2),
            "max": None if present == 0 else round(float(s.max()), 2),
        }
        lo, hi = PLAUSIBLE_RANGES.get(column, (None, None))
        if lo is not None:
            entry["below_plausible"] = int((s < lo).sum())
            entry["above_plausible"] = int((s > hi).sum())
            entry["plausible_range"] = [lo, hi]
        if column == "load_mw":
            entry["negative"] = int((s < 0).sum())
        report["series"][column] = entry
        report["flags"].extend(_flags(column, entry))
    return report


def _flags(column: str, entry: dict) -> list[str]:
    flags = []
    if entry["missing_share"] > 0.01:
        flags.append(f"{column}: {entry['missing_share']:.1%} of hours missing")
    if entry.get("negative"):
        flags.append(f"{column}: {entry['negative']} negative values")
    outside = entry.get("below_plausible", 0) + entry.get("above_plausible", 0)
    if outside:
        lo, hi = entry["plausible_range"]
        flags.append(f"{column}: {outside} values outside {lo:,} to {hi:,}")
    return flags


def format_quality(report: dict) -> str:
    lines = [f"Data quality ({report['expected_hours']:,} hours expected):"]
    for column, e in report["series"].items():
        extra = ""
        if "plausible_range" in e:
            outside = e["below_plausible"] + e["above_plausible"]
            extra = f", {outside} outside plausible range"
        lines.append(
            f"  {column:<14} missing {e['missing_hours']:>6,} h ({e['missing_share']:.2%}), "
            f"min {e['min']}, max {e['max']}{extra}"
        )
    lines.append("  flags: " + ("; ".join(report["flags"]) if report["flags"] else "none"))
    return "\n".join(lines)
