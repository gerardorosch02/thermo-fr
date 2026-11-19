"""Coverage of each model input over an evaluation window.

Written on 2026-11-19 after a 52-day hole in two inputs went unnoticed through every holdout evaluation of the project:
gradient boosting does not fail on a missing input, it sends it down a default branch, so the scores were computed and logged
without anything flagging that the model had been run without three of its inputs.

`input_coverage` reports, for every column of a feature table, the first and last delivery day with a value and what is
missing inside the window. `check_coverage` raises unless every input is present on every day of the window, apart from the
columns the caller names as allowed to be missing. An evaluation calls it before any score is computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CoverageError(ValueError):
    pass


def _longest_run(days: list[pd.Timestamp]) -> int:
    best = run = 0
    previous = None
    for d in sorted(days):
        run = run + 1 if previous is not None and d - previous == pd.Timedelta(days=1) else 1
        best, previous = max(best, run), d
    return best


def input_coverage(X: pd.DataFrame, delivery_day: pd.Series, first, last) -> pd.DataFrame:
    """One row per input: where it has values at all, and what is missing on the delivery days from `first` to `last`
    inclusive. A day is fully missing when the input has no value on any of its hours."""
    days = pd.to_datetime(pd.Series(np.asarray(delivery_day), index=X.index))
    window = (days >= pd.Timestamp(first)) & (days <= pd.Timestamp(last))
    rows = []
    for column in X.columns:
        present = X[column].notna()
        valid_days = days[present]
        inside = present[window].groupby(days[window])
        any_value, every_value = inside.any(), inside.all()
        fully = list(any_value.index[~any_value])
        rows.append({"input": column, "first_valid_day": valid_days.min() if len(valid_days) else pd.NaT,
                     "last_valid_day": valid_days.max() if len(valid_days) else pd.NaT, "days_in_window": int(len(any_value)),
                     "days_fully_missing": int(len(fully)), "days_partly_missing": int((any_value & ~every_value).sum()),
                     "hours_missing": int((~present[window]).sum()), "longest_missing_run_days": _longest_run(fully),
                     "first_missing_day": min(fully) if fully else pd.NaT, "last_missing_day": max(fully) if fully else pd.NaT})
    return pd.DataFrame(rows).set_index("input")


def check_coverage(report: pd.DataFrame, max_missing_days: int = 0, allow=()) -> None:
    """Raise unless every input not named in `allow` is fully missing on at most `max_missing_days` days of the window."""
    bad = report[(report["days_fully_missing"] > max_missing_days) & ~report.index.isin(list(allow))]
    if len(bad):
        lines = [f"{name}: {int(r.days_fully_missing)} of {int(r.days_in_window)} days missing"
                 + (f", {r.first_missing_day:%Y-%m-%d} to {r.last_missing_day:%Y-%m-%d}" if pd.notna(r.first_missing_day) else "")
                 + (f" (last value {r.last_valid_day:%Y-%m-%d})" if pd.notna(r.last_valid_day) else " (never present)") for name, r in bad.iterrows()]
        raise CoverageError("inputs missing inside the evaluation window, so the model would be scored without them:\n  " + "\n  ".join(lines))
