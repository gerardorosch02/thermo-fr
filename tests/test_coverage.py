"""Per-input coverage of an evaluation window (forecast/coverage.py)."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast.coverage import CoverageError, check_coverage, input_coverage


def table(days=20):
    index = pd.date_range("2026-08-10", periods=days * 24, freq="h", tz="UTC")
    X = pd.DataFrame({"full": 1.0, "hole": 1.0, "partial": 1.0, "never": np.nan}, index=index)
    day = pd.Series(index.tz_convert(None).normalize(), index=index)
    X.loc[(day >= "2026-08-20") & (day <= "2026-08-24"), "hole"] = np.nan  # five whole days
    X.loc[index[5], "partial"] = np.nan  # one hour of the first day
    return X, day


def test_report_names_the_hole():
    X, day = table()
    report = input_coverage(X, day, "2026-08-15", "2026-08-29")
    assert report.loc["full", ["days_fully_missing", "days_partly_missing", "hours_missing"]].tolist() == [0, 0, 0]
    hole = report.loc["hole"]
    assert (hole.days_in_window, hole.days_fully_missing, hole.longest_missing_run_days, hole.hours_missing) == (15, 5, 5, 120)
    assert (hole.first_missing_day, hole.last_missing_day) == (pd.Timestamp("2026-08-20"), pd.Timestamp("2026-08-24"))
    assert hole.last_valid_day == pd.Timestamp("2026-08-29")
    assert report.loc["never", "days_fully_missing"] == 15 and pd.isna(report.loc["never", "first_valid_day"])
    assert report.loc["partial", "days_partly_missing"] == 0  # the partial hour is outside this window


def test_check_raises_on_a_hole_and_names_it():
    X, day = table()
    report = input_coverage(X, day, "2026-08-15", "2026-08-29")
    with pytest.raises(CoverageError, match="hole: 5 of 15 days missing, 2026-08-20 to 2026-08-24"):
        check_coverage(report, allow=("never",))
    check_coverage(report, allow=("never", "hole"))
    check_coverage(report, max_missing_days=5, allow=("never",))


def test_partly_missing_days_are_reported_not_refused():
    X, day = table()
    report = input_coverage(X, day, "2026-08-10", "2026-08-12")
    assert report.loc["partial", ["days_fully_missing", "days_partly_missing", "hours_missing"]].tolist() == [0, 1, 1]
    check_coverage(report, allow=("never",))
