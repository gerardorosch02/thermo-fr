import numpy as np
import pandas as pd
import pytest

from thermo_fr.data.dataset import daily_frame, hdd, to_hourly_utc


def hourly_utc(start, end):
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    return pd.DataFrame({"temperature": 10.0, "load_mw": 50000.0, "price_eur_mwh": 80.0}, index=idx)


def test_spring_dst_day_has_23_hours_and_is_kept():
    # Paris clocks go forward on 31 March 2024.
    daily = daily_frame(hourly_utc("2024-03-29", "2024-04-03"))
    assert daily.loc["2024-03-31", "hours"] == 23
    assert daily.loc["2024-03-30", "hours"] == 24


def test_autumn_dst_day_has_25_hours():
    # Paris clocks go back on 27 October 2024.
    daily = daily_frame(hourly_utc("2024-10-25", "2024-10-30"))
    assert daily.loc["2024-10-27", "hours"] == 25


def test_incomplete_days_are_dropped():
    frame = hourly_utc("2024-01-10", "2024-01-13")
    frame.loc["2024-01-11 05:00":"2024-01-11 12:00", "load_mw"] = np.nan  # 8 missing hours
    daily = daily_frame(frame)
    assert pd.Timestamp("2024-01-11") not in daily.index


def test_naive_timestamps_are_rejected():
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="1h"))
    with pytest.raises(ValueError):
        to_hourly_utc(naive)


def test_quarter_hours_average_to_hourly():
    idx = pd.date_range("2025-10-01", periods=8, freq="15min", tz="UTC")
    series = pd.Series([10, 20, 30, 40, 50, 50, 50, 50], index=idx, dtype=float)
    hourly = to_hourly_utc(series)
    assert list(hourly) == [25.0, 50.0]


def test_calendar_features():
    daily = daily_frame(hourly_utc("2024-12-23", "2024-12-28"))
    assert bool(daily.loc["2024-12-25", "holiday"]) is True
    assert bool(daily.loc["2024-12-23", "holiday"]) is False
    assert daily.loc["2024-12-25", "dow"] == 2  # Wednesday


def test_hdd():
    assert list(hdd([20.0, 15.0, 10.0], 15.0)) == [0.0, 0.0, 5.0]
