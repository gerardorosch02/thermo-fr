"""Timing rules: the 12:00 Paris gate, issue times per input, and the look-ahead check."""

import pandas as pd
import pytest

from thermo_fr.forecast import timing
from thermo_fr.forecast.timing import (
    LookaheadError,
    check_point_in_time,
    delivery_days,
    gate_closure,
    gate_for,
    issue_times,
    lateness,
)


def hours(day: str) -> pd.DatetimeIndex:
    """The UTC hours of one Paris delivery day."""
    start = pd.Timestamp(day, tz="Europe/Paris")
    end = (pd.Timestamp(day) + pd.DateOffset(days=1)).tz_localize("Europe/Paris")
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def test_gate_is_noon_paris_on_the_day_before_in_both_seasons():
    assert gate_closure(["2024-07-15"])[0] == pd.Timestamp("2024-07-14T10:00Z")  # CEST
    assert gate_closure(["2024-01-15"])[0] == pd.Timestamp("2024-01-14T11:00Z")  # CET


def test_delivery_day_is_the_paris_calendar_day():
    idx = hours("2024-07-15")
    assert len(idx) == 24 and idx[0] == pd.Timestamp("2024-07-14T22:00Z")
    assert (delivery_days(idx) == pd.Timestamp("2024-07-15")).all()
    assert (gate_for(idx) == pd.Timestamp("2024-07-14T10:00Z")).all()


def test_dst_days_have_23_and_25_hours_and_one_gate_each():
    spring, autumn = hours("2024-03-31"), hours("2024-10-27")
    assert len(spring) == 23 and len(autumn) == 25
    assert gate_for(spring).nunique() == 1 and gate_for(autumn).nunique() == 1


def test_price_lags_are_known_but_todays_price_is_not():
    idx = hours("2024-07-15")
    assert (issue_times(idx, "price_lag1") == pd.Timestamp("2024-07-13T11:00Z")).all()
    assert (lateness(idx, "price_lag1") < pd.Timedelta(0)).all()
    assert (lateness(idx, "price_lag7") < pd.Timedelta(0)).all()
    # A lag of zero days would be the auction result itself, published after the gate.
    today = timing.price_lag_issue(0)(idx)
    assert (today > gate_for(idx)).all()


def test_load_forecast_before_gate_and_wind_solar_after():
    idx = hours("2024-07-15")
    assert (lateness(idx, "load_forecast") == pd.Timedelta(hours=-2)).all()
    assert (lateness(idx, "wind_solar_forecast") == pd.Timedelta(hours=6)).all()


def test_weather_issued_uses_runs_of_two_days_before():
    idx = hours("2024-07-15")
    issued = issue_times(idx, "weather_issued")
    # 00:00 Paris on 15 July is 22:00 UTC on the 14th; minus 48 h is 22:00 UTC on the 12th, run 18 UTC, plus 6 h.
    assert issued[0] == pd.Timestamp("2024-07-13T00:00Z")
    assert issued.max() <= pd.Timestamp("2024-07-14T00:00Z")
    assert (lateness(idx, "weather_issued") < pd.Timedelta(0)).all()
    # Lead day 1 would take the 12 and 18 UTC runs of D-1 for the afternoon: after the gate.
    day1 = timing.weather_issued_issue(idx, lead_days=1)
    assert (day1[-6:] > gate_for(idx)[-6:]).all()


def test_wind_proxy_is_the_later_of_weather_run_and_calibration_data():
    mid = hours("2024-07-15")
    assert (issue_times(mid, "wind_proxy") == issue_times(mid, "weather_issued")).all()
    assert (lateness(mid, "wind_proxy") < pd.Timedelta(0)).all()
    first = hours("2024-07-01")  # calibration data runs to 29 June, public at 01:00 Paris on 30 June
    calibration = timing.wind_proxy_calibration_issue(first)
    assert (calibration == pd.Timestamp("2024-06-29T23:00Z")).all()
    issued = issue_times(first, "wind_proxy")
    assert issued[0] == pd.Timestamp("2024-06-29T23:00Z") and issued[-1] == issue_times(first, "weather_issued")[-1]
    assert (lateness(first, "wind_proxy") < pd.Timedelta(0)).all()
    # Using generation up to the day before the month would be published after the gate of the 1st.
    one_day = timing._local_clock(pd.DatetimeIndex([pd.Timestamp("2024-07-01")]), 0, 1)
    assert one_day[0] > gate_for(first)[0]


def test_weather_proxy_is_never_point_in_time():
    idx = hours("2024-07-15")
    assert (lateness(idx, "weather_proxy") > pd.Timedelta(0)).all()


def test_check_passes_for_honest_features_and_fails_for_late_ones():
    idx = hours("2024-01-15").append(hours("2024-07-15"))
    honest = {"price_lag1": "price_lag1", "load_fc_mw": "load_forecast", "temp_fc_c": "weather_issued", "hour": "calendar"}
    check_point_in_time(idx, honest)
    with pytest.raises(LookaheadError, match="wind_fc_mw"):
        check_point_in_time(idx, {**honest, "wind_fc_mw": "wind_solar_forecast"})
    with pytest.raises(LookaheadError, match="temp_proxy_c"):
        check_point_in_time(idx, {"temp_proxy_c": "weather_proxy"})
    check_point_in_time(idx[:0], {"wind_fc_mw": "wind_solar_forecast"})  # nothing to check


def test_every_timing_has_a_description():
    for name, item in timing.TIMINGS.items():
        assert item.name == name and item.description
