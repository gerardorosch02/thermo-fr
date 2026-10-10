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


def test_solar_proxy_shares_the_generation_proxy_rule():
    first, mid = hours("2024-07-01"), hours("2024-07-15")
    assert (issue_times(first, "solar_proxy") == issue_times(first, "wind_proxy")).all()
    assert (issue_times(mid, "solar_proxy") == issue_times(mid, "weather_issued")).all()
    assert (lateness(first, "solar_proxy") < pd.Timedelta(0)).all()


def test_same_type_price_lag_is_the_comparable_days_price_published_before_the_gate():
    monday = hours("2024-07-15")  # comparable day: Friday 12 July, price published 13:00 Paris on 11 July
    assert (issue_times(monday, "price_lag_same_type") == pd.Timestamp("2024-07-11T11:00Z")).all()
    tuesday = hours("2024-07-16")  # comparable day: Monday, so the same as price_lag1
    assert (issue_times(tuesday, "price_lag_same_type") == issue_times(tuesday, "price_lag1")).all()
    ascension = hours("2024-05-09")  # comparable day: Wednesday 8 May, a holiday
    assert (issue_times(ascension, "price_lag_same_type") == pd.Timestamp("2024-05-07T11:00Z")).all()
    year = pd.DatetimeIndex([h for d in pd.date_range("2025-01-01", "2025-12-31", freq="D") for h in hours(d.strftime("%Y-%m-%d"))])
    assert (lateness(year, "price_lag_same_type") <= pd.Timedelta(hours=-23)).all()


def test_nuclear_lags_are_known_before_the_pre_market_issue_time():
    idx = hours("2024-07-15")  # D-1 is 14 July; Paris is UTC+2
    assert (issue_times(idx, "nuclear_d2") == pd.Timestamp("2024-07-14T00:00Z")).all()  # 02:00 Paris on D-1
    assert (issue_times(idx, "nuclear_d1") == pd.Timestamp("2024-07-14T08:00Z")).all()  # 10:00 Paris on D-1
    premarket = timing.premarket_issue_for(idx)
    assert (premarket == pd.Timestamp("2024-07-14T08:05Z")).all()
    assert (issue_times(idx, "nuclear_d1") < premarket).all() and (issue_times(idx, "nuclear_d2") < premarket).all()
    winter = hours("2024-01-15")
    assert (issue_times(winter, "nuclear_d1") == pd.Timestamp("2024-01-14T09:00Z")).all()
    # a D-1 hour ending after the cutoff would be known only after the first run started
    late = timing._local_clock(delivery_days(winter), -1, timing.NUCLEAR_D1_CUTOFF_HOUR + 2 + timing.NUCLEAR_PUBLICATION_LAG_HOURS)
    assert (late > timing.premarket_issue_for(winter)).all()


def test_neighbour_prices_and_residual_v2_meet_both_deadlines():
    idx = hours("2024-07-15")
    assert (issue_times(idx, "neighbour_price_lag1") == issue_times(idx, "price_lag1")).all()
    # the residual is known with its latest component, the load forecast at 10:00 Paris on D-1
    assert (issue_times(idx, "residual_v2") == issue_times(idx, "load_forecast")).all()
    first = hours("2024-07-01")  # at the start of a month the proxy calibration data is later than the nuclear hours but still earlier
    assert (issue_times(first, "residual_v2") == issue_times(first, "load_forecast")).all()
    v2 = {"nuclear_d2_mw": "nuclear_d2", "nuclear_d1_early_mw": "nuclear_d1", "price_de_lu_lag1": "neighbour_price_lag1",
          "residual_v2_mw": "residual_v2", "load_fc_mw": "load_forecast", "wind_proxy_mw": "wind_proxy"}
    year = pd.DatetimeIndex([h for d in pd.date_range("2025-01-01", "2025-12-31", freq="D") for h in hours(d.strftime("%Y-%m-%d"))])
    check_point_in_time(year, v2)
    check_point_in_time(year, v2, deadline="premarket")
    # the Swiss auction result for D clears before the gate but is not reliably out by the pre-market issue time: never a feature
    swiss_today = {"price_ch_lag0": "wind_solar_forecast"}
    with pytest.raises(LookaheadError):
        check_point_in_time(idx, swiss_today, deadline="premarket")
    with pytest.raises(LookaheadError, match="pre-market"):
        check_point_in_time(idx, {"load_fc_mw": "load_forecast", "x": "wind_solar_forecast"}, deadline="premarket")
    with pytest.raises(ValueError):
        check_point_in_time(idx, v2, deadline="noon")


def test_fuel_price_rules_for_the_private_collector():
    summer, winter = hours("2024-07-15"), hours("2024-01-15")
    # the end-of-day indices for gas day D-1 are known from 22:00 CET on D-2, the same UTC hour all year
    assert (issue_times(summer, "fuel_index_lag1") == pd.Timestamp("2024-07-13T21:00Z")).all()
    assert (issue_times(winter, "fuel_index_lag1") == pd.Timestamp("2024-01-13T21:00Z")).all()
    assert (issue_times(summer, "fuel_index_lag1") < timing.premarket_issue_for(summer)).all()
    # the index for gas day D itself would be published on the evening of D-1, after the gate: never a feature
    same_day = timing.fuel_index_lag1_issue(summer) + pd.Timedelta(days=1)
    assert (same_day > gate_for(summer)).all()
    # the morning trades are taken up to the issue time, so the feature is known exactly then and passes the pre-market check
    assert (issue_times(summer, "fuel_morning_trades") == timing.premarket_issue_for(summer)).all()
    rules = {"gas_index_lag1": "fuel_index_lag1", "carbon_index_lag1": "fuel_index_lag1", "gas_front_month_morning": "fuel_morning_trades"}
    check_point_in_time(summer.append(winter), rules)
    check_point_in_time(summer.append(winter), rules, deadline="premarket")
    # a trade one minute after the issue time would be late for the pre-market deadline
    late = {"x": "wind_solar_forecast"}
    with pytest.raises(LookaheadError):
        check_point_in_time(summer, late, deadline="premarket")


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
