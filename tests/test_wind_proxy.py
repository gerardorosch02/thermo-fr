"""Wind generation proxy: power curve, calibration, point-in-time monthly refits, weights file."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.data.wind_points import POINT_COLUMNS, WindPointsSource, point_column
from thermo_fr.forecast.timing import delivery_days
from thermo_fr.forecast.wind_proxy import (
    apply_weights,
    calibrate,
    latest_weights,
    load_weights,
    power_curve,
    rolling_proxy,
    save_weights,
)

POINTS = POINT_COLUMNS[:4]
TRUE_MW = {POINTS[0]: 9000.0, POINTS[1]: 6000.0, POINTS[2]: 0.0, POINTS[3]: 3000.0}


def synthetic_points(start="2024-01-01", end="2024-07-01", seed=1, capacity_growth=0.0):
    """Point speeds and the actual generation implied by TRUE_MW (plus noise and optional growth)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    base = np.clip(7 + 4 * rng.standard_normal(len(index)).cumsum() / 30, 0, 25)
    points = pd.DataFrame({c: np.clip(base + rng.normal(0, 1.5, len(index)), 0, None) for c in POINTS}, index=index)
    growth = 1 + capacity_growth * np.arange(len(index)) / len(index)
    actual = pd.Series(sum(TRUE_MW[c] * power_curve(points[c]) for c in POINTS) * growth + rng.normal(0, 150, len(index)), index=index)
    return points, actual.clip(lower=0)


def test_power_curve_shape():
    v = np.array([0.0, 3.0, 7.5, 12.0, 20.0, 25.0, 30.0, np.nan])
    cf = power_curve(v)
    assert cf[0] == 0 and cf[1] == 0 and cf[3] == 1 and cf[4] == 1 and cf[5] == 0 and cf[6] == 0 and np.isnan(cf[7])
    assert cf[2] == pytest.approx(0.5**3)


def test_calibration_recovers_the_point_capacities():
    points, actual = synthetic_points()
    weights = calibrate(points, actual)
    for c, mw in TRUE_MW.items():
        assert weights["weights_mw"][c] == pytest.approx(mw, abs=400)
    assert weights["fit"]["capacity_mw"] == pytest.approx(18000, abs=800) and weights["fit"]["r2"] > 0.95
    assert weights["hub_height_m"] == 100 and weights["fit"]["hours"] == len(points)


def test_apply_weights_rescales_missing_points_and_is_nan_without_any():
    points, actual = synthetic_points(end="2024-01-03")
    weights = calibrate(points, actual)
    full = apply_weights(points, weights)
    assert full.name == "wind_proxy_mw" and full.notna().all()
    holes = points.copy()
    holes.loc[holes.index[:5], POINTS[0]] = np.nan
    holes.loc[holes.index[5], :] = np.nan
    partial = apply_weights(holes, weights)
    assert partial.iloc[:5].notna().all() and np.isnan(partial.iloc[5]) and partial.iloc[6:].equals(full.iloc[6:])
    with pytest.raises(ValueError, match="None of the calibrated"):
        apply_weights(points[[]], weights)


def test_rolling_proxy_is_point_in_time_and_follows_capacity_growth():
    points, actual = synthetic_points(start="2024-01-01", end="2024-10-01", capacity_growth=0.5)
    proxy, fits = rolling_proxy(points, actual, window_days=120, min_days=50)
    days = delivery_days(proxy.index)
    # Nothing before 50 days of calibration data exist: January and February are NaN, March onwards is filled.
    assert proxy[days < "2024-03-01"].isna().all()
    assert proxy[(days >= "2024-03-01") & (days < "2024-10-01")].notna().all()
    assert [f["month"] for f in fits][:7] == ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"]
    assert fits[0]["last_train_day"] == "2024-02-28" and fits[0]["train_days"] == 59  # Jan 1 to Feb 28 (two days before March)
    # Point in time: changing actual generation inside June (or after) leaves June's proxy untouched.
    tampered = actual.copy()
    tampered[days >= "2024-05-31"] *= 3  # June is fitted on data to 30 May
    proxy2, _ = rolling_proxy(points, tampered, window_days=120, min_days=50)
    june = (days >= "2024-06-01") & (days < "2024-07-01")
    assert np.allclose(proxy[june], proxy2[june], equal_nan=True)
    assert not np.allclose(proxy[days >= "2024-07-01"].dropna(), proxy2[days >= "2024-07-01"].dropna())
    # The trailing-window capacity rises with the fleet.
    assert fits[-1]["capacity_mw"] > fits[0]["capacity_mw"] * 1.15


def test_latest_weights_save_and_load(tmp_path):
    points, actual = synthetic_points(end="2024-04-01")
    weights = latest_weights(points, actual, window_days=30, min_days=20)
    assert pd.Timestamp(weights["fit"]["from"]) >= pd.Timestamp("2024-03-01T23:00Z") and weights["fit"]["to"] == "2024-03-31 23:00:00+00:00"
    path = save_weights(weights, tmp_path / "model" / "wind_proxy.json")
    assert load_weights(path) == weights and load_weights(tmp_path / "missing.json") is None
    assert latest_weights(points, actual, window_days=30, min_days=200) is None
    assert latest_weights(points[[]], actual) is None


def test_wind_points_source_requests_hub_height_previous_runs(tmp_path, client_factory):
    from conftest import FakeResponse, FakeSession

    def handler(url, params):
        assert "previous-runs" in url and params["hourly"] == "wind_speed_100m_previous_day2"
        times = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="1h")
        speed = 36.0 if params["latitude"] == 49.95 else 18.0  # km/h
        return FakeResponse(200, payload={"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                                                     "wind_speed_100m_previous_day2": [speed] * len(times)}})

    source = WindPointsSource(cache_dir=tmp_path, client=client_factory(FakeSession(handler=handler)), now=pd.Timestamp("2026-10-05", tz="UTC"))
    frame = source.fetch_points("2023-12-30", "2024-01-03")
    assert list(frame.columns) == POINT_COLUMNS and len(frame) == 96
    assert frame.loc["2023-12-31", point_column("Somme")].isna().all()  # before the archive
    assert frame.loc["2024-01-02", point_column("Somme")].eq(10.0).all() and frame.loc["2024-01-02", point_column("Aude")].eq(5.0).all()
    assert source.details["hub_height_m"] == 100 and len(source.details["points"]) == len(POINT_COLUMNS)
    assert frame.loc[:, point_column("Aude")].first_valid_index() == pd.Timestamp("2024-01-01", tz="UTC")
