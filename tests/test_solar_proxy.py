"""Solar generation proxy: irradiance ratio, calibration, point-in-time monthly refits, and the shared proxy machinery."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.data.solar_points import POINT_COLUMNS, SolarPointsSource, point_column
from thermo_fr.forecast import gen_proxy
from thermo_fr.forecast.solar_proxy import SOLAR, apply_weights, calibrate, latest_weights, pv_factor, rolling_proxy
from thermo_fr.forecast.timing import delivery_days
from thermo_fr.forecast.wind_proxy import WIND

POINTS = POINT_COLUMNS[:3]
TRUE_MW = {POINTS[0]: 12000.0, POINTS[1]: 0.0, POINTS[2]: 8000.0}


def synthetic_points(start="2024-01-01", end="2024-07-01", seed=3, capacity_growth=0.0):
    """Radiation at three points with a diurnal and seasonal cycle, and the generation implied by TRUE_MW."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    hour = index.hour.to_numpy()
    season = 0.6 + 0.4 * np.sin((index.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi)
    clear = np.clip(np.sin((hour - 5) / 14 * np.pi), 0, None) * 900 * season
    points = pd.DataFrame({c: clear * np.clip(rng.uniform(0.3, 1.0, len(index)), 0, 1) for c in POINTS}, index=index)
    growth = 1 + capacity_growth * np.arange(len(index)) / len(index)
    actual = sum(TRUE_MW[c] * pv_factor(points[c]) for c in POINTS) * growth + rng.normal(0, 100, len(index))
    return points, pd.Series(np.clip(actual, 0, None), index=index)


def test_pv_factor_is_the_ratio_to_the_reference_irradiance():
    cf = pv_factor(np.array([0.0, 250.0, 1000.0, 1100.0, -5.0, np.nan]))
    assert cf[0] == 0 and cf[1] == 0.25 and cf[2] == 1.0 and cf[3] == pytest.approx(1.1) and cf[4] == 0 and np.isnan(cf[5])


def test_calibration_recovers_the_point_capacities():
    points, actual = synthetic_points()
    weights = calibrate(points, actual)
    for c, mw in TRUE_MW.items():
        assert weights["weights_mw"][c] == pytest.approx(mw, abs=300)
    assert weights["proxy"] == "solar" and weights["reference_wm2"] == 1000.0 and weights["fit"]["r2"] > 0.95
    assert weights["fit"]["capacity_mw"] == pytest.approx(20000, abs=500)


def test_apply_weights_names_the_column_and_rejects_the_other_proxy():
    points, actual = synthetic_points(end="2024-01-05")
    weights = calibrate(points, actual)
    proxy = apply_weights(points, weights)
    assert proxy.name == "solar_proxy_mw" and proxy.notna().all() and (proxy[points.index.hour == 1] == 0).all()
    with pytest.raises(ValueError, match="belong to the solar proxy"):
        gen_proxy.apply_weights(points, weights, WIND)
    with pytest.raises(ValueError, match="None of the calibrated solar"):
        apply_weights(points[[]], weights)


def test_rolling_proxy_follows_the_window_and_is_point_in_time():
    points, actual = synthetic_points(start="2024-01-01", end="2024-10-01", capacity_growth=0.4)
    assert SOLAR.window_days == WIND.window_days == 365 and SOLAR.min_days == WIND.min_days == 60
    proxy, fits = rolling_proxy(points, actual, window_days=120, min_days=45)
    days = delivery_days(proxy.index)
    # 45 days of data are first available for March (fitted to 28 February on 59 days); February is not.
    assert proxy[days < "2024-03-01"].isna().all() and proxy[(days >= "2024-03-01") & (days < "2024-10-01")].notna().all()
    assert fits[0]["month"] == "2024-03" and fits[0]["train_days"] == 59 and fits[0]["last_train_day"] == "2024-02-28"
    assert fits[-1]["train_days"] == 120  # the trailing window, not the whole history
    tampered = actual.copy()
    tampered[days >= "2024-05-31"] *= 3
    proxy2, _ = rolling_proxy(points, tampered, window_days=120, min_days=45)
    june = (days >= "2024-06-01") & (days < "2024-07-01")
    assert np.allclose(proxy[june], proxy2[june], equal_nan=True)
    assert fits[-1]["capacity_mw"] > fits[0]["capacity_mw"] * 1.1


def test_latest_weights_default_to_the_spec_window(tmp_path):
    points, actual = synthetic_points(end="2024-06-01")
    weights = latest_weights(points, actual, window_days=120)
    assert pd.Timestamp(weights["fit"]["from"]) >= pd.Timestamp("2024-02-01T23:00Z")  # 120 days before the end
    assert latest_weights(points, actual)["fit"]["from"] == str(points.index[0])  # the default window covers the whole short table
    assert latest_weights(points, actual, min_days=400) is None


def test_solar_points_source_requests_radiation_from_previous_runs(tmp_path, client_factory):
    from conftest import FakeResponse, FakeSession

    def handler(url, params):
        assert "previous-runs" in url and params["hourly"] == "shortwave_radiation_previous_day2"
        times = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="1h")
        value = 500.0 if params["latitude"] == 43.9 and params["longitude"] == -0.5 else 100.0
        return FakeResponse(200, payload={"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                                                     "shortwave_radiation_previous_day2": [value] * len(times)}})

    source = SolarPointsSource(cache_dir=tmp_path, client=client_factory(FakeSession(handler=handler)), now=pd.Timestamp("2026-10-05", tz="UTC"))
    frame = source.fetch_points("2023-12-30", "2024-01-03")
    assert list(frame.columns) == POINT_COLUMNS and len(frame) == 96 and len(POINT_COLUMNS) == 21
    assert frame.loc["2023-12-31", point_column("Landes")].isna().all()  # before the archive
    assert frame.loc["2024-01-02", point_column("Landes")].eq(500.0).all() and frame.loc["2024-01-02", point_column("Lille")].eq(100.0).all()
    assert source.details["variable"] == "shortwave_radiation" and source.details["first_valid"][point_column("Paris")] == "2024-01-01 00:00:00+00:00"
    # cache keys carry the source prefix, so a solar point called Paris or Lyon never reads the city or wind-point cache
    assert [p.name for p in tmp_path.iterdir()] and all(p.name.startswith("solarpt_issued_") for p in tmp_path.iterdir())
