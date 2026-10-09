"""Open-Meteo forecast source: endpoints, variables, unit conversion, caching and aggregation."""

import json

import pandas as pd

from conftest import FakeResponse, FakeSession
from thermo_fr.config import City
from thermo_fr.data.weather_forecast import HISTORICAL_FORECAST_URL, PREVIOUS_RUNS_URL, OpenMeteoForecastSource, column_name

CITIES = (City("A", 48.0, 2.0, 3.0), City("B", 45.0, 5.0, 1.0))


def payload(params, offset: float, missing_wind: bool = False):
    times = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="1h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for variable in params["hourly"].split(","):
        if variable.startswith("temperature"):
            hourly[variable] = [10.0 + offset] * len(times)
        elif variable.startswith("wind"):
            hourly[variable] = [None] * len(times) if missing_wind else [36.0 + offset] * len(times)  # km/h
        else:
            hourly[variable] = [100.0 + offset] * len(times)
    return FakeResponse(200, payload={"hourly": hourly, "hourly_units": {}})


def test_issued_and_proxy_use_their_own_endpoints_and_variables(tmp_path, client_factory):
    def handler(url, params):
        assert params["timezone"] == "UTC" and params["models"] == "best_match"
        offset = 0.0 if params["latitude"] == 48.0 else 4.0
        if url == PREVIOUS_RUNS_URL:
            assert params["hourly"] == (
                "temperature_2m_previous_day2,wind_speed_100m_previous_day2,shortwave_radiation_previous_day2"
            )
            return payload(params, offset)
        assert url == HISTORICAL_FORECAST_URL
        assert params["hourly"] == "temperature_2m,wind_speed_100m,shortwave_radiation"
        return payload(params, offset + 1.0)

    session = FakeSession(handler=handler)
    source = OpenMeteoForecastSource(cities=CITIES, cache_dir=tmp_path, client=client_factory(session), now=pd.Timestamp("2026-10-05", tz="UTC"))
    frame = source.fetch("2024-06-01", "2024-06-03")
    assert list(frame.columns) == [
        "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2", "temp_proxy_c", "wind100_proxy_ms", "radiation_proxy_wm2",
    ]
    assert len(frame) == 48 and frame.index[0] == pd.Timestamp("2024-06-01", tz="UTC")
    # temperature weighted 3:1 -> 10 + 4/4 = 11; wind and radiation simple means; km/h to m/s
    assert frame["temp_fc_c"].iloc[0] == 11.0 and frame["temp_proxy_c"].iloc[0] == 12.0
    assert abs(frame["wind100_fc_ms"].iloc[0] - (36.0 + 40.0) / 2 / 3.6) < 1e-9
    assert frame["radiation_fc_wm2"].iloc[0] == 102.0
    assert len(session.calls) == 4  # 2 cities x 2 endpoints, one year


def test_requests_are_split_by_year_cached_and_future_windows_not_cached(tmp_path, client_factory):
    def handler(url, params):
        return payload(params, 0.0)

    session = FakeSession(handler=handler)
    source = OpenMeteoForecastSource(cities=CITIES[:1], cache_dir=tmp_path, client=client_factory(session), now=pd.Timestamp("2026-10-05", tz="UTC"))
    source.fetch("2023-12-30", "2024-01-03", kinds=("issued",))
    assert [(c[1]["start_date"], c[1]["end_date"]) for c in session.calls] == [("2023-12-30", "2023-12-31"), ("2024-01-01", "2024-01-02")]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "issued_best_match_lead2_A_2023-12-30_2023-12-31.json",
        "issued_best_match_lead2_A_2024-01-01_2024-01-02.json",
    ]
    source.fetch("2023-12-30", "2024-01-03", kinds=("issued",))
    assert len(session.calls) == 2  # cache hit

    source.fetch("2026-10-01", "2026-10-07", kinds=("issued",))
    source.fetch("2026-10-01", "2026-10-07", kinds=("issued",))
    assert len(session.calls) == 4  # window reaching today is fetched again
    assert len(list(tmp_path.iterdir())) == 2


def test_missing_variables_stay_nan_and_first_valid_is_recorded(tmp_path, client_factory):
    def handler(url, params):
        return payload(params, 0.0, missing_wind=True)

    source = OpenMeteoForecastSource(cities=CITIES, cache_dir=tmp_path, client=client_factory(FakeSession(handler=handler)))
    frame = source.fetch("2022-01-01", "2022-01-02", kinds=("issued",))
    assert frame["wind100_fc_ms"].isna().all() and frame["temp_fc_c"].notna().all()
    assert source.details["first_valid"]["wind100_fc_ms"] is None
    assert source.details["first_valid"]["temp_fc_c"] == "2022-01-01 00:00:00+00:00"


def test_column_names():
    assert column_name("temp_c", "issued") == "temp_fc_c"
    assert column_name("wind100_ms", "proxy") == "wind100_proxy_ms"
    assert column_name("radiation_wm2", "issued") == "radiation_fc_wm2"
    assert json.dumps({"ok": True})  # keep json imported for payload helpers above
