import json

import numpy as np
import pandas as pd
import pytest

from conftest import FakeResponse, FakeSession
from thermo_fr.data.energy_charts import EnergyChartsSource
from thermo_fr.data.http import ServiceUnavailableError
from thermo_fr.data.sources import UnsupportedSeriesError


def unix(start, end, freq):
    idx = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    return idx, [int(t.timestamp()) for t in idx]


def price_payload(start, end, freq="1h", value=50.0):
    idx, seconds = unix(start, end, freq)
    return {
        "license_info": "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur | SMARD.de",
        "unix_seconds": seconds,
        "price": [value + i for i in range(len(seconds))],
        "unit": "EUR/MWh",
        "deprecated": False,
    }


def power_payload(start, end, freq="15min"):
    idx, seconds = unix(start, end, freq)
    return {
        "unix_seconds": seconds,
        "production_types": [
            {"name": "Nuclear", "data": [40000.0] * len(seconds)},
            {"name": "Load", "data": [60000.0] * len(seconds)},
        ],
        "deprecated": False,
    }


def handler_for(payload_fn):
    def handler(url, params):
        return FakeResponse(200, payload_fn(url, params))

    return handler


def make_source(tmp_path, session, client_factory, **kwargs):
    return EnergyChartsSource(cache_dir=tmp_path / "cache", client=client_factory(session, min_interval=0), **kwargs)


def test_prices_are_hourly_utc_and_clipped(tmp_path, client_factory):
    session = FakeSession(handler_for(lambda url, p: price_payload("2024-12-31", "2025-01-10")))
    series = make_source(tmp_path, session, client_factory).day_ahead_prices("2025-01-01", "2025-01-08")
    assert series.name == "price_eur_mwh"
    assert str(series.index.tz) == "UTC"
    assert series.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert series.index[-1] == pd.Timestamp("2025-01-07 23:00", tz="UTC")
    assert len(series) == 7 * 24
    url, params = session.calls[0]
    assert url.endswith("/price") and params["bzn"] == "FR"
    assert params["start"] == "2025-01-01T00:00Z" and params["end"] == "2025-01-08T00:00Z"


def test_quarter_hour_prices_are_averaged(tmp_path, client_factory):
    session = FakeSession(handler_for(lambda url, p: price_payload("2025-10-01", "2025-10-02", freq="15min", value=0.0)))
    series = make_source(tmp_path, session, client_factory).day_ahead_prices("2025-10-01", "2025-10-02")
    assert len(series) == 24
    assert series.iloc[0] == pytest.approx(np.mean([0, 1, 2, 3]))


def test_range_is_split_into_yearly_chunks_and_cached(tmp_path, client_factory):
    session = FakeSession(handler_for(lambda url, p: price_payload(p["start"][:10], p["end"][:10])))
    source = make_source(tmp_path, session, client_factory)
    series = source.day_ahead_prices("2023-01-01", "2025-01-01")
    assert len(session.calls) == 2
    assert [p["start"] for _, p in session.calls] == ["2023-01-01T00:00Z", "2024-01-01T00:00Z"]
    assert len(series) == (365 + 366) * 24
    assert sorted(f.name for f in (tmp_path / "cache").iterdir()) == [
        "price_FR_2023-01-01_2024-01-01.json",
        "price_FR_2024-01-01_2025-01-01.json",
    ]
    # Second run: served from cache, nothing downloaded.
    again = make_source(tmp_path, session, client_factory).day_ahead_prices("2023-01-01", "2025-01-01")
    assert len(session.calls) == 2
    pd.testing.assert_series_equal(series, again)


def test_unexpected_unit_is_rejected(tmp_path, client_factory):
    payload = price_payload("2025-01-01", "2025-01-02")
    payload["unit"] = "ct/kWh"
    session = FakeSession(handler_for(lambda url, p: payload))
    with pytest.raises(RuntimeError, match="ct/kWh"):
        make_source(tmp_path, session, client_factory).day_ahead_prices("2025-01-01", "2025-01-02")


def test_outage_raises_clear_error_and_caches_nothing(tmp_path, client_factory, sleeps):
    session = FakeSession(responses=[FakeResponse(503)] * 4)
    source = make_source(tmp_path, session, client_factory)
    with pytest.raises(ServiceUnavailableError, match="down"):
        source.day_ahead_prices("2025-01-01", "2025-01-02")
    assert not (tmp_path / "cache").exists()
    assert len(sleeps) == 3


def test_load_uses_public_power_load_series(tmp_path, client_factory):
    session = FakeSession(handler_for(lambda url, p: power_payload("2025-01-01", "2025-01-02")))
    series = make_source(tmp_path, session, client_factory).load("2025-01-01", "2025-01-02")
    assert series.name == "load_mw" and len(series) == 24
    assert (series == 60000.0).all()
    assert session.calls[0][0].endswith("/public_power") and session.calls[0][1]["country"] == "fr"


def test_load_without_load_series_fails_loudly(tmp_path, client_factory):
    payload = power_payload("2025-01-01", "2025-01-02")
    payload["production_types"] = payload["production_types"][:1]
    session = FakeSession(handler_for(lambda url, p: payload))
    with pytest.raises(UnsupportedSeriesError, match="Nuclear"):
        make_source(tmp_path, session, client_factory).load("2025-01-01", "2025-01-02")


def test_dst_days_survive(tmp_path, client_factory):
    """Unix seconds are unambiguous, so both DST days come through with 23 and 25 local hours."""
    from thermo_fr.data.dataset import daily_frame

    session = FakeSession(handler_for(lambda url, p: price_payload("2024-03-29", "2024-10-30")))
    price = make_source(tmp_path, session, client_factory).day_ahead_prices("2024-03-29", "2024-10-30")
    frame = pd.DataFrame({"temperature": 10.0, "load_mw": 50000.0, "price_eur_mwh": price})
    daily = daily_frame(frame)
    assert daily.loc["2024-03-31", "hours"] == 23
    assert daily.loc["2024-10-27", "hours"] == 25


def test_cached_file_is_raw_response(tmp_path, client_factory):
    payload = price_payload("2025-01-01", "2025-01-02")
    session = FakeSession(handler_for(lambda url, p: payload))
    make_source(tmp_path, session, client_factory).day_ahead_prices("2025-01-01", "2025-01-02")
    cached = json.loads((tmp_path / "cache" / "price_FR_2025-01-01_2025-01-02.json").read_text())
    assert cached == payload
