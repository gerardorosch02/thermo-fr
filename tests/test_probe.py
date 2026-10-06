"""The timing probe logs presence or absence of tomorrow's items, hashes values and detects revisions."""

import json

import pandas as pd

from conftest import FakeResponse, FakeSession
from test_entsoe_rest import ack_document, gl_document, gl_timeseries, period_xml
from thermo_fr.data.entsoe_rest import EntsoeApi
from thermo_fr.forecast import probe as probe_module
from thermo_fr.forecast.probe import poll, probe, run_probe


def make_handler(load_value=1, wind_absent=True):
    def handler(url, params):
        start = pd.Timestamp(params["periodStart"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
        end = pd.Timestamp(params["periodEnd"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
        if params["documentType"] == "A69":
            if wind_absent:
                return FakeResponse(200, content=ack_document("999", "No matching data found"))
            series = [gl_timeseries([period_xml(start, end, "PT60M", {1: 100})], psr="B16"),
                      gl_timeseries([period_xml(start, end, "PT60M", {1: 200})], psr="B19")]
            return FakeResponse(200, content=gl_document(series, revision="3"))
        ts = gl_timeseries([period_xml(start, end, "PT15M", {1: load_value})])
        return FakeResponse(200, content=gl_document([ts], revision="2"))

    return handler


def make_api(tmp_path, client_factory, handler):
    return EntsoeApi(api_key="SECRET-TOKEN", cache_dir=tmp_path / "entsoe", client=client_factory(FakeSession(handler=handler)),
                     now=pd.Timestamp("2026-10-05T08:00Z"))


def test_probe_reports_present_and_absent_items(tmp_path, client_factory):
    api = make_api(tmp_path, client_factory, make_handler())
    snapshots = {}
    rows = pd.DataFrame(probe(api, pd.Timestamp("2026-10-06"), snapshots=snapshots)).set_index("item")
    assert rows.loc["load_forecast", "status"] == "present" and rows.loc["load_forecast", "hours"] == 24
    assert rows.loc["load_forecast", "revision"] == "2" and rows.loc["load_forecast", "value_hash"]
    assert rows.loc["wind_solar_forecast", "status"] == "absent" and rows.loc["wind_solar_forecast", "hours"] == 0
    assert rows.loc["wind_solar_forecast", "value_hash"] is None
    assert rows.loc["prices", "status"] == "present"
    assert len(snapshots["load_forecast"]["load_forecast"]) == 24 and snapshots["wind_solar_forecast"] == {}
    assert not (tmp_path / "entsoe").exists()  # tomorrow's window is never cached


def test_wind_solar_values_are_split_by_type(tmp_path, client_factory):
    api = make_api(tmp_path, client_factory, make_handler(wind_absent=False))
    snapshots = {}
    rows = pd.DataFrame(probe(api, pd.Timestamp("2026-10-06"), items=("wind_solar_forecast",), snapshots=snapshots))
    assert rows.iloc[0]["hours"] == 24 and rows.iloc[0]["revision"] == "3"
    assert set(snapshots["wind_solar_forecast"]) == {"solar", "wind_onshore"}
    assert snapshots["wind_solar_forecast"]["solar"][0] == 100.0


def test_run_probe_appends_to_the_log_and_saves_snapshots(tmp_path, client_factory, monkeypatch):
    monkeypatch.setattr(probe_module, "EntsoeApi", lambda cache_dir: make_api(tmp_path, client_factory, make_handler()))
    log = tmp_path / "probe.csv"
    run_probe(log_path=log, cache_dir=tmp_path / "entsoe", delivery_day="2026-10-06", snapshot_dir=tmp_path / "values")
    run_probe(log_path=log, cache_dir=tmp_path / "entsoe", delivery_day="2026-10-06")
    logged = pd.read_csv(log)
    assert len(logged) == 6 and set(logged["item"]) == {"load_forecast", "wind_solar_forecast", "prices"}
    assert "SECRET" not in log.read_text()
    files = list((tmp_path / "values").iterdir())
    assert len(files) == 1 and "SECRET" not in files[0].read_text()
    assert set(json.loads(files[0].read_text())) == {"load_forecast", "wind_solar_forecast", "prices"}


def test_poll_detects_a_revision_between_polls(tmp_path, client_factory, monkeypatch):
    values = iter([1, 1, 5])
    monkeypatch.setattr(probe_module, "EntsoeApi", lambda cache_dir: make_api(tmp_path, client_factory, make_handler(load_value=next(values))))
    state = {"now": pd.Timestamp("2026-10-06T10:00", tz="Europe/Paris")}
    monkeypatch.setattr(
        probe_module.pd.Timestamp, "now",
        classmethod(lambda cls, tz=None: state["now"].tz_convert(tz) if tz else state["now"].tz_localize(None)),
    )
    lines, sleeps = [], []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        state["now"] = state["now"] + pd.Timedelta(seconds=seconds)

    rows = poll("10:30", every_minutes=15, log_path=tmp_path / "log.csv", snapshot_dir=tmp_path / "values",
                cache_dir=tmp_path / "entsoe", items=("load_forecast",), delivery_day="2026-10-07", log=lines.append, sleep=fake_sleep)
    assert len(rows) == 3 and rows["value_hash"].nunique() == 2
    assert sum("CHANGED" in line for line in lines) == 1 and "CHANGED" in lines[-1]
    assert sleeps == [15 * 60, 15 * 60]
    assert len(list((tmp_path / "values").iterdir())) == 3
