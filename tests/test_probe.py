"""The timing probe logs presence or absence of tomorrow's items without caching future windows."""

import pandas as pd

from conftest import FakeResponse, FakeSession
from test_entsoe_rest import ack_document, gl_document, gl_timeseries, period_xml
from thermo_fr.data.entsoe_rest import EntsoeApi
from thermo_fr.forecast.probe import probe, run_probe


def handler(url, params):
    start = pd.Timestamp(params["periodStart"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
    end = pd.Timestamp(params["periodEnd"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
    if params["documentType"] == "A69":
        return FakeResponse(200, content=ack_document("999", "No matching data found"))
    ts = gl_timeseries([period_xml(start, end, "PT15M", {1: 1})])
    return FakeResponse(200, content=gl_document([ts], revision="2"))


def test_probe_reports_present_and_absent_items(tmp_path, client_factory):
    api = EntsoeApi(api_key="SECRET-TOKEN", cache_dir=tmp_path / "entsoe", client=client_factory(FakeSession(handler=handler)),
                    now=pd.Timestamp("2026-10-05T08:00Z"))
    rows = pd.DataFrame(probe(api, pd.Timestamp("2026-10-06")))
    by_item = rows.set_index("item")
    assert by_item.loc["load_forecast", "status"] == "present" and by_item.loc["load_forecast", "hours"] == 24
    assert by_item.loc["load_forecast", "revision"] == "2"
    assert by_item.loc["wind_solar_forecast", "status"] == "absent" and by_item.loc["wind_solar_forecast", "hours"] == 0
    assert by_item.loc["prices", "status"] == "present"
    assert not (tmp_path / "entsoe").exists()  # tomorrow's window is never cached


def test_run_probe_appends_to_the_log(tmp_path, client_factory, monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "SECRET-TOKEN")
    monkeypatch.setattr("thermo_fr.forecast.probe.EntsoeApi", lambda cache_dir: EntsoeApi(
        api_key="SECRET-TOKEN", cache_dir=cache_dir, client=client_factory(FakeSession(handler=handler)), now=pd.Timestamp("2026-10-05T08:00Z")))
    log = tmp_path / "probe.csv"
    run_probe(log_path=log, cache_dir=tmp_path / "entsoe", delivery_day="2026-10-06")
    run_probe(log_path=log, cache_dir=tmp_path / "entsoe", delivery_day="2026-10-06")
    logged = pd.read_csv(log)
    assert len(logged) == 6 and set(logged["item"]) == {"load_forecast", "wind_solar_forecast", "prices"}
    assert "SECRET" not in log.read_text()
