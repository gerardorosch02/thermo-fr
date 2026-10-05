"""ENTSO-E REST client: XML parsing (A03 curves, mixed resolutions), chunking, caching and token hygiene."""

import pandas as pd
import pytest

from conftest import FakeResponse, FakeSession
from thermo_fr.data import entsoe_rest
from thermo_fr.data.entsoe_rest import (
    EntsoeApi,
    EntsoeApiError,
    combine_resolutions,
    parse_document,
    year_boundaries,
)

GL_NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
PUB_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"


def period_xml(start: str, end: str, resolution: str, points: dict, value_tag: str = "quantity") -> str:
    body = "".join(
        f"<Point><position>{pos}</position><{value_tag}>{val}</{value_tag}></Point>" for pos, val in points.items()
    )
    return (
        f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
        f"<resolution>{resolution}</resolution>{body}</Period>"
    )


def gl_document(timeseries: list[str], revision: str = "1") -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><GL_MarketDocument xmlns="{GL_NS}">'
        f"<mRID>x</mRID><revisionNumber>{revision}</revisionNumber><type>A69</type>"
        f"<createdDateTime>2026-10-05T14:41:52Z</createdDateTime>{''.join(timeseries)}</GL_MarketDocument>"
    ).encode()


def gl_timeseries(periods: list[str], psr: str | None = None, curve: str = "A03") -> str:
    psr_xml = f"<MktPSRType><psrType>{psr}</psrType></MktPSRType>" if psr else ""
    return f"<TimeSeries><mRID>1</mRID><businessType>A94</businessType><curveType>{curve}</curveType>{psr_xml}{''.join(periods)}</TimeSeries>"


def price_document(periods: list[str]) -> bytes:
    return (
        f'<?xml version="1.0" encoding="utf-8"?><Publication_MarketDocument xmlns="{PUB_NS}">'
        f"<mRID>p</mRID><revisionNumber>1</revisionNumber><type>A44</type>"
        f"<createdDateTime>2026-10-05T14:41:09Z</createdDateTime>"
        f"<TimeSeries><mRID>1</mRID><curveType>A03</curveType>{''.join(periods)}</TimeSeries></Publication_MarketDocument>"
    ).encode()


def ack_document(code: str, text: str) -> bytes:
    return (
        '<?xml version="1.0"?><Acknowledgement_MarketDocument xmlns="urn:x">'
        f"<Reason><code>{code}</code><text>{text}</text></Reason></Acknowledgement_MarketDocument>"
    ).encode()


def test_a03_curve_forward_fills_omitted_points():
    # Solar: zero until position 7, then values, and position 19 repeats to the end of the day.
    period = period_xml("2024-03-10T00:00Z", "2024-03-11T00:00Z", "PT60M", {1: 0, 7: 1.5, 8: 850, 19: 3})
    parts = parse_document(gl_document([gl_timeseries([period], psr="B16")], revision="3"))
    assert len(parts) == 1
    s = parts[0].series
    assert len(s) == 24 and s.index.tz is not None
    assert s.iloc[0:6].tolist() == [0.0] * 6
    assert s.iloc[6] == 1.5 and s.iloc[7] == 850.0
    assert s.iloc[18:].tolist() == [3.0] * 6
    assert parts[0].psr_type == "B16" and parts[0].meta["revision"] == "3"
    assert parts[0].meta["created"] == "2026-10-05T14:41:52Z"


def test_non_a03_curve_keeps_gaps_as_nan():
    period = period_xml("2024-03-10T00:00Z", "2024-03-11T00:00Z", "PT60M", {1: 10, 3: 30})
    parts = parse_document(gl_document([gl_timeseries([period], curve="A01")]))
    s = parts[0].series
    assert s.iloc[0] == 10 and pd.isna(s.iloc[1]) and s.iloc[2] == 30 and s.iloc[3:].isna().all()


def test_quarter_hours_are_averaged_and_finest_resolution_wins():
    hourly = period_xml("2025-09-30T22:00Z", "2025-10-01T22:00Z", "PT60M", {i: 100 for i in range(1, 25)}, "price.amount")
    quarter = period_xml(
        "2025-09-30T22:00Z", "2025-10-01T22:00Z", "PT15M", {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}, "price.amount"
    )
    parts = parse_document(price_document([hourly, quarter]))
    assert sorted(p.resolution for p in parts) == ["PT15M", "PT60M"]
    s = combine_resolutions(parts)
    assert len(s) == 24
    assert s.iloc[0] == 25.0  # mean of 10, 20, 30, 40 beats the hourly 100
    assert s.iloc[1] == 50.0  # 50 repeated by A03 forward fill
    assert s.index[0] == pd.Timestamp("2025-09-30T22:00Z")


def test_acknowledgement_no_data_is_empty_and_other_reasons_raise():
    assert parse_document(ack_document("999", "No matching data found")) == []
    with pytest.raises(EntsoeApiError, match="Unauthorized"):
        parse_document(ack_document("401", "Unauthorized"))


def test_year_boundaries_split_on_calendar_years():
    chunks = list(year_boundaries(pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC")))
    assert [(str(a.date()), str(b.date())) for a, b in chunks] == [
        ("2023-06-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
        ("2025-01-01", "2025-03-01"),
    ]


def make_api(tmp_path, handler, client_factory, now="2026-10-05T12:00Z"):
    session = FakeSession(handler=handler)
    client = client_factory(session, min_interval=0.0)
    api = EntsoeApi(api_key="SECRET-TOKEN", cache_dir=tmp_path / "entsoe", client=client, now=pd.Timestamp(now))
    return api, session


def wind_solar_handler(url, params):
    start = pd.Timestamp(params["periodStart"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
    end = pd.Timestamp(params["periodEnd"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
    assert params["documentType"] == "A69" and params["processType"] == "A01"
    assert params["in_Domain"] == "10YFR-RTE------C"
    series = [
        gl_timeseries([period_xml(start, end, "PT60M", {1: 100})], psr="B16"),
        gl_timeseries([period_xml(start, end, "PT60M", {1: 200})], psr="B19"),
    ]
    return FakeResponse(200, content=gl_document(series))


def test_requests_are_chunked_by_calendar_year_and_cached_without_the_token(tmp_path, client_factory):
    api, session = make_api(tmp_path, wind_solar_handler, client_factory)
    frame = api.wind_solar_forecast("2024-06-01", "2025-02-01")
    assert len(session.calls) == 2
    assert [c[1]["periodStart"] for c in session.calls] == ["202406010000", "202501010000"]
    assert all(c[1]["securityToken"] == "SECRET-TOKEN" for c in session.calls)
    assert list(frame.columns) == ["solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw"]
    assert len(frame) == len(pd.date_range("2024-06-01", "2025-02-01", freq="1h", inclusive="left"))
    assert frame["solar_fc_mw"].eq(100).all() and frame["wind_onshore_fc_mw"].eq(200).all()
    assert frame["wind_offshore_fc_mw"].isna().all()

    cached = sorted(p.name for p in (tmp_path / "entsoe").iterdir())
    assert cached == [
        "wind_solar_forecast_FR_202406010000_202501010000.xml",
        "wind_solar_forecast_FR_202501010000_202502010000.xml",
    ]
    assert not any("SECRET" in name for name in cached)
    assert all(b"SECRET" not in p.read_bytes() for p in (tmp_path / "entsoe").iterdir())

    api.wind_solar_forecast("2024-06-01", "2025-02-01")
    assert len(session.calls) == 2  # served from the cache
    assert api.details["wind_solar_forecast"][0]["resolutions"] == ["PT60M"]


def test_windows_ending_in_the_future_are_not_cached(tmp_path, client_factory):
    def handler(url, params):
        start = pd.Timestamp(params["periodStart"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
        end = pd.Timestamp(params["periodEnd"]).tz_localize("UTC").strftime("%Y-%m-%dT%H:%MZ")
        ts = gl_timeseries([period_xml(start, end, "PT15M", {1: 40000})])
        return FakeResponse(200, content=gl_document([ts]))

    api, session = make_api(tmp_path, handler, client_factory, now="2026-10-05T12:00Z")
    s = api.load_forecast("2026-10-05", "2026-10-07")
    assert s.name == "load_fc_mw" and len(s) == 48 and s.eq(40000).all()
    assert not (tmp_path / "entsoe").exists() or not any((tmp_path / "entsoe").iterdir())
    api.load_forecast("2026-10-05", "2026-10-07")
    assert len(session.calls) == 2


def test_http_errors_never_leak_the_token(tmp_path, client_factory):
    def handler(url, params):
        return FakeResponse(400, content=b"Bad request: securityToken=SECRET-TOKEN is not valid")

    api, _ = make_api(tmp_path, handler, client_factory)
    with pytest.raises(EntsoeApiError) as excinfo:
        api.day_ahead_prices("2024-01-01", "2024-02-01")
    assert "SECRET-TOKEN" not in str(excinfo.value) and "<token>" in str(excinfo.value)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ENTSOE_API_KEY"):
        EntsoeApi()


def test_prices_clip_to_requested_window(tmp_path, client_factory):
    def handler(url, params):
        # The platform pads to whole local days; here it answers one extra hour on each side.
        start = pd.Timestamp(params["periodStart"]).tz_localize("UTC") - pd.Timedelta(hours=1)
        end = pd.Timestamp(params["periodEnd"]).tz_localize("UTC") + pd.Timedelta(hours=1)
        period = period_xml(
            start.strftime("%Y-%m-%dT%H:%MZ"), end.strftime("%Y-%m-%dT%H:%MZ"), "PT60M", {1: 50}, "price.amount"
        )
        return FakeResponse(200, content=price_document([period]))

    api, _ = make_api(tmp_path, handler, client_factory)
    s = api.day_ahead_prices("2024-03-01", "2024-03-03")
    assert s.index[0] == pd.Timestamp("2024-03-01", tz="UTC") and len(s) == 48
    assert s.name == "price_eur_mwh"
    assert entsoe_rest.BASE_URL.startswith("https://web-api.tp.entsoe.eu")


def test_gateway_timeouts_are_retried(tmp_path, sleeps):
    from thermo_fr.data.http import HttpClient

    answers = iter([FakeResponse(599, content=b'{"message":"Unable to access service within time limit."}'),
                    FakeResponse(200, content=price_document([period_xml("2024-03-01T00:00Z", "2024-03-02T00:00Z", "PT60M", {1: 5}, "price.amount")]))])
    session = FakeSession(handler=lambda url, params: next(answers))
    client = HttpClient(session=session, retries=2, backoff=1.0, retry_statuses=entsoe_rest.ENTSOE_RETRY_STATUSES)
    api = EntsoeApi(api_key="SECRET-TOKEN", cache_dir=tmp_path, client=client, now=pd.Timestamp("2026-10-05T12:00Z"))
    s = api.day_ahead_prices("2024-03-01", "2024-03-02")
    assert len(session.calls) == 2 and len(s) == 24 and s.eq(5).all()
