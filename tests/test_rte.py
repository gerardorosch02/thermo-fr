import numpy as np
import pandas as pd
import pytest

from conftest import FakeResponse, FakeSession
from thermo_fr.data.dataset import daily_frame
from thermo_fr.data.rte_eco2mix import CONSOLIDATED, REALTIME, RteEco2mixSource, parse_export
from thermo_fr.data.sources import UnsupportedSeriesError


def export_csv(start, end, nature="Données définitives", step="30min", value=50000.0, drop=(), dup=()):
    """ODRE-style export: UTC timestamps with +00:00, blank quarter-hour consumption."""
    lines = ["﻿date_heure;consommation;nature"]
    for t in pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left"):
        if t in [pd.Timestamp(d, tz="UTC") for d in drop]:
            continue
        stamp = t.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        cons = "" if (step == "30min" and t.minute % 30) else f"{int(value + t.hour * 100)}"
        lines.append(f"{stamp};{cons};{nature}")
        if t in [pd.Timestamp(d, tz="UTC") for d in dup]:
            lines.append(f"{stamp};{cons};{nature}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_handler(cons_end: str, cons_body=None, tr_body=None):
    def handler(url, params):
        if url.endswith("/records"):
            return FakeResponse(200, {"results": [{"hi": cons_end}]})
        if f"/{CONSOLIDATED}/exports/csv" in url:
            return FakeResponse(200, content=cons_body(params) if callable(cons_body) else cons_body)
        if f"/{REALTIME}/exports/csv" in url:
            return FakeResponse(200, content=tr_body(params) if callable(tr_body) else tr_body)
        raise AssertionError(url)

    return handler


def window(params):
    """Parse the ODSQL where clause back into UTC bounds."""
    a, b = [s.split("date'")[1].split("'")[0] for s in params["where"].split(" AND ")]
    return a, b


def make_source(tmp_path, session, client_factory):
    return RteEco2mixSource(cache_dir=tmp_path / "cache", client=client_factory(session, min_interval=0))


def test_parse_export_handles_bom_blank_rows_and_utc():
    frame = parse_export(export_csv("2024-01-01", "2024-01-01T02:00"))
    assert list(frame.columns) == ["consommation", "nature"]
    assert str(frame.index.tz) == "UTC"
    assert len(frame) == 4  # half-hourly rows only; quarter-hour rows are blank
    assert frame["consommation"].iloc[0] == 50000.0


def test_load_is_hourly_utc_in_mw(tmp_path, client_factory):
    session = FakeSession(make_handler("2025-06-30T21:45:00+00:00", lambda p: export_csv(*window(p))))
    series = make_source(tmp_path, session, client_factory).load("2025-01-01", "2025-01-08")
    assert series.name == "load_mw"
    assert len(series) == 7 * 24 and series.notna().all()
    assert series.iloc[5] == pytest.approx(50500.0)  # 05:00 UTC: mean of two half-hour readings
    url, params = session.calls[1]
    assert params["timezone"] == "UTC" and params["select"] == "date_heure,consommation,nature"
    assert window(params) == ("2025-01-01T00:00:00", "2025-01-08T00:00:00")


def test_only_consolidated_dataset_when_range_is_covered(tmp_path, client_factory):
    session = FakeSession(make_handler("2026-06-30T21:45:00+00:00", lambda p: export_csv(*window(p))))
    source = make_source(tmp_path, session, client_factory)
    source.load("2023-01-01", "2025-01-01")
    datasets = [d["dataset"] for d in source.details["load"]["datasets"]]
    assert datasets == [CONSOLIDATED, CONSOLIDATED]  # one chunk per year
    assert all(CONSOLIDATED in url for url, _ in session.calls[1:])


def test_realtime_dataset_fills_the_tail(tmp_path, client_factory):
    handler = make_handler(
        "2025-06-30T21:45:00+00:00",
        lambda p: export_csv(*window(p)),
        lambda p: export_csv(*window(p), nature="Données temps réel", step="15min"),
    )
    session = FakeSession(handler)
    source = make_source(tmp_path, session, client_factory)
    series = source.load("2025-06-28", "2025-07-03")
    assert len(series) == 5 * 24 and series.notna().all()
    periods = source.details["load"]["datasets"]
    assert [p["dataset"] for p in periods] == [CONSOLIDATED, REALTIME]
    assert periods[0]["end"] == "2025-06-30 22:00:00+00:00"
    assert periods[1]["start"] == "2025-06-30 22:00:00+00:00"
    assert source.details["load"]["rows_by_nature"] == {"Données définitives": 2 * 24 * 2 + 22 * 2, "Données temps réel": (2 * 24 + 2) * 4}


def test_spring_dst_duplicates_are_dropped_and_autumn_gap_is_kept(tmp_path, client_factory):
    """The platform repeats the spring jump instants and omits the autumn repeated hour."""
    body = lambda p: export_csv(
        *window(p), dup=("2024-03-31T01:00", "2024-03-31T01:30"),
        drop=("2024-10-27T00:00", "2024-10-27T00:15", "2024-10-27T00:30", "2024-10-27T00:45"),
    )
    session = FakeSession(make_handler("2026-01-01T00:00:00+00:00", body))
    series = make_source(tmp_path, session, client_factory).load("2024-03-29", "2024-10-30")
    assert not series.index.duplicated().any()
    assert np.isnan(series[pd.Timestamp("2024-10-27 00:00", tz="UTC")])  # the gap is not filled
    assert series.notna().sum() == len(series) - 1
    daily = daily_frame(pd.DataFrame({"temperature": 10.0, "load_mw": series, "price_eur_mwh": 50.0}))
    assert daily.loc["2024-03-31", "hours"] == 23
    assert daily.loc["2024-10-27", "hours"] == 24  # 25 local hours, one missing


def test_cache_prevents_second_download(tmp_path, client_factory):
    session = FakeSession(make_handler("2026-01-01T00:00:00+00:00", lambda p: export_csv(*window(p))))
    make_source(tmp_path, session, client_factory).load("2025-01-01", "2025-01-03")
    make_source(tmp_path, session, client_factory).load("2025-01-01", "2025-01-03")
    exports = [url for url, _ in session.calls if "exports" in url]
    assert len(exports) == 1
    assert (tmp_path / "cache" / f"{CONSOLIDATED}_2025-01-01_2025-01-03.csv").exists()


def test_prices_are_unsupported(tmp_path, client_factory):
    with pytest.raises(UnsupportedSeriesError):
        make_source(tmp_path, FakeSession(), client_factory).day_ahead_prices("2025-01-01", "2025-01-02")


def test_truncated_export_is_returned_but_not_cached(tmp_path, client_factory, capsys):
    """While ODRE reprocesses a dataset an export can come back nearly empty; never cache that."""
    short = lambda p: export_csv(window(p)[0], pd.Timestamp(window(p)[0]) + pd.Timedelta(hours=2))
    session = FakeSession(make_handler("2026-01-01T00:00:00+00:00", short))
    source = make_source(tmp_path, session, client_factory)
    series = source.load("2025-01-01", "2025-01-08")
    assert series.notna().sum() == 2
    assert "not caching" in capsys.readouterr().out
    source.load("2025-01-01", "2025-01-08")
    assert len([u for u, _ in session.calls if "exports" in u]) == 2  # downloaded again
