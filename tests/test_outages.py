"""Outage snapshots: paging, file names with the retrieval time, the as-of loader and its look-ahead rule, git-ignore."""

import io
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from thermo_fr import cli
from thermo_fr.data import outages
from thermo_fr.data.outages import (
    Notice,
    latest_snapshot_before,
    notices_as_of,
    parse_answer,
    planned_nuclear,
    snapshot,
    snapshot_stamps,
    unavailability,
)

NS = "urn:iec62325.351:tc57wg16:451-6:outagedocument:3:0"


def notice_xml(mrid, revision, created, unit, nominal, start, end, available, psr="B14", status=None, business="A53", doc_type="A80",
               points=None):
    """A minimal Unavailability_MarketDocument in the platform's shape (dotted tag names, one Available_Period)."""
    status_xml = f"<docStatus><value>{status}</value></docStatus>" if status else ""
    points = points or [(1, available)]
    points_xml = "".join(f"<Point><position>{p}</position><quantity>{q}</quantity></Point>" for p, q in points)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Unavailability_MarketDocument xmlns="{NS}">
  <mRID>{mrid}</mRID><revisionNumber>{revision}</revisionNumber><type>{doc_type}</type><process.processType>A26</process.processType>
  <createdDateTime>{created}</createdDateTime>
  <unavailability_Time_Period.timeInterval><start>{start}</start><end>{end}</end></unavailability_Time_Period.timeInterval>
  {status_xml}
  <TimeSeries>
    <mRID>1</mRID><businessType>{business}</businessType><biddingZone_Domain.mRID>10YFR-RTE------C</biddingZone_Domain.mRID>
    <quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name><curveType>A03</curveType>
    <production_RegisteredResource.mRID>17W{unit.replace(' ', '')}</production_RegisteredResource.mRID>
    <production_RegisteredResource.name>{unit}</production_RegisteredResource.name>
    <production_RegisteredResource.pSRType.psrType>{psr}</production_RegisteredResource.pSRType.psrType>
    <production_RegisteredResource.pSRType.powerSystemResources.nominalP>{nominal}</production_RegisteredResource.pSRType.powerSystemResources.nominalP>
    <Available_Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval><resolution>PT1M</resolution>{points_xml}</Available_Period>
  </TimeSeries>
</Unavailability_MarketDocument>"""


def zipped(*documents: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i, doc in enumerate(documents):
            archive.writestr(f"{i:03d}.xml", doc)
    return buffer.getvalue()


ACK = b"""<?xml version="1.0"?><Acknowledgement_MarketDocument xmlns="urn:x"><Reason><code>999</code><text>No matching data found</text></Reason></Acknowledgement_MarketDocument>"""


class FakeApi:
    """Serves pages per document type; records every request; carries no key at all."""

    def __init__(self, pages: dict):
        self.pages, self.requests = pages, []

    def download_raw(self, params):
        self.requests.append(dict(params))
        pages = self.pages.get(params["documentType"], [])
        index = params["offset"] // outages.PAGE
        return pages[index] if index < len(pages) else ACK


def test_parse_reads_dotted_tags_status_and_segments():
    doc = notice_xml("m1", 2, "2026-10-01T06:00:00Z", "UNIT A", 900.0, "2026-10-05T22:00Z", "2026-10-07T22:00Z", 0, status="A09")
    [n] = parse_answer(doc.encode())
    assert n.mrid == "m1" and n.revision == 2 and n.doc_type == "A80" and n.status == "A09" and not n.active
    assert n.unit_name == "UNIT A" and n.psr_type == "B14" and n.nominal_mw == 900.0 and n.business_type == "A53"
    assert n.segments == [(pd.Timestamp("2026-10-05T22:00Z"), pd.Timestamp("2026-10-07T22:00Z"), 0.0)]
    stepped = notice_xml("m2", 1, "2026-10-01T06:00:00Z", "UNIT B", 1300.0, "2026-10-05T00:00Z", "2026-10-05T03:00Z", 0,
                         points=[(1, 0), (121, 650)])  # PT1M: full outage for two hours, then half power
    [s] = parse_answer(zipped(stepped))
    assert s.segments == [(pd.Timestamp("2026-10-05T00:00Z"), pd.Timestamp("2026-10-05T02:00Z"), 0.0),
                          (pd.Timestamp("2026-10-05T02:00Z"), pd.Timestamp("2026-10-05T03:00Z"), 650.0)]
    assert parse_answer(ACK) == []


def test_snapshot_pages_past_the_cap_and_names_files_with_the_retrieval_time(tmp_path, monkeypatch):
    monkeypatch.setattr(outages, "PAGE", 2)
    docs = [notice_xml(f"m{i}", 1, "2026-10-01T06:00:00Z", f"UNIT {i}", 900.0, "2026-10-05T22:00Z", "2026-10-07T22:00Z", 0) for i in range(5)]
    api = FakeApi({"A80": [zipped(*docs[:2]), zipped(*docs[2:4]), zipped(docs[4])], "A77": [zipped(docs[0])]})
    manifest = snapshot(api, out_dir=tmp_path, now=pd.Timestamp("2026-10-10T08:00:05Z"), log=lambda *_: None)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["20261010T080005Z_A77_FR_p00.zip", "20261010T080005Z_A80_FR_p00.zip", "20261010T080005Z_A80_FR_p01.zip",
                     "20261010T080005Z_A80_FR_p02.zip", "20261010T080005Z_manifest.json"]
    assert manifest["documents"] == 6 and [f["offset"] for f in manifest["files"] if f["doc_type"] == "A80"] == [0, 2, 4]
    offsets = [(r["documentType"], r["offset"]) for r in api.requests]
    assert offsets == [("A80", 0), ("A80", 2), ("A80", 4), ("A77", 0)]  # the last A80 page was short, so no fourth request
    assert all(r["biddingZone_Domain"] == "10YFR-RTE------C" and "psrType" not in r and "securityToken" not in r for r in api.requests)
    assert api.requests[0]["periodStart"] == "202610090000" and api.requests[0]["periodEnd"] == "202710050000"
    saved = json.loads((tmp_path / "20261010T080005Z_manifest.json").read_text())
    assert saved == manifest and snapshot_stamps(tmp_path) == [pd.Timestamp("2026-10-10T08:00:05Z")]


@pytest.fixture
def archive(tmp_path):
    """Two snapshots. The first has a planned outage of UNIT A (rev 1, two days) and a cancelled one; the second extends it and adds UNIT B."""
    first = [
        notice_xml("a", 1, "2026-10-08T09:00:00Z", "UNIT A", 900.0, "2026-10-13T22:00Z", "2026-10-15T22:00Z", 0),
        notice_xml("c", 1, "2026-10-08T09:00:00Z", "UNIT C", 1300.0, "2026-10-13T22:00Z", "2026-10-20T22:00Z", 0, status="A09"),
        notice_xml("g", 1, "2026-10-08T09:00:00Z", "GAS UNIT", 400.0, "2026-10-13T22:00Z", "2026-10-20T22:00Z", 0, psr="B04"),
        notice_xml("f", 1, "2020-06-30T00:00:00Z", "FESSENHEIM 1", 880.0, "2020-06-30T00:00Z", "2099-12-31T00:00Z", 0),
        notice_xml("p", 1, "2026-10-08T09:00:00Z", "PLANT A", 1800.0, "2026-10-13T22:00Z", "2026-10-15T22:00Z", 0, doc_type="A77"),
    ]
    second = [
        notice_xml("a", 2, "2026-10-11T07:30:00Z", "UNIT A", 900.0, "2026-10-13T22:00Z", "2026-10-17T22:00Z", 0),
        notice_xml("b", 1, "2026-10-11T07:40:00Z", "UNIT B", 1300.0, "2026-10-14T10:00Z", "2026-10-14T16:30Z", 650),
    ]
    for stamp, docs in (("20261009T080000Z", first), ("20261011T080000Z", second)):
        (tmp_path / f"{stamp}_A80_FR_p00.zip").write_bytes(zipped(*docs))
        (tmp_path / f"{stamp}_manifest.json").write_text("{}")
    return tmp_path


def test_loader_uses_only_snapshots_taken_before_as_of(archive):
    # 10 October 10:05 Paris: only the 9 October snapshot exists by then
    as_of = pd.Timestamp("2026-10-10T08:05Z")
    assert latest_snapshot_before(as_of, archive) == pd.Timestamp("2026-10-09T08:00Z")
    stamp, notices = notices_as_of(as_of, archive)
    assert {n.mrid for n in notices} == {"a"} and notices[0].revision == 1  # cancelled, gas, retired and A77 notices are excluded
    frame = planned_nuclear("2026-10-14", as_of, archive, installed_mw=60_000)
    assert len(frame) == 24 and (frame["unavailable_mw"] == 900.0).all() and (frame["available_mw"] == 59_100.0).all()
    assert frame.attrs["snapshot"] == "2026-10-09T08:00:00Z" and frame.attrs["notice_count"] == 1
    # 16 October: on the 9 October snapshot UNIT A was due back on the 15th; the extension is only in the later snapshot
    assert (planned_nuclear("2026-10-16", as_of, archive)["unavailable_mw"] == 0.0).all()
    # 12 October 10:05 Paris: the 11 October snapshot is in: revision 2 of UNIT A and the partial outage of UNIT B
    later = pd.Timestamp("2026-10-12T08:05Z")
    stamp, notices = notices_as_of(later, archive)
    assert stamp == pd.Timestamp("2026-10-11T08:00Z") and {(n.mrid, n.revision) for n in notices} == {("a", 2), ("b", 1)}
    frame = planned_nuclear("2026-10-14", later, archive, installed_mw=60_000)
    assert frame.loc["2026-10-14T09:00Z", "unavailable_mw"] == 900.0  # before UNIT B's outage starts at 10:00Z
    assert frame.loc["2026-10-14T12:00Z", "unavailable_mw"] == 900.0 + 650.0
    assert frame.loc["2026-10-14T16:00Z", "unavailable_mw"] == pytest.approx(900.0 + 650.0 * 0.5)  # ends 16:30Z: half an hour
    assert frame.loc["2026-10-14T12:00Z", "notices"] == 2 and (planned_nuclear("2026-10-16", later, archive)["unavailable_mw"] == 900.0).all()
    # a snapshot taken at exactly as_of counts, one second later does not; nothing before the first snapshot
    assert latest_snapshot_before(pd.Timestamp("2026-10-11T08:00:00Z"), archive) == pd.Timestamp("2026-10-11T08:00Z")
    assert latest_snapshot_before(pd.Timestamp("2026-10-11T07:59:59Z"), archive) == pd.Timestamp("2026-10-09T08:00Z")
    empty = planned_nuclear("2026-10-14", pd.Timestamp("2026-10-01T00:00Z"), archive)
    assert empty.empty is False and empty["unavailable_mw"].isna().all() and empty.attrs["snapshot"] is None
    # a notice whose creation time is after as_of is ignored even if a snapshot somehow carried it
    doc = notice_xml("z", 1, "2026-10-12T09:00:00Z", "UNIT Z", 900.0, "2026-10-13T22:00Z", "2026-10-15T22:00Z", 0)
    (archive / "20261011T080000Z_A80_FR_p01.zip").write_bytes(zipped(doc))
    assert "z" not in {n.mrid for n in notices_as_of(later, archive)[1]}


def test_unavailability_is_pro_rata_and_never_negative():
    hours = pd.date_range("2026-10-14T00:00Z", periods=3, freq="1h")
    n = Notice("x", 1, "A80", pd.Timestamp("2026-10-01T00:00Z"), "", "A53", "u", "U", "B14", 1000.0,
               [(pd.Timestamp("2026-10-14T00:30Z"), pd.Timestamp("2026-10-14T01:45Z"), 400.0)])
    series = unavailability([n], hours)
    assert series.tolist() == pytest.approx([300.0, 450.0, 0.0])
    over = Notice("y", 1, "A80", pd.Timestamp("2026-10-01T00:00Z"), "", "A53", "u", "U", "B14", 1000.0,
                  [(pd.Timestamp("2026-10-14T00:00Z"), pd.Timestamp("2026-10-14T03:00Z"), 1200.0)])
    assert (unavailability([over], hours) == 0.0).all()


def test_cli_snapshot_and_availability(tmp_path, monkeypatch, capsys):
    docs = [notice_xml("a", 1, "2026-10-08T09:00:00Z", "UNIT A", 900.0, "2026-10-13T22:00Z", "2026-10-15T22:00Z", 0)]
    fake = FakeApi({"A80": [zipped(*docs)], "A77": []})
    monkeypatch.setattr(cli, "_outage_api", lambda zone: fake)
    cli.main(["outage-snapshot", "--out", str(tmp_path), "--now", "2026-10-10T08:00:00Z"])
    out = capsys.readouterr().out
    assert "1 notices" in out and (tmp_path / "20261010T080000Z_A80_FR_p00.zip").exists()
    cli.main(["nuclear-availability", "--date", "2026-10-14", "--as-of", "2026-10-13T08:05:00Z", "--snapshots", str(tmp_path)])
    out = capsys.readouterr().out
    assert "snapshot 2026-10-10T08:00:00Z" in out and "900.0" in out


def test_snapshot_folder_is_git_ignored():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "check-ignore", "-q", str(outages.SNAPSHOT_DIR / "20261010T080000Z_A80_FR_p00.zip")], cwd=repo,
                            capture_output=True, text=True)
    assert result.returncode == 0
