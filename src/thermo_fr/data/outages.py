"""Daily snapshots of the ENTSO-E unavailability notices of French nuclear units, and planned availability as of a time.

Why snapshots. The Transparency Platform serves only the latest revision of
each outage notice (checked on 2026-10-10: one revision per mRID in every
answer, createdDateTime being that revision's time, earlier revisions not
retrievable). The availability a forecaster knew at a given moment can
therefore not be rebuilt from the API after the fact. It can be rebuilt from
an archive of what the API said each day, which is what this module keeps:
one snapshot per run under data/entsoe/outage_snapshots/ (git-ignored), the
retrieval time in every file name, the raw zip answers untouched.

What is fetched. documentType A80, unavailability of generation units (one
notice per reactor, nominal power per unit), and A77, unavailability of
production units (plant level), for bidding zone FR, from one day back to
360 days ahead (the endpoint refuses a span of a year or more), so that planned outages announced far in advance are
in the archive. The endpoint caps an answer at 200 documents and pages with
an `offset` parameter (checked: 200 then 88 documents for one week of French
notices); every page is saved. The psrType parameter is not honoured by this
endpoint (an answer filtered for B14 carried gas, hydro and wind units too),
so the nuclear filter is applied when the archive is read.

Reading the archive. notices_as_of(as_of) opens the latest snapshot taken at
or before `as_of`, keeps the A80 notices of nuclear units (psrType B14) whose
createdDateTime is at or before `as_of`, drops cancelled and withdrawn
documents (docStatus A09, A13), keeps the highest revision of each mRID and
ignores the retired Fessenheim units, whose perpetual notices would otherwise
count as 1,760 MW of outage for ever. planned_nuclear() turns them into an
hourly unavailable power for a delivery day and an available power against
INSTALLED_NUCLEAR_MW. Only snapshots taken before `as_of` are ever opened:
that is the look-ahead rule, and the test in tests/test_outages.py checks it.

Each Available_Period carries a resolution and points; curve type A03 means
a point holds until the next one, so a single point at PT1M covers the whole
interval. The available quantity is in MW (MAW); unavailable power is the
unit's nominal power minus it.

The API key is read by EntsoeApi from ENTSOE_API_KEY and travels only as a
query parameter; nothing here logs, prints or stores it.
"""

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .entsoe_rest import EIC, EntsoeApi

SNAPSHOT_DIR = Path("data/entsoe/outage_snapshots")
DOC_TYPES = ("A80", "A77")
PAGE = 200  # documents per answer, the platform's cap
MAX_PAGES = 25  # 5,000 documents: the platform's own offset limit is 4,800
DAYS_BACK = 1
DAYS_AHEAD = 360  # with DAYS_BACK the window stays under a year: the endpoint refuses a span of one year or more (checked 2026-10-10)
NUCLEAR = "B14"
CANCELLED = {"A09", "A13"}  # docStatus: cancelled, withdrawn
INSTALLED_NUCLEAR_MW = 63_020  # 56 reactors after Flamanville 3 (1,650 MW) joined the 61,370 MW fleet; a reference level, not a feature
RETIRED_UNITS = ("FESSENHEIM 1", "FESSENHEIM 2")  # closed in 2020, still carrying open-ended notices
STAMP = "%Y%m%dT%H%M%SZ"
STAMP_RE = re.compile(r"^(\d{8}T\d{6}Z)_")
RESOLUTIONS = {"PT1M": pd.Timedelta(minutes=1), "PT15M": pd.Timedelta(minutes=15), "PT30M": pd.Timedelta(minutes=30),
               "PT60M": pd.Timedelta(hours=1), "P1D": pd.Timedelta(days=1)}


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _text(element, name: str) -> str | None:
    """The text of the first descendant whose local tag is `name` or ends with '.' + name (ENTSO-E tags are dotted paths)."""
    for child in element.iter():
        local = _local(child.tag)
        if local == name or local.endswith("." + name):
            return (child.text or "").strip()
    return None


def _children(element, name: str):
    return [c for c in element if _local(c.tag) == name]


@dataclass
class Notice:
    """One unavailability notice as the platform served it in a snapshot."""

    mrid: str
    revision: int
    doc_type: str
    created: pd.Timestamp
    status: str
    business_type: str  # A53 planned, A54 forced
    unit_mrid: str
    unit_name: str
    psr_type: str
    nominal_mw: float
    segments: list = field(default_factory=list)  # (start, end, available_mw)

    @property
    def active(self) -> bool:
        return self.status not in CANCELLED


def parse_notice(content: bytes) -> Notice | None:
    """One Unavailability_MarketDocument; None for an acknowledgement."""
    root = ET.fromstring(content)
    if _local(root.tag) == "Acknowledgement_MarketDocument":
        return None
    series = _children(root, "TimeSeries")
    ts = series[0] if series else root
    segments = []
    for ts_i in series:
        for period in _children(ts_i, "Available_Period"):
            interval = _children(period, "timeInterval")[0]
            start = pd.Timestamp(_text(interval, "start"))
            end = pd.Timestamp(_text(interval, "end"))
            step = RESOLUTIONS.get(_text(period, "resolution") or "", pd.Timedelta(minutes=1))
            points = sorted(((int(_text(p, "position")), float(_text(p, "quantity"))) for p in _children(period, "Point")))
            for i, (position, quantity) in enumerate(points):
                seg_start = start + (position - 1) * step
                seg_end = start + (points[i + 1][0] - 1) * step if i + 1 < len(points) else end
                segments.append((seg_start, min(seg_end, end), quantity))
    status = _text(_children(root, "docStatus")[0], "value") if _children(root, "docStatus") else ""
    return Notice(
        mrid=_text(root, "mRID") or "",
        revision=int(_text(root, "revisionNumber") or 0),
        doc_type=_text(root, "type") or "",
        created=pd.Timestamp(_text(root, "createdDateTime")),
        status=status or "",
        business_type=_text(ts, "businessType") or "",
        unit_mrid=_text(ts, "production_RegisteredResource.mRID") or "",
        unit_name=_text(ts, "production_RegisteredResource.name") or "",
        psr_type=_text(ts, "psrType") or "",
        nominal_mw=float(_text(ts, "nominalP") or 0.0),
        segments=segments,
    )


def parse_answer(content: bytes) -> list[Notice]:
    """Every notice in an answer: a zip of documents, or a single document."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            parsed = [parse_notice(archive.read(info.filename)) for info in archive.infolist()]
    else:
        parsed = [parse_notice(content)]
    return [n for n in parsed if n is not None]


def snapshot(api: EntsoeApi | None = None, out_dir=SNAPSHOT_DIR, now=None, zone: str = "FR", doc_types=DOC_TYPES, days_back: int = DAYS_BACK,
             days_ahead: int = DAYS_AHEAD, log=print) -> dict:
    """Fetch every page of every document type and save the raw answers; returns the manifest, which is also written."""
    api = api or EntsoeApi(zone=zone)
    retrieved = (pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")).floor("s")
    stamp = retrieved.strftime(STAMP)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = (retrieved - pd.Timedelta(days=days_back)).floor("D")
    end = (retrieved + pd.Timedelta(days=days_ahead)).floor("D")
    fmt = "%Y%m%d%H%M"
    manifest = {"retrieved_at_utc": retrieved.strftime("%Y-%m-%dT%H:%M:%SZ"), "zone": zone, "period_start": start.strftime(fmt),
                "period_end": end.strftime(fmt), "files": [], "documents": 0}
    for doc_type in doc_types:
        for page in range(MAX_PAGES):
            params = {"documentType": doc_type, "biddingZone_Domain": EIC.get(zone, zone), "periodStart": start.strftime(fmt),
                      "periodEnd": end.strftime(fmt), "offset": page * PAGE}
            content = api.download_raw(params)
            notices = parse_answer(content)
            if not notices:
                break
            suffix = "zip" if content[:2] == b"PK" else "xml"
            name = f"{stamp}_{doc_type}_{zone}_p{page:02d}.{suffix}"
            (out / name).write_bytes(content)
            manifest["files"].append({"name": name, "doc_type": doc_type, "offset": page * PAGE, "documents": len(notices)})
            manifest["documents"] += len(notices)
            log(f"{doc_type} page {page}: {len(notices)} notices saved as {name}")
            if len(notices) < PAGE:
                break
    (out / f"{stamp}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def snapshot_stamps(snapshot_dir=SNAPSHOT_DIR) -> list[pd.Timestamp]:
    """Retrieval times of the snapshots in the folder, from the file names, oldest first."""
    stamps = set()
    for path in Path(snapshot_dir).glob("*_manifest.json"):
        match = STAMP_RE.match(path.name)
        if match:
            stamps.add(pd.Timestamp(pd.to_datetime(match.group(1), format=STAMP), tz="UTC"))
    return sorted(stamps)


def latest_snapshot_before(as_of, snapshot_dir=SNAPSHOT_DIR) -> pd.Timestamp | None:
    """The most recent snapshot retrieved at or before `as_of`; None when there is none. Later snapshots are never consulted."""
    limit = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of, tz="UTC")
    earlier = [s for s in snapshot_stamps(snapshot_dir) if s <= limit]
    return earlier[-1] if earlier else None


def load_snapshot(stamp: pd.Timestamp, snapshot_dir=SNAPSHOT_DIR) -> list[Notice]:
    """Every notice of one snapshot, from its raw files."""
    prefix = pd.Timestamp(stamp).strftime(STAMP)
    notices = []
    for path in sorted(Path(snapshot_dir).glob(f"{prefix}_*")):
        if path.suffix in (".zip", ".xml"):
            notices.extend(parse_answer(path.read_bytes()))
    return notices


def notices_as_of(as_of, snapshot_dir=SNAPSHOT_DIR, doc_type: str = "A80", psr_type: str = NUCLEAR, retired=RETIRED_UNITS) -> tuple[pd.Timestamp | None, list[Notice]]:
    """The active nuclear notices known at `as_of`: from the latest snapshot taken by then, created by then, highest revision per mRID."""
    stamp = latest_snapshot_before(as_of, snapshot_dir)
    if stamp is None:
        return None, []
    limit = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of, tz="UTC")
    best: dict[str, Notice] = {}
    for n in load_snapshot(stamp, snapshot_dir):
        if n.doc_type != doc_type or n.psr_type != psr_type or n.created > limit or n.unit_name in retired:
            continue
        if n.mrid not in best or n.revision > best[n.mrid].revision:
            best[n.mrid] = n
    return stamp, [n for n in best.values() if n.active]


def delivery_hours(delivery_day) -> pd.DatetimeIndex:
    day = pd.Timestamp(delivery_day).normalize()
    start = day.tz_localize(LOCAL_TZ)
    end = (day + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def unavailability(notices: list[Notice], index: pd.DatetimeIndex) -> pd.Series:
    """Unavailable power in MW for each hour of `index`, summed over notices, pro rata for partial hours."""
    hours = pd.DatetimeIndex(index)
    out = np.zeros(len(hours))
    hour_start = hours.asi8.astype(float)
    hour_end = (hours + pd.Timedelta(hours=1)).asi8.astype(float)
    for n in notices:
        for start, end, available in n.segments:
            unavailable = max(n.nominal_mw - available, 0.0)
            if unavailable == 0.0:
                continue
            s, e = pd.Timestamp(start).value, pd.Timestamp(end).value
            overlap = np.clip(np.minimum(hour_end, e) - np.maximum(hour_start, s), 0, None) / 3.6e12
            out += overlap * unavailable
    return pd.Series(out, index=hours, name="unavailable_mw")


def planned_nuclear(delivery_day, as_of, snapshot_dir=SNAPSHOT_DIR, installed_mw: float = INSTALLED_NUCLEAR_MW, doc_type: str = "A80",
                    psr_type: str = NUCLEAR) -> pd.DataFrame:
    """Hourly planned nuclear unavailability and availability for a delivery day, using only snapshots taken at or before `as_of`.

    Columns unavailable_mw, available_mw (installed_mw minus unavailable),
    notices (how many notices touch the hour). attrs carry the snapshot used
    and the number of notices; an empty frame with the same columns means no
    snapshot existed by `as_of`.
    """
    hours = delivery_hours(delivery_day)
    stamp, notices = notices_as_of(as_of, snapshot_dir, doc_type=doc_type, psr_type=psr_type)
    frame = pd.DataFrame(index=hours)
    if stamp is None:
        frame = frame.reindex(columns=["unavailable_mw", "available_mw", "notices"])
        frame.attrs["snapshot"] = None
        return frame
    frame["unavailable_mw"] = unavailability(notices, hours).round(1)
    frame["available_mw"] = (installed_mw - frame["unavailable_mw"]).round(1)
    counts = np.zeros(len(hours), dtype=int)
    for n in notices:
        touched = np.zeros(len(hours), dtype=bool)
        for start, end, available in n.segments:
            if n.nominal_mw - available > 0:
                touched |= (hours < pd.Timestamp(end)) & (hours + pd.Timedelta(hours=1) > pd.Timestamp(start))
        counts += touched
    frame["notices"] = counts
    frame.attrs["snapshot"] = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame.attrs["notice_count"] = len(notices)
    frame.attrs["as_of"] = str(as_of)
    return frame
