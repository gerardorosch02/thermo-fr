"""Direct client for the ENTSO-E Transparency Platform RESTful API.

Endpoint: GET https://web-api.tp.entsoe.eu/api?securityToken=...&documentType=...
The token comes from the ENTSOE_API_KEY environment variable. It is sent as a
query parameter and nowhere else: it is not part of cache file names, log
lines or error messages.

Data items used (codes from the platform's RESTful API guide):

- Day-ahead prices        documentType A44, in_Domain = out_Domain = zone
- Day-ahead load forecast documentType A65, processType A01, outBiddingZone_Domain
- Actual total load       documentType A65, processType A16, outBiddingZone_Domain
- Wind and solar forecast documentType A69, processType A01, in_Domain,
                          one TimeSeries per psrType: B16 solar, B18 wind
                          offshore, B19 wind onshore

Limits, as published by ENTSO-E: at most one year per request for these
items, at most 400 requests per minute per token, and a ten minute ban after
a 429. Requests are therefore split into calendar-year chunks and spaced out,
and every raw XML answer is cached under data/cache/entsoe/ so a rerun costs
nothing. A chunk whose end lies in the future is never cached, since the
platform may still fill it in.

Timestamps in the XML are UTC. The response for a historical query is a
document generated at request time: its createdDateTime is the moment the
query ran, not when the data was first published, so it carries no
information about publication timing (checked on 2026-10-05, see
docs/forecast.md). revisionNumber counts resubmissions of the item.

Curve type A03 means "variable sized blocks": a Point is omitted when its
value repeats the previous one, so each Period is expanded to its full
length and forward filled. Periods come at PT60M or PT15M resolution (both
appear in the same document since the move to 15-minute market time units);
everything is averaged to hourly UTC.
"""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import BIDDING_ZONE
from .dataset import to_hourly_utc
from .http import RETRY_STATUSES, FileCache, HttpClient, HttpError
from .sources import clip

BASE_URL = "https://web-api.tp.entsoe.eu/api"
ATTRIBUTION = "ENTSO-E Transparency Platform, https://transparency.entsoe.eu, RESTful API."
EIC = {"FR": "10YFR-RTE------C"}
FREQ = {"PT60M": "60min", "PT30M": "30min", "PT15M": "15min"}
PSR_NAMES = {"B16": "solar", "B19": "wind_onshore", "B18": "wind_offshore"}
NO_DATA_REASON = "999"  # Acknowledgement reason code for "No matching data found"
# The gateway in front of the platform answers 599 "Unable to access service within time limit" or 527 when a
# large query (a year of prices is one TimeSeries per day) takes too long; every 5xx is treated as transient.
ENTSOE_RETRY_STATUSES = RETRY_STATUSES + tuple(range(500, 600))


class EntsoeApiError(RuntimeError):
    """The platform answered with an error document or a non-transient HTTP status."""


@dataclass
class TimeSeriesData:
    series: pd.Series
    resolution: str
    psr_type: str | None = None
    business_type: str | None = None
    curve_type: str | None = None
    meta: dict = field(default_factory=dict)


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _child_text(element, name: str) -> str | None:
    for child in element.iter():
        if _local(child.tag) == name:
            return (child.text or "").strip()
    return None


def _children(element, name: str):
    return [child for child in element if _local(child.tag) == name]


def expand_period(period, curve_type: str | None) -> tuple[pd.Series, str]:
    """One Period element to a series over its full interval at its resolution.

    Missing positions are forward filled for curve type A03 (omitted repeats);
    for other curve types they stay NaN.
    """
    interval = _children(period, "timeInterval")[0]
    start = pd.Timestamp(_child_text(interval, "start"))
    end = pd.Timestamp(_child_text(interval, "end"))
    resolution = _child_text(period, "resolution")
    if resolution not in FREQ:
        raise EntsoeApiError(f"Unsupported resolution {resolution!r}")
    index = pd.date_range(start, end, freq=FREQ[resolution], inclusive="left", tz="UTC")
    values = pd.Series(float("nan"), index=index, dtype=float)
    for point in _children(period, "Point"):
        position = int(_child_text(point, "position"))
        raw = _child_text(point, "quantity")
        if raw is None:
            raw = _child_text(point, "price.amount")
        if 1 <= position <= len(index):
            values.iloc[position - 1] = float(raw)
    if curve_type == "A03":
        values = values.ffill()
    return values, resolution


def parse_document(content: bytes) -> list[TimeSeriesData]:
    """All TimeSeries of a market document. An acknowledgement with "no data" gives []."""
    root = ET.fromstring(content)
    kind = _local(root.tag)
    if kind == "Acknowledgement_MarketDocument":
        reasons = [(_child_text(r, "code"), _child_text(r, "text")) for r in root.iter() if _local(r.tag) == "Reason"]
        if any(code == NO_DATA_REASON for code, _ in reasons):
            return []
        raise EntsoeApiError("ENTSO-E rejected the request: " + "; ".join(f"{c} {t}" for c, t in reasons))
    meta = {
        "created": _child_text(root, "createdDateTime"),
        "revision": _child_text(root, "revisionNumber"),
        "document_type": _child_text(root, "type"),
    }
    out = []
    for ts in _children(root, "TimeSeries"):
        curve_type = _child_text(ts, "curveType")
        for period in _children(ts, "Period"):
            values, resolution = expand_period(period, curve_type)
            out.append(
                TimeSeriesData(
                    series=values,
                    resolution=resolution,
                    psr_type=_child_text(ts, "psrType"),
                    business_type=_child_text(ts, "businessType"),
                    curve_type=curve_type,
                    meta=meta,
                )
            )
    return out


def combine_resolutions(parts: list[TimeSeriesData]) -> pd.Series:
    """Hourly UTC series from pieces at mixed resolutions, finest resolution winning on overlap."""
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    by_resolution: dict[str, list[pd.Series]] = {}
    for part in parts:
        by_resolution.setdefault(part.resolution, []).append(part.series)
    hourly = None
    for resolution in sorted(by_resolution, key=lambda r: pd.Timedelta(FREQ[r])):
        raw = pd.concat(by_resolution[resolution]).sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        piece = to_hourly_utc(raw)
        hourly = piece if hourly is None else hourly.combine_first(piece)
    return hourly.dropna()


def ack_reason(message: str) -> str:
    """Shorten an HTTP error whose body is an Acknowledgement document to its status and reason text.

    A 400 from the platform carries the reason inside XML; keeping the whole
    document in logs and status rows hides the one line that matters.
    """
    if "Acknowledgement_MarketDocument" not in message:
        return message
    status = re.match(r"(HTTP \d+ from \S+)", message)
    reasons = re.findall(r"<code>([^<]*)</code>\s*<text>([^<]*)</text>", message, re.S)
    reason = "; ".join(f"{c.strip()} {t.strip()}" for c, t in reasons) if reasons else "acknowledgement without reason text"
    return f"{status.group(1) if status else 'HTTP error'}: {reason}"


def year_boundaries(start: pd.Timestamp, end: pd.Timestamp):
    """Split [start, end) at calendar-year boundaries, so cache files line up with years."""
    cursor = start
    while cursor < end:
        next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz=cursor.tz)
        nxt = min(next_year, end)
        yield cursor, nxt
        cursor = nxt


class EntsoeApi:
    """Cached, rate-limited access to a few ENTSO-E data items for one bidding zone."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir=Path("data/cache/entsoe"),
        zone: str = BIDDING_ZONE,
        client: HttpClient | None = None,
        now=None,
    ):
        key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not key:
            raise RuntimeError(
                "No ENTSO-E API key. Set the ENTSOE_API_KEY environment variable "
                "(see README for how to request one)."
            )
        self._key = key
        self.zone = zone
        self.eic = EIC.get(zone, zone)
        self.cache = FileCache(cache_dir)
        self.client = client or HttpClient(
            min_interval=0.5, backoff=15.0, max_wait=120.0, timeout=300.0, retry_statuses=ENTSOE_RETRY_STATUSES
        )
        self._now = now  # injectable clock for tests
        self.details: dict = {}

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "<token>")

    def _params(self, item: str) -> dict:
        if item == "prices":
            return {"documentType": "A44", "in_Domain": self.eic, "out_Domain": self.eic}
        if item == "load_forecast":
            return {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": self.eic}
        if item == "load_actual":
            return {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": self.eic}
        if item == "wind_solar_forecast":
            return {"documentType": "A69", "processType": "A01", "in_Domain": self.eic}
        raise ValueError(f"Unknown data item {item!r}")

    def _download(self, params: dict) -> bytes:
        try:
            response = self.client.get(BASE_URL, params={**params, "securityToken": self._key})
        except HttpError as exc:
            raise EntsoeApiError(self._scrub(ack_reason(str(exc)))) from None
        return response.content

    def fetch_chunk(self, item: str, start: pd.Timestamp, end: pd.Timestamp) -> bytes:
        """Raw XML for [start, end), from the cache when the window is fully in the past."""
        fmt = "%Y%m%d%H%M"
        params = {**self._params(item), "periodStart": start.strftime(fmt), "periodEnd": end.strftime(fmt)}
        key = f"{item}_{self.zone}_{start.strftime(fmt)}_{end.strftime(fmt)}.xml"
        now = self._now or pd.Timestamp.now(tz="UTC")
        if end > now:
            return self._download(params)
        return self.cache.fetch(key, lambda: self._download(params))

    def query(self, item: str, start: str, end: str) -> list[TimeSeriesData]:
        """All TimeSeries for [start, end) (YYYY-MM-DD, UTC), fetched in calendar-year chunks."""
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        parts: list[TimeSeriesData] = []
        meta: list[dict] = []
        for a, b in year_boundaries(s, e):
            try:
                chunk = parse_document(self.fetch_chunk(item, a, b))
            except EntsoeApiError as exc:
                raise EntsoeApiError(f"{item} {a.date()} to {b.date()}: {exc}") from None
            if chunk:
                meta.append(
                    {
                        "from": str(a.date()),
                        "to": str(b.date()),
                        **chunk[0].meta,
                        "resolutions": sorted({c.resolution for c in chunk}),
                    }
                )
            parts.extend(chunk)
        self.details[item] = meta
        return parts

    # Public series, all hourly UTC and clipped to [start, end).

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        parts = self.query("prices", start, end)
        return clip(combine_resolutions(parts), start, end).rename("price_eur_mwh")

    def load_forecast(self, start: str, end: str) -> pd.Series:
        parts = self.query("load_forecast", start, end)
        return clip(combine_resolutions(parts), start, end).rename("load_fc_mw")

    def load_actual(self, start: str, end: str) -> pd.Series:
        parts = self.query("load_actual", start, end)
        return clip(combine_resolutions(parts), start, end).rename("load_mw")

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        """Columns solar_fc_mw, wind_onshore_fc_mw, wind_offshore_fc_mw (absent types are NaN)."""
        parts = self.query("wind_solar_forecast", start, end)
        columns = {}
        for psr, name in PSR_NAMES.items():
            series = combine_resolutions([p for p in parts if p.psr_type == psr])
            columns[f"{name}_fc_mw"] = clip(series, start, end)
        return pd.DataFrame(columns)
