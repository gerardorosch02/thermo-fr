"""French national load from RTE's eCO2mix data on the ODRE open data platform.

Platform: https://opendata.reseaux-energies.fr (Opendatasoft, Explore API v2.1).
No key is needed. Two datasets carry the national series:

- eco2mix-national-cons-def   "Donnees eCO2mix nationales consolidees et definitives".
  Definitive data (field `nature` = "Donnees definitives") from 2012 until the
  last fully audited year, then consolidated data ("Donnees consolidees") up to
  about the middle of the previous month. Consumption is at a half-hour step;
  the quarter-hour rows carry forecasts only and have an empty `consommation`.
- eco2mix-national-tr         "Donnees eCO2mix nationales temps reel".
  Real-time data at a quarter-hour step for the period after the consolidated
  dataset ends. It is used only for that tail of the requested range.

Which dataset covered which dates is recorded in `details` after a fetch.

Export endpoint (bulk CSV, not paginated records):
  GET /api/explore/v2.1/catalog/datasets/{dataset_id}/exports/csv
      ?select=date_heure,consommation,nature&where=...&order_by=date_heure
      &timezone=UTC&delimiter=;

Timestamps: `date_heure` is a datetime field. With `timezone=UTC` the export
writes it as ISO 8601 with a +00:00 offset and the `where` date literals are
read as UTC, so a filter on [start, end) returns exactly 35,040 quarter-hour
rows for a normal year. The platform maps the French local-time source data
onto UTC itself: on the spring daylight-saving day the instants around the
jump appear twice with identical values (dropped by `to_hourly_utc`), and on
the autumn day the repeated local hour is missing from the source, which
leaves one empty UTC hour. Nothing is filled in.

Units: `consommation` is in MW. Licence: Licence Ouverte v2.0 (Etalab).
"""

import io
from pathlib import Path

import pandas as pd

from .dataset import to_hourly_utc
from .http import FileCache, HttpClient
from .sources import UnsupportedSeriesError, clip, date_chunks

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
CONSOLIDATED = "eco2mix-national-cons-def"
REALTIME = "eco2mix-national-tr"
ATTRIBUTION = (
    "RTE eCO2mix national consumption via ODRE, https://opendata.reseaux-energies.fr, "
    "Licence Ouverte v2.0 (Etalab)."
)


def _literal(ts: pd.Timestamp) -> str:
    """ODSQL date literal in UTC, matching `timezone=UTC` on the export."""
    return "date'" + ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S") + "'"


def parse_export(content: bytes) -> pd.DataFrame:
    """Parse a CSV export into a frame indexed by UTC time with `consommation` and `nature`."""
    frame = pd.read_csv(io.BytesIO(content), sep=";", encoding="utf-8-sig")
    if "date_heure" not in frame.columns:
        raise RuntimeError(f"Unexpected ODRE export columns: {list(frame.columns)}")
    frame["date_heure"] = pd.to_datetime(frame["date_heure"], utc=True)
    frame["consommation"] = pd.to_numeric(frame.get("consommation"), errors="coerce")
    frame = frame.set_index("date_heure").sort_index()
    return frame.dropna(subset=["consommation"])


class RteEco2mixSource:
    """Hourly UTC load from RTE eCO2mix (ODRE). Prices are not available here."""

    name = "rte"
    attribution = ATTRIBUTION

    def __init__(self, cache_dir=Path("data/cache/rte"), client: HttpClient | None = None, chunk_years: int = 1):
        self.cache = FileCache(cache_dir)
        self.client = client or HttpClient(min_interval=1.0)
        self.chunk_years = chunk_years
        self.details: dict = {}

    def coverage_end(self, dataset: str = CONSOLIDATED) -> pd.Timestamp | None:
        """Last timestamp present in `dataset`, or None when it is empty."""
        response = self.client.get(f"{BASE_URL}/{dataset}/records", params={"select": "max(date_heure) as hi"})
        results = response.json().get("results") or [{}]
        hi = results[0].get("hi")
        return pd.Timestamp(hi).tz_convert("UTC") if hi else None

    def _export(self, dataset: str, a: pd.Timestamp, b: pd.Timestamp, key: str) -> pd.DataFrame:
        params = {
            "select": "date_heure,consommation,nature",
            "where": f"date_heure >= {_literal(a)} AND date_heure < {_literal(b)}",
            "order_by": "date_heure",
            "timezone": "UTC",
            "delimiter": ";",
        }

        cached = self.cache.get(key)
        if cached is not None:
            return parse_export(cached)
        content = self.client.get(f"{BASE_URL}/{dataset}/exports/csv", params=params).content
        frame = parse_export(content)
        # ODRE occasionally serves a near-empty export while a dataset is being reprocessed.
        # Such a response is returned but not cached, so the next run tries again.
        expected_rows = (b - a) / pd.Timedelta(hours=1)  # at least hourly data is expected
        if len(frame) < 0.5 * expected_rows:
            print(f"Warning: {dataset} returned {len(frame)} rows for {a.date()} to {b.date()}, "
                  f"expected about {expected_rows:.0f}; not caching this response.")
        else:
            self.cache.put(key, content)
        return frame

    def load(self, start: str, end: str) -> pd.Series:
        """National consumption in MW, averaged to hourly UTC."""
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        cons_end = self.coverage_end(CONSOLIDATED)
        cons_until = min(e, cons_end + pd.Timedelta(minutes=15)) if cons_end is not None else s

        frames, periods = [], []
        for a, b in date_chunks(s, cons_until, self.chunk_years):
            # Chunks that touch the moving end of the consolidated data carry it in their
            # cache key, so the cache refreshes as RTE publishes more consolidated months.
            tag = f"_until_{cons_end.strftime('%Y%m%dT%H%M')}" if b == cons_until and b < e else ""
            frames.append(self._export(CONSOLIDATED, a, b, f"{CONSOLIDATED}_{a.date()}_{b.date()}{tag}.csv"))
            periods.append({"dataset": CONSOLIDATED, "start": str(a), "end": str(b)})
        if cons_until < e:
            a = max(s, cons_until)
            key = f"{REALTIME}_{a.date()}_{e.date()}_{pd.Timestamp.utcnow().strftime('%Y%m%d')}.csv"
            frames.append(self._export(REALTIME, a, e, key))
            periods.append({"dataset": REALTIME, "start": str(a), "end": str(e)})

        data = pd.concat(frames) if frames else parse_export(b"date_heure;consommation;nature\n")
        nature = data["nature"].value_counts().to_dict() if "nature" in data else {}
        self.details["load"] = {
            "datasets": periods,
            "consolidated_data_ends": str(cons_end) if cons_end is not None else None,
            "rows_by_nature": {str(k): int(v) for k, v in nature.items()},
            "unit": "MW",
        }
        hourly = to_hourly_utc(data["consommation"]) if len(data) else pd.Series(dtype=float)
        return clip(hourly, start, end).rename("load_mw")

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        raise UnsupportedSeriesError(
            "RTE eCO2mix carries no day-ahead prices; use --price-source energy-charts, entsoe or csv."
        )
