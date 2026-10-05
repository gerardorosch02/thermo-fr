"""French load and day-ahead prices from the Energy-Charts API (Fraunhofer ISE).

No API key is needed. Documentation: https://api.energy-charts.info/ and
https://api.energy-charts.info/llms.txt. Endpoints used:

- /price?bzn=FR   day-ahead spot price in EUR/MWh for the French bidding zone.
  Response fields: unix_seconds, price, unit, license_info, deprecated.
  The FR series is published unchanged from Bundesnetzagentur | SMARD.de under
  CC BY 4.0.
- /public_power?country=fr   public net production per type in MW, including
  a series named "Load". Response fields: unix_seconds, production_types
  (list of {name, data}), deprecated.

Timestamps are given to the API as ISO 8601 UTC and come back as unix seconds.
The API is rate limited (the price endpoint at about 2 requests per minute), so
the date range is split into yearly chunks, requests are spaced out, and every
raw response is cached under data/cache/energy-charts/.

Data license: CC BY 4.0, attribution to Energy-Charts.info required.
"""

import json
from pathlib import Path

import pandas as pd

from ..config import BIDDING_ZONE
from .dataset import to_hourly_utc
from .http import FileCache, HttpClient
from .sources import UnsupportedSeriesError, clip, date_chunks

BASE_URL = "https://api.energy-charts.info"
LOAD_SERIES = "Load"
ATTRIBUTION = (
    "Energy-Charts (Fraunhofer ISE), https://energy-charts.info, licensed CC BY 4.0. "
    "French day-ahead prices originate from Bundesnetzagentur | SMARD.de."
)


def _iso_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M") + "Z"


class EnergyChartsSource:
    """Hourly UTC load and price series from Energy-Charts, cached and rate limited."""

    name = "energy-charts"
    attribution = ATTRIBUTION

    def __init__(
        self,
        cache_dir=Path("data/cache/energy-charts"),
        zone: str = BIDDING_ZONE,
        country: str = "fr",
        client: HttpClient | None = None,
        chunk_years: int = 1,
    ):
        self.zone = zone
        self.country = country
        self.cache = FileCache(cache_dir)
        # About 2 requests per minute are allowed on /price; 30 s spacing stays inside that.
        self.client = client or HttpClient(min_interval=30.0)
        self.chunk_years = chunk_years
        self.details: dict = {}

    def _fetch_json(self, endpoint: str, params: dict, key: str) -> dict:
        def download() -> bytes:
            return self.client.get(f"{BASE_URL}/{endpoint}", params=params).content

        payload = json.loads(self.cache.fetch(key, download))
        if not isinstance(payload, dict) or "unix_seconds" not in payload:
            raise RuntimeError(f"Unexpected Energy-Charts response for {endpoint}: {str(payload)[:200]}")
        return payload

    def _chunks(self, start: str, end: str):
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        for a, b in date_chunks(s, e, self.chunk_years):
            yield a, b, {"start": _iso_utc(a), "end": _iso_utc(b)}

    @staticmethod
    def _series(payload: dict, values, name: str) -> pd.Series:
        index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
        return pd.Series(values, index=index, name=name, dtype=float)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Day-ahead price in EUR/MWh, averaged to hourly (15-minute products since 2025)."""
        parts, units, licenses = [], set(), set()
        for a, b, window in self._chunks(start, end):
            key = f"price_{self.zone}_{a.date()}_{b.date()}.json"
            payload = self._fetch_json("price", {"bzn": self.zone, **window}, key)
            units.add(payload.get("unit"))
            licenses.add(payload.get("license_info"))
            parts.append(self._series(payload, payload["price"], "price_eur_mwh"))
        if units - {"EUR/MWh", None}:
            raise RuntimeError(f"Energy-Charts returned prices in {units}, expected EUR/MWh.")
        self.details["price"] = {
            "endpoint": "/price",
            "bzn": self.zone,
            "unit": sorted(u for u in units if u),
            "license_info": sorted(l for l in licenses if l),
        }
        return clip(to_hourly_utc(pd.concat(parts)), start, end).rename("price_eur_mwh")

    def load(self, start: str, end: str) -> pd.Series:
        """Total load in MW from the "Load" series of /public_power, averaged to hourly."""
        parts = []
        for a, b, window in self._chunks(start, end):
            key = f"public_power_{self.country}_{a.date()}_{b.date()}.json"
            payload = self._fetch_json("public_power", {"country": self.country, **window}, key)
            names = [p["name"] for p in payload.get("production_types", [])]
            match = [p for p in payload.get("production_types", []) if p["name"].lower() == LOAD_SERIES.lower()]
            if not match:
                raise UnsupportedSeriesError(
                    f"No '{LOAD_SERIES}' series in Energy-Charts /public_power for {self.country}; "
                    f"available: {names}"
                )
            parts.append(self._series(payload, match[0]["data"], "load_mw"))
        self.details["load"] = {"endpoint": "/public_power", "country": self.country, "series": LOAD_SERIES, "unit": "MW"}
        return clip(to_hourly_utc(pd.concat(parts)), start, end).rename("load_mw")
