"""French load and day-ahead prices from the ENTSO-E Transparency Platform."""

import os

import pandas as pd

from ..config import BIDDING_ZONE
from .dataset import to_hourly_utc
from .sources import date_chunks

ATTRIBUTION = "ENTSO-E Transparency Platform, https://transparency.entsoe.eu, via entsoe-py."


def year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Split [start, end) into pieces of at most one year, the API's request limit."""
    return date_chunks(start, end, years=1)


class EntsoeSource:
    """Thin wrapper around entsoe-py that returns clean hourly UTC series."""

    name = "entsoe"
    attribution = ATTRIBUTION

    def __init__(self, api_key: str | None = None, zone: str = BIDDING_ZONE):
        key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not key:
            raise RuntimeError(
                "No ENTSO-E API key. Set the ENTSOE_API_KEY environment variable "
                "(see README for how to request one)."
            )
        from entsoe import EntsoePandasClient  # imported here so tests do not need it

        self.client = EntsoePandasClient(api_key=key)
        self.zone = zone
        self.details: dict = {"zone": zone}

    @staticmethod
    def _bounds(start: str, end: str):
        tz = "Europe/Brussels"  # entsoe-py expects tz-aware timestamps
        return pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)

    def load(self, start: str, end: str) -> pd.Series:
        """Actual total load in MW, averaged to hourly."""
        s, e = self._bounds(start, end)
        parts = []
        for a, b in year_chunks(s, e):
            frame = self.client.query_load(self.zone, start=a, end=b)
            column = "Actual Load" if "Actual Load" in frame.columns else frame.columns[0]
            parts.append(frame[column])
        return to_hourly_utc(pd.concat(parts)).rename("load_mw")

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Day-ahead price in EUR/MWh, averaged to hourly (handles 15-minute products)."""
        s, e = self._bounds(start, end)
        parts = [
            self.client.query_day_ahead_prices(self.zone, start=a, end=b)
            for a, b in year_chunks(s, e)
        ]
        return to_hourly_utc(pd.concat(parts)).rename("price_eur_mwh")
