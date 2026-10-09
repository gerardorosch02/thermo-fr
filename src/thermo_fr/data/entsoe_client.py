"""French load, day-ahead prices and day-ahead forecasts from the ENTSO-E Transparency Platform.

This is the `entsoe` source of the Source protocol, built on the direct REST
client in entsoe_rest.py (no third-party ENTSO-E package is needed). Besides
the two series every source carries, it exposes the day-ahead inputs used by
the price forecast: the total load forecast and the wind and solar forecasts.
"""

from pathlib import Path

import pandas as pd

from ..config import BIDDING_ZONE
from .entsoe_rest import ATTRIBUTION, EntsoeApi


class EntsoeSource:
    """Hourly UTC series from the ENTSO-E RESTful API, cached under data/cache/entsoe/."""

    name = "entsoe"
    attribution = ATTRIBUTION

    def __init__(self, api_key: str | None = None, zone: str = BIDDING_ZONE, cache_dir=Path("data/cache/entsoe"), **options):
        self.api = EntsoeApi(api_key=api_key, zone=zone, cache_dir=cache_dir, **options)
        self.zone = zone

    @property
    def details(self) -> dict:
        return {"zone": self.zone, **self.api.details}

    def load(self, start: str, end: str) -> pd.Series:
        """Actual total load in MW, averaged to hourly."""
        return self.api.load_actual(start, end)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Day-ahead price in EUR/MWh, averaged to hourly (15-minute products are averaged)."""
        return self.api.day_ahead_prices(start, end)

    def load_forecast(self, start: str, end: str) -> pd.Series:
        """Day-ahead total load forecast in MW, hourly."""
        return self.api.load_forecast(start, end)

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        """Day-ahead solar, wind onshore and wind offshore forecasts in MW, hourly."""
        return self.api.wind_solar_forecast(start, end)

    def wind_generation_actual(self, start: str, end: str) -> pd.DataFrame:
        """Actual wind onshore and offshore generation in MW, hourly means."""
        return self.api.wind_generation_actual(start, end)
