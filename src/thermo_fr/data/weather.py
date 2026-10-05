"""Hourly temperatures from the Open-Meteo archive API (free, no key needed)."""

import json
import urllib.parse
import urllib.request

import pandas as pd

from ..config import CITIES, City

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ATTRIBUTION = "Weather data by Open-Meteo.com, https://open-meteo.com, CC BY 4.0."


def weighted_temperature(frames: dict, weights: dict) -> pd.Series:
    """Population-weighted average across cities.

    Weights are renormalised hour by hour over the cities that have data, so a
    missing reading for one city does not drag the national figure down.
    """
    df = pd.DataFrame(frames)
    w = pd.Series(weights, dtype=float)[df.columns]
    available = df.notna()
    numerator = (df.fillna(0.0) * w).sum(axis=1)
    denominator = (available * w).sum(axis=1)
    out = numerator / denominator.where(denominator > 0)
    return out.rename("temperature")


class OpenMeteoSource:
    def __init__(self, cities=CITIES, timeout: int = 60):
        self.cities = cities
        self.timeout = timeout

    def fetch_city(self, city: City, start: str, end: str) -> pd.Series:
        """Hourly 2m temperature for one city. Dates are YYYY-MM-DD, both inclusive."""
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            payload = json.load(response)
        index = pd.to_datetime(payload["hourly"]["time"], utc=True)
        values = payload["hourly"]["temperature_2m"]
        return pd.Series(values, index=index, name=city.name, dtype=float)

    def fetch(self, start: str, end: str) -> pd.Series:
        frames = {c.name: self.fetch_city(c, start, end) for c in self.cities}
        weights = {c.name: c.weight for c in self.cities}
        return weighted_temperature(frames, weights)
