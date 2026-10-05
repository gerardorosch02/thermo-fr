"""Weather forecasts as they were issued, from Open-Meteo (free, no key).

Two Open-Meteo endpoints are used, and the difference matters for a
point-in-time forecast (documentation read on 2026-10-05, facts checked
against live responses):

- Previous Runs API, https://previous-runs-api.open-meteo.com/v1/forecast.
  A variable such as `temperature_2m_previous_day2` is "the value that was
  predicted 48 hours before valid time": for each valid hour it is taken from
  the model run initialised 48 to 53 hours earlier (the run cycle at or
  before valid time minus 48 hours; global models run at 00, 06, 12 and
  18 UTC). This was verified against the Single Runs API: the previous_day1
  values for 2026-09-15 matched the runs of 2026-09-14 at the same 6-hour
  cycle, and previous_day2 the runs of 2026-09-13. Coverage for the French
  cities: temperature from 2021-03-25, 100 m wind speed and shortwave
  radiation from 2024-02-17 (temperature and radiation from 2024-01-20).
  These are the point-in-time inputs. Lead day 2 is used rather than day 1
  because day 1 takes the run issued 24 hours before each hour, which for the
  afternoon and evening of the delivery day is the 12 or 18 UTC run of the
  day before, after the 12:00 Paris auction.

- Historical Forecast API, https://historical-forecast-api.open-meteo.com/v1/forecast.
  "A continuous hourly timeseries built by stitching the first hours of each
  successive model run", so every hour comes from the latest run available
  before it. It is close to the actual weather and is not what a forecaster
  knew the day before. It is fetched as a proxy for the training years before
  the previous-run archive starts, and is never used as a point-in-time input.

Both accept start_date and end_date (inclusive), hourly variable lists,
timezone=UTC and models=best_match. Free use is limited to 600 calls per
minute, 5,000 per hour and 10,000 per day, so requests are spaced out and
cached under data/cache/open-meteo/, one JSON file per city, year, endpoint
and model. Units: temperature in degrees C, wind in km/h (converted to m/s),
radiation in W/m2. Licence CC BY 4.0, attribution to Open-Meteo required.
"""

import json
from pathlib import Path

import pandas as pd

from ..config import CITIES, City
from .http import FileCache, HttpClient
from .sources import clip
from .weather import weighted_temperature

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Weather data by Open-Meteo.com, https://open-meteo.com, CC BY 4.0."

# Open-Meteo variable -> short column stem (units after conversion: C, m/s, W/m2)
VARIABLES = {"temperature_2m": "temp_c", "wind_speed_100m": "wind100_ms", "shortwave_radiation": "radiation_wm2"}
KMH_TO_MS = 1000.0 / 3600.0
LEAD_DAYS = 2
KINDS = ("issued", "proxy")


def column_name(stem: str, kind: str) -> str:
    """temp_c + issued -> temp_fc_c; temp_c + proxy -> temp_proxy_c."""
    base, unit = stem.rsplit("_", 1)
    return f"{base}_{'fc' if kind == 'issued' else 'proxy'}_{unit}"


class OpenMeteoForecastSource:
    """Hourly UTC weather forecasts for the eight cities, aggregated to national series."""

    name = "open-meteo-forecast"
    attribution = ATTRIBUTION

    def __init__(
        self,
        cities=CITIES,
        cache_dir=Path("data/cache/open-meteo"),
        client: HttpClient | None = None,
        lead_days: int = LEAD_DAYS,
        model: str = "best_match",
        now=None,
    ):
        self.cities = cities
        self.cache = FileCache(cache_dir)
        self.client = client or HttpClient(min_interval=0.5, backoff=10.0)
        self.lead_days = lead_days
        self.model = model
        self._now = now
        self.details: dict = {"model": model, "lead_days": lead_days, "cities": [c.name for c in cities]}

    def _variables(self, kind: str) -> dict:
        if kind == "issued":
            return {f"{v}_previous_day{self.lead_days}": stem for v, stem in VARIABLES.items()}
        return dict(VARIABLES)

    def _download(self, kind: str, city: City, start: str, end: str) -> bytes:
        url = PREVIOUS_RUNS_URL if kind == "issued" else HISTORICAL_FORECAST_URL
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(self._variables(kind)),
            "timezone": "UTC",
            "models": self.model,
        }
        return self.client.get(url, params=params).content

    def fetch_city(self, city: City, start: str, end: str, kind: str) -> pd.DataFrame:
        """One city, [start, end) in YYYY-MM-DD, fetched per calendar year and cached."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        today = (self._now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").normalize().tz_localize(None)
        frames = []
        cursor = s
        while cursor < e:
            year_end = min(pd.Timestamp(year=cursor.year + 1, month=1, day=1), e)
            last_day = year_end - pd.Timedelta(days=1)  # Open-Meteo end_date is inclusive
            a, b = cursor.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
            key = f"{kind}_{self.model}_lead{self.lead_days}_{city.name}_{a}_{b}.json"
            download = lambda a=a, b=b: self._download(kind, city, a, b)  # noqa: E731
            content = download() if year_end > today else self.cache.fetch(key, download)
            frames.append(self._parse(content, kind))
            cursor = year_end
        return pd.concat(frames).sort_index()

    def _parse(self, content: bytes, kind: str) -> pd.DataFrame:
        payload = json.loads(content)
        hourly = payload["hourly"]
        index = pd.to_datetime(hourly["time"], utc=True)
        frame = pd.DataFrame(index=index)
        for variable, stem in self._variables(kind).items():
            values = pd.Series(hourly.get(variable, [None] * len(index)), index=index, dtype=float)
            if stem == "wind100_ms":
                values = values * KMH_TO_MS
            frame[stem] = values
        return frame

    def fetch(self, start: str, end: str, kinds=KINDS) -> pd.DataFrame:
        """National series for [start, end): weighted temperature, mean wind and radiation.

        Temperature is population weighted, like the thermosensitivity model, since
        it stands for heating demand. Wind and radiation are plain means over the
        cities: they stand for renewable output, which is not where people live.
        """
        out = {}
        for kind in kinds:
            per_city = {c.name: self.fetch_city(c, start, end, kind) for c in self.cities}
            weights = {c.name: c.weight for c in self.cities}
            temps = {name: f["temp_c"] for name, f in per_city.items()}
            out[column_name("temp_c", kind)] = weighted_temperature(temps, weights)
            for stem in ("wind100_ms", "radiation_wm2"):
                out[column_name(stem, kind)] = pd.DataFrame({n: f[stem] for n, f in per_city.items()}).mean(axis=1)
        frame = pd.DataFrame(out)
        frame = clip(frame, start, end)
        self.details["first_valid"] = {c: (str(frame[c].first_valid_index()) if frame[c].notna().any() else None) for c in frame}
        return frame
