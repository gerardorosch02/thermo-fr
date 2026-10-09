"""Shortwave radiation forecasts, as issued, at points covering France's main solar regions.

French photovoltaic capacity (about 25 GW in 2026) is concentrated in the
south and west: Nouvelle-Aquitaine and Occitanie hold over a third of it,
Provence-Alpes-Cote d'Azur and Auvergne-Rhone-Alpes most of the rest of the
south, with Pays de la Loire, Centre-Val de Loire, Grand Est and Bourgogne
following and the north (Hauts-de-France, Normandy, Ile-de-France, Brittany)
smaller but not negligible. The points below sit in those regions; the
calibration of forecast/solar_proxy.py decides how much each one weighs.

Radiation comes from the Open-Meteo Previous Runs API as `shortwave_radiation`
(global horizontal irradiance, W/m2, hourly mean of the preceding hour), with
the same `previous_day2` convention as weather_forecast.py: every hour of
delivery day D comes from a run of D-2, available before the 12:00 Paris gate
on D-1. The archive holds radiation from 2024-01-20 for the best_match model
(checked live on 2026-10-09 at the Manosque point). The same cache and rate
limiting apply; each point is one request per calendar year.
"""

from pathlib import Path

import pandas as pd

from ..config import City
from .sources import clip
from .weather_forecast import LEAD_DAYS, OpenMeteoForecastSource

ARCHIVE_START = "2024-01-01"  # the previous-runs archive holds no radiation before 2024-01-20

# name, latitude, longitude; the weight field of City is unused here (the calibration fits one per point)
SOLAR_POINTS = (
    City("Landes", 43.90, -0.50, 1.0),  # Mont-de-Marsan, the Landes forest parks
    City("Gironde", 44.85, -0.60, 1.0),  # Bordeaux
    City("LotGaronne", 44.20, 0.60, 1.0),  # Agen
    City("Charente", 45.70, -0.30, 1.0),  # Cognac, northern Nouvelle-Aquitaine
    City("Toulouse", 43.60, 1.45, 1.0),
    City("Herault", 43.50, 3.50, 1.0),  # Montpellier and Beziers
    City("Gard", 43.85, 4.35, 1.0),  # Nimes
    City("Roussillon", 42.70, 2.90, 1.0),  # Perpignan
    City("Provence", 43.50, 5.40, 1.0),  # Aix and Marseille
    City("Var", 43.50, 6.40, 1.0),  # Draguignan
    City("AlpesHP", 43.90, 5.95, 1.0),  # Manosque, Les Mees parks
    City("Drome", 44.90, 4.90, 1.0),  # Valence, Rhone valley
    City("Lyon", 45.75, 4.85, 1.0),
    City("Nantes", 47.20, -1.55, 1.0),
    City("Orleans", 47.90, 1.90, 1.0),  # Centre-Val de Loire
    City("Champagne", 49.25, 4.00, 1.0),  # Reims, Grand Est
    City("Alsace", 48.60, 7.75, 1.0),  # Strasbourg
    City("Bourgogne", 47.30, 5.00, 1.0),  # Dijon
    City("Paris", 48.85, 2.35, 1.0),
    City("Lille", 50.60, 3.10, 1.0),
    City("Rennes", 48.10, -1.70, 1.0),
)


def point_column(name: str) -> str:
    return f"solar_pt_{name}_wm2"


POINT_COLUMNS = [point_column(p.name) for p in SOLAR_POINTS]


class SolarPointsSource(OpenMeteoForecastSource):
    """Hourly UTC shortwave radiation (W/m2) at each point, forecast two days ahead."""

    name = "open-meteo-solar-points"
    cache_prefix = "solarpt_"

    def __init__(self, points=SOLAR_POINTS, cache_dir=Path("data/cache/open-meteo"), client=None, lead_days: int = LEAD_DAYS,
                 model: str = "best_match", now=None):
        super().__init__(cities=points, cache_dir=cache_dir, client=client, lead_days=lead_days, model=model, now=now)
        self.details = {
            "model": model, "lead_days": lead_days, "variable": "shortwave_radiation",
            "points": [{"name": p.name, "lat": p.lat, "lon": p.lon} for p in points],
        }

    def _variables(self, kind: str) -> dict:
        return {f"shortwave_radiation_previous_day{self.lead_days}" if kind == "issued" else "shortwave_radiation": "radiation_wm2"}

    def fetch_points(self, start: str, end: str) -> pd.DataFrame:
        """One column per point for [start, end), NaN before the archive starts."""
        s = max(pd.Timestamp(start), pd.Timestamp(ARCHIVE_START))
        index = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1h", inclusive="left")
        frame = pd.DataFrame(index=index, columns=[point_column(p.name) for p in self.cities], dtype=float)
        if s >= pd.Timestamp(end):
            return frame
        for point in self.cities:
            series = self.fetch_city(point, s.strftime("%Y-%m-%d"), end, "issued")["radiation_wm2"]
            frame[point_column(point.name)] = clip(series, start, end).reindex(index)
        self.details["first_valid"] = {c: (str(frame[c].first_valid_index()) if frame[c].notna().any() else None) for c in frame}
        return frame
