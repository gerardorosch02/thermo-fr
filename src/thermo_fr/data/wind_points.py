"""Hub-height wind speed forecasts, as issued, at points covering France's main wind regions.

The eight cities of the temperature model are where people live, not where
the turbines are. For a wind generation proxy the forecast is taken at a set
of points in the regions that hold most of the French onshore fleet (Hauts-
de-France, Grand Est, Centre-Val de Loire, Occitanie, Brittany, Pays de la
Loire, Normandy, Nouvelle-Aquitaine, Bourgogne and the Rhone valley) plus the
three offshore farms in operation (Saint-Nazaire, Fecamp, Saint-Brieuc).

Speeds come from the Open-Meteo Previous Runs API at 100 m, the closest
archived level to the hub height of the French fleet (80 to 100 m onshore;
the API also offers 80 m and 120 m, all archived from 2024-02-17 for the
best_match model). As in weather_forecast.py, `previous_day2` is the value
the run initialised 48 to 53 hours before the valid hour predicted, so every
hour of delivery day D comes from a run of D-2, available before the 12:00
Paris gate on D-1. The same cache and rate limiting apply; each point is one
request per calendar year.
"""

from pathlib import Path

import pandas as pd

from ..config import City
from .sources import clip
from .weather_forecast import LEAD_DAYS, OpenMeteoForecastSource

HUB_HEIGHT_M = 100
ARCHIVE_START = "2024-01-01"  # the previous-runs archive holds no wind at these levels before 2024

# name, latitude, longitude; the weight field of City is unused here (the calibration fits one per point)
WIND_POINTS = (
    City("Somme", 49.95, 2.30, 1.0),  # Somme and Pas-de-Calais plateaux, the densest onshore area
    City("Aisne", 49.60, 3.60, 1.0),
    City("Champagne", 48.95, 4.40, 1.0),  # Marne and Aube
    City("Lorraine", 48.80, 5.80, 1.0),  # Meuse and Moselle
    City("Beauce", 48.30, 1.60, 1.0),  # Eure-et-Loir
    City("Indre", 46.90, 1.80, 1.0),
    City("Finistere", 48.40, -3.60, 1.0),
    City("Vendee", 46.80, -1.40, 1.0),
    City("Eure", 49.20, 1.00, 1.0),
    City("Poitou", 46.40, 0.00, 1.0),
    City("Aude", 43.20, 2.90, 1.0),  # Narbonne, tramontane
    City("Lauragais", 43.45, 1.95, 1.0),  # autan wind
    City("Bourgogne", 47.40, 4.60, 1.0),
    City("Rhone", 44.50, 4.70, 1.0),  # Rhone valley, mistral
    City("OffSaintNazaire", 47.15, -2.65, 1.0),
    City("OffFecamp", 49.85, 0.25, 1.0),
    City("OffSaintBrieuc", 48.85, -2.55, 1.0),
)


def point_column(name: str) -> str:
    return f"wind_pt_{name}_ms"


POINT_COLUMNS = [point_column(p.name) for p in WIND_POINTS]


class WindPointsSource(OpenMeteoForecastSource):
    """Hourly UTC 100 m wind speed (m/s) at each point, forecast two days ahead."""

    name = "open-meteo-wind-points"
    cache_prefix = "windpt_"

    def __init__(self, points=WIND_POINTS, cache_dir=Path("data/cache/open-meteo"), client=None, lead_days: int = LEAD_DAYS,
                 model: str = "best_match", now=None):
        super().__init__(cities=points, cache_dir=cache_dir, client=client, lead_days=lead_days, model=model, now=now)
        self.details = {
            "model": model, "lead_days": lead_days, "hub_height_m": HUB_HEIGHT_M,
            "points": [{"name": p.name, "lat": p.lat, "lon": p.lon} for p in points],
        }

    def _variables(self, kind: str) -> dict:
        variable = f"wind_speed_{HUB_HEIGHT_M}m"
        return {f"{variable}_previous_day{self.lead_days}" if kind == "issued" else variable: "wind100_ms"}

    def fetch_points(self, start: str, end: str) -> pd.DataFrame:
        """One column per point for [start, end), NaN before the archive starts."""
        s = max(pd.Timestamp(start), pd.Timestamp(ARCHIVE_START))
        index = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1h", inclusive="left")
        frame = pd.DataFrame(index=index, columns=[point_column(p.name) for p in self.cities], dtype=float)
        if s >= pd.Timestamp(end):
            return frame
        for point in self.cities:
            series = self.fetch_city(point, s.strftime("%Y-%m-%d"), end, "issued")["wind100_ms"]
            frame[point_column(point.name)] = clip(series, start, end).reindex(index)
        self.details["first_valid"] = {c: (str(frame[c].first_valid_index()) if frame[c].notna().any() else None) for c in frame}
        return frame
