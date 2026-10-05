"""Settings shared across the package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    weight: float  # approximate metropolitan population, millions


# Approximate metro populations, used to weight temperatures so that the
# national figure reflects where heating demand actually is.
CITIES = (
    City("Paris", 48.8566, 2.3522, 12.3),
    City("Lyon", 45.7640, 4.8357, 2.3),
    City("Marseille", 43.2965, 5.3698, 1.9),
    City("Toulouse", 43.6047, 1.4442, 1.5),
    City("Lille", 50.6292, 3.0573, 1.5),
    City("Bordeaux", 44.8378, -0.5792, 1.4),
    City("Nantes", 47.2184, -1.5536, 1.0),
    City("Strasbourg", 48.5734, 7.7521, 0.8),
)

BIDDING_ZONE = "FR"
LOCAL_TZ = "Europe/Paris"

# Candidate heating thresholds (degrees C) searched when fitting: start, stop, step.
THRESHOLD_GRID = (10.0, 20.0, 0.25)
