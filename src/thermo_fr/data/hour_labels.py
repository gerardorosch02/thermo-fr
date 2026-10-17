"""Which hour a value is labelled with: the one setting for the solar alignment fix (branch solar-align, 2026-10-17).

Open-Meteo documents two kinds of hourly variable (API documentation, read on 2026-10-17: the hourly parameter definition
tables of the forecast APIs, and the Previous Runs API page, "Solar radiation is averaged over the past hour. Use instant for
radiation at the indicated time"):

- Instant, the value at the labelled time: temperature_2m, wind_speed_10m and the hub-height wind speeds (80, 100, 120 m),
  cloud cover, humidity, pressure.
- Preceding hour, the mean (or sum) over the hour that ends at the label: shortwave_radiation and the other radiation
  variables (direct, diffuse, direct normal, global tilted) as means; precipitation, snowfall and sunshine_duration as sums.

This project fetches three Open-Meteo variables. temperature_2m and wind_speed_100m are instant, and a value at T stands
for [T, T+1) as well as any single time can. shortwave_radiation is a preceding-hour mean, so the value labelled T is the
mean over [T-1, T). ENTSO-E series are labelled at the start of their period (position 1 is the start of the interval, and
the hourly resampling labels each hour at its start), so actual solar labelled T is the generation over [T, T+1). Stored
as received, radiation therefore ran one hour behind generation: in the radiation features radiation_fc_wm2 and
radiation_proxy_wm2 and in the 21 solar_pt_<point>_wm2 columns, hence in the solar proxy.

HOUR_LABELS is the one setting:

- "as_logged" (the default, and what honest_v2, S2 and every logged result used): Open-Meteo values on the labels as
  received, ENTSO-E actual solar as published.
- "start_of_hour" (the fix): every preceding-hour Open-Meteo variable relabelled to the start of its hour, so the value for
  [T, T+1) sits at T in both the history path (forecast-fetch, refit) and the morning-run path; and ENTSO-E actual solar
  relabelled on the days that the physical rule of amendment 2 flags (forecast/solar_labels.py).

The default stays "as_logged" until the corrected models are released, because the published model and proxy files were
fitted on the old labels: switching the setting without refitting them would feed a model inputs one hour away from those
it was trained on. The release changes this constant and refits in the same commit.
"""

import numpy as np
import pandas as pd

AS_LOGGED = "as_logged"
START_OF_HOUR = "start_of_hour"
CHOICES = (AS_LOGGED, START_OF_HOUR)

HOUR_LABELS = AS_LOGGED  # the setting; see the module docstring before changing it

# Open-Meteo variables averaged (or summed) over the preceding hour, among those this project fetches or might fetch
PRECEDING_HOUR_VARIABLES = ("shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
                            "global_tilted_irradiance", "terrestrial_radiation", "precipitation", "rain", "showers", "snowfall",
                            "sunshine_duration")
INSTANT_VARIABLES = ("temperature_2m", "wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m", "cloud_cover")
NIGHT_HOUR_UTC = 23  # the sun is below the horizon everywhere in metropolitan France from 23:00 to 24:00 UTC all year


def current(labels: str | None = None) -> str:
    value = HOUR_LABELS if labels is None else labels
    if value not in CHOICES:
        raise ValueError(f"hour labels must be one of {CHOICES}, not {value!r}")
    return value


def aligned(labels: str | None = None) -> bool:
    return current(labels) == START_OF_HOUR


def preceding_hour_columns() -> list[str]:
    """The inputs-table columns that hold Open-Meteo preceding-hour means: the radiation features and the solar points."""
    from .solar_points import POINT_COLUMNS

    return ["radiation_fc_wm2", "radiation_proxy_wm2", *POINT_COLUMNS]


def valid_offset(labels: str | None = None) -> pd.Timedelta:
    """How far the end of the averaging hour of a preceding-hour value lies after its label: the Open-Meteo valid time that
    the timing rules (forecast/timing.py) must use. 0 as logged (label = valid time), 1 hour on start-of-hour labels."""
    return pd.Timedelta(hours=1) if aligned(labels) else pd.Timedelta(0)


def to_start_of_hour(series: pd.Series, end=None) -> pd.Series:
    """A preceding-hour series (value at T is the mean over [T-1, T)) moved to start-of-hour labels (the same value at T-1).

    The result has the input's index. The label T takes the value received for T+1; where that value is missing it stays
    missing, except at 23:00 UTC, which is night in France all year and is set to 0. `end`, when given, is the exclusive
    end of the requested range: the last label before it has no received value after it and gets the night rule too.
    """
    index = pd.DatetimeIndex(series.index)
    shifted = pd.Series(series.reindex(index + pd.Timedelta(hours=1)).to_numpy(), index=index, name=series.name, dtype=float)
    night = (index.tz_convert("UTC").hour == NIGHT_HOUR_UTC) & np.isnan(shifted.to_numpy()) & series.notna().to_numpy()
    shifted[night] = 0.0
    if end is not None:
        last = pd.Timestamp(end, tz="UTC") - pd.Timedelta(hours=1)
        if last in shifted.index and last.hour == NIGHT_HOUR_UTC and pd.notna(series.get(last)):
            shifted[last] = 0.0
    return shifted


def align_table(table: pd.DataFrame, columns=None) -> pd.DataFrame:
    """An inputs table stored as logged, with every preceding-hour column moved to start-of-hour labels."""
    out = table.copy()
    for column in (columns or preceding_hour_columns()):
        if column in out:
            out[column] = to_start_of_hour(out[column])
    return out
