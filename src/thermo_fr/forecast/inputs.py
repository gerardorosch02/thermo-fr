"""Download and store every input of the day-ahead price forecast.

One hourly UTC table, data/forecast/inputs.csv, with these columns:

    price_eur_mwh         ENTSO-E day-ahead price (target and, lagged, a feature)
    load_fc_mw            ENTSO-E day-ahead total load forecast
    solar_fc_mw           ENTSO-E day-ahead solar forecast
    wind_onshore_fc_mw    ENTSO-E day-ahead wind onshore forecast
    wind_offshore_fc_mw   ENTSO-E day-ahead wind offshore forecast (NaN before the first farms)
    temp_fc_c             Open-Meteo temperature as forecast two days ahead, population weighted
    wind100_fc_ms         Open-Meteo 100 m wind speed as forecast two days ahead, city mean
    radiation_fc_wm2      Open-Meteo shortwave radiation as forecast two days ahead, city mean
    temp_proxy_c, wind100_proxy_ms, radiation_proxy_wm2
                          the same from the historical-forecast archive (not point in time)
    wind_onshore_mw       ENTSO-E actual wind onshore generation (calibration target, never a feature)
    wind_offshore_mw      ENTSO-E actual wind offshore generation
    wind_proxy_mw         pre-gate wind generation proxy (forecast/wind_proxy.py), weights refitted monthly
    solar_mw              ENTSO-E actual solar generation (calibration target, never a feature)
    solar_proxy_mw        pre-gate solar generation proxy (forecast/solar_proxy.py), weights refitted monthly
    wind_pt_<point>_ms    Open-Meteo 100 m wind speed as forecast two days ahead at each wind-region point
    solar_pt_<point>_wm2  Open-Meteo shortwave radiation as forecast two days ahead at each solar-region point

Next to it, sources.json records where each series came from, the ENTSO-E
document metadata per year (revision numbers, resolutions) and the first
valid hour of each weather series, and price_comparison.json holds the
comparison between the API prices and the CSV exports in data/csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.csv_source import CsvSource
from ..data.entsoe_client import EntsoeSource
from ..data.sources import UnsupportedSeriesError
from ..data.solar_points import POINT_COLUMNS as SOLAR_POINT_COLUMNS
from ..data.solar_points import SolarPointsSource
from ..data.weather_forecast import OpenMeteoForecastSource
from ..data.wind_points import POINT_COLUMNS, WindPointsSource
from . import solar_proxy, wind_proxy

INPUT_COLUMNS = [
    "price_eur_mwh", "load_fc_mw", "solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw",
    "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2", "temp_proxy_c", "wind100_proxy_ms", "radiation_proxy_wm2",
    "wind_onshore_mw", "wind_offshore_mw", "wind_proxy_mw", "solar_mw", "solar_proxy_mw",
] + POINT_COLUMNS + SOLAR_POINT_COLUMNS


def compare_prices(api: pd.Series, csv: pd.Series, tolerance: float = 0.005) -> dict:
    """Where the API and the CSV exports overlap, how often and by how much they differ."""
    both = pd.concat([api.rename("api"), csv.rename("csv")], axis=1, join="inner").dropna()
    if both.empty:
        return {"overlap_hours": 0}
    diff = (both["api"] - both["csv"]).abs()
    differing = both[diff > tolerance]
    by_year = {
        str(year): {
            "hours": int(len(group)),
            "differing": int((group["api"] - group["csv"]).abs().gt(tolerance).sum()),
            "max_abs_diff": round(float((group["api"] - group["csv"]).abs().max()), 3),
        }
        for year, group in both.groupby(both.index.year)
    }
    only_api = api.dropna().index.difference(csv.dropna().index)
    only_csv = csv.dropna().index.difference(api.dropna().index)
    return {
        "overlap_hours": int(len(both)),
        "differing_hours": int(len(differing)),
        "max_abs_diff": round(float(diff.max()), 3),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "tolerance": tolerance,
        "by_year": by_year,
        "hours_only_in_api": int(len(only_api)),
        "hours_only_in_csv": int(len(only_csv)),
        "first_differences": [
            {"timestamp_utc": str(ts), "api": float(row["api"]), "csv": float(row["csv"])}
            for ts, row in differing.head(10).iterrows()
        ],
    }


def fetch_inputs(start: str, end: str, cache_dir=Path("data/cache"), csv_dir=Path("data/csv"), log=print) -> tuple[pd.DataFrame, dict, dict]:
    """Fetch all inputs for [start, end) (YYYY-MM-DD, end exclusive). Returns (hourly table, sources)."""
    cache_dir = Path(cache_dir)
    entsoe = EntsoeSource(cache_dir=cache_dir / "entsoe")
    log("Fetching ENTSO-E day-ahead prices ...")
    price = entsoe.day_ahead_prices(start, end)
    log("Fetching ENTSO-E day-ahead load forecast ...")
    load_fc = entsoe.load_forecast(start, end)
    log("Fetching ENTSO-E day-ahead wind and solar forecasts ...")
    wind_solar = entsoe.wind_solar_forecast(start, end)

    log("Fetching ENTSO-E actual wind generation (onshore and offshore) ...")
    wind_actual = entsoe.wind_generation_actual(start, end)
    log("Fetching ENTSO-E actual solar generation ...")
    solar_actual = entsoe.solar_generation_actual(start, end)

    weather = OpenMeteoForecastSource(cache_dir=cache_dir / "open-meteo")
    log("Fetching Open-Meteo forecasts as issued (previous runs) and the historical-forecast proxy ...")
    weather_frame = weather.fetch(start, end)
    points_source = WindPointsSource(cache_dir=cache_dir / "open-meteo")
    log("Fetching Open-Meteo 100 m wind forecasts as issued at the wind-region points ...")
    points = points_source.fetch_points(start, end)
    log("Calibrating the wind proxy month by month on actual generation ...")
    proxy, fits = wind_proxy.rolling_proxy(points, wind_actual.sum(axis=1, min_count=1), log=log)
    solar_points_source = SolarPointsSource(cache_dir=cache_dir / "open-meteo")
    log("Fetching Open-Meteo radiation forecasts as issued at the solar-region points ...")
    solar_points = solar_points_source.fetch_points(start, end)
    log("Calibrating the solar proxy month by month on actual generation ...")
    sproxy, solar_fits = solar_proxy.rolling_proxy(solar_points, solar_actual, log=log)

    hourly = pd.concat([price, load_fc, wind_solar, weather_frame, wind_actual, proxy, solar_actual, sproxy, points, solar_points],
                       axis=1).sort_index()
    hourly = hourly.reindex(columns=INPUT_COLUMNS)

    comparison = {"note": "no CSV price export found"}
    try:
        csv_prices = CsvSource(csv_dir).day_ahead_prices(start, end)
        comparison = compare_prices(price, csv_prices)
        log(f"Price comparison with {csv_dir}: {comparison.get('overlap_hours', 0):,} overlapping hours, "
            f"{comparison.get('differing_hours', 0):,} differ by more than {comparison.get('tolerance')} EUR/MWh")
    except (FileNotFoundError, UnsupportedSeriesError) as exc:
        log(f"No CSV prices to compare with ({exc})")

    sources = {
        "start": start,
        "end": end,
        "fetched_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entsoe": {"attribution": entsoe.attribution, "details": entsoe.details},
        "weather": {"source": weather.name, "attribution": weather.attribution, "details": weather.details},
        "wind_points": {"source": points_source.name, "attribution": points_source.attribution, "details": points_source.details},
        "wind_proxy": {"monthly_fits": fits},
        "solar_points": {"source": solar_points_source.name, "attribution": solar_points_source.attribution, "details": solar_points_source.details},
        "solar_proxy": {"monthly_fits": solar_fits},
        "coverage": {c: coverage(hourly[c]) for c in hourly.columns},
    }
    return hourly, sources, comparison


def coverage(series: pd.Series) -> dict:
    present = series.dropna()
    if present.empty:
        return {"hours": 0}
    return {
        "hours": int(len(present)),
        "first": str(present.index.min()),
        "last": str(present.index.max()),
        "missing_between": int(np.sum(series.loc[present.index.min(): present.index.max()].isna())),
    }


def save_inputs(hourly: pd.DataFrame, sources: dict, comparison: dict, out=Path("data/forecast")) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out / "inputs.csv", index_label="timestamp_utc")
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    (out / "price_comparison.json").write_text(json.dumps(comparison, indent=2))
    return out / "inputs.csv"


def load_inputs(path=Path("data/forecast/inputs.csv")) -> pd.DataFrame:
    hourly = pd.read_csv(path, index_col=0)
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    return hourly.reindex(columns=INPUT_COLUMNS)  # tables saved before a column was added get NaN for it
