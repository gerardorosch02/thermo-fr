"""The interface every load or price source implements, and a factory to pick one.

A source turns a date range into clean hourly UTC series:

- `load(start, end)` returns a `pd.Series` named "load_mw"
- `day_ahead_prices(start, end)` returns a `pd.Series` named "price_eur_mwh"

Dates are YYYY-MM-DD strings, start inclusive and end exclusive. A source
that only carries one of the two series raises `UnsupportedSeriesError` for
the other, so that the command line can mix sources freely.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

SOURCE_NAMES = ("entsoe", "energy-charts", "rte", "csv")


class UnsupportedSeriesError(RuntimeError):
    """The source does not carry the requested series (for example, RTE has no prices)."""


@runtime_checkable
class Source(Protocol):
    name: str
    attribution: str

    def load(self, start: str, end: str) -> pd.Series:
        """Actual total load in MW, hourly, tz-aware UTC, named "load_mw"."""

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Day-ahead price in EUR/MWh, hourly, tz-aware UTC, named "price_eur_mwh"."""


def date_chunks(start: pd.Timestamp, end: pd.Timestamp, years: int = 1):
    """Split [start, end) into pieces of at most `years` years."""
    cursor = start
    while cursor < end:
        nxt = min(cursor + pd.DateOffset(years=years), end)
        yield cursor, nxt
        cursor = nxt


def clip(series: pd.Series, start: str, end: str) -> pd.Series:
    """Keep the [start, end) window of a UTC-indexed series."""
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return series[(series.index >= lo) & (series.index < hi)]


def get_source(name: str, cache_dir=Path("data/cache"), csv_dir=Path("data/csv"), **options) -> Source:
    """Build a source by name. Imports are local so unused sources cost nothing."""
    if name == "entsoe":
        from .entsoe_client import EntsoeSource

        return EntsoeSource(**options)
    if name == "energy-charts":
        from .energy_charts import EnergyChartsSource

        return EnergyChartsSource(cache_dir=Path(cache_dir) / "energy-charts", **options)
    if name == "rte":
        from .rte_eco2mix import RteEco2mixSource

        return RteEco2mixSource(cache_dir=Path(cache_dir) / "rte", **options)
    if name == "csv":
        from .csv_source import CsvSource

        return CsvSource(csv_dir, **options)
    raise ValueError(f"Unknown source {name!r}; choose from {', '.join(SOURCE_NAMES)}.")
