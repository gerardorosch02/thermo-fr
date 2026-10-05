"""Load and prices from CSV files exported by hand from the ENTSO-E Transparency Platform.

Export "Actual Total Load" (Load > Total Load - Day Ahead / Actual) and
"Day-ahead Prices" (Market > Energy Prices) for the France bidding zone, as
CSV, into one directory. Files are recognised by their header, so names do
not matter and several files per series (one per year, say) are fine.

The platform writes one row per market time unit, with the interval in the
first column, for example:

    "MTU (CET/CEST)","Day-ahead Price (EUR/MWh)",...
    "01.01.2024 00:00 - 01.01.2024 01:00 (CET/CEST)",68.21,...

Older exports call the column "Time (CET/CEST)" and omit the trailing zone
label. The interval start is parsed as Paris local time. On the autumn
daylight-saving day the 02:00 interval appears twice in file order; the first
copy is summer time and the second winter time, which is how the two are
localised. The spring day simply has no 02:00 interval. Both 15-minute and
hourly files are accepted and averaged to hourly UTC. Missing values such as
"n/e" or "-" become NaN and are never filled.
"""

import csv
import io
import re
from pathlib import Path

import pandas as pd

from ..config import LOCAL_TZ
from .dataset import to_hourly_utc
from .sources import UnsupportedSeriesError, clip

ATTRIBUTION = "ENTSO-E Transparency Platform, https://transparency.entsoe.eu, manual CSV export."
TIMESTAMP = re.compile(r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})")


def localize_cet_cest(naive: pd.Series) -> pd.DatetimeIndex:
    """Localise naive Paris timestamps in file order.

    A timestamp that occurs twice (autumn switch) is summer time the first time
    and winter time the second. A timestamp that does not exist (spring switch)
    becomes NaT rather than being shifted.
    """
    stamps = pd.to_datetime(naive, format="%d.%m.%Y %H:%M")
    first_occurrence = ~pd.Series(stamps).duplicated(keep="first").to_numpy()
    return pd.DatetimeIndex(stamps).tz_localize(LOCAL_TZ, ambiguous=first_occurrence, nonexistent="NaT")


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _pick(columns, *needles: str) -> str | None:
    for column in columns:
        lowered = column.lower()
        if all(n in lowered for n in needles):
            return column
    return None


def classify(columns) -> str | None:
    """"load", "price" or None, from an export's header."""
    if _pick(columns, "actual", "load"):
        return "load"
    if _pick(columns, "price"):
        return "price"
    return None


def read_entsoe_csv(path) -> tuple[str, pd.Series]:
    """Parse one export. Returns (kind, series) with a tz-aware UTC index at the file's resolution."""
    text = Path(path).read_text(encoding="utf-8-sig")
    frame = pd.read_csv(io.StringIO(text), sep=_sniff_delimiter(text), dtype=str)
    kind = classify(frame.columns)
    time_col = _pick(frame.columns, "mtu") or _pick(frame.columns, "time")
    if kind is None or time_col is None:
        raise ValueError(f"{path}: not an ENTSO-E load or price export (columns {list(frame.columns)})")
    value_col = _pick(frame.columns, "actual", "load") if kind == "load" else _pick(frame.columns, "price")

    starts = frame[time_col].astype(str).str.extract(TIMESTAMP)[0]
    index = localize_cet_cest(starts)
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=index, dtype=float)
    series = series[series.index.notna()]
    return kind, series.tz_convert("UTC")


class CsvSource:
    """Hourly UTC series from a directory of ENTSO-E CSV exports."""

    name = "csv"
    attribution = ATTRIBUTION

    def __init__(self, directory=Path("data/csv")):
        self.directory = Path(directory)
        self.details: dict = {}

    def _collect(self, wanted: str) -> pd.Series:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"CSV directory {self.directory} does not exist.")
        parts, files = [], []
        for path in sorted(self.directory.glob("*.csv")):
            kind, series = read_entsoe_csv(path)
            if kind == wanted:
                parts.append(series)
                files.append(path.name)
        if not parts:
            raise UnsupportedSeriesError(f"No ENTSO-E {wanted} export found in {self.directory}.")
        self.details[wanted] = {"files": files}
        return pd.concat(parts)

    def load(self, start: str, end: str) -> pd.Series:
        """Actual total load in MW, averaged to hourly."""
        return clip(to_hourly_utc(self._collect("load")), start, end).rename("load_mw")

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Day-ahead price in EUR/MWh, averaged to hourly."""
        return clip(to_hourly_utc(self._collect("price")), start, end).rename("price_eur_mwh")
