"""Load and prices from CSV files exported by hand from the ENTSO-E Transparency Platform.

Export "Actual Total Load" (Load > Total Load - Day Ahead / Actual) and
"Day-ahead Prices" (Market > Energy Prices) for the France bidding zone, as
CSV, into one directory. Files are recognised by their header, so names do
not matter and several files per series (one per year, say) are fine.

The platform writes one row per market time unit, with the interval in the
first column. The current export (checked on real files in October 2026)
looks like this, with the time zone named in the header:

    "MTU (UTC)","Area","Sequence","Day-ahead Price (EUR/MWh)","Intraday Period (UTC)","Intraday Price (EUR/MWh)"
    "01/01/2025 00:00:00 - 01/01/2025 00:15:00","BZN|FR","Without Sequence","18.92","",""

Older exports use "MTU (CET/CEST)" or "Time (CET/CEST)" with "dd.mm.yyyy
HH:MM" timestamps and no seconds. Both date styles are accepted. The interval
start is used as the timestamp. When the header says UTC the values are taken
as UTC. When it says CET/CEST they are Paris local time: on the autumn
daylight-saving day the 02:00 interval appears twice in file order, the first
copy is summer time and the second winter time, and the spring day simply has
no 02:00 interval. Both 15-minute and hourly files are accepted and averaged
to hourly UTC. Missing values such as "n/e" or "-" become NaN and are never
filled. Empty files are skipped with a message.

Sequence column: ENTSO-E publishes the single day-ahead coupling result as
"Without Sequence" (or "Sequence 1" on some zones) and any later auction on
the same delivery day as "Sequence 2" and up. Only the first is kept, so the
series is the main day-ahead price.
"""

import csv
import io
import re
from pathlib import Path

import pandas as pd

from ..config import BIDDING_ZONE, LOCAL_TZ
from .dataset import to_hourly_utc
from .sources import UnsupportedSeriesError, clip

ATTRIBUTION = "ENTSO-E Transparency Platform, https://transparency.entsoe.eu, manual CSV export."
TIMESTAMP = re.compile(r"(\d{2})[./](\d{2})[./](\d{4}) (\d{2}:\d{2})")


def localize_cet_cest(naive: pd.Series) -> pd.DatetimeIndex:
    """Localise naive Paris timestamps ("dd.mm.yyyy HH:MM") in file order.

    A timestamp that occurs twice (autumn switch) is summer time the first time
    and winter time the second. A timestamp that does not exist (spring switch)
    becomes NaT rather than being shifted.
    """
    stamps = pd.to_datetime(naive, format="%d.%m.%Y %H:%M")
    first_occurrence = ~pd.Series(stamps).duplicated(keep="first").to_numpy()
    return pd.DatetimeIndex(stamps).tz_localize(LOCAL_TZ, ambiguous=first_occurrence, nonexistent="NaT")


def interval_starts(column: pd.Series) -> pd.Series:
    """Interval start as a naive "dd.mm.yyyy HH:MM" string, from either date style."""
    parts = column.astype(str).str.extract(TIMESTAMP)
    return parts[0] + "." + parts[1] + "." + parts[2] + " " + parts[3]


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
    if _pick(columns, "day-ahead", "price") or _pick(columns, "price"):
        return "price"
    return None


def main_sequence(values: pd.Series) -> str:
    """The sequence label of the main day-ahead auction: "Without Sequence", else the lowest number."""
    labels = sorted(values.dropna().unique())
    if "Without Sequence" in labels:
        return "Without Sequence"

    def number(label: str) -> float:
        found = re.search(r"\d+", label)
        return float(found.group()) if found else float("inf")

    return min(labels, key=number)


def read_entsoe_csv(path) -> tuple[str, pd.Series]:
    """Parse one export. Returns (kind, series) with a tz-aware UTC index at the file's resolution."""
    text = Path(path).read_text(encoding="utf-8-sig")
    frame = pd.read_csv(io.StringIO(text), sep=_sniff_delimiter(text), dtype=str)
    kind = classify(frame.columns)
    time_col = _pick(frame.columns, "mtu") or _pick(frame.columns, "time")
    if kind is None or time_col is None:
        raise ValueError(f"{path}: not an ENTSO-E load or price export (columns {list(frame.columns)})")
    if kind == "load":
        value_col = _pick(frame.columns, "actual", "load")
    else:
        value_col = _pick(frame.columns, "day-ahead", "price") or _pick(frame.columns, "price")
        if "eur/mwh" not in value_col.lower():
            raise ValueError(f"{path}: price column {value_col!r} is not in EUR/MWh")

    area_col, seq_col = _pick(frame.columns, "area"), _pick(frame.columns, "sequence")
    if area_col is not None and frame[area_col].nunique() > 1:
        frame = frame[frame[area_col].str.contains(BIDDING_ZONE, na=False)]
    if seq_col is not None and frame[seq_col].nunique() > 1:
        frame = frame[frame[seq_col] == main_sequence(frame[seq_col])]

    starts = interval_starts(frame[time_col])
    if "utc" in time_col.lower():
        index = pd.DatetimeIndex(pd.to_datetime(starts, format="%d.%m.%Y %H:%M")).tz_localize("UTC")
    else:
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
        parts, files, skipped = [], [], []
        for path in sorted(self.directory.glob("*.csv")):
            if path.stat().st_size == 0:
                print(f"Skipping empty file {path.name}")
                skipped.append(path.name)
                continue
            kind, series = read_entsoe_csv(path)
            if kind == wanted:
                parts.append(series)
                files.append(path.name)
        if not parts:
            raise UnsupportedSeriesError(f"No ENTSO-E {wanted} export found in {self.directory}.")
        raw = pd.concat(parts).sort_index()
        duplicates = int(raw.index.duplicated().sum())
        self.details[wanted] = {"files": files, "skipped_empty": skipped, "duplicate_timestamps_dropped": duplicates}
        return raw[~raw.index.duplicated(keep="first")]

    def load(self, start: str, end: str) -> pd.Series:
        """Actual total load in MW, averaged to hourly."""
        return clip(to_hourly_utc(self._collect("load")), start, end).rename("load_mw")

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Day-ahead price in EUR/MWh, averaged to hourly."""
        return clip(to_hourly_utc(self._collect("price")), start, end).rename("price_eur_mwh")
