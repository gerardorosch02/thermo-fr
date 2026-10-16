"""Weekly hydro reservoir series (branch nuclear-hydro): ENTSO-E A72 and the Swiss BFE ogd17 table.

- ENTSO-E "Aggregated filling rate of water reservoirs and hydro storage plants" (document type A72, process A16), one XML
  document per zone and year, weekly points (resolution P7D, curve type A03: a point covers its week and every later week up
  to the next point's position). Weeks run Monday 00:00 to Monday 00:00 Paris time; `week_start` and `week_end` (the Sunday)
  are local calendar dates. The value is the stored energy in MWh as reported for that week.
- BFE ogd17 (Swiss Federal Office of Energy open data): weekly stored energy of the Swiss storage lakes by region, in GWh, with
  the regions' maximum contents. `Datum` is the reading date given by the source (a Monday); it is kept as `date`.

When each value is published is not part of either file; the publication rule is for the pre-registration. Nothing here reads
prices.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

LOCAL_TZ = "Europe/Paris"


def _strip(tag: str) -> str:
    return tag.split("}", 1)[-1]


def parse_a72(path) -> pd.DataFrame:
    """Weekly points of one A72 document: zone, week_start (local date), week_end (the Sunday), stored_mwh, created (UTC)."""
    root = ET.parse(Path(path)).getroot()
    ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}

    def find(node, name):
        return node.find(f"n:{name}", ns) if ns else node.find(name)

    created = pd.Timestamp(find(root, "createdDateTime").text)
    rows = []
    for ts in (root.findall("n:TimeSeries", ns) if ns else root.findall("TimeSeries")):
        zone = find(ts, "inBiddingZone_Domain.mRID").text
        for period in (ts.findall("n:Period", ns) if ns else ts.findall("Period")):
            interval = find(period, "timeInterval")
            start = pd.Timestamp(find(interval, "start").text)
            end = pd.Timestamp(find(interval, "end").text)
            resolution = find(period, "resolution").text
            if resolution != "P7D":
                raise ValueError(f"unexpected A72 resolution {resolution}")
            local_days = (end.tz_convert(LOCAL_TZ).tz_localize(None).normalize() - start.tz_convert(LOCAL_TZ).tz_localize(None).normalize()).days
            n_weeks = int(round(local_days / 7))
            points = {}
            for p in (period.findall("n:Point", ns) if ns else period.findall("Point")):
                points[int(find(p, "position").text)] = float(find(p, "quantity").text)
            value = None
            for pos in range(1, n_weeks + 1):  # curve type A03: a missing position repeats the previous point
                value = points.get(pos, value)
                # weeks counted in local calendar days, so a clock change inside the period does not shift them
                week = start.tz_convert(LOCAL_TZ).tz_localize(None).normalize() + pd.Timedelta(days=7 * (pos - 1))
                rows.append({"zone": zone, "week_start": week, "week_end": week + pd.Timedelta(days=6), "stored_mwh": value,
                             "repeated": pos not in points, "created": created})
    return pd.DataFrame(rows)


def load_a72(paths) -> pd.DataFrame:
    """Every A72 file into one weekly series per zone; a week present in two files keeps the later document's value."""
    frames = [parse_a72(p) for p in sorted(Path(p) for p in paths)]
    frame = pd.concat(frames, ignore_index=True).sort_values(["zone", "week_start", "created"])
    return frame.drop_duplicates(["zone", "week_start"], keep="last").reset_index(drop=True)


def load_bfe_ogd17(path) -> pd.DataFrame:
    """The BFE weekly lake storage table, columns renamed to snake case, `date` the source's reading date."""
    raw = pd.read_csv(Path(path))
    frame = raw.rename(columns={"Datum": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame.columns = [c.lower() for c in frame.columns]
    return frame.sort_values("date").reset_index(drop=True)
