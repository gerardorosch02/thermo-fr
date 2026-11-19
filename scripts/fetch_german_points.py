"""Fetch day-ahead weather forecasts, as issued, at points covering Germany's wind and solar fleets.

Standard library only, so it runs with any Python 3.8 or later:

    python fetch_german_points.py --out <folder> [--start 2024-01-01] [--end YYYY-MM-DD]

It mirrors the French point fetch of the project (thermo_fr/data/wind_points.py and solar_points.py): the Open-Meteo Previous
Runs API, model best_match, the value forecast two days ahead (`_previous_day2`: every hour of delivery day D comes from a run
of D-2, available before the morning of D-1), one request per point and calendar year, wind at 100 m converted from km/h to
m/s, global horizontal radiation in W/m2 on the API's own labels. The archive holds these variables from early 2024.

Output, in the folder given: german_wind_points.csv.gz, german_solar_points.csv.gz (hourly UTC, one column per point),
german_points.json (points, request count, sha256 of each file), and a cache folder so a rerun resumes where it stopped.

Weather data by Open-Meteo.com, https://open-meteo.com, CC BY 4.0.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEAD_DAYS = 2
KMH_TO_MS = 1000.0 / 3600.0

# name, latitude, longitude. The calibration fits one weight per point, so the list only has to cover the regions.
WIND_POINTS = (
    ("NordseeKueste", 54.50, 9.00),  # Schleswig-Holstein west coast
    ("Ostholstein", 54.25, 10.60),
    ("Ostfriesland", 53.50, 7.40),
    ("Niedersachsen", 52.75, 8.90),
    ("Mecklenburg", 53.80, 12.20),
    ("Uckermark", 53.20, 13.90),
    ("Lausitz", 51.80, 13.90),
    ("Boerde", 52.00, 11.50),  # Saxony-Anhalt
    ("Paderborn", 51.60, 8.80),  # North Rhine-Westphalia
    ("Hunsrueck", 49.90, 7.40),  # Rhineland-Palatinate
    ("Hessen", 50.60, 9.20),
    ("Thueringen", 51.00, 11.00),
    ("Franken", 49.60, 10.40),
    ("Alb", 48.60, 9.90),  # Baden-Wuerttemberg
    ("OffNordseeWest", 54.30, 6.60),  # German Bight clusters
    ("OffNordseeOst", 54.00, 7.60),
    ("OffOstsee", 54.80, 13.80),
)
SOLAR_POINTS = (
    ("Oberbayern", 48.40, 12.00),
    ("Franken", 49.50, 11.00),
    ("Schwaben", 48.30, 10.50),
    ("Wuerttemberg", 48.50, 9.00),
    ("Brandenburg", 52.30, 13.50),
    ("Sachsen", 51.20, 13.30),
    ("SachsenAnhalt", 51.80, 11.60),
    ("Niedersachsen", 52.60, 9.50),
    ("Rheinland", 51.40, 7.50),
    ("Pfalz", 49.60, 7.80),
    ("Hessen", 50.40, 8.90),
    ("Mecklenburg", 53.60, 12.40),
    ("SchleswigHolstein", 54.20, 9.60),
)
KINDS = {
    "wind": {"points": WIND_POINTS, "variable": f"wind_speed_100m_previous_day{LEAD_DAYS}", "column": "de_wind_pt_{}_ms", "scale": KMH_TO_MS, "file": "german_wind_points.csv.gz"},
    "solar": {"points": SOLAR_POINTS, "variable": f"shortwave_radiation_previous_day{LEAD_DAYS}", "column": "de_solar_pt_{}_wm2", "scale": 1.0, "file": "german_solar_points.csv.gz"},
}


def year_ranges(start: dt.date, end: dt.date):
    """Inclusive (first, last) dates, one pair per calendar year, covering start to end."""
    cursor = start
    while cursor <= end:
        last = min(dt.date(cursor.year, 12, 31), end)
        yield cursor, last
        cursor = last + dt.timedelta(days=1)


def download(lat: float, lon: float, variable: str, first: dt.date, last: dt.date, retries: int = 5) -> bytes:
    query = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": first.isoformat(), "end_date": last.isoformat(),
                                    "hourly": variable, "timezone": "UTC", "models": "best_match"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{URL}?{query}", headers={"User-Agent": "thermo-fr research"}), timeout=180) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    {exc}; retrying in {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse(content: bytes, variable: str, scale: float) -> dict:
    """{timestamp string: value or None}, hourly UTC."""
    hourly = json.loads(content)["hourly"]
    values = hourly.get(variable) or [None] * len(hourly["time"])
    return {t: (None if v is None else v * scale) for t, v in zip(hourly["time"], values)}


def fetch_kind(kind: str, start: dt.date, end: dt.date, out: Path, pause: float) -> dict:
    spec = KINDS[kind]
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date()
    columns, table, requests = [], {}, 0
    for name, lat, lon in spec["points"]:
        column = spec["column"].format(name)
        columns.append(column)
        for first, last in year_ranges(start, end):
            path = cache / f"{kind}_{name}_{first}_{last}.json"
            if path.exists() and last < today:  # a finished year is final; the current one is fetched again
                content = path.read_bytes()
            else:
                content = download(lat, lon, spec["variable"], first, last)
                path.write_bytes(content)
                requests += 1
                time.sleep(pause)
            for stamp, value in parse(content, spec["variable"], spec["scale"]).items():
                table.setdefault(stamp, {})[column] = value
        print(f"  {kind} {name}: done", flush=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["timestamp_utc"] + columns)
    filled = {c: 0 for c in columns}
    for stamp in sorted(table):
        row = [table[stamp].get(c) for c in columns]
        for c, v in zip(columns, row):
            filled[c] += v is not None
        writer.writerow([stamp.replace("T", " ") + ":00+00:00"] + ["" if v is None else f"{v:.3f}" for v in row])
    data = gzip.compress(buffer.getvalue().encode(), mtime=0)
    (out / spec["file"]).write_bytes(data)
    return {"file": spec["file"], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "hours": len(table), "requests": requests,
            "first_hour": min(table) if table else None, "last_hour": max(table) if table else None, "hours_with_value": filled,
            "points": [{"name": n, "lat": la, "lon": lo} for n, la, lo in spec["points"]], "variable": spec["variable"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="German wind and solar weather points, forecasts as issued two days ahead (Open-Meteo)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None, help="last day, inclusive; default yesterday (UTC)")
    parser.add_argument("--pause", type=float, default=0.6, help="seconds between requests")
    args = parser.parse_args(argv)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": "Open-Meteo Previous Runs API, model best_match, lead day 2", "attribution": "Weather data by Open-Meteo.com, https://open-meteo.com, CC BY 4.0.",
            "start": start.isoformat(), "end": end.isoformat(), "fetched_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for kind in KINDS:
        print(f"{kind}: {len(KINDS[kind]['points'])} points, {start} to {end}", flush=True)
        meta[kind] = fetch_kind(kind, start, end, out, args.pause)
    (out / "german_points.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: {x: meta[k][x] for x in ("file", "bytes", "hours", "requests", "first_hour", "last_hour")} for k in KINDS}, indent=1))


if __name__ == "__main__":
    main()
