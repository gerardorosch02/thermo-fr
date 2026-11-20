r"""One-off probe of the weather sources proposed for the morning collector and the history fetch.

It makes a few dozen small requests, saves every answer untouched under a new folder beside this script, and prints a short
table. It sends no key, no email address and no personal data, and it changes nothing else on the computer. Standard
library only.

    python weather_probe.py

What it asks:
  1. Open-Meteo: when each model's latest run started and when it became available (metadata).
  2. Open-Meteo Single Runs: today's 00 UTC run and yesterday's 18 UTC run of GFS, ICON-EU and ECMWF at one German point.
  3. Open-Meteo Ensemble: the members of the latest GFS and ECMWF ensembles at that point.
  4. dynamical.org: its API documentation, and the historical point queries quoted in the review of 20 November 2026
     (1 January 2022, 00 UTC), then the same query for a recent run.
  5. NOAA's public file listings for today's 00 UTC GFS and GEFS, which carry the time each file arrived.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LAT, LON = 53.55, 9.99  # near Hamburg, the point used in the review
UA = "thermo-fr-weather-probe/1.0 (research; no personal data)"
now = dt.datetime.now(dt.timezone.utc)
stamp = now.strftime("%Y%m%dT%H%M%SZ")
out = Path(__file__).resolve().parent / "probe" / stamp
out.mkdir(parents=True, exist_ok=True)
today00 = now.replace(hour=0, minute=0, second=0, microsecond=0)
if now.hour < 6:  # before about 06 UTC the midnight run may not be out anywhere: probe yesterday's instead
    today00 -= dt.timedelta(days=1)
prev18 = today00 - dt.timedelta(hours=6)
recent = today00 - dt.timedelta(days=3)
ymd = today00.strftime("%Y%m%d")
rows = []


def get(label: str, url: str, note: str = "") -> bytes | None:
    """One GET. The body is saved as it came; the table gets status, seconds and size."""
    started = time.time()
    status, body, error = None, b"", ""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(request, timeout=90) as response:
            status, body = response.status, response.read(60_000_000)
    except urllib.error.HTTPError as exc:
        status, body, error = exc.code, exc.read(200_000), f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001  a source that is down is written down
        error = f"{exc.__class__.__name__}: {exc}"[:200]
    seconds = round(time.time() - started, 2)
    name = f"{len(rows) + 1:02d}_{''.join(c if c.isalnum() else '_' for c in label)[:60]}"
    (out / (name + ".body")).write_bytes(body)
    row = {"n": len(rows) + 1, "label": label, "url": url, "status": status, "seconds": seconds, "bytes": len(body), "error": error,
           "sha256": hashlib.sha256(body).hexdigest(), "file": name + ".body", "note": note, "asked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    try:
        parsed = json.loads(body)
        row["json"] = describe(parsed)
    except Exception:  # noqa: BLE001
        row["starts_with"] = body[:160].decode("utf-8", "replace")
    rows.append(row)
    print(f"{row['n']:>2} {str(status):>4} {seconds:>6}s {len(body):>9} B  {label}  {error}", flush=True)
    time.sleep(0.4)
    return body if status == 200 else None


def describe(value, depth: int = 0):
    """The shape of a JSON answer: keys, list lengths, first and last items of short lists. No bulk values."""
    if isinstance(value, dict):
        if depth >= 3:
            return {"keys": list(value)[:40], "n_keys": len(value)}
        return {k: describe(v, depth + 1) for k, v in list(value.items())[:80]}
    if isinstance(value, list):
        present = [v for v in value if v is not None]
        info = {"len": len(value), "non_null": len(present)}
        if value and not isinstance(value[0], (dict, list)):
            info["first"], info["last"] = value[0], value[-1]
        elif value:
            info["first_item"] = describe(value[0], depth + 1)
        return info
    return value


def q(base: str, **params) -> str:
    return base + "?" + urllib.parse.urlencode(params, safe=",:")


# 1. Open-Meteo metadata: run start and availability times
for host, model in (("api", "ncep_gfs013"), ("api", "ncep_gfs025"), ("api", "dwd_icon_eu"), ("api", "dwd_icon_d2"), ("api", "ecmwf_ifs025"), ("api", "ecmwf_ifs"),
                    ("ensemble-api", "ncep_gefs025"), ("ensemble-api", "ncep_gefs05"), ("ensemble-api", "ecmwf_ifs025_ensemble"), ("ensemble-api", "dwd_icon_eu_eps")):
    get(f"open-meteo meta {model}", f"https://{host}.open-meteo.com/data/{model}/static/meta.json")

# 2. Open-Meteo Single Runs: one explicit run of one named model
point = {"latitude": LAT, "longitude": LON, "hourly": "wind_speed_100m,wind_speed_10m,shortwave_radiation,temperature_2m", "wind_speed_unit": "ms", "timezone": "GMT"}
for model in ("gfs_global", "gfs_seamless", "icon_eu", "icon_seamless", "ecmwf_ifs025", "ecmwf_ifs"):
    for run in (today00, prev18):
        get(f"single run {model} {run:%Y-%m-%dT%H}", q("https://single-runs-api.open-meteo.com/v1/forecast", **point, models=model, run=run.strftime("%Y-%m-%dT%H:%M")))
get("single run ecmwf_ifs 2024-06-01T00 (archive depth)", q("https://single-runs-api.open-meteo.com/v1/forecast", **point, models="ecmwf_ifs", run="2024-06-01T00:00"))
get("single run gfs_global 2026-05-04T00 (archive depth)", q("https://single-runs-api.open-meteo.com/v1/forecast", **point, models="gfs_global", run="2026-05-04T00:00"))
get("previous runs, as the project fetches now", q("https://previous-runs-api.open-meteo.com/v1/forecast", latitude=LAT, longitude=LON, hourly="wind_speed_100m,wind_speed_100m_previous_day1,wind_speed_100m_previous_day2",
                                                    wind_speed_unit="ms", timezone="GMT", past_days=2, forecast_days=3))

# 3. Open-Meteo Ensemble: members of the latest run
for model in ("gfs025", "gfs05", "gfs_seamless", "ecmwf_ifs025", "icon_eu"):
    get(f"ensemble {model}", q("https://ensemble-api.open-meteo.com/v1/ensemble", latitude=LAT, longitude=LON, hourly="wind_speed_100m", wind_speed_unit="ms", timezone="GMT", models=model, forecast_days=3))

# 4. dynamical.org: documentation, then the point queries of the review, then a recent run
get("dynamical api documentation page", "https://dynamical.org/api/")
get("dynamical catalogue page gefs", "https://dynamical.org/catalog/noaa-gefs-forecast-35-day/")
get("dynamical catalogue page gfs", "https://dynamical.org/catalog/noaa-gfs-forecast/")
get("dynamical api root", "https://api.dynamical.org/v1/")
get("dynamical api data list", "https://api.dynamical.org/v1/data")
DATASETS = {"gefs": ("noaa-gefs-forecast-35-day", "DF9V9TYTMCT39M11QH50"), "ifs-ens": ("ecmwf-ifs-ens-forecast-15-day-0-25-degree", "X4WGD034D8KZE219QATG")}
for name, (dataset, snapshot) in DATASETS.items():
    get(f"dynamical dataset {name}", f"https://api.dynamical.org/v1/data/{dataset}")
    for label, init, hours in (("2022-01-01 +24h (the review's query)", "2022-01-01T00:00:00Z", 24), ("2022-01-01 +48h", "2022-01-01T00:00:00Z", 48),
                               (f"{recent:%Y-%m-%d} +48h", recent.strftime("%Y-%m-%dT%H:%M:%SZ"), 48), (f"{today00:%Y-%m-%d} +48h (this morning's run)", today00.strftime("%Y-%m-%dT%H:%M:%SZ"), 48)):
        get(f"dynamical {name} point {label}", q(f"https://api.dynamical.org/v1/data/{dataset}/{snapshot}/points/146,760", variables="wind_u_100m,wind_v_100m", initTime=init, maxLeadTimeHours=hours))
get("dynamical dataset gfs", "https://api.dynamical.org/v1/data/noaa-gfs-forecast")

# 5. NOAA public listings: arrival time of this morning's files
get("noaa gfs listing 00z f02x-f04x", q("https://noaa-gfs-bdp-pds.s3.amazonaws.com/", **{"list-type": 2, "prefix": f"gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f04", "max-keys": 50}))
get("noaa gfs index f024", f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx")
get("noaa gefs listing control", q("https://noaa-gefs-pds.s3.amazonaws.com/", **{"list-type": 2, "prefix": f"gefs.{ymd}/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f04", "max-keys": 50}))
get("noaa gefs index control f024", f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{ymd}/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f024.idx")

summary = {"probe": "thermo-fr weather sources", "started_utc": now.isoformat(timespec="seconds"), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "python": sys.version.split()[0], "platform": sys.platform, "point": [LAT, LON], "run_probed_utc": today00.isoformat(), "requests": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
ok = sum(1 for r in rows if r["status"] == 200)
print(f"\n{ok} of {len(rows)} requests answered 200. Everything is saved in {out}")
