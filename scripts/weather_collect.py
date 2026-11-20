r"""Weather collector and history fetch for thermo-fr. Standard library only. Separate from the spread logger and from the
test of W: it reads nothing of theirs and writes only under its own folder (the parent of the snapshot this file sits in).

    python weather_collect.py collect [--dry-run]     one morning snapshot of what each source offers right now
    python weather_collect.py history --product gfs   forecasts of past runs, resumable (also: --product gefs)
    python weather_collect.py pack --product gfs      one compact table from the saved history answers
    python weather_collect.py selftest                 parsing and planning checks, no network
    python weather_collect.py status

Sources, all public, no key, no personal data sent:
  dynamical.org point API   NOAA GFS (0.25 degrees, hourly), NOAA GEFS (31 members, 3-hourly), ECMWF IFS ENS (51 members),
                            by exact run time; CC BY 4.0
  Open-Meteo                when each model's latest run became available, and the 00 UTC runs of ICON-EU and ECMWF IFS
  NOAA public listings      the time each original file of the 00 UTC GFS and GEFS arrived

What a morning snapshot records: which runs each source held at that moment, the forecasts of the 00 UTC run (or, if it was
not there, of the latest run that was, marked as a fallback) of 100 m wind at the German and French wind points, every
answer saved untouched with its checksum, and the time of each request by this computer's clock. One record per day and
slot, never overwritten. Slots follow the Paris clock: "early" before 09:00, "gate" from 09:00 to 10:05, "late" after.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UTC = dt.timezone.utc
UA = "thermo-fr-weather/1.0 (research; no personal data)"
API = "https://api.dynamical.org"
PRODUCTS = {"gfs": "noaa-gfs-forecast", "gefs": "noaa-gefs-forecast-35-day", "ifs_ens": "ecmwf-ifs-ens-forecast-15-day-0-25-degree"}
WIND = ["wind_u_100m", "wind_v_100m"]
MAX_VALUES = 100_000  # the API's ceiling on values in one answer
# name, latitude, longitude: the wind points the project's proxies are fitted on (scripts/fetch_german_points.py, data/wind_points.py)
POINTS = {
    "de": [("NordseeKueste", 54.50, 9.00), ("Ostholstein", 54.25, 10.60), ("Ostfriesland", 53.50, 7.40), ("Niedersachsen", 52.75, 8.90),
           ("Mecklenburg", 53.80, 12.20), ("Uckermark", 53.20, 13.90), ("Lausitz", 51.80, 13.90), ("Boerde", 52.00, 11.50), ("Paderborn", 51.60, 8.80),
           ("Hunsrueck", 49.90, 7.40), ("Hessen", 50.60, 9.20), ("Thueringen", 51.00, 11.00), ("Franken", 49.60, 10.40), ("Alb", 48.60, 9.90),
           ("OffNordseeWest", 54.30, 6.60), ("OffNordseeOst", 54.00, 7.60), ("OffOstsee", 54.80, 13.80)],
    "fr": [("Somme", 49.95, 2.30), ("Aisne", 49.60, 3.60), ("Champagne", 48.95, 4.40), ("Lorraine", 48.80, 5.80), ("Beauce", 48.30, 1.60),
           ("Indre", 46.90, 1.80), ("Finistere", 48.40, -3.60), ("Vendee", 46.80, -1.40), ("Eure", 49.20, 1.00), ("Poitou", 46.40, 0.00),
           ("Aude", 43.20, 2.90), ("Lauragais", 43.45, 1.95), ("Bourgogne", 47.40, 4.60), ("Rhone", 44.50, 4.70),
           ("OffSaintNazaire", 47.15, -2.65), ("OffFecamp", 49.85, 0.25), ("OffSaintBrieuc", 48.85, -2.55)],
}
MEMBERS = {"gfs": 1, "gefs": 31, "ifs_ens": 51}
LEADS = {"gfs": 55, "gefs": 17, "ifs_ens": 17}  # values per variable for the horizons asked below
HORIZON = {"gfs": 54, "gefs": 48, "ifs_ens": 48}  # hours, enough for the whole next delivery day from the 00 UTC run


def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def paris_offset(t: dt.datetime) -> int:
    """Hours Paris is ahead of UTC at instant t: 2 between the last Sundays of March and October at 01:00 UTC, else 1."""
    def last_sunday(year, month):
        day = dt.datetime(year, month, 31, 1, tzinfo=UTC)
        return day - dt.timedelta(days=(day.weekday() + 1) % 7)
    return 2 if last_sunday(t.year, 3) <= t.astimezone(UTC) < last_sunday(t.year, 10) else 1


def slot_of(t: dt.datetime) -> tuple[str, str]:
    """(issue date in Paris, slot) for instant t."""
    paris = t.astimezone(UTC) + dt.timedelta(hours=paris_offset(t))
    minutes = paris.hour * 60 + paris.minute
    return paris.strftime("%Y-%m-%d"), "early" if minutes < 9 * 60 else "gate" if minutes <= 10 * 60 + 5 else "late"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_whole(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(content)
    for attempt in range(10):
        try:
            os.replace(part, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def call(url: str, body: dict | None = None, tries: int = 4, pause: float = 0.7) -> dict:
    """One request. Returns status, body bytes, seconds, times and error text; retries on 429 and on server or network errors."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if data is not None:
        headers["content-type"] = "application/json"
    out = {}
    for attempt in range(tries):
        asked = now_utc()
        started = time.time()
        status, payload, error, wait = None, b"", "", 0.0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as response:
                status, payload = response.status, response.read()
                out["work_units"] = response.headers.get("X-Dynamical-Work-Units")
        except urllib.error.HTTPError as exc:
            status, payload, error = exc.code, exc.read(400_000), f"HTTP {exc.code}"
            wait = float(exc.headers.get("Retry-After") or 0) if exc.headers else 0.0
        except Exception as exc:  # noqa: BLE001  a source that is down is written down
            error = f"{exc.__class__.__name__}: {exc}"[:300]
        out.update({"url": url, "status": status, "body": payload, "seconds": round(time.time() - started, 2), "asked_utc": iso(asked), "answered_utc": iso(now_utc()),
                    "error": error, "attempts": attempt + 1, "request_sha256": sha(data) if data is not None else None})
        time.sleep(pause)
        if status == 200 or (status is not None and status < 500 and status != 429) or attempt == tries - 1:
            break
        time.sleep(max(wait, 5.0 * (attempt + 1)))
    return out


def batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def per_request(product: str, runs: int, leads: int) -> int:
    """How many points fit in one answer under the API's value ceiling (at most 10 queries a request)."""
    values = runs * leads * len(WIND) * MEMBERS[product]
    return max(1, min(10, MAX_VALUES // values))


def query(product: str, name: str, lat: float, lon: float, horizon: int, init: str | None = None, after: str | None = None, before: str | None = None) -> dict:
    q = {"dataProductId": PRODUCTS[product], "location": {"latitude": lat, "longitude": lon}, "variables": WIND, "maxLeadTimeHours": horizon, "queryId": name}
    if init:
        q["initTime"] = init
    else:
        q["initTimesAfter"], q["initTimesBefore"] = after, before
    return q


def speeds(result: dict) -> list[dict]:
    """One answer of the point API for one point -> rows {init, lead_h, valid, member, speed}. Speed is hypot(u, v) in m/s."""
    rows = []
    for forecast in result.get("forecasts", []):
        members = forecast.get("ensembleMembers") or [{"member": 0, "data": forecast.get("data", {})}]
        for member in members:
            u, v = member["data"].get("wind_u_100m", []), member["data"].get("wind_v_100m", [])
            for lead, valid, a, b in zip(forecast["leadTimeHours"], forecast["validTimes"], u, v):
                rows.append({"init": forecast["initTime"], "lead_h": int(round(lead)), "valid": valid, "member": member["member"],
                             "speed": None if a is None or b is None else round(math.hypot(a, b), 2)})
    return rows


# ---- the morning snapshot --------------------------------------------------------------------------------------------------

def collect(root: Path, dry_run: bool = False, clock=now_utc, fetch=call, log=print) -> dict:
    started = clock()
    day, slot = slot_of(started)
    base = root / "live" / ("dry_run" if dry_run else "") / day
    record_path = base / f"{slot}.json"
    if record_path.exists() and not dry_run:
        log(f"{day} {slot}: already recorded")
        return {"status": "already recorded", "record": str(record_path)}
    folder = base / (slot if not dry_run else f"{slot}_{started:%H%M%S}")
    target = started.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    record = {"issue_day_paris": day, "slot": slot, "started_utc": iso(started), "target_run_utc": iso(target), "dry_run": dry_run, "requests": [], "flags": [],
              "gate_utc": iso(dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC) + dt.timedelta(hours=10, minutes=5) - dt.timedelta(hours=paris_offset(started))),
              "clock": "every time in this record is this computer's own clock", "code_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")), "products": {}}

    def keep(label: str, answer: dict) -> bytes | None:
        name = f"{len(record['requests']) + 1:02d}_{''.join(c if c.isalnum() else '_' for c in label)[:70]}.body"
        write_whole(folder / name, answer["body"])
        record["requests"].append({"label": label, "file": name, "sha256": sha(answer["body"]), "bytes": len(answer["body"]),
                                   **{k: answer.get(k) for k in ("url", "status", "seconds", "asked_utc", "answered_utc", "error", "attempts", "request_sha256", "work_units")}})
        if answer["status"] != 200:
            record["flags"].append(f"{label}: {answer['error'] or answer['status']}")
        return answer["body"] if answer["status"] == 200 else None

    points = [(f"{zone}_{name}", lat, lon) for zone in ("de", "fr") for name, lat, lon in POINTS[zone]]
    for product, product_id in PRODUCTS.items():
        info = {"runs_listed": [], "run_used": None, "fallback": None, "points_answered": 0, "values": 0, "missing_values": 0}
        record["products"][product] = info
        listing = keep(f"runs {product}", fetch(f"{API}/v1/forecasts/runs?dataProductId={product_id}&limit=8"))
        if listing is None:
            continue
        listed = json.loads(listing)
        info["snapshot_id"], info["runs_listed"] = listed.get("snapshotId"), [r["initTime"] for r in listed.get("runs", [])]
        midnight = [r for r in info["runs_listed"] if r <= iso(target) and r.endswith("T00:00:00Z")]
        if not midnight:
            record["flags"].append(f"{product}: no 00 UTC run listed")
            continue
        info["run_used"], info["fallback"] = midnight[0], midnight[0] != iso(target)
        if info["fallback"]:
            record["flags"].append(f"{product}: the 00 UTC run of today was not there, the run of {midnight[0]} was saved instead")
        for group in batches(points, per_request(product, 1, LEADS[product])):
            body = {"queries": [query(product, name, lat, lon, HORIZON[product], init=info["run_used"]) for name, lat, lon in group]}
            answer = keep(f"forecast {product} {group[0][0]}", fetch(f"{API}/v1/forecasts", body))
            if answer is None:
                continue
            for result in json.loads(answer).get("results", []):
                rows = speeds(result)
                info["points_answered"] += 1
                info["values"] += len(rows)
                info["missing_values"] += sum(r["speed"] is None for r in rows)
        if info["points_answered"] < len(points):
            record["flags"].append(f"{product}: {info['points_answered']} of {len(points)} points answered")
        log(f"{product}: run {info['run_used']}{' (fallback)' if info['fallback'] else ''}, {info['points_answered']} points, {info['values']} values")

    record["open_meteo_meta"] = {}
    for host, model in (("api", "ncep_gfs013"), ("api", "ncep_gfs025"), ("api", "dwd_icon_eu"), ("api", "dwd_icon_d2"), ("api", "ecmwf_ifs"), ("api", "ecmwf_ifs025"),
                        ("ensemble-api", "ncep_gefs05"), ("ensemble-api", "ecmwf_ifs025_ensemble")):
        answer = keep(f"open-meteo meta {model}", fetch(f"https://{host}.open-meteo.com/data/{model}/static/meta.json", pause=0.2))
        if answer is not None:
            meta = json.loads(answer)
            record["open_meteo_meta"][model] = {k: iso(dt.datetime.fromtimestamp(meta[k], UTC)) for k in ("last_run_initialisation_time", "last_run_availability_time") if k in meta}
    place = {"latitude": ",".join(str(lat) for _, lat, _ in points), "longitude": ",".join(str(lon) for _, _, lon in points)}
    record["open_meteo_runs"] = {}
    for model in ("icon_eu", "ecmwf_ifs"):
        url = "https://single-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
            {**place, "hourly": "wind_speed_100m", "wind_speed_unit": "ms", "timezone": "GMT", "models": model, "run": target.strftime("%Y-%m-%dT%H:%M")}, safe=",:")
        answer = keep(f"open-meteo single run {model}", fetch(url))
        if answer is not None:
            places = json.loads(answer)
            places = places if isinstance(places, list) else [places]
            present = [sum(v is not None for v in p.get("hourly", {}).get("wind_speed_100m", [])) for p in places]
            record["open_meteo_runs"][model] = {"points": len(places), "hours_present_min": min(present) if present else 0, "hours_present_max": max(present) if present else 0}
            if not present or min(present) == 0:
                record["flags"].append(f"open-meteo {model}: the 00 UTC run has no 100 m wind at one point or more")
    ymd = target.strftime("%Y%m%d")
    record["noaa_arrival"] = {}
    for label, url in (("gfs", f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/?list-type=2&max-keys=220&prefix=gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f0"),
                       ("gefs_control", f"https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&max-keys=100&prefix=gefs.{ymd}/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f0")):
        answer = keep(f"noaa listing {label}", fetch(url, pause=0.2))
        if answer is not None:
            record["noaa_arrival"][label] = arrival(answer.decode("utf-8", "replace"), 54 if label == "gfs" else 48)

    record["finished_utc"] = iso(clock())
    record["finished_by_gate"] = record["finished_utc"] <= record["gate_utc"]
    if slot == "late":
        record["flags"].append("started after 10:05 Paris")
    content = json.dumps(record, indent=1).encode()
    write_whole(record_path if not dry_run else folder / "record.json", content)
    if not dry_run:
        manifest = root / "live" / "manifest.jsonl"
        previous = manifest.read_text(encoding="utf-8").splitlines()[-1] if manifest.exists() and manifest.stat().st_size else ""
        line = json.dumps({"day": day, "slot": slot, "record_sha256": sha(content), "listed_utc": iso(clock()), "prev": sha(previous.encode()) if previous else "0" * 64}, sort_keys=True)
        with manifest.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    log(json.dumps({"day": day, "slot": slot, "status": "dry run written" if dry_run else "recorded", "record": str(record_path if not dry_run else folder / "record.json"),
                    "requests": len(record["requests"]), "flags": record["flags"], "seconds": round((clock() - started).total_seconds(), 1)}, indent=1))
    return {"status": "dry run written" if dry_run else "recorded", "record": record}


def arrival(listing_xml: str, last_lead: int) -> dict:
    """From a public bucket listing: how many forecast files up to the last lead are there, and when the last of them arrived."""
    import re
    files = [(key, modified) for key, modified in re.findall(r"<Key>([^<]+)</Key><LastModified>([^<]+)</LastModified>", listing_xml) if not key.endswith(".idx")]
    wanted = [(key, modified) for key, modified in files if key[-3:].isdigit() and int(key[-3:]) <= last_lead]
    return {"files_to_lead": len(wanted), "last_lead_asked": last_lead, "latest_arrival_utc": max((m for _, m in wanted), default=None), "earliest_arrival_utc": min((m for _, m in wanted), default=None)}


# ---- history ---------------------------------------------------------------------------------------------------------------

def chunks(product: str, first: dt.date, last: dt.date) -> list[dict]:
    """Run ranges for the history. GFS: every run of two days (8 runs). GEFS: the 00 UTC run of ten days (10 runs)."""
    days, out, day = (2 if product == "gfs" else 10), [], first
    while day <= last:
        end = min(day + dt.timedelta(days=days - 1), last)
        start = dt.datetime(day.year, day.month, day.day, tzinfo=UTC)
        after = start - dt.timedelta(hours=1)  # both bounds are exclusive in the API
        before = dt.datetime(end.year, end.month, end.day, tzinfo=UTC) + (dt.timedelta(hours=19) if product == "gfs" else dt.timedelta(hours=1))
        runs = ((end - day).days + 1) * (4 if product == "gfs" else 1)
        out.append({"first": day.isoformat(), "last": end.isoformat(), "after": iso(after), "before": iso(before), "runs": runs})
        day = end + dt.timedelta(days=1)
    return out


def history(root: Path, product: str, zone: str, first: dt.date, last: dt.date, fetch=call, log=print, limit: int | None = None) -> dict:
    horizon = 54 if product == "gfs" else 72  # GEFS to 72 hours: the 00 UTC run of two days before reaches the whole delivery day
    leads = 55 if product == "gfs" else 25
    points = [(f"{zone}_{name}", lat, lon) for name, lat, lon in POINTS[zone]]
    folder = root / "history" / product / zone
    plan = chunks(product, first, last)
    done = failed = asked = 0
    started = time.time()
    for n, chunk in enumerate(plan):
        size = per_request(product, chunk["runs"], leads)
        for group in batches(points, size):
            path = folder / chunk["first"][:4] / f"{chunk['first']}_{chunk['last']}_{group[0][0]}_{len(group)}.json.gz"
            if path.exists():
                done += 1
                continue
            if limit is not None and asked >= limit:
                continue
            body = {"queries": [query(product, name, lat, lon, horizon, after=chunk["after"], before=chunk["before"]) for name, lat, lon in group]}
            answer = fetch(f"{API}/v1/forecasts", body)
            asked += 1
            if answer["status"] != 200:
                failed += 1
                write_whole(path.with_suffix(".error.txt"), json.dumps({k: answer.get(k) for k in ("status", "error", "asked_utc")}).encode() + b"\n" + answer["body"][:2000])
                log(f"  {chunk['first']} {group[0][0]}: {answer['error'] or answer['status']} (left for the next run)")
                if failed >= 12:
                    log("Twelve requests failed. Stopping; run the same command again later, it continues where it stopped.")
                    return {"asked": asked, "saved": done, "failed": failed, "stopped": True}
                continue
            envelope = {"product": product, "zone": zone, "chunk": chunk, "points": [g[0] for g in group], "asked_utc": answer["asked_utc"], "answered_utc": answer["answered_utc"],
                        "work_units": answer.get("work_units"), "answer_sha256": sha(answer["body"]), "answer": json.loads(answer["body"])}
            write_whole(path, gzip.compress(json.dumps(envelope).encode(), mtime=0))
            done += 1
        if n % 20 == 0 or n == len(plan) - 1:
            rate = (time.time() - started) / max(asked, 1)
            left = sum(1 for c in plan[n + 1:] for _ in batches(points, per_request(product, c["runs"], leads)))
            log(f"{product} {zone}: to {chunk['last']}, {done} saved, {failed} failed, about {left * rate / 60:.0f} minutes left")
    return {"asked": asked, "saved": done, "failed": failed, "stopped": False}


def pack(root: Path, product: str, zone: str, log=print) -> Path:
    """One compact table from the saved answers: a row per run, lead and point; GFS one speed, GEFS one speed per member."""
    folder = root / "history" / product / zone
    out = root / "history" / f"{product}_{zone}_wind100m.csv.gz"
    n_members = MEMBERS[product]
    header = "init_utc,lead_h,point," + ",".join(f"m{m:02d}" for m in range(n_members)) if n_members > 1 else "init_utc,lead_h,point,speed_ms"
    rows = 0
    with gzip.open(out.with_name(out.name + ".part"), "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(header + "\n")
        for path in sorted(folder.rglob("*.json.gz")):
            envelope = json.loads(gzip.decompress(path.read_bytes()))
            for result in envelope["answer"].get("results", []):
                table: dict = {}
                for row in speeds(result):
                    table.setdefault((row["init"], row["lead_h"]), {})[row["member"]] = row["speed"]
                for (init, lead), members in sorted(table.items()):
                    values = [members.get(m) for m in sorted(members)]
                    handle.write(f"{init},{lead},{result.get('queryId')}," + ",".join("" if v is None else f"{v:.2f}" for v in values) + "\n")
                    rows += 1
    os.replace(out.with_name(out.name + ".part"), out)
    log(f"{out} written: {rows} rows, sha256 {sha(out.read_bytes())}")
    return out


# ---- checks without a network ----------------------------------------------------------------------------------------------

def selftest() -> dict:
    checks = {}
    forecast = {"initTime": "2026-11-20T00:00:00Z", "validTimes": ["2026-11-20T00:00:00Z", "2026-11-20T03:00:00Z"], "leadTimeHours": [0.0, 3.0]}
    ens = {"queryId": "de_X", "forecasts": [{**forecast, "ensembleMembers": [{"member": 0, "data": {"wind_u_100m": [3.0, None], "wind_v_100m": [4.0, 1.0]}},
                                                                            {"member": 1, "data": {"wind_u_100m": [0.0, 6.0], "wind_v_100m": [0.0, 8.0]}}]}]}
    det = {"queryId": "de_X", "forecasts": [{**forecast, "data": {"wind_u_100m": [3.0, 6.0], "wind_v_100m": [4.0, 8.0]}}]}
    checks["ensemble_speeds"] = [r["speed"] for r in speeds(ens)] == [5.0, None, 0.0, 10.0]
    checks["deterministic_speeds"] = [(r["member"], r["lead_h"], r["speed"]) for r in speeds(det)] == [(0, 0, 5.0), (0, 3, 10.0)]
    checks["clock_changes"] = [paris_offset(dt.datetime(2026, 3, 29, 0, 59, tzinfo=UTC)), paris_offset(dt.datetime(2026, 3, 29, 1, 0, tzinfo=UTC)),
                               paris_offset(dt.datetime(2026, 10, 25, 0, 59, tzinfo=UTC)), paris_offset(dt.datetime(2026, 10, 25, 1, 0, tzinfo=UTC))] == [1, 2, 2, 1]
    checks["slots"] = [slot_of(dt.datetime(2026, 11, 21, h, m, tzinfo=UTC)) for h, m in ((6, 59), (8, 0), (9, 5), (9, 6))] == [
        ("2026-11-21", "early"), ("2026-11-21", "gate"), ("2026-11-21", "gate"), ("2026-11-21", "late")]
    gfs, gefs = chunks("gfs", dt.date(2021, 5, 1), dt.date(2021, 5, 5)), chunks("gefs", dt.date(2021, 5, 1), dt.date(2021, 5, 25))
    checks["gfs_chunks"] = [(c["after"], c["before"], c["runs"]) for c in gfs] == [("2021-04-30T23:00:00Z", "2021-05-02T19:00:00Z", 8), ("2021-05-02T23:00:00Z", "2021-05-04T19:00:00Z", 8),
                                                                                  ("2021-05-04T23:00:00Z", "2021-05-05T19:00:00Z", 4)]
    checks["gefs_chunks"] = [(c["after"], c["before"], c["runs"]) for c in gefs] == [("2021-04-30T23:00:00Z", "2021-05-10T01:00:00Z", 10), ("2021-05-10T23:00:00Z", "2021-05-20T01:00:00Z", 10),
                                                                                    ("2021-05-20T23:00:00Z", "2021-05-25T01:00:00Z", 5)]
    checks["answers_stay_under_the_ceiling"] = all(per_request(p, r, n) * r * n * 2 * MEMBERS[p] <= MAX_VALUES for p, r, n in (("gfs", 8, 55), ("gefs", 10, 25), ("gfs", 1, 55), ("gefs", 1, 17), ("ifs_ens", 1, 17)))
    checks["points"] = (len(POINTS["de"]), len(POINTS["fr"])) == (17, 17) and len({n for z in POINTS.values() for n, _, _ in z}) == 34
    listing = "<Key>a/gfs.t00z.pgrb2.0p25.f054</Key><LastModified>2026-11-20T03:50:00.000Z</LastModified><Key>a/gfs.t00z.pgrb2.0p25.f054.idx</Key><LastModified>2026-11-20T03:59:00.000Z</LastModified>" \
              "<Key>a/gfs.t00z.pgrb2.0p25.f055</Key><LastModified>2026-11-20T04:50:00.000Z</LastModified><Key>a/gfs.t00z.pgrb2.0p25.f000</Key><LastModified>2026-11-20T03:30:00.000Z</LastModified>"
    checks["arrival"] = arrival(listing, 54) == {"files_to_lead": 2, "last_lead_asked": 54, "latest_arrival_utc": "2026-11-20T03:50:00.000Z", "earliest_arrival_utc": "2026-11-20T03:30:00.000Z"}
    return {"checks": checks, "pass": all(checks.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("collect", "history", "pack", "selftest", "status"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--product", choices=("gfs", "gefs"), default="gfs")
    parser.add_argument("--zone", choices=("de", "fr"), default="de")
    parser.add_argument("--first", default="2021-05-01")
    parser.add_argument("--last", default="2026-06-30")
    parser.add_argument("--limit", type=int, default=None, help="history: stop after this many requests")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "selftest":
        result = selftest()
        print(json.dumps(result, indent=1))
        return 0 if result["pass"] else 1
    if args.command == "collect":
        logs = root / "live" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"{now_utc():%Y%m%dT%H%M%SZ}_collect.log").open("a", encoding="utf-8") as handle:
            def log(text):
                print(text, flush=True)
                handle.write(f"{iso(now_utc())} {text}\n")
                handle.flush()
            try:
                collect(root, dry_run=args.dry_run, log=log)
            except Exception as exc:  # noqa: BLE001  a failure leaves a trace
                log(f"failed: {exc.__class__.__name__}: {exc}")
                return 1
        return 0
    if args.command == "history":
        if args.last > "2026-06-30":
            print("The history fetch stops at 2026-06-30: later days are outside the development window.")
            return 2
        result = history(root, args.product, args.zone, dt.date.fromisoformat(args.first), dt.date.fromisoformat(args.last), limit=args.limit)
        print(json.dumps(result))
        return 0 if not result["failed"] else 1
    if args.command == "pack":
        pack(root, args.product, args.zone)
        return 0
    live = root / "live"
    records = sorted(p for p in live.glob("20*/*.json")) if live.exists() else []
    print(json.dumps({"root": str(root), "records": len(records), "last": str(records[-1]) if records else None,
                      "history_files": {p: sum(1 for _ in (root / "history" / p).rglob("*.json.gz")) for p in ("gfs", "gefs") if (root / "history" / p).exists()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
