r"""One-off probe: which public sources hold the day-ahead prices that ENTSO-E has lacked since the auction of 19 November 2026?

It asks each candidate for the prices of delivery days 16 to 22 November, saves every answer untouched under a new folder
beside this script, and prints, for each source and zone, how many values it holds for each delivery day. It prints no price.
It asks ENTSO-E the same question at the same moment, with the key already set on this computer (ENTSOE_API_KEY), which is
never printed or saved. Standard library only. Nothing else on the computer is changed.

    python price_probe.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

UTC = dt.timezone.utc
UA = "thermo-fr-price-probe/1.0 (research; no personal data)"
FIRST, LAST = dt.date(2026, 11, 16), dt.date(2026, 11, 22)
now = dt.datetime.now(UTC)
out = Path(__file__).resolve().parent / "price_probe" / now.strftime("%Y%m%dT%H%M%SZ")
out.mkdir(parents=True, exist_ok=True)
rows = []


def paris_day(t: dt.datetime) -> str:
    """Delivery day in Paris of a UTC instant (winter time throughout this week: UTC+1)."""
    return (t.astimezone(UTC) + dt.timedelta(hours=1)).strftime("%Y-%m-%d")


def get(label: str, url: str, shown: str | None = None) -> bytes | None:
    started, status, body, error = time.time(), None, b"", ""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"}), timeout=90) as response:
            status, body = response.status, response.read(30_000_000)
    except urllib.error.HTTPError as exc:
        status, body, error = exc.code, exc.read(200_000), f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}"[:200]
    name = f"{len(rows) + 1:02d}_{''.join(c if c.isalnum() else '_' for c in label)[:60]}.body"
    (out / name).write_bytes(body)
    rows.append({"n": len(rows) + 1, "label": label, "url": shown or url, "status": status, "seconds": round(time.time() - started, 2), "bytes": len(body), "error": error,
                 "sha256": hashlib.sha256(body).hexdigest(), "file": name, "asked_utc": dt.datetime.now(UTC).isoformat(timespec="seconds")})
    time.sleep(0.5)
    return body if status == 200 else None


def report(label: str, stamps: list[dt.datetime], extra: str = "") -> None:
    """Values held per delivery day, and the spacing of the values in minutes."""
    days = Counter(paris_day(t) for t in stamps)
    gaps = Counter(int((b - a).total_seconds() // 60) for a, b in zip(sorted(stamps), sorted(stamps)[1:]))
    wanted = [(FIRST + dt.timedelta(days=i)).isoformat() for i in range((LAST - FIRST).days + 1)]
    rows[-1]["values_by_delivery_day"] = {d: days.get(d, 0) for d in wanted}
    rows[-1]["spacing_minutes"] = dict(gaps.most_common(3))
    rows[-1]["note"] = extra
    print(f"{label:34s} " + "  ".join(f"{d[5:]}:{days.get(d, 0):>3}" for d in wanted) + f"   spacing {dict(gaps.most_common(2))} {extra}", flush=True)


print("values held for each delivery day (16 to 22 November); 24 or 96 is a full day\n")

# 1. energy-charts.info (Fraunhofer ISE)
for zone in ("FR", "DE-LU", "BE", "NL", "CH", "IT-North", "AT"):
    body = get(f"energy-charts {zone}", "https://api.energy-charts.info/price?" + urllib.parse.urlencode({"bzn": zone, "start": FIRST.isoformat(), "end": LAST.isoformat()}))
    if body:
        try:
            j = json.loads(body)
            stamps = [dt.datetime.fromtimestamp(s, UTC) for s, p in zip(j.get("unix_seconds", []), j.get("price", [])) if p is not None]
            report(f"energy-charts {zone}", stamps, "licence: " + str(j.get("license_info", ""))[:90])
        except Exception as exc:  # noqa: BLE001
            print(f"energy-charts {zone}: could not read the answer ({exc})")
    else:
        print(f"energy-charts {zone}: {rows[-1]['error'] or rows[-1]['status']}")

# 2. SMARD (Bundesnetzagentur): 4169 is the DE-LU price, 254 the French one
for label, flt, region in (("smard DE-LU", 4169, "DE-LU"), ("smard FR", 254, "FR"), ("smard FR (region DE-LU)", 254, "DE-LU")):
    for resolution in ("quarterhour", "hour"):
        index = get(f"{label} index {resolution}", f"https://www.smard.de/app/chart_data/{flt}/{region}/index_{resolution}.json")
        if not index:
            print(f"{label} {resolution}: {rows[-1]['error'] or rows[-1]['status']}")
            continue
        try:
            chunks = sorted(json.loads(index)["timestamps"])[-3:]
            stamps = []
            for chunk in chunks:
                body = get(f"{label} {resolution} {chunk}", f"https://www.smard.de/app/chart_data/{flt}/{region}/{flt}_{region}_{resolution}_{chunk}.json")
                if body:
                    stamps += [dt.datetime.fromtimestamp(ms / 1000, UTC) for ms, value in json.loads(body).get("series", []) if value is not None]
            report(f"{label} {resolution}", stamps)
        except Exception as exc:  # noqa: BLE001
            print(f"{label} {resolution}: could not read the answer ({exc})")

# 3. aWATTar (hourly DE-LU prices from the exchange)
start = int(dt.datetime(FIRST.year, FIRST.month, FIRST.day, tzinfo=UTC).timestamp() * 1000) - 3_600_000
end = int(dt.datetime(LAST.year, LAST.month, LAST.day, tzinfo=UTC).timestamp() * 1000) + 23 * 3_600_000
body = get("awattar DE", f"https://api.awattar.de/v1/marketdata?start={start}&end={end}")
if body:
    try:
        report("awattar DE", [dt.datetime.fromtimestamp(x["start_timestamp"] / 1000, UTC) for x in json.loads(body).get("data", []) if x.get("marketprice") is not None])
    except Exception as exc:  # noqa: BLE001
        print(f"awattar DE: could not read the answer ({exc})")
else:
    print(f"awattar DE: {rows[-1]['error'] or rows[-1]['status']}")

# 4. ENTSO-E at the same moment, for comparison
key = os.environ.get("ENTSOE_API_KEY", "")
if not key:
    print("ENTSO-E: no ENTSOE_API_KEY in this window, skipped")
for zone, eic in (("FR", "10YFR-RTE------C"), ("DE-LU", "10Y1001A1001A82H")):
    if not key:
        break
    params = {"documentType": "A44", "in_Domain": eic, "out_Domain": eic, "periodStart": "202611152300", "periodEnd": "202611222300"}
    url = "https://web-api.tp.entsoe.eu/api?" + urllib.parse.urlencode({"securityToken": key, **params})
    body = get(f"entsoe {zone}", url, shown="https://web-api.tp.entsoe.eu/api?" + urllib.parse.urlencode({"securityToken": "(not saved)", **params}))
    if not body:
        print(f"entsoe {zone}: {rows[-1]['error'] or rows[-1]['status']}")
        continue
    text = body.decode("utf-8", "replace")
    for sequence in ("1", "2", None):
        stamps = []
        for series in re.findall(r"<TimeSeries>.*?</TimeSeries>", text, flags=re.S):
            found = re.search(r"classificationSequence_AttributeInstanceComponent.position>(\d+)<", series)
            if (found.group(1) if found else None) != sequence:
                continue
            for period in re.findall(r"<Period>.*?</Period>", series, flags=re.S):
                first = dt.datetime.strptime(re.search(r"<start>([^<]+)</start>", period).group(1), "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
                step = {"PT15M": 15, "PT30M": 30, "PT60M": 60}[re.search(r"<resolution>([^<]+)</resolution>", period).group(1)]
                last = dt.datetime.strptime(re.search(r"<end>([^<]+)</end>", period).group(1), "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
                stamps += [first + dt.timedelta(minutes=step * k) for k in range(int((last - first).total_seconds() // 60 // step))]  # every interval of the period: the file leaves out repeats
        if stamps:
            report(f"entsoe {zone} sequence {sequence or 'none'}", stamps)

(out / "summary.json").write_text(json.dumps({"probe": "second source for day-ahead prices", "started_utc": now.isoformat(timespec="seconds"), "python": sys.version.split()[0], "requests": rows}, indent=1), encoding="utf-8")
print(f"\n{sum(1 for r in rows if r['status'] == 200)} of {len(rows)} requests answered 200. Everything is saved in {out}")
