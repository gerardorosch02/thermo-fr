r"""Read the saved answers of price_probe.py and say how far the sources agree. Prints counts and differences only, never a price.

For each pair of sources and each delivery day (Paris): the number of quarter hours (or hours) both hold, how many agree to
0.005 EUR/MWh (a tie at exactly half a cent counts as agreeing), and the largest difference. A difference between two published prices is not a forecast error.

    python price_probe_read.py <folder written by price_probe.py>
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

UTC = dt.timezone.utc
folder = Path(sys.argv[1])
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
files = {r["label"]: folder / r["file"] for r in summary["requests"] if r["status"] == 200}


def day(t: dt.datetime) -> str:
    return (t + dt.timedelta(hours=1)).strftime("%m-%d")  # Paris, winter time


def energy_charts(label: str) -> dict:
    j = json.loads(files[label].read_bytes())
    return {dt.datetime.fromtimestamp(s, UTC): float(p) for s, p in zip(j["unix_seconds"], j["price"]) if p is not None}


def smard(prefix: str) -> dict:
    out = {}
    for label, path in files.items():
        if label.startswith(prefix) and "index" not in label:
            for ms, value in json.loads(path.read_bytes()).get("series", []):
                if value is not None:
                    out[dt.datetime.fromtimestamp(ms / 1000, UTC)] = float(value)
    return out


def awattar() -> dict:
    return {dt.datetime.fromtimestamp(x["start_timestamp"] / 1000, UTC): float(x["marketprice"]) for x in json.loads(files["awattar DE"].read_bytes())["data"]}


def entsoe(label: str, sequence: str | None) -> dict:
    """Quarter-hourly values of one sequence; a position the file leaves out repeats the value before it."""
    out = {}
    text = files[label].read_text(encoding="utf-8", errors="replace")
    for series in re.findall(r"<TimeSeries>.*?</TimeSeries>", text, flags=re.S):
        found = re.search(r"classificationSequence_AttributeInstanceComponent.position>(\d+)<", series)
        if (found.group(1) if found else None) != sequence:
            continue
        for period in re.findall(r"<Period>.*?</Period>", series, flags=re.S):
            first = dt.datetime.strptime(re.search(r"<start>([^<]+)</start>", period).group(1), "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
            last = dt.datetime.strptime(re.search(r"<end>([^<]+)</end>", period).group(1), "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
            step = {"PT15M": 15, "PT30M": 30, "PT60M": 60}[re.search(r"<resolution>([^<]+)</resolution>", period).group(1)]
            points = {int(p): float(v) for p, v in re.findall(r"<Point>\s*<position>(\d+)</position>\s*<price.amount>([^<]+)</price.amount>", period)}
            value = None
            for k in range(int((last - first).total_seconds() // 60 // step)):
                value = points.get(k + 1, value)
                for q in range(step // 15):
                    out[first + dt.timedelta(minutes=step * k + 15 * q)] = value
    return out


def hourly(series: dict) -> dict:
    """Mean of the four quarter hours of each hour, only where all four are held."""
    groups = defaultdict(list)
    for t, v in series.items():
        groups[t.replace(minute=0)].append(v)
    return {t: sum(v) / 4 for t, v in groups.items() if len(v) == 4}


def compare(name: str, a: dict, b: dict) -> None:
    by_day = defaultdict(list)
    for t in sorted(set(a) & set(b)):
        by_day[day(t)].append(abs(a[t] - b[t]))
    cells = [f"{d}: {len(v):>2} both, {sum(1 for x in v if x <= 0.005 + 1e-9):>2} agree, largest {max(v):6.2f}" for d, v in sorted(by_day.items()) if "11-16" <= d <= "11-22"]
    print(f"{name}\n   " + "\n   ".join(cells or ["no value in common"]), flush=True)


ec_de, ec_fr = energy_charts("energy-charts DE-LU"), energy_charts("energy-charts FR")
e_de1, e_de2, e_fr = entsoe("entsoe DE-LU", "1"), entsoe("entsoe DE-LU", "2"), entsoe("entsoe FR", None)
sm_de, sm_fr, aw = smard("smard DE-LU quarterhour"), smard("smard FR (region DE-LU) quarterhour"), awattar()

compare("energy-charts DE-LU against ENTSO-E DE-LU sequence 1 (the coupled auction)", ec_de, e_de1)
compare("energy-charts DE-LU against ENTSO-E DE-LU sequence 2 (the early Austrian auction)", ec_de, e_de2)
compare("ENTSO-E DE-LU sequence 1 against sequence 2 (how far the two auctions differ anyway)", e_de1, e_de2)
compare("energy-charts DE-LU, hourly mean, against aWATTar", hourly(ec_de), aw)
compare("ENTSO-E DE-LU sequence 1, hourly mean, against aWATTar", hourly(e_de1), aw)
compare("ENTSO-E DE-LU sequence 2, hourly mean, against aWATTar", hourly(e_de2), aw)
compare("SMARD DE-LU against ENTSO-E DE-LU sequence 1", sm_de, e_de1)
compare("energy-charts FR against ENTSO-E FR", ec_fr, e_fr)
compare("SMARD FR against ENTSO-E FR", sm_fr, e_fr)
for other in ("BE", "CH", "AT"):
    if f"energy-charts {other}" in files:
        compare(f"energy-charts DE-LU against energy-charts {other} (a check that the zones are different series)", ec_de, energy_charts(f"energy-charts {other}"))
