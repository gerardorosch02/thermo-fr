"""Download the raw ENTSO-E files the spread test needs for 1 June to 10 October 2026. Standard library only.

    python fetch_spread_test_raw.py --out .

It asks ENTSO-E for three items of the DE-LU bidding zone and saves each answer untouched as XML:
  prices_DE_LU               day-ahead prices (document A44), every series the zone publishes
  load_forecast_DE_LU        day-ahead total load forecast (A65, process A01)
  wind_solar_forecast_DE_LU  day-ahead wind and solar generation forecast (A69, process A01)
in five windows from 2026-05-31 to 2026-10-11 (UTC), 15 requests in all. June is fetched so the files can be compared with
the stored ones before anything later is read. Nothing is parsed here; thermo_fr.forecast.spread_eval does that.

The key is read from the ENTSOE_API_KEY environment variable. It is sent to ENTSO-E only, never printed and never written
to a file, and it is removed from any error text. Rerunning skips the files already saved.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"
WINDOWS = [("202605310000", "202607010000"), ("202607010000", "202608010000"), ("202608010000", "202609010000"),
           ("202609010000", "202610010000"), ("202610010000", "202610110000")]
ITEMS = {"prices_DE_LU": {"documentType": "A44", "in_Domain": DE_LU, "out_Domain": DE_LU},
         "load_forecast_DE_LU": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": DE_LU},
         "wind_solar_forecast_DE_LU": {"documentType": "A69", "processType": "A01", "in_Domain": DE_LU}}
NO_DATA = "<code>999</code>"


def download(params: dict, key: str, tries: int = 5) -> bytes:
    url = BASE_URL + "?" + urllib.parse.urlencode({**params, "securityToken": key})
    wait = 15.0
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "thermo-fr spread test"}), timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if exc.code == 400 and NO_DATA.encode() in body:
                return body  # an acknowledgement saying there is no data: saved as it is, and reported
            if exc.code in (429, 500, 502, 503, 504) and attempt < tries:
                time.sleep(wait); wait *= 2; continue
            raise RuntimeError(f"HTTP {exc.code} from ENTSO-E: {body[:300].decode('utf-8', 'replace').replace(key, '<token>')}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < tries:
                time.sleep(wait); wait *= 2; continue
            raise RuntimeError(f"could not reach ENTSO-E: {str(exc).replace(key, '<token>')}") from None
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        print("ENTSOE_API_KEY is not set in this window. Open a new PowerShell window (the variable is set at user level) and run again.")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "ENTSO-E Transparency Platform, https://transparency.entsoe.eu, RESTful API", "zone": "DE_LU", "files": {}}
    for item, params in ITEMS.items():
        for start, end in WINDOWS:
            path = out / f"{item}_{start}_{end}.xml"
            if path.exists() and path.stat().st_size > 0:
                state = "kept"
            else:
                content = download({**params, "periodStart": start, "periodEnd": end}, key)
                if key.encode() in content:
                    raise RuntimeError("the answer contains the key; nothing was written")
                path.write_bytes(content)
                state = "downloaded"
                time.sleep(0.6)
            content = path.read_bytes()
            empty = NO_DATA.encode() in content and b"Acknowledgement_MarketDocument" in content
            manifest["files"][path.name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "no_data": bool(empty)}
            print(f"  {path.name}: {state}, {len(content):,} bytes" + (" (ENTSO-E says no data)" if empty else ""))
    manifest["fetched_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "spread_test_raw.json").write_text(json.dumps(manifest, indent=1))
    missing = [name for name, f in manifest["files"].items() if f["no_data"]]
    print(f"{len(manifest['files'])} files, {sum(f['bytes'] for f in manifest['files'].values()):,} bytes" + (f"; no data in: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
