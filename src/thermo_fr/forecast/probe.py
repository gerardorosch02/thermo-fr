"""Record, at the moment it runs, which day-ahead items ENTSO-E already serves for tomorrow.

The API gives no publication timestamps for past data, so the only way to
measure when the load forecast, the wind and solar forecast and the prices for
delivery day D appear is to ask repeatedly on D-1 and note the first time each
answers. `run_probe` does one poll and appends it to a CSV log; `poll` repeats
it at a fixed interval until a deadline and also keeps the hourly values of
every poll, so that a revision between two polls shows up as a change in the
value hash. The token is read from the environment and never written.
"""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from ..config import LOCAL_TZ
from ..data.entsoe_rest import EntsoeApi, EntsoeApiError, combine_resolutions, parse_document

ITEMS = ("load_forecast", "wind_solar_forecast", "prices")


def _series_for(item: str, parts) -> dict[str, pd.Series]:
    """Named hourly series out of the parsed TimeSeries (one per psrType for wind and solar)."""
    if item == "wind_solar_forecast":
        out = {}
        for psr, name in (("B16", "solar"), ("B19", "wind_onshore"), ("B18", "wind_offshore")):
            s = combine_resolutions([p for p in parts if p.psr_type == psr])
            if len(s):
                out[name] = s
        return out
    s = combine_resolutions(parts)
    return {item: s} if len(s) else {}


def probe(api: EntsoeApi, delivery_day: pd.Timestamp, items=ITEMS, snapshots: dict | None = None) -> list[dict]:
    """One row per item: presence, hour count, revision and a hash of the values for `delivery_day`.

    When `snapshots` is a dict, the hourly values are stored in it under the
    item name, so a caller can compare polls.
    """
    now = pd.Timestamp.now(tz="UTC")
    start = delivery_day.tz_localize(LOCAL_TZ).tz_convert("UTC")
    end = (delivery_day + pd.DateOffset(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    rows = []
    for item in items:
        try:
            parts = parse_document(api.fetch_chunk(item, start, end))
            series = {k: v.loc[start:end - pd.Timedelta(hours=1)] for k, v in _series_for(item, parts).items()}
            hours = max((len(s) for s in series.values()), default=0)
            revision = parts[0].meta.get("revision") if parts else None
            status = "present" if hours else "absent"
            payload = {k: [round(float(x), 3) for x in s.to_numpy()] for k, s in series.items()}
            digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12] if hours else None
            if snapshots is not None:
                snapshots[item] = payload
        except EntsoeApiError as exc:
            hours, revision, status, digest = 0, None, f"error: {exc}"[:80], None
        rows.append(
            {
                "probed_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "probed_at_paris": now.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                "delivery_day": delivery_day.strftime("%Y-%m-%d"),
                "item": item,
                "status": status,
                "hours": hours,
                "revision": revision,
                "value_hash": digest,
            }
        )
    return rows


def tomorrow() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LOCAL_TZ).normalize().tz_localize(None) + pd.DateOffset(days=1)


def run_probe(log_path=Path("data/timing_probe.csv"), cache_dir=Path("data/cache/entsoe"), delivery_day=None, items=ITEMS,
              snapshot_dir=None) -> pd.DataFrame:
    """One poll appended to `log_path`; with `snapshot_dir`, the hourly values are saved as JSON too."""
    api = EntsoeApi(cache_dir=cache_dir)
    day = pd.Timestamp(delivery_day) if delivery_day else tomorrow()
    snapshots: dict = {}
    rows = pd.DataFrame(probe(api, day, items=items, snapshots=snapshots))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = rows["probed_at_utc"].iloc[0].replace(":", "")
        (snapshot_dir / f"values_{day:%Y-%m-%d}_{stamp}.json").write_text(json.dumps(snapshots, indent=0))
    return rows


def poll(until_paris: str, every_minutes: int = 15, log_path=Path("reports/timing_probe/log.csv"), snapshot_dir=Path("reports/timing_probe/values"),
         cache_dir=Path("data/cache/entsoe"), items=("load_forecast", "wind_solar_forecast"), delivery_day=None, log=print,
         sleep=time.sleep) -> pd.DataFrame:
    """Poll now and then every `every_minutes` until `until_paris` (HH:MM today, Paris), printing changes."""
    day = pd.Timestamp(delivery_day) if delivery_day else tomorrow()
    deadline = pd.Timestamp(f"{pd.Timestamp.now(tz=LOCAL_TZ):%Y-%m-%d} {until_paris}", tz=LOCAL_TZ)
    previous: dict[str, str | None] = {}
    frames = []
    while True:
        rows = run_probe(log_path=log_path, cache_dir=cache_dir, delivery_day=day, items=items, snapshot_dir=snapshot_dir)
        frames.append(rows)
        for _, r in rows.iterrows():
            change = ""
            if r["item"] in previous and previous[r["item"]] != r["value_hash"]:
                change = f"  CHANGED from {previous[r['item']]}"
            previous[r["item"]] = r["value_hash"]
            log(f"{r['probed_at_paris']} Paris  {r['item']:20s} {r['status']:8s} hours={r['hours']:>3} rev={r['revision']} hash={r['value_hash']}{change}")
        nxt = pd.Timestamp.now(tz=LOCAL_TZ) + pd.Timedelta(minutes=every_minutes)
        if nxt > deadline + pd.Timedelta(minutes=1):
            break
        sleep(max(0.0, (nxt - pd.Timestamp.now(tz=LOCAL_TZ)).total_seconds()))
    return pd.concat(frames, ignore_index=True)
