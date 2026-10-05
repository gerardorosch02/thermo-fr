"""Record, at the moment it runs, which day-ahead items ENTSO-E already serves for tomorrow.

The API gives no publication timestamps for past data, so the only way to
measure when the load forecast, the wind and solar forecast and the prices for
delivery day D appear is to ask repeatedly on D-1 and note the first time each
answers. Run this a few times during a morning (for example 08:00, 10:30,
11:55, 13:15 and 18:30 Paris) and the log in data/timing_probe.csv shows when
each item crossed from "absent" to "present" relative to the 12:00 gate.
"""

from pathlib import Path

import pandas as pd

from ..config import LOCAL_TZ
from ..data.entsoe_rest import EntsoeApi, EntsoeApiError

ITEMS = ("load_forecast", "wind_solar_forecast", "prices")


def probe(api: EntsoeApi, delivery_day: pd.Timestamp) -> list[dict]:
    """One row per item: whether the API has any hours for `delivery_day`, and how many."""
    now = pd.Timestamp.now(tz="UTC")
    start = delivery_day.tz_localize(LOCAL_TZ).tz_convert("UTC")
    end = (delivery_day + pd.DateOffset(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    rows = []
    for item in ITEMS:
        try:
            content = api.fetch_chunk(item, start, end)
            from ..data.entsoe_rest import combine_resolutions, parse_document

            parts = parse_document(content)
            hours = len(combine_resolutions(parts).loc[start:end]) if parts else 0
            revision = parts[0].meta.get("revision") if parts else None
            status = "present" if hours else "absent"
        except EntsoeApiError as exc:
            hours, revision, status = 0, None, f"error: {exc}"[:80]
        rows.append(
            {
                "probed_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "probed_at_paris": now.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                "delivery_day": delivery_day.strftime("%Y-%m-%d"),
                "item": item,
                "status": status,
                "hours": hours,
                "revision": revision,
            }
        )
    return rows


def run_probe(log_path=Path("data/timing_probe.csv"), cache_dir=Path("data/cache/entsoe"), delivery_day=None) -> pd.DataFrame:
    api = EntsoeApi(cache_dir=cache_dir)
    day = pd.Timestamp(delivery_day) if delivery_day else pd.Timestamp.now(tz=LOCAL_TZ).normalize().tz_localize(None) + pd.DateOffset(days=1)
    rows = pd.DataFrame(probe(api, day))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return rows
