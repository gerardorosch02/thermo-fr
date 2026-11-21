"""The morning logger of the spread forecasts (docs/prereg/spread_w_amendment_2.md, changed by spread_w_amendment_4.md and
spread_w_amendment_5.md).

Each morning it fetches what is public, forecasts the France minus Germany spread of the next delivery day with H, HW, G,
GK, GM and W, and writes one time-stamped record that is never rewritten. It computes no error against any price, ever.

    python -m thermo_fr.forecast.spread_live selftest --root <folder>                      no network: rebuild a pre-window day
    python -m thermo_fr.forecast.spread_live dry-run  --root <folder> --production <folder> full run, written under live/dry_run
    python -m thermo_fr.forecast.spread_live issue    --root <folder> --production <folder> the scheduled run
    python -m thermo_fr.forecast.spread_live status   --root <folder>                      counts only

Folder layout under --root:
  static/   inputs.csv, neighbour_fundamentals.csv, neighbour_sequences.csv.gz, raw_first_test/*.xml,
            german_wind_points.csv.gz, german_solar_points.csv.gz, reference.json          never changed after installation
  live/     french_windows/, german_raw/<stamp>/, german_price_second_source/, german_points/, records/issued/,
            records/manifest.jsonl, records/attempts.jsonl, cache/, dry_run/               written by the logger
--production is the production folder of the public forecast. The logger only reads there: the French input windows saved by
the morning runs (data/shadow/fresh) and the proxy weights (published/model).

What "available by 10:05 Paris" means here. The forecast of delivery day D is meant to be issued at 10:05 Paris on D-1.
  - Nothing is fetched before 10:00 Paris on D-1, the deadline of the load forecast of D. A run that starts earlier writes
    nothing.
  - Every source is fetched that morning, the four fetches side by side, and the time each one finished is written into the
    record, with a flag if it finished after 10:05.
  - German prices and ENTSO-E forecasts of D or later are dropped before any feature is built, whatever the fetch returned.
  - The features of D are then built a second time from the same tables with every value that cannot be known at the gate
    replaced by noise (spread_eval.scrambled). The record states whether the inputs of D came out the same to one part in
    a billion, and separately whether they came out bit for bit identical. This tests the position of every value used. It
    cannot test whether a value was revised between its publication and the fetch.
  - One record per delivery day, created atomically, written before the auction closes at 12:00 Paris or not at all.
  - If the French window, the German ENTSO-E files or the German points fail on the first run of the morning, or the
    French load forecast of D is not there yet, or a price request failed and left a price of the eve missing, nothing is
    written and the second run tries again. The second run issues with what there is, and the record says what was missing.
    The second source of the German price failing holds the first run up only if the German price of the eve is then
    missing.
  - A price of D-1 that the source answered for but does not hold (yesterday's spread) does not postpone anything: the first
    run issues with those inputs missing, and says so. Waiting could only make the fetch late.
  - The coupled German price has a second source, energy-charts.info (amendment 5). ENTSO-E comes first. An hour of D-1 or
    earlier for which the files fetched from ENTSO-E hold no coupled price is taken from the second source, and only on a
    morning when the two agree on every hour both hold over the ten days before D, with at least 24 such hours. That is
    decided afresh each morning from every answer saved so far. The record counts the hours taken and says so in a flag.
    Nothing of D or later is ever taken from it. The French price has no second source.
All time stamps are the laptop's own clock. Nothing outside the laptop witnesses them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import forward_q as fq
from . import spread_eval as se
from . import spread_hw as shw
from . import spread_w as sw
from .inputs import INPUT_COLUMNS, load_inputs
from .timing import delivery_days

PARIS = "Europe/Paris"
EARLIEST_HOUR = 10  # nothing is fetched before 10:00 Paris: the deadline of the load forecast of D
CUTOFF_HOUR, CUTOFF_MINUTE = 10, 5
LAST_CHANCE_HOUR, LAST_CHANCE_MINUTE = 10, 25  # before this a run with a failed source writes nothing and leaves it to the next trigger
AUCTION_HOUR = 12
WAIT_IF_EARLY_BY = pd.Timedelta(minutes=5)  # a task that fires a little before 10:00 waits for it
RUNNING_STALE_AFTER = pd.Timedelta(minutes=30)
FETCH_BUDGET_SECONDS = {"french_window": 300, "german_raw": 150, "german_price_second_source": 150, "german_points": 150}
FRENCH_WINDOW_SPAN_DAYS = 11  # one production fetch covers D-10 to D+1
MAX_FRENCH_BACKFILL_WINDOWS = 8
MODELS = ("H", "HW", "G", "GK", "GM", "W")
GERMAN_LIVE_FROM = "2026-10-10"  # first UTC day the logger fetches German data for; the stored raw files end at 2026-10-11 00:00 UTC
GERMAN_REFRESH_DAYS = 10  # each morning the last ten days are fetched again
POINT_REFRESH_DAYS = 5
SELFTEST_DAY = "2026-10-10"  # the last day of the opened window: rebuilt from the static files and compared with the frozen batch
SELFTEST_TOLERANCE = 1e-6
# The models answer a change in the last bits of their inputs with a different fit (docs/prereg/spread_w_amendment_3.md): on the
# self-test day, inputs perturbed by one part in 1e15 to 1e9 moved the six forecasts by 1.3 to 5.6 EUR/MWh on average and by up to
# 12.3 at one hour. A machine whose tables agree with the reference to 1e-9 but not bit for bit therefore passes if its forecasts
# stay inside these bands, set at about one and a half times the largest values seen.
NUMERICAL_BAND_MEAN, NUMERICAL_BAND_MAX = 8.0, 20.0
SCRAMBLE_TOLERANCE = 1e-9  # relative; a difference under the scramble below this is rounding, not a value from the future (amendment 4)
ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"
GERMAN_ITEMS = {"prices_DE_LU": {"documentType": "A44", "in_Domain": DE_LU, "out_Domain": DE_LU},
                "load_forecast_DE_LU": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": DE_LU},
                "wind_solar_forecast_DE_LU": {"documentType": "A69", "processType": "A01", "in_Domain": DE_LU}}
SECOND_SOURCE = "german_price_second_source"  # the name of the fetch, of its folder under live/ and of its entry in the record
SECOND_SOURCE_URL = "https://api.energy-charts.info/price"
SECOND_SOURCE_NAME = "energy-charts.info (Fraunhofer ISE), day-ahead price of bidding zone DE-LU, CC BY 4.0"
SECOND_SOURCE_AGREE = 0.001  # EUR/MWh on the hourly mean: one quarter hour apart by a single cent moves the mean by 0.0025 and fails
SECOND_SOURCE_PLAUSIBLE = (-1000.0, 10000.0)  # EUR/MWh; a number outside this is not read as a price
SECOND_SOURCE_MIN_HOURS_IN_COMMON = 24  # fewer hours held by both over the ten days before D, and the second source is not used
POINTS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
KMH_TO_MS = 1000.0 / 3600.0
DUE_BY_THE_GATE = ("load_forecast", "weather_issued", "wind_points", "solar_points", "wind_proxy", "solar_proxy")  # French items that should exist for D at 10:00 Paris
LIVE_DATA_FROM = "2026-10-11"  # delivery days the stored files do not hold: only rows from here on are ever touched by live fixes
KNOWN_BY = {  # the timing rule of every input group, written into each record
    "French prices and neighbours' prices of D-1": "about 13:00 Paris on D-2",
    "French load forecast of D": "10:00 Paris on D-1 (ENTSO-E deadline)",
    "French nuclear output": "within about an hour of each period; used to 08:00 Paris on D-1",
    "French weather and points of D": "Open-Meteo run of 18 UTC on D-2 at the latest (forecast as issued two days ahead)",
    "German coupled price of D-1": "about 13:00 Paris on D-2",
    "German coupled price of days before D from the second source": "the auction result as energy-charts.info publishes it, used only where the files fetched from ENTSO-E hold none; "
                                                                    "that it is the same result is checked on the hours both hold, not on the hours taken",
    "German load forecast of D-1 and D-7": "10:00 Paris on D-2 and D-8",
    "German wind and solar forecasts of D-1": "18:00 Paris on D-2",
    "German wind and solar points of D": "Open-Meteo run of 18 UTC on D-2 at the latest (forecast as issued two days ahead)",
}


@dataclass
class Paths:
    root: Path
    production: Path | None = None
    live_dir: Path | None = None  # a dry run writes everything, fetches included, under live/dry_run

    @property
    def static(self) -> Path:
        return Path(self.root) / "static"

    @property
    def live(self) -> Path:
        return Path(self.live_dir) if self.live_dir else Path(self.root) / "live"

    @property
    def records(self) -> Path:
        return self.live / "records"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def stamp_of(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")


def iso(t) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def next_delivery_day(now: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(now).tz_convert(PARIS).tz_localize(None).normalize() + pd.Timedelta(days=1))


def gates(day: pd.Timestamp) -> dict:
    """The moments of the eve of `day`, in UTC: earliest start (10:00 Paris), cutoff (10:05), last chance (10:25), auction (12:00)."""
    eve = pd.Timestamp(day) - pd.Timedelta(days=1)
    at = lambda hour, minute=0: (eve + pd.Timedelta(hours=hour, minutes=minute)).tz_localize(PARIS).tz_convert("UTC")  # noqa: E731
    return {"earliest": at(EARLIEST_HOUR), "cutoff": at(CUTOFF_HOUR, CUTOFF_MINUTE), "last_chance": at(LAST_CHANCE_HOUR, LAST_CHANCE_MINUTE), "auction": at(AUCTION_HOUR)}


def code_files() -> list[Path]:
    """Every source file a forecast runs through: the forecast and data packages and the point list of the fetch script."""
    package = Path(__file__).resolve().parents[1]
    return sorted([*(package / "forecast").glob("*.py"), *(package / "data").glob("*.py")]) + [Path(__file__).resolve().parents[3] / "scripts" / "fetch_german_points.py"]


def code_sha256() -> str:
    """One hash over those files as they are on disk now (line endings normalised, names relative to the package)."""
    h = hashlib.sha256()
    for path in code_files():
        h.update(f"{path.parent.name}/{path.name}".encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def folder_sha256(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(folder).glob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def code_commit() -> str | None:
    """The commit the code snapshot was taken at: a COMMIT file beside src/, else the checkout's HEAD."""
    root = Path(__file__).resolve().parents[3]
    marker = root / "COMMIT"
    return marker.read_text().strip() if marker.exists() else fq.code_commit(root)


# ---- French inputs -----------------------------------------------------------------------------------------------------------

class _Statuses:
    """Stands in for the store the production fetch reports to. Nothing is written to any database."""

    def __init__(self):
        self.items = {}

    def record_status(self, run_id, delivery_day, item, status, hours=0, revision=None, hash_=None, message="", now=None):
        self.items[item] = {"status": status, "hours": int(hours or 0), "revision": revision, "message": message}


def write_whole(path: Path, content: bytes) -> None:
    """Write under a temporary name, then rename: a file the logger later reads is either whole or not there. A fetch that
    is cut short (the process ends, a thread is given up on) cannot leave half a file under a name that is read."""
    part = path.with_name(path.name + ".part")
    part.write_bytes(content)
    for attempt in range(10):  # on Windows a scanner can hold the new file for a moment, and the rename is then refused
        try:
            os.replace(part, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def fetch_french_window(day: pd.Timestamp, paths: Paths, now: pd.Timestamp | None = None) -> dict:
    """The production fetch of the French inputs around `day` (jobs.fetch_fresh), saved as a window file of the logger's own.
    The file is stamped with the moment it is written. If the files already held end more than one window back (the laptop was
    off for more than ten days), earlier windows are fetched too, so the history has no hole."""
    from .jobs import fetch_fresh

    weights = Path(paths.production) / "published" / "model"
    folder = paths.live / "french_windows"
    folder.mkdir(parents=True, exist_ok=True)
    held = _overlay(paths, utc_now())["price_eur_mwh"].dropna()
    latest = delivery_days(held.index).max() if len(held) else pd.Timestamp(day) - pd.Timedelta(days=FRENCH_WINDOW_SPAN_DAYS * MAX_FRENCH_BACKFILL_WINDOWS)
    targets = [pd.Timestamp(day)]
    while targets[-1] - pd.Timedelta(days=FRENCH_WINDOW_SPAN_DAYS - 1) > latest + pd.Timedelta(days=1) and len(targets) < MAX_FRENCH_BACKFILL_WINDOWS:
        targets.append(targets[-1] - pd.Timedelta(days=FRENCH_WINDOW_SPAN_DAYS))
    out = {"files": {}, "items": {}, "windows_fetched": len(targets), "errors": {}}
    for target in reversed(targets):  # oldest first, so the window of `day` is written last and wins
        statuses = _Statuses()
        fresh = fetch_fresh(f"{target:%Y-%m-%d}", paths.live / "cache", statuses, 0, wind_weights=weights / "wind_proxy.json", solar_weights=weights / "solar_proxy.json")
        path = folder / f"{target:%Y-%m-%d}_spread_{stamp_of(utc_now())}.csv"
        write_whole(path, fresh.to_csv().encode("utf-8"))
        out["files"][path.name] = {"sha256": fq.sha256_file(path), "rows": int(len(fresh))}
        if target == pd.Timestamp(day):
            out["items"] = statuses.items
    # Prices, nuclear output and neighbours' prices of D itself never exist at the gate, and the operator's wind and solar
    # forecast of D is published after it: their absence for D is expected. A failed call is an error for any item.
    out["errors"] = {item: v["message"] or "failed" for item, v in out["items"].items() if v["status"] == "error"}
    out["absent_for_D_though_due_by_the_gate"] = sorted(item for item, v in out["items"].items() if v["status"] == "absent" and item in DUE_BY_THE_GATE)
    return out


def window_files(paths: Paths, upto: pd.Timestamp, own: bool = True) -> list[tuple[pd.Timestamp, Path]]:
    """Every saved French window issued at or before `upto`: the production ones and the logger's own, oldest first."""
    found = []
    folders = [Path(paths.production) / "data" / "shadow" / "fresh"] if paths.production else []  # production windows are stamped with the start of their run
    if own:
        folders.append(paths.live / "french_windows")
    for folder in folders:
        for path in sorted(folder.glob("*.csv")) if folder.exists() else []:
            try:
                issued = pd.Timestamp(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            issued = issued.tz_localize("UTC") if issued.tzinfo is None else issued.tz_convert("UTC")
            if issued <= upto:
                found.append((issued, path))
    return sorted(found, key=lambda item: (item[0], item[1].name))


def _overlay(paths: Paths, upto: pd.Timestamp, windows: bool = True, used: list | None = None) -> pd.DataFrame:
    """static/inputs.csv with every saved window issued by `upto` laid over it, later issue winning."""
    table = load_inputs(paths.static / "inputs.csv").reindex(columns=INPUT_COLUMNS).astype(float)
    for item in window_files(paths, upto) if windows else []:
        window = load_inputs(item[1]).reindex(columns=INPUT_COLUMNS).astype(float)
        table = window.combine_first(table)
        if used is not None:
            used.append(item)
    return table


def french_history(paths: Paths, day: pd.Timestamp, upto: pd.Timestamp, windows: bool = True) -> tuple[pd.DataFrame, dict]:
    """The overlay with the proxy gaps filled as in every evaluation, and rows for every hour to the end of `day` so the
    calendar and the lags of `day` exist even if an input did not arrive."""
    used = []
    table = _overlay(paths, upto, windows=windows, used=used)
    last_hour = fq.day_hours(day)[-1]
    table = table.reindex(pd.date_range(table.index.min(), max(table.index.max(), last_hour), freq="h"))
    table.index.name = "timestamp_utc"
    history, filled = fq.fill_proxy_gaps(table.sort_index())
    info = {"windows_used": len(used), "last_window": used[-1][1].name if used else None, "last_window_issued_utc": iso(used[-1][0]) if used else None,
            "proxy_gaps_filled": json.loads(json.dumps(filled, default=str))}
    return se.cut(history, f"{day:%Y-%m-%d}"), info


# ---- German prices and ENTSO-E forecasts ---------------------------------------------------------------------------------------

def chunks(first: pd.Timestamp, last: pd.Timestamp, days: int = 31) -> list[tuple[str, str]]:
    """(periodStart, periodEnd) pairs of at most `days` UTC days covering first 00:00 to last 00:00."""
    out, cursor = [], pd.Timestamp(first).normalize()
    last = pd.Timestamp(last).normalize()
    while cursor < last:
        end = min(cursor + pd.Timedelta(days=days), last)
        out.append((cursor.strftime("%Y%m%d0000"), end.strftime("%Y%m%d0000")))
        cursor = end
    return out


def _entsoe_get(params: dict, key: str, opener=urllib.request.urlopen, tries: int = 4) -> bytes:
    url = ENTSOE_URL + "?" + urllib.parse.urlencode({**params, "securityToken": key})
    wait = 10.0
    for attempt in range(1, tries + 1):
        try:
            with opener(urllib.request.Request(url, headers={"User-Agent": "thermo-fr spread logger"}), timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if exc.code == 400 and b"<code>999</code>" in body:
                return body  # ENTSO-E's "no data" answer: kept as it is
            if exc.code in (429, 500, 502, 503, 504) and attempt < tries:
                time.sleep(wait); wait *= 2; continue
            raise RuntimeError(f"HTTP {exc.code} from ENTSO-E: {body[:200].decode('utf-8', 'replace').replace(key, '<token>')}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < tries:
                time.sleep(wait); wait *= 2; continue
            raise RuntimeError(f"could not reach ENTSO-E: {str(exc).replace(key, '<token>')}") from None
    raise RuntimeError("unreachable")


def fetch_german_raw(first: pd.Timestamp, last: pd.Timestamp, folder: Path, key: str | None = None, opener=urllib.request.urlopen) -> dict:
    """Raw ENTSO-E XML for DE-LU, first 00:00 UTC to last 00:00 UTC: prices, load forecast, wind and solar forecast."""
    key = key or os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("ENTSOE_API_KEY is not set")
    folder.mkdir(parents=True, exist_ok=True)
    files, errors = {}, {}
    for item, params in GERMAN_ITEMS.items():
        for start, end in chunks(first, last):
            name = f"{item}_{start}_{end}.xml"
            try:
                content = _entsoe_get({**params, "periodStart": start, "periodEnd": end}, key, opener=opener)
                if key.encode() in content:
                    raise RuntimeError("the answer contains the key; not written")
                write_whole(folder / name, content)
                files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "no_data": b"<code>999</code>" in content}
            except Exception as exc:  # noqa: BLE001  one item that fails must not stop the others
                errors[name] = f"{exc.__class__.__name__}: {exc}"
            time.sleep(0.6)
    return {"folder": folder.name, "files": files, "errors": errors}


def parse_german_folder(folder: Path) -> dict:
    """spread_eval.parse_raw, tolerant of an item that is absent or empty in this folder."""
    def parts(stem):
        out = []
        for f in sorted(Path(folder).glob(f"{stem}_*.xml")):
            try:
                out += se.er.parse_document(f.read_bytes())
            except Exception:  # noqa: BLE001  an acknowledgement or a broken file holds no series
                continue
        return out

    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    price_parts, ws, load_parts = parts("prices_DE_LU"), parts("wind_solar_forecast_DE_LU"), parts("load_forecast_DE_LU")
    tso = pd.DataFrame({name.replace("wind_", ""): se.er.combine_resolutions([p for p in ws if p.psr_type == psr]) if ws else empty for psr, name in se.er.PSR_NAMES.items()})
    load = se.er.combine_resolutions(load_parts) if load_parts else empty
    tso = tso.reindex(tso.index.union(load.index))
    tso["load"] = load.reindex(tso.index)
    return {"de": se.coupled_german_price(price_parts) if price_parts else empty, "tso": tso.reindex(columns=list(se.TSO_COLUMNS))}


def german_refetched(paths: Paths, upto: pd.Timestamp, live: bool = True) -> tuple[dict, list[str]]:
    """The raw files of the first test, then every folder the logger fetched up to `upto`, later fetch winning."""
    new = parse_german_folder(paths.static / "raw_first_test")
    used = []
    folders = sorted((paths.live / "german_raw").glob("*")) if live and (paths.live / "german_raw").exists() else []
    for folder in folders:
        try:
            fetched = pd.Timestamp(folder.name).tz_localize("UTC") if pd.Timestamp(folder.name).tzinfo is None else pd.Timestamp(folder.name).tz_convert("UTC")
        except ValueError:
            continue
        if fetched > upto:
            continue
        later = parse_german_folder(folder)
        new = {"de": later["de"].combine_first(new["de"]), "tso": later["tso"].combine_first(new["tso"])}
        used.append(folder.name)
    return new, used


# ---- the second source of the coupled German price (amendment 5) --------------------------------------------------------------

def _second_source_get(first: pd.Timestamp, last: pd.Timestamp, opener=urllib.request.urlopen, tries: int = 4, wait: float = 10.0) -> bytes:
    url = SECOND_SOURCE_URL + "?" + urllib.parse.urlencode({"bzn": "DE-LU", "start": f"{first:%Y-%m-%d}", "end": f"{last:%Y-%m-%d}"})
    for attempt in range(1, tries + 1):
        try:
            with opener(urllib.request.Request(url, headers={"User-Agent": "thermo-fr spread logger (research, one request a morning)"}), timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < tries:
                time.sleep(wait); wait *= 2; continue
            raise RuntimeError(f"HTTP {exc.code} from energy-charts.info") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < tries:
                time.sleep(wait); wait *= 2; continue
            raise RuntimeError(f"could not reach energy-charts.info: {exc}") from None
    raise RuntimeError("unreachable")


def parse_second_source(content: bytes) -> pd.Series:
    """Hourly UTC prices from one saved answer: the mean of the four quarter hours of an hour, and only of an hour for which
    all four are held. Anything else in the answer (a missing value, a stamp off the quarter-hour grid, a unit other than
    EUR per MWh, a number outside -1000 to 10000) is not read as a price."""
    answer = json.loads(content)
    if str(answer.get("unit", "")).replace(" ", "") != "EUR/MWh":
        raise ValueError(f"unexpected unit {answer.get('unit')!r}")
    stamps, prices = answer["unix_seconds"], answer["price"]
    if len(stamps) != len(prices):
        raise ValueError("stamps and prices differ in number")
    low, high = SECOND_SOURCE_PLAUSIBLE
    pairs = [(t, float(v)) for t, v in zip(stamps, prices)  # a number only: not a missing value, a text or a truth value, and inside the plausible range
             if isinstance(t, (int, float)) and not isinstance(t, bool) and isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and low <= v <= high
             and 1_600_000_000 <= t <= 4_100_000_000]
    raw = pd.Series([v for _, v in pairs], index=pd.to_datetime([t for t, _ in pairs], unit="s", utc=True), dtype=float)
    raw = raw[(raw.index.minute % 15 == 0) & (raw.index.second == 0)]
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    if raw.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    hourly = raw.resample("1h")  # the same mean of quarter hours as the ENTSO-E path (dataset.to_hourly_utc)
    return hourly.mean()[hourly.count() == 4]


def fetch_second_source(first: pd.Timestamp, last: pd.Timestamp, folder: Path, stamp: str, opener=urllib.request.urlopen) -> dict:
    """One request: the day-ahead price of DE-LU for the delivery days `first` to `last`, saved untouched. An answer that
    cannot be read is saved and reported as a failure."""
    content = _second_source_get(first, last, opener=opener)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{stamp}.json"
    write_whole(folder / name, content)
    try:
        held = parse_second_source(content)
    except Exception as exc:  # noqa: BLE001  the answer stays on disk and the record names it
        raise RuntimeError(f"the answer could not be read ({exc.__class__.__name__}: {str(exc)[:120]}); saved as {name}, sha256 {hashlib.sha256(content).hexdigest()}, {len(content)} bytes") from None
    try:
        licence = str(json.loads(content).get("license_info", ""))[:200]
    except ValueError:
        licence = ""
    return {"source": SECOND_SOURCE_NAME, "file": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "asked_for_delivery_days": [f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}"],
            "complete_hours_held": int(len(held)), "licence_stated_in_the_answer": licence, "errors": {}}


def second_source_held(paths: Paths, upto: pd.Timestamp, live: bool = True) -> tuple[pd.Series, list[dict]]:
    """Every answer of the second source the logger saved up to `upto`, with the hash of each file read. A later answer wins
    where it holds a value; an hour it no longer holds is still read from an earlier one. A file that cannot be read holds
    nothing."""
    held, used = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")), []
    folder = paths.live / SECOND_SOURCE
    for file in sorted(folder.glob("*.json")) if live and folder.exists() else []:
        try:
            fetched = pd.Timestamp(file.stem)
            fetched = fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
        except ValueError:
            continue
        if fetched > upto:
            continue
        try:
            content = file.read_bytes()
            later = parse_second_source(content)
        except Exception:  # noqa: BLE001  a broken answer holds no price
            continue
        held = later.combine_first(held)
        used.append({"file": file.name, "sha256": hashlib.sha256(content).hexdigest()})
    return held, used


def with_second_source(de: pd.Series, second: pd.Series, day: pd.Timestamp) -> tuple[pd.Series, dict]:
    """The German price with the hours ENTSO-E lacks taken from the second source, and what was done.

    Only delivery days from LIVE_DATA_FROM to D-1 can be filled. The second source is trusted on a morning only if, over the
    ten delivery days before D, it agrees with ENTSO-E to a tenth of a cent on every hour both hold and both hold at least
    24 hours. Otherwise nothing is taken from it that morning, including hours taken on earlier mornings. An hour ENTSO-E
    holds is never replaced."""
    days = delivery_days(second.index) if len(second) else pd.DatetimeIndex([])
    later = int((np.asarray(days >= day)).sum()) if len(second) else 0
    second = second[np.asarray((days >= pd.Timestamp(LIVE_DATA_FROM)) & (days < day))] if len(second) else second
    mine = de.dropna()
    recent = second[np.asarray(delivery_days(second.index) >= day - pd.Timedelta(days=GERMAN_REFRESH_DAYS))] if len(second) else second
    common = recent.index.intersection(mine.index)
    gaps = (recent.reindex(common) - mine.reindex(common)).abs()
    agree = int((gaps <= SECOND_SOURCE_AGREE).sum())
    trusted = bool(len(common) >= SECOND_SOURCE_MIN_HOURS_IN_COMMON and agree == len(common))
    lacking = second[~second.index.isin(mine.index)]
    info = {"source": SECOND_SOURCE_NAME, "hours_both_hold_over_the_ten_days_before_D": int(len(common)), "of_which_agree": agree, "agreement_tolerance_eur_mwh": SECOND_SOURCE_AGREE,
            "largest_difference_eur_mwh": float(gaps.max()) if len(common) else None, "trusted_this_morning": trusted,
            "hours_it_holds_that_entsoe_lacks": int(len(lacking)), "hours_taken_from_it": 0, "hours_taken_from_it_by_delivery_day": {},
            "hours_of_D_or_later_in_its_answers_never_read": later}
    if not trusted or lacking.empty:
        return de, info
    by_day = pd.Series(1, index=lacking.index).groupby(np.asarray(delivery_days(lacking.index))).sum()
    info["hours_taken_from_it"] = int(len(lacking))
    info["hours_taken_from_it_by_delivery_day"] = {f"{d:%Y-%m-%d}": int(n) for d, n in by_day.items()}
    return de.combine_first(lacking).sort_index(), info


# ---- German weather points -------------------------------------------------------------------------------------------------------

def _points_get(lat, lon, variable, first, last, opener=urllib.request.urlopen, tries: int = 4) -> bytes:
    query = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": f"{first:%Y-%m-%d}", "end_date": f"{last:%Y-%m-%d}", "hourly": variable,
                                    "timezone": "UTC", "models": "best_match"})
    for attempt in range(1, tries + 1):
        try:
            with opener(urllib.request.Request(f"{POINTS_URL}?{query}", headers={"User-Agent": "thermo-fr spread logger"}), timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def fetch_german_points(first: pd.Timestamp, last: pd.Timestamp, folder: Path, stamp: str, columns: dict, opener=urllib.request.urlopen, pause: float = 0.25) -> dict:
    """Open-Meteo forecasts as issued two days ahead at the German points, UTC days first to last inclusive. `columns` maps
    kind to the column names of the static files, which carry the point names; the coordinates are those of the first fetch."""
    from importlib import util

    spec = util.spec_from_file_location("fetch_german_points", Path(__file__).resolve().parents[3] / "scripts" / "fetch_german_points.py")
    points = util.module_from_spec(spec)
    spec.loader.exec_module(points)  # the point lists and the units live in the fetch script of 2026-11-19
    folder.mkdir(parents=True, exist_ok=True)
    out = {"errors": {}}
    for kind, cfg in points.KINDS.items():
        table = {}
        for name, lat, lon in cfg["points"]:
            column = cfg["column"].format(name)
            if column not in columns[kind]:
                raise ValueError(f"{column} is not a column of the static {kind} file")
            try:
                content = _points_get(lat, lon, cfg["variable"], first, last, opener=opener)
                for t, v in points.parse(content, cfg["variable"], cfg["scale"]).items():
                    table.setdefault(t, {})[column] = v
            except Exception as exc:  # noqa: BLE001  a point that fails stays missing and is reported
                out["errors"][column] = f"{exc.__class__.__name__}: {exc}"
            time.sleep(pause)
        frame = pd.DataFrame.from_dict(table, orient="index").reindex(columns=list(columns[kind]))
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = "timestamp_utc"
        path = folder / f"{stamp}_{kind}.csv.gz"
        buffer = io.StringIO()
        frame.sort_index().to_csv(buffer, lineterminator="\n")
        write_whole(path, gzip.compress(buffer.getvalue().encode(), mtime=0))
        out[kind] = {"file": path.name, "sha256": fq.sha256_file(path), "hours": int(len(frame)), "hours_all_points_present": int(frame.notna().all(axis=1).sum())}
    return out


def points_tables(paths: Paths, stored: se.Data, upto: pd.Timestamp, live: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The hashed static point files, then the logger's own fetches oldest first. The first value seen for an hour is kept,
    so the value a forecast was issued with is also the value later rebuilds use."""
    tables, used = {"wind": stored.wind_pts.copy(), "solar": stored.solar_pts.copy()}, []
    folder = paths.live / "german_points"
    for path in sorted(folder.glob("*.csv.gz")) if live and folder.exists() else []:
        stamp, kind = path.name[: -len(".csv.gz")].rsplit("_", 1)
        fetched = pd.Timestamp(stamp)
        fetched = fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
        if fetched > upto or kind not in tables:
            continue
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.to_datetime(frame.index, utc=True)
        tables[kind] = tables[kind].combine_first(frame.reindex(columns=tables[kind].columns))
        used.append(path.name)
    return tables["wind"].sort_index().asfreq("h"), tables["solar"].sort_index().asfreq("h"), used


# ---- the tables of one morning ---------------------------------------------------------------------------------------------------

def build_data(paths: Paths, day: pd.Timestamp, upto: pd.Timestamp, live: bool = True) -> tuple[se.Data, dict]:
    """Everything the forecast of `day` may read, as fetched up to `upto`. German prices and ENTSO-E forecasts of `day` or later
    are dropped here, whatever the files hold."""
    stored = se.load_stored(paths.static / "inputs.csv", paths.static / "neighbour_fundamentals.csv", paths.static / "neighbour_sequences.csv.gz",
                            paths.static / "german_wind_points.csv.gz", paths.static / "german_solar_points.csv.gz")
    new, raw_used = german_refetched(paths, upto, live=live)
    data = se.extend(stored, new, last=f"{day:%Y-%m-%d}")
    try:
        second, second_used = second_source_held(paths, upto, live=live)
        filled, second_info = with_second_source(data.de, second, day)  # before the cut below, so nothing of D or later can pass
    except Exception as exc:  # noqa: BLE001  whatever the saved answers hold, they cannot stop a record: nothing is taken
        filled, second_used = data.de, []
        second_info = {"source": SECOND_SOURCE_NAME, "could_not_be_read": f"{exc.__class__.__name__}: {str(exc)[:200]}", "trusted_this_morning": False, "hours_taken_from_it": 0,
                       "hours_taken_from_it_by_delivery_day": {}}
    data = replace(data, de=filled)
    before = lambda obj: obj[np.asarray(delivery_days(obj.index) < day)]  # noqa: E731
    dropped = {"german_price_hours_of_D_or_later": int((delivery_days(data.de.index) >= day).sum()), "entsoe_forecast_hours_of_D_or_later": int((delivery_days(data.tso.index) >= day).sum())}
    # The frozen feature code adds onshore and offshore with a missing value read as 0 MW. On the days only the logger holds, a row
    # with no wind forecast at all is therefore removed whole, so its lags come out missing and not as zero. Stored days are untouched.
    no_wind = np.asarray((delivery_days(data.tso.index) >= pd.Timestamp(LIVE_DATA_FROM)) & data.tso["onshore"].isna() & data.tso["offshore"].isna())
    dropped["entsoe_rows_without_a_wind_forecast_removed"] = int(no_wind.sum())
    data = replace(data, tso=data.tso[~no_wind])
    french, french_info = french_history(paths, day, upto, windows=live)
    wind, solar, points_used = points_tables(paths, stored, upto, live=live)
    data = replace(data, french=french, de=before(data.de), tso=before(data.tso), wind_pts=wind, solar_pts=solar, last=f"{day:%Y-%m-%d}")
    info = {"french": french_info, "german_raw_folders_used": raw_used, "german_point_files_used": points_used, "dropped": dropped,
            SECOND_SOURCE: {**second_info, "files_used": second_used},
            "latest_german_price_hour_utc": iso(data.de.dropna().index.max()), "latest_entsoe_forecast_hour_utc": iso(data.tso.dropna(how="all").index.max()),
            "latest_french_price_hour_utc": iso(french["price_eur_mwh"].dropna().index.max())}
    return data, info


def differing_columns(a: pd.DataFrame, b: pd.DataFrame, tolerance: float = 0.0) -> dict:
    """Columns whose values differ between two tables of the same hours, by value and not by storage type: {column: detail}.
    With a tolerance, two values differ when they are further apart than tolerance * max(1, |x|, |y|); a value missing in one
    table only always differs."""
    out = {}
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {"<shape>": f"{a.shape} against {b.shape}"}
    for column in a.columns:
        x, y = pd.to_numeric(a[column], errors="coerce").to_numpy(float), pd.to_numeric(b[column], errors="coerce").to_numpy(float)
        both = ~(np.isnan(x) | np.isnan(y))
        with np.errstate(invalid="ignore"):
            close = np.isfinite(x) & np.isfinite(y) & (np.abs(x - y) <= tolerance * np.maximum(1.0, np.maximum(np.abs(x), np.abs(y))))
        differ = (np.isnan(x) != np.isnan(y)) | (both & (x != y) & ~close)
        if differ.any():
            with np.errstate(invalid="ignore", over="ignore"):
                gaps = (x - y)[differ & both]  # the hours that differ; an infinite value on one side gives no finite gap
            out[column] = {"hours": int(differ.sum()), "missing_in_one_only": int((np.isnan(x) != np.isnan(y)).sum()),
                           "largest_difference": float(np.max(np.abs(gaps[np.isfinite(gaps)]))) if np.isfinite(gaps).any() else None}
    return out


def fingerprint(table: pd.DataFrame) -> dict:
    """Per column: missing count, sum and sum of squares. Enough to see whether two machines built the same table, and where not."""
    values = table.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return {"rows": int(len(table)), "columns": {c: [int(np.isnan(values[:, k]).sum()), float(np.nansum(values[:, k])), float(np.nansum(values[:, k] ** 2))]
                                                 for k, c in enumerate(table.columns)},
            "bits": {c: hashlib.sha256(np.ascontiguousarray(values[:, k]).tobytes()).hexdigest()[:16] for k, c in enumerate(table.columns)}}


def compare_fingerprints(mine: dict, reference: dict, rtol: float = 1e-9) -> dict:
    """Columns whose fingerprint differs from the reference: {column: [mine, reference]}."""
    out = {}
    if mine["rows"] != reference["rows"]:
        out["<rows>"] = [mine["rows"], reference["rows"]]
    for column in sorted(set(mine["columns"]) | set(reference["columns"])):
        a, b = mine["columns"].get(column), reference["columns"].get(column)
        if a is None or b is None or a[0] != b[0] or any(abs(x - y) > rtol * max(1.0, abs(x), abs(y)) for x, y in zip(a[1:], b[1:])):
            out[column] = [a, b]
    return out


def lightgbm_probe() -> dict:
    """LightGBM on a small synthetic table that is bit for bit the same on every machine (numpy's seeded generator), with the
    frozen settings and with a single-threaded deterministic variant. If the forecasts differ between machines here, the
    difference is in the library on that machine and not in the data."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 12))
    y = 30 * np.tanh(X[:, 0]) + 10 * X[:, 1] * X[:, 2] + 5 * np.abs(X[:, 3]) + rng.normal(scale=20, size=6000)
    out = {"input_checksum": float(X.sum() + y.sum())}
    for name, extra in (("frozen_settings", {}), ("single_thread_deterministic", {"n_jobs": 1, "deterministic": True, "force_row_wise": True})):
        model = lgb.LGBMRegressor(**{**se.PARAMS, **extra}).fit(X[:5000], y[:5000])
        out[name] = [float(v) for v in model.predict(X[5000:5024])]
    return out


def forecasts(data: se.Data, day: pd.Timestamp, log=lambda *_: None) -> tuple[pd.DataFrame, dict, dict]:
    """The six forecasts of `day`, how each model was trained, and the evidence about its inputs."""
    feat = se.features(data)
    frame = feat["frame"]
    rows = np.asarray(frame["delivery_day"] == day)
    if not rows.any():
        raise fq.ForwardInputError(f"no rows for {day:%Y-%m-%d}")
    out = pd.DataFrame(index=frame.index[rows])
    month = day.replace(day=1)
    monthly = se.fit_months(feat, [month], models=("H", "G"))
    out["H"], out["G"] = monthly.loc[rows, "H"], monthly.loc[rows, "G"]
    out["W"] = sw.fit_w(feat, f"{day:%Y-%m-%d}", f"{day:%Y-%m-%d}")[rows]
    for name in ("HW", "GK", "GM"):
        out[name] = shw.fit_comparator(feat, name, f"{day:%Y-%m-%d}", f"{day:%Y-%m-%d}")[rows]
        log(f"{name} done")
    week = sw.refit_days(f"{day:%Y-%m-%d}", f"{day:%Y-%m-%d}")[0]
    labelled = frame.loc[feat["y"].notna(), "delivery_day"]

    def training(block_start, weekly, half_life):
        days = labelled[labelled < block_start]
        return {"block_start": f"{block_start:%Y-%m-%d}", "refit": "weekly" if weekly else "monthly", "half_life_days": half_life, "training_hours": int(len(days)),
                "last_training_day": f"{days.max():%Y-%m-%d}" if len(days) else None}

    models = {"H": training(month, False, None), "G": training(month, False, None), "W": training(week, True, sw.HALF_LIFE_DAYS),
              "HW": training(week, True, sw.HALF_LIFE_DAYS), "GK": training(week, True, None), "GM": training(month, False, sw.HALF_LIFE_DAYS)}
    g_rows, h_rows = feat["sets"]["G"].loc[rows], feat["sets"]["H"].loc[rows]
    missing = {c: int(n) for c, n in g_rows.isna().sum().items() if n}
    scrambled = se.features(se.scrambled(data, f"{day:%Y-%m-%d}"))
    rows_s = np.asarray(scrambled["frame"]["delivery_day"] == day)
    pairs = [(name, table, scrambled["sets"][name].loc[rows_s]) for name, table in (("H", h_rows), ("G", g_rows))]
    exact = {name: differing_columns(table, other) for name, table, other in pairs}
    moved = {name: differing_columns(table, other, SCRAMBLE_TOLERANCE) for name, table, other in pairs}
    evidence = {"hours": int(rows.sum()), "input_hours_missing_for_D": missing, "inputs_with_a_missing_hour": len(missing),
                "scramble_tolerance": SCRAMBLE_TOLERANCE,
                "inputs_identical_when_everything_unknown_at_the_gate_is_scrambled": {name: not columns for name, columns in moved.items()},
                "inputs_bit_identical_under_the_scramble": {name: not columns for name, columns in exact.items()},
                "inputs_that_moved_under_the_scramble": moved["G"],
                "inputs_that_differ_only_by_rounding_under_the_scramble": {c: d for c, d in exact["G"].items() if c not in moved["G"]},
                "german_proxies_present_hours": int((feat["px"].loc[rows, ["wind", "solar"]].notna().all(axis=1)).sum()),
                "yesterdays_spread_present_hours": int(feat["yesterday"][rows].notna().sum()),
                "german_price_hours_missing_on_the_eve": max(0, int((frame["delivery_day"] == day - pd.Timedelta(days=1)).sum())
                                                                 - int(data.de[np.asarray(delivery_days(data.de.index) == day - pd.Timedelta(days=1))].notna().sum()))}
    known = np.asarray(frame["delivery_day"] <= day)
    evidence["fingerprint"] = {"inputs_of_D": fingerprint(g_rows), "all_rows_to_D": fingerprint(feat["sets"]["G"].loc[known]),
                               "target": fingerprint(feat["y"][known].to_frame("spread")), "W_weights_sum": float(sw.weights(labelled[labelled < week], week).sum())}
    return out[list(MODELS)], models, evidence


# ---- records -----------------------------------------------------------------------------------------------------------------

def _append(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def record_path(paths: Paths, label: str) -> Path:
    return paths.records / "issued" / f"{label}.json"


def issued_records(paths: Paths) -> tuple[list[dict], list[str]]:
    """Every readable record, and the names of files that cannot be read (reported, never repaired)."""
    good, bad = [], []
    for path in sorted((paths.records / "issued").glob("*.json")):
        try:
            good.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            bad.append(path.name)
    return good, bad


def write_once(path: Path, record: dict) -> Path:
    """Write the whole record to a temporary file, then link it to its final name. The link is atomic and fails if the name
    exists, so a day has at most one record, it is never replaced, and a half-written record never carries the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "x") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise
    except OSError:  # a file system without hard links: exclusive creation still refuses a second record
        with open(path, "x") as f:
            f.write(tmp.read_text())
    finally:
        os.unlink(tmp)
    return path


def run_fetches(steps: dict, clock, budgets: dict) -> dict:
    """Run the fetches side by side, each in its own thread. Returns {name: {"began", "done", and "result" or "error"}} in
    the order of `steps`. A budget is the least time a fetch is waited for, counted from the common start: the fetches are
    looked at in order, so one listed after a slower one is waited for until that one is done, and the whole never waits
    longer than the largest budget. A fetch still running when it is looked at after its budget is given up on. Its thread
    dies with the process and can no longer change what is returned here; a file it finishes later is written whole
    (write_whole) and is named in no record."""
    boxes = {name: {} for name in steps}

    def work(box, step):
        try:
            box["began"] = clock()
            box["result"] = step()
        except BaseException as exc:  # noqa: BLE001  whatever ends a fetch is written down, not lost with the thread
            box["error"] = exc
        try:
            box["done"] = clock()
        except BaseException as exc:  # noqa: BLE001
            box.setdefault("error", exc)
            box["done"] = None

    threads = {name: threading.Thread(target=work, args=(boxes[name], step), daemon=True) for name, step in steps.items()}
    start, started_at = time.monotonic(), clock()
    for thread in threads.values():
        thread.start()
    out = {}
    for name, thread in threads.items():
        seconds = budgets.get(name, 150)
        thread.join(max(0.0, seconds - (time.monotonic() - start)))
        alive = thread.is_alive()  # read before the copy: a thread seen as ended has written everything it will write
        box = dict(boxes[name])  # a copy: a thread that is still running writes to the original, not to this
        if alive:
            box = {"began": box.get("began", started_at), "done": clock(), "error": TimeoutError(f"not finished after {seconds:.0f} seconds")}
        elif "error" in box or "result" not in box:
            box = {"began": box.get("began", started_at), "done": box.get("done") or clock(), "error": box.get("error", RuntimeError("ended without a result"))}
        elif box.get("done") is None:
            box["done"] = clock()
        out[name] = box
    return out


def fetch_steps(paths: Paths, day: pd.Timestamp, now: pd.Timestamp, stamp: str) -> dict:
    """The four fetches of a morning. Each range starts where the files already held end, so a missed stretch leaves no hole."""
    def german():
        held, _ = german_refetched(paths, now)
        latest = held["de"].dropna().index.max()
        first = min(day - pd.Timedelta(days=GERMAN_REFRESH_DAYS), latest.tz_localize(None).normalize() - pd.Timedelta(days=1))
        result = fetch_german_raw(max(first, pd.Timestamp(GERMAN_LIVE_FROM)), day + pd.Timedelta(days=1), paths.live / "german_raw" / stamp)
        result["request_errors"] = sorted(result["errors"])  # requests that failed, as distinct from an answer of "no data"
        for name, meta in result["files"].items():
            if meta["no_data"]:
                result["errors"][name] = "ENTSO-E answered that it has no data"
        return result

    def points():
        stored = {kind: pd.read_csv(paths.static / f"german_{kind}_points.csv.gz", index_col=0) for kind in ("wind", "solar")}
        columns = {kind: list(frame.columns) for kind, frame in stored.items()}
        held = se.Data(french=None, de=None, tso=None, wind_pts=_utc(stored["wind"]), solar_pts=_utc(stored["solar"]), last="")
        wind, _, _ = points_tables(paths, held, now)
        latest = wind.dropna(how="all").index.max()
        first = min(day - pd.Timedelta(days=POINT_REFRESH_DAYS), latest.tz_localize(None).normalize() - pd.Timedelta(days=1))
        return fetch_german_points(first, day, paths.live / "german_points", stamp, columns)

    def second():  # the ten delivery days before D, never D itself
        return fetch_second_source(max(day - pd.Timedelta(days=GERMAN_REFRESH_DAYS), pd.Timestamp(LIVE_DATA_FROM)), day - pd.Timedelta(days=1), paths.live / SECOND_SOURCE, stamp)

    return {"french_window": lambda: fetch_french_window(day, paths), "german_raw": german, SECOND_SOURCE: second, "german_points": points}


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index().asfreq("h")


def issue(paths: Paths, dry_run: bool = False, clock=utc_now, fetch: bool = True, log=print, sleep=time.sleep) -> dict:
    """One morning: fetch, forecast, write. Every way out leaves a line in attempts.jsonl. A dry run does everything under
    live/dry_run, at any time of day, and its record says so."""
    if dry_run:
        paths = replace(paths, live_dir=Path(paths.root) / "live" / "dry_run")
    started = clock()
    day = next_delivery_day(started)
    gate = gates(day)
    label = f"{day:%Y-%m-%d}"
    attempts = paths.records / "attempts.jsonl"

    def out(status: str, **more) -> dict:
        _append(attempts, {"at_utc": iso(clock()), "started_utc": iso(started), "delivery_day": label, "outcome": status, **more})
        return {"delivery_day": label, "status": status, **more}

    if not dry_run:
        if started < gate["earliest"]:
            if gate["earliest"] - started > WAIT_IF_EARLY_BY:
                return out("too early", reason="started before 10:00 Paris on the eve; nothing fetched, nothing written")
            sleep((gate["earliest"] - started).total_seconds())
            started = clock()
        if started >= gate["auction"]:
            return out("missed", reason="started after the auction closed")
        if record_path(paths, label).exists():
            return out("already issued")
    running = paths.records / "running" / f"{label}.json"
    running.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(running, "x") as f:
            json.dump({"started_utc": iso(started), "pid": os.getpid()}, f)
    except FileExistsError:
        try:
            other = pd.Timestamp(json.loads(running.read_text())["started_utc"])
        except (OSError, ValueError, KeyError):
            other = started - RUNNING_STALE_AFTER
        if started - other < RUNNING_STALE_AFTER:
            return out("another run is in progress", since_utc=iso(other))
        running.write_text(json.dumps({"started_utc": iso(started), "pid": os.getpid(), "took_over_a_stale_marker": True}))
    try:
        return _issue(paths, day, gate, started, dry_run, clock, fetch, log, out)
    except FileExistsError:
        return out("already issued", reason="another run wrote the record first")
    except Exception as exc:  # noqa: BLE001  nothing may fail without a trace
        trace = traceback.format_exc().splitlines()[-12:]
        log("\n".join(trace))
        return out("failed", error=f"{exc.__class__.__name__}: {exc}", trace=trace)
    finally:
        try:
            running.unlink()
        except OSError:
            pass


def _issue(paths: Paths, day: pd.Timestamp, gate: dict, started: pd.Timestamp, dry_run: bool, clock, fetch: bool, log, out) -> dict:
    label = f"{day:%Y-%m-%d}"
    record = {"delivery_day": label, "started_at_utc": iso(started), "earliest_start_utc": iso(gate["earliest"]), "cutoff_utc": iso(gate["cutoff"]),
              "auction_close_utc": iso(gate["auction"]), "dry_run": bool(dry_run), "models": list(MODELS), "code_commit": code_commit(), "code_sha256": code_sha256(),
              "versions": fq.versions_used(), "known_by": KNOWN_BY, "flags": [], "fetch": {}, "scored": False,
              "clock": "every time in this record is the laptop's own clock", "note": "No error against any price is computed by the logger."}
    if paths.production is None or not (Path(paths.production) / "data" / "shadow" / "fresh").exists():
        record["flags"].append("production folder not found: no production window is read")
    failed = []
    if fetch:
        for name, box in run_fetches(fetch_steps(paths, day, started, stamp_of(started)), clock, FETCH_BUDGET_SECONDS).items():
            begun, done = box["began"], box["done"]
            if "error" in box:  # a source that is down is written down
                result = {"error": f"{box['error'].__class__.__name__}: {box['error']}"}
                if name != SECOND_SOURCE:  # the second source failing postpones only if the German price of the eve is then missing (below)
                    failed.append(name)
                record["flags"].append(f"{name} failed")
            else:
                result = box["result"]
            record["fetch"][name] = {**result, "began_utc": iso(begun), "finished_utc": iso(done), "finished_by_cutoff": bool(done <= gate["cutoff"])}
            if result.get("errors"):
                record["flags"].append(f"{name}: {len(result['errors'])} item(s) failed")
            if result.get("absent_for_D_though_due_by_the_gate"):
                record["flags"].append(f"{name}: not on the source for D though due by the gate: {', '.join(result['absent_for_D_though_due_by_the_gate'])}")
            log(f"{name}: finished {iso(done)}")
        record["fetches_ran"] = "side by side"
        record["all_fetches_finished_by_cutoff"] = all(v["finished_by_cutoff"] for v in record["fetch"].values())
        if not record["all_fetches_finished_by_cutoff"]:
            record["flags"].append("a fetch finished after 10:05 Paris")
        if failed and not dry_run and clock() < gate["last_chance"]:
            return out("postponed", reason=f"{', '.join(failed)} failed; nothing written, the next trigger tries again", fetch=record["fetch"])
    data, info = build_data(paths, day, upto=clock())
    record["inputs"] = info
    last_window = info["french"]["last_window_issued_utc"]
    if last_window and pd.Timestamp(last_window) > gate["cutoff"]:
        record["flags"].append("the latest French window used was written after 10:05 Paris")
    if info["dropped"]["entsoe_rows_without_a_wind_forecast_removed"]:
        record["flags"].append("ENTSO-E rows without a German wind forecast were removed whole, so their lags are missing and not zero")
    record["static_sha256"] = {**{f.name: fq.sha256_file(f) for f in sorted(paths.static.glob("*")) if f.is_file()}, "raw_first_test": folder_sha256(paths.static / "raw_first_test")}
    curves, models, evidence = forecasts(data, day, log=log)
    record["fingerprint"] = evidence.pop("fingerprint")
    record["training"], record["evidence"] = models, evidence
    # Only a missing load forecast of D postpones: it is due at 10:00 and a few minutes can bring it. A missing price of D-1 was
    # due the day before, so waiting half an hour gains nothing and makes every fetch late (amendment 4).
    # A price request that failed this morning is a different case from a price the source does not hold: trying again can
    # bring it, so it postpones as a failed source does.
    not_there_yet = ["the French load forecast of D"] if "load_fc_mw" in evidence["input_hours_missing_for_D"] else []
    price_requests_failed = [what for what, failed_request in (
        ("the French price request failed", "prices" in (record["fetch"].get("french_window", {}).get("errors") or {})),
        ("the German price request failed", bool(evidence["german_price_hours_missing_on_the_eve"])  # with the second source in, a failed ENTSO-E request can leave nothing missing
         and any(n.startswith("prices_DE_LU") for n in record["fetch"].get("german_raw", {}).get("request_errors") or []))) if failed_request]
    if price_requests_failed and evidence["yesterdays_spread_present_hours"] < evidence["hours"]:
        not_there_yet.append(f"yesterday's spread ({' and '.join(price_requests_failed)})")
    elif "error" in record["fetch"].get(SECOND_SOURCE, {}) and evidence["german_price_hours_missing_on_the_eve"]:
        not_there_yet.append("the German price of the eve (the files fetched from ENTSO-E hold none and the request to the second source failed)")
    if not_there_yet and not dry_run and clock() < gate["last_chance"]:
        return out("postponed", reason=f"not on the sources yet: {'; '.join(not_there_yet)}; nothing written, the next trigger tries again", fetch=record["fetch"])
    if not all(evidence["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values()):
        record["flags"].append("an input of D depends on a value not known at the gate")
    elif not all(evidence["inputs_bit_identical_under_the_scramble"].values()):
        record["flags"].append("an input of D differs by rounding only (under one part in a billion) when everything unknown at the gate is scrambled")
    if evidence["inputs_with_a_missing_hour"]:
        record["flags"].append(f"{evidence['inputs_with_a_missing_hour']} input(s) missing for part of D and left missing")
    if "load_fc_mw" in evidence["input_hours_missing_for_D"]:
        record["flags"].append("the French load forecast of D is missing for part of D: issued without it")
    if evidence["yesterdays_spread_present_hours"] < evidence["hours"]:
        record["flags"].append("yesterday's spread is missing for part of D: its lags are missing, and decoupled_share_l1 reads 0 (frozen feature code)")
    if evidence["german_proxies_present_hours"] < evidence["hours"]:
        record["flags"].append("a German proxy is missing for part of D")
    second = info.get(SECOND_SOURCE) or {}
    if second.get("hours_taken_from_it"):
        record["flags"].append(f"the coupled German price of {second['hours_taken_from_it']} hour(s) before D came from the second source (energy-charts.info) because the files fetched from ENTSO-E held none")
    elif second.get("could_not_be_read"):
        record["flags"].append("the saved answers of the second source could not be read: nothing was taken from it")
    elif second.get("hours_it_holds_that_entsoe_lacks") and not second.get("trusted_this_morning"):
        record["flags"].append("the second source holds German prices that the files fetched from ENTSO-E lack but was not used: "
                               + (f"it disagrees with ENTSO-E on {second['hours_both_hold_over_the_ten_days_before_D'] - second['of_which_agree']} of {second['hours_both_hold_over_the_ten_days_before_D']} hours both hold"
                                  if second["hours_both_hold_over_the_ten_days_before_D"] >= SECOND_SOURCE_MIN_HOURS_IN_COMMON else f"only {second['hours_both_hold_over_the_ten_days_before_D']} hour(s) held by both over the ten days before D"))
    if curves.isna().any().any():
        record["flags"].append("a forecast is missing")
    record["timestamps_utc"] = [iso(t) for t in curves.index]
    record["forecasts"] = {name: [None if pd.isna(v) else float(v) for v in curves[name].to_numpy()] for name in MODELS}
    issued = clock()  # taken immediately before the write
    record["issued_at_utc"] = iso(issued)
    record["issued_by_cutoff"] = bool(issued <= gate["cutoff"])
    record["saved_before_auction"] = bool(issued < gate["auction"])
    record["seconds"] = round((issued - started).total_seconds(), 1)
    if dry_run:
        path = write_once(paths.records / "issued" / f"{label}_{stamp_of(issued)}.json", record)
        return out("dry run written", record=path.as_posix(), flags=record["flags"], seconds=record["seconds"])
    if not record["saved_before_auction"]:
        return out("missed", reason="finished after the auction closed")
    path = write_once(record_path(paths, label), record)
    written = clock()
    fq.manifest_append(paths.records / "manifest.jsonl", [{"kind": "spread_record", "path": path.name, "sha256": fq.sha256_file(path), "bytes": path.stat().st_size,
                                                             "issued_at_utc": record["issued_at_utc"], "listed_at_utc": iso(written)}])
    return out("issued", record=path.as_posix(), flags=record["flags"], seconds=record["seconds"])


def selftest_material(paths: Paths, log=print) -> tuple[pd.DataFrame, dict, dict]:
    day = pd.Timestamp(SELFTEST_DAY)
    data, _ = build_data(paths, day, upto=utc_now(), live=False)
    curves, _, evidence = forecasts(data, day, log=log)
    return curves, evidence, lightgbm_probe()


def selftest(paths: Paths, log=print) -> dict:
    """No network. Rebuild the forecasts of SELFTEST_DAY from the static files alone, through the logger's own code path, and
    compare them with the frozen batch forecasts shipped as static/reference.json. It involves no price of the test window.

    Three things are compared with the reference machine, in this order: the tables (every input column and the target, over
    all rows), LightGBM on a synthetic table that is identical everywhere, and the six forecasts. The verdict says which of
    them, if any, differs."""
    reference = json.loads((paths.static / "reference.json").read_text())
    curves, evidence, probe = selftest_material(paths, log=log)
    stamps = [iso(t) for t in curves.index]
    prints = evidence.pop("fingerprint")
    tables = {name: compare_fingerprints(prints[name], reference["fingerprint"][name]) for name in ("inputs_of_D", "all_rows_to_D", "target")}
    weights_same = abs(prints["W_weights_sum"] - reference["fingerprint"]["W_weights_sum"]) <= 1e-9 * abs(reference["fingerprint"]["W_weights_sum"])
    last_bits = {name: sorted(c for c, h in prints[name]["bits"].items() if reference["fingerprint"][name]["bits"].get(c) != h) for name in ("inputs_of_D", "all_rows_to_D", "target")}
    library = {name: float(np.max(np.abs(np.asarray(probe[name]) - np.asarray(reference["lightgbm_probe"][name])))) for name in ("frozen_settings", "single_thread_deterministic")}
    result = {"day": SELFTEST_DAY, "hours": len(stamps), "timestamps_match": stamps == reference["timestamps_utc"], "code_sha256": code_sha256(),
              "code_matches_reference": code_sha256() == reference.get("code_sha256"), "versions": fq.versions_used(), "reference_machine": reference.get("reference_machine"),
              "tables": {"identical_to_the_reference_machine": not any(tables.values()) and weights_same,
                         "bit_for_bit_identical": not any(last_bits.values()),
                         "columns_not_bit_for_bit_identical": {name: len(columns) for name, columns in last_bits.items()},
                         "examples_not_bit_for_bit": last_bits["all_rows_to_D"][:12],
                         "number_of_columns_that_differ": {name: len(diff) for name, diff in tables.items()},
                         "columns_that_differ": {name: dict(list(diff.items())[:12]) for name, diff in tables.items() if diff}},
              "lightgbm_on_identical_synthetic_data": {"input_identical": abs(probe["input_checksum"] - reference["lightgbm_probe"]["input_checksum"]) < 1e-9,
                                                       "largest_difference_from_the_reference_machine": library},
              "forecasts": {"largest_difference": {}, "mean_absolute_difference": {}}, "evidence": evidence}
    for name in MODELS:
        gap = np.abs(curves[name].to_numpy() - np.asarray(reference["forecasts"][name], dtype=float))
        result["forecasts"]["largest_difference"][name] = float(gap.max())
        result["forecasts"]["mean_absolute_difference"][name] = float(gap.mean())
    exact = max(result["forecasts"]["largest_difference"].values()) <= SELFTEST_TOLERANCE
    same_tables = result["tables"]["identical_to_the_reference_machine"]
    bitwise = result["tables"]["bit_for_bit_identical"]
    same_library = max(library.values()) <= SELFTEST_TOLERANCE
    in_band = (max(result["forecasts"]["mean_absolute_difference"].values()) <= NUMERICAL_BAND_MEAN and max(result["forecasts"]["largest_difference"].values()) <= NUMERICAL_BAND_MAX)
    result["numerical_band"] = {"mean": NUMERICAL_BAND_MEAN, "largest": NUMERICAL_BAND_MAX, "inside": bool(in_band)}
    if not same_tables:
        verdict, ok = "the tables differ from the reference machine: the data or the feature code behaves differently here", False
    elif not same_library:
        verdict, ok = "the tables agree, but LightGBM gives different numbers on this machine even on identical synthetic data", False
    elif exact:
        verdict, ok = "forecasts identical to the reference machine", True
    elif not bitwise and in_band:
        verdict, ok = ("the tables agree with the reference machine to one part in a billion but not bit for bit, LightGBM is identical on identical data, and the "
                       "forecasts differ by an amount the models show for any change in the last bits of their inputs: same model, machine-dependent numbers"), True
    else:
        verdict, ok = "the forecasts differ from the reference machine by more than last-bit differences in the inputs explain", False
    result["verdict"] = verdict
    result["pass"] = bool(ok and result["timestamps_match"] and all(evidence["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values()))
    return result


def verify(paths: Paths) -> dict:
    """The manifest chain and the hash of every listed record, recomputed. Reported, never repaired."""
    entries = fq.read_manifest(paths.records / "manifest.jsonl")
    prev, broken, missing, changed = fq.GENESIS, [], [], []
    for k, entry in enumerate(entries):
        if entry.get("prev") != prev or fq._chain_hash(entry) != entry.get("line_sha256"):
            broken.append(k)
        prev = entry.get("line_sha256")
        file = paths.records / "issued" / entry["path"]
        if not file.exists():
            missing.append(entry["path"])
        elif fq.sha256_file(file) != entry["sha256"]:
            changed.append(entry["path"])
    listed = {e["path"] for e in entries}
    unlisted = sorted(f.name for f in (paths.records / "issued").glob("*.json") if f.name not in listed)
    return {"lines": len(entries), "broken_links": broken, "missing_files": missing, "changed_files": changed, "records_not_in_the_manifest": unlisted,
            "ok": not (broken or missing or changed or unlisted)}


def status(paths: Paths) -> dict:
    """Counts only. No error of any forecast is computed here or anywhere in this module."""
    records, unreadable = issued_records(paths)
    attempts_file = paths.records / "attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_file.read_text().splitlines() if line.strip()] if attempts_file.exists() else []
    not_issued = [a for a in attempts if a.get("outcome") not in ("issued", "already issued")]
    return {"records": len(records), "first_day": records[0]["delivery_day"] if records else None, "last_day": records[-1]["delivery_day"] if records else None,
            "issued_by_cutoff": sum(bool(r.get("issued_by_cutoff")) for r in records), "all_fetches_by_cutoff": sum(bool(r.get("all_fetches_finished_by_cutoff")) for r in records),
            "with_flags": {r["delivery_day"]: r["flags"] for r in records if r["flags"]}, "unreadable_records": unreadable, "manifest": verify(paths),
            "attempts_without_a_record": not_issued[-20:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Morning logger of the spread forecasts")
    parser.add_argument("command", choices=["issue", "dry-run", "selftest", "status"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--production", default=None)
    args = parser.parse_args(argv)
    paths = Paths(Path(args.root), Path(args.production) if args.production else None)
    if args.command == "selftest":
        result = selftest(paths)
    elif args.command == "status":
        result = status(paths)
    else:
        if paths.production is None:
            parser.error("--production is required")
        logs = paths.live / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / f"{stamp_of(utc_now())}_{args.command}.log", "a") as handle:  # the scheduled task captures no output of its own
            def log(*parts):
                line = " ".join(str(part) for part in parts)
                print(line)
                handle.write(f"{iso(utc_now())} {line}\n")
                handle.flush()

            result = issue(paths, dry_run=args.command == "dry-run", log=log)
            log(json.dumps(result, default=str))
    print(json.dumps(result, indent=1, default=str))
    return 0 if result.get("pass", True) and result.get("status") != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
