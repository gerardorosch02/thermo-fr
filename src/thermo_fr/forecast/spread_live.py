"""The morning logger of the spread forecasts (docs/prereg/spread_w_amendment_2.md, changed by spread_w_amendment_4.md).

Each morning it fetches what is public, forecasts the France minus Germany spread of the next delivery day with H, HW, G,
GK, GM and W, and writes one time-stamped record that is never rewritten. It computes no error against any price, ever.

    python -m thermo_fr.forecast.spread_live selftest --root <folder>                      no network: rebuild a pre-window day
    python -m thermo_fr.forecast.spread_live dry-run  --root <folder> --production <folder> full run, written under live/dry_run
    python -m thermo_fr.forecast.spread_live issue    --root <folder> --production <folder> the scheduled run
    python -m thermo_fr.forecast.spread_live status   --root <folder>                      counts only

Folder layout under --root:
  static/   inputs.csv, neighbour_fundamentals.csv, neighbour_sequences.csv.gz, raw_first_test/*.xml,
            german_wind_points.csv.gz, german_solar_points.csv.gz, reference.json          never changed after installation
  live/     french_windows/, german_raw/<stamp>/, german_points/, records/issued/, records/manifest.jsonl,
            records/attempts.jsonl, cache/, dry_run/                                       written by the logger
--production is the production folder of the public forecast. The logger only reads there: the French input windows saved by
the morning runs (data/shadow/fresh) and the proxy weights (published/model).

What "available by 10:05 Paris" means here. The forecast of delivery day D is meant to be issued at 10:05 Paris on D-1.
  - Nothing is fetched before 10:00 Paris on D-1, the deadline of the load forecast of D. A run that starts earlier writes
    nothing.
  - Every source is fetched that morning, the three fetches side by side, and the time each one finished is written into the
    record, with a flag if it finished after 10:05.
  - German prices and ENTSO-E forecasts of D or later are dropped before any feature is built, whatever the fetch returned.
  - The features of D are then built a second time from the same tables with every value that cannot be known at the gate
    replaced by noise (spread_eval.scrambled). The record states whether the inputs of D came out the same to one part in
    a billion, and separately whether they came out bit for bit identical. This tests the position of every value used. It
    cannot test whether a value was revised between its publication and the fetch.
  - One record per delivery day, created atomically, written before the auction closes at 12:00 Paris or not at all.
  - If a source fails on the first run of the morning, or the French load forecast of D is not there yet, nothing is
    written and the second run tries again. The second run issues with what there is, and the record says what was missing.
  - A price of D-1 that is not on the source (yesterday's spread) does not postpone anything: the first run issues with
    those inputs missing, and says so. Waiting could only make the fetch late.
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
FETCH_BUDGET_SECONDS = {"french_window": 300, "german_raw": 150, "german_points": 150}
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
        fresh.to_csv(path)
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
                (folder / name).write_bytes(content)
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
        path.write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
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
            apart = np.abs(x - y) > tolerance * np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))
        differ = (np.isnan(x) != np.isnan(y)) | (both & apart)
        if differ.any():
            out[column] = {"hours": int(differ.sum()), "missing_in_one_only": int((np.isnan(x) != np.isnan(y)).sum()),
                           "largest_difference": float(np.max(np.abs(x[both] - y[both]))) if both.any() else None}
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
                "yesterdays_spread_present_hours": int(feat["yesterday"][rows].notna().sum())}
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
    """Run the fetches side by side, each in its own thread with its own time budget counted from the common start.
    Returns {name: {"began", "done", and "result" or "error"}} in the order of `steps`. A source that hangs is given up on
    when its budget ends; its thread dies with the process and can no longer change what is returned."""
    boxes = {name: {} for name in steps}

    def work(box, step):
        box["began"] = clock()
        try:
            box["result"] = step()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        box["done"] = clock()

    threads = {name: threading.Thread(target=work, args=(boxes[name], step), daemon=True) for name, step in steps.items()}
    start, started_at = time.monotonic(), clock()
    for thread in threads.values():
        thread.start()
    out = {}
    for name, thread in threads.items():
        seconds = budgets.get(name, 150)
        thread.join(max(0.0, seconds - (time.monotonic() - start)))
        box = dict(boxes[name])  # a copy: a thread that is still running writes to the original, not to this
        if thread.is_alive() or "done" not in box:
            box = {"began": box.get("began", started_at), "done": clock(), "error": TimeoutError(f"not finished after {seconds:.0f} seconds")}
        out[name] = box
    return out


def fetch_steps(paths: Paths, day: pd.Timestamp, now: pd.Timestamp, stamp: str) -> dict:
    """The three fetches of a morning. Each range starts where the files already held end, so a missed stretch leaves no hole."""
    def german():
        held, _ = german_refetched(paths, now)
        latest = held["de"].dropna().index.max()
        first = min(day - pd.Timedelta(days=GERMAN_REFRESH_DAYS), latest.tz_localize(None).normalize() - pd.Timedelta(days=1))
        result = fetch_german_raw(max(first, pd.Timestamp(GERMAN_LIVE_FROM)), day + pd.Timedelta(days=1), paths.live / "german_raw" / stamp)
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

    return {"french_window": lambda: fetch_french_window(day, paths), "german_raw": german, "german_points": points}


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
    not_there_yet = ["the French load forecast of D"] if "load_fc_mw" in evidence["input_hours_missing_for_D"] else []
    if not_there_yet and not dry_run and clock() < gate["last_chance"]:
        return out("postponed", reason=f"not on the sources yet: {'; '.join(not_there_yet)}; nothing written, the next trigger tries again", fetch=record["fetch"])
    if not all(evidence["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values()):
        record["flags"].append("an input of D depends on a value not known at the gate")
    elif not all(evidence["inputs_bit_identical_under_the_scramble"].values()):
        record["flags"].append("an input of D differs by rounding only (under one part in a billion) when everything unknown at the gate is scrambled")
    if evidence["inputs_with_a_missing_hour"]:
        record["flags"].append(f"{evidence['inputs_with_a_missing_hour']} input(s) missing for part of D and left missing")
    if evidence["yesterdays_spread_present_hours"] < evidence["hours"]:
        record["flags"].append("yesterday's spread is missing for part of D: its lags are missing, and decoupled_share_l1 reads 0 (frozen feature code)")
    if evidence["german_proxies_present_hours"] < evidence["hours"]:
        record["flags"].append("a German proxy is missing for part of D")
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
