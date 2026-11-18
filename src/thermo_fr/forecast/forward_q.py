"""Forward live log of Q: the live entry point, the write-once record and the end-of-test statistics
(docs/prereg/forward_q_prereg.md).

Nothing here writes to the production store, to published/ or to git. The logger reads the laptop's store read-only, reads
the morning windows that `morning-run --save-shadow-window` already saves, and writes only under its own folder
(default data/forward/, git-ignored):

    issued/<D>_run<run>_<stamp>.json   one file per issue, never overwritten
    manifest.jsonl                     every saved window and every record with its sha256, each line chained to the last
    looks.jsonl                        any use of the scoring command before the end date

Commands:

    python -m thermo_fr.forecast.forward_q capture [--out DIR]           # add new saved windows to the manifest
    python -m thermo_fr.forecast.forward_q issue   [--date D] [--out DIR] # issue L0, P, Q, S1, S1-Q next to the pre-market honest_v2 version
    python -m thermo_fr.forecast.forward_q status  [--out DIR]           # counts only: scored, late, lost, fallbacks
    python -m thermo_fr.forecast.forward_q verify  [--out DIR]           # recompute every hash of the manifest
    python -m thermo_fr.forecast.forward_q score   [--out DIR]           # the statistics, once, from STATS_FROM

The live entry point is q_live(history, day). It cuts the history to what the timing rules place before 10:05 Paris on D-1
(shadow.as_known, plus Q's registered stand-in: the operator's wind and solar forecast on delivery days before 2024-05-01
only) and fits L0, P and Q with neighbour_rl.fit_predict_day, the function behind the stored research forecasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from . import french_rl, lear, yardstick
from . import neighbour_rl as nr
from .features import build_features
from .market import premarket_cutoff
from .timing import delivery_days, premarket_issue_for, wind_solar_forecast_issue

FIRST_DAY = pd.Timestamp("2026-12-01")  # first counted delivery day (issued Monday 30 November)
LAST_DAY = pd.Timestamp("2027-04-17")  # last counted delivery day; no extension
STATS_FROM = pd.Timestamp("2027-04-18")  # the statistics are computed once, from this date
EXPECTED_WEEKDAYS = (1, 2, 3, 4, 5)  # delivery days Tuesday to Saturday (pandas dayofweek)
MIN_SCORED_DAYS = 90
MIN_Q_SHARE = 0.90
LEVEL = 0.95  # one-sided
BLOCK, RESAMPLES, SEED = 7, 5000, 0
SENSITIVITY_BLOCKS = (1, 3, 14)
CANDIDATES = ("L0", "P", "Q")
PREMARKET_WINDOW = (pd.Timestamp("10:05").time(), pd.Timestamp("11:00").time())  # Paris, start of the morning run on D-1
OUT_DIR = Path("data/forward")
FRESH_DIR = Path("data/shadow/fresh")
GENESIS = "0" * 64


class ForwardInputError(ValueError):
    """An input the forecast of D needs is missing or not known at the issue time."""


# --- the live entry point --------------------------------------------------------------------------------------------------

def day_hours(day) -> pd.DatetimeIndex:
    """The hourly rows of delivery day D (Paris), in UTC: 23, 24 or 25 of them."""
    day = pd.Timestamp(day)
    return pd.date_range(day.tz_localize(LOCAL_TZ), (day + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ), freq="1h",
                         inclusive="left").tz_convert("UTC")


def as_known_q(table: pd.DataFrame, tau) -> pd.DataFrame:
    """shadow.as_known, plus the operator's wind and solar forecast on delivery days before HONEST_START (Q's stand-in). On every
    later day those columns stay blank, so no forecast of D, or of any day from 2024-05-01, can read them."""
    from .shadow import as_known

    tau = pd.Timestamp(tau).tz_convert("UTC")
    out = as_known(table, tau)
    index = pd.DatetimeIndex(table.index)
    early = np.asarray(delivery_days(index) < french_rl.HONEST_START) & np.asarray(wind_solar_forecast_issue(index) <= tau)
    for column in french_rl.OPERATOR_COLUMNS:
        if column in table.columns:
            out.loc[early, column] = table.loc[early, column]
    return out


def fit_known(known: pd.DataFrame, day, windows=lear.LEAR_WINDOWS, candidates=CANDIDATES) -> dict:
    """L0, P and Q for D from a table that already holds only what may be used. Returns {"index", "hour", name: {"windows":
    {w: Series by clock hour}, "kept": {w: hourly models that kept the regressor, or None}, "hourly": Series on the rows of D}}."""
    day = pd.Timestamp(day)
    known = known.sort_index()
    design, price, blocks = lear.lear_design(known)
    info = build_features(known, "honest_v2").info
    rows = np.asarray(info["delivery_day"] == day)
    if not rows.any():
        raise ForwardInputError(f"no rows for {day:%Y-%m-%d} in the inputs")
    lear.assert_gate(info.index[rows], blocks)
    french_rl.assert_gate(info.index[rows], "Q")
    french, _ = french_rl.french_tables(known, info, design.index)
    extras = {"L0": ({}, ()), "P": ({"fr": french["honest"]}, ("fr",)), "Q": ({"fr": french["q"]}, ())}
    day_info = info[rows]
    out = {"index": day_info.index, "hour": day_info["hour"].to_numpy()}
    for name in candidates:
        extra, partial = extras[name]
        if name != "L0" and day < french_rl.FIRST_DAY[name]:
            extra, partial = {}, ()  # before its first day a candidate has no forecast of its own: it is L0, as in the research frame
        per_window, kept = {}, {}
        for w in windows:
            pred, k = nr.fit_predict_day(design, price, blocks, extra, day, int(w), partial=partial)
            per_window[int(w)] = pred
            kept[int(w)] = int(k["fr"]) if "fr" in k else None
        mean = sum(per_window[int(w)] for w in windows) / len(windows)  # every window is needed: a missing one leaves the candidate missing
        hourly = pd.Series(mean.reindex(out["hour"]).to_numpy(), index=day_info.index, name=name)
        out[name] = {"windows": per_window, "kept": kept, "hourly": hourly}
    return out


def q_live(history: pd.DataFrame, day, windows=lear.LEAR_WINDOWS, candidates=CANDIDATES) -> dict:
    """L0, P and Q for every hourly row of D from the history as known at 10:05 Paris on D-1."""
    day = pd.Timestamp(day)
    hours = day_hours(day)
    tau = premarket_issue_for(hours[:1])[0]
    known = as_known_q(history, tau)
    known = known[known.index < hours[-1] + pd.Timedelta(hours=1)]
    if known.reindex(hours)["load_fc_mw"].isna().any():
        raise ForwardInputError(f"load forecast for {day:%Y-%m-%d} incomplete at the issue time")
    out = fit_known(known, day, windows, candidates)
    out["tau_utc"] = str(tau)
    return out


def fill_proxy_gaps(history: pd.DataFrame, log=None) -> tuple[pd.DataFrame, dict]:
    """The wind and solar proxies where the history lacks them although the point forecasts are there, computed by the project's
    own point-in-time rule (gen_proxy.rolling_proxy: weights refitted each month on actual generation up to two days before the
    month). Values already in the history are never replaced. Found necessary on 2026-11-18: data/forecast/inputs.csv on the
    laptop holds no proxy from 2026-08-20 to 2026-10-09, which would otherwise leave a hole in the regressor's training days."""
    from . import solar_proxy, wind_proxy

    out = history.copy()
    filled = {}
    for module, spec, actual in ((wind_proxy, wind_proxy.WIND, wind_proxy.actual_wind(history)), (solar_proxy, solar_proxy.SOLAR, solar_proxy.actual_solar(history))):
        columns = [c for c in spec.columns if c in history.columns]
        proxy, _ = module.rolling_proxy(history[columns], actual)
        missing = out[spec.output].isna() & proxy.notna()
        out.loc[missing, spec.output] = proxy[missing]
        days = delivery_days(out.index[missing.to_numpy()])
        filled[spec.output] = {"hours": int(missing.sum()), "first_day": str(days.min().date()) if len(days) else None, "last_day": str(days.max().date()) if len(days) else None}
    return out, filled


# --- hashes, the manifest and the write-once record --------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _chain_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "line_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def manifest_append(path, entries: list[dict]) -> list[dict]:
    """Append entries, each carrying the hash of the line before it. Lines are only ever added."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_manifest(path)
    prev = existing[-1]["line_sha256"] if existing else GENESIS
    written = []
    with open(path, "a") as f:
        for entry in entries:
            line = {**entry, "prev": prev}
            line["line_sha256"] = _chain_hash(line)
            f.write(json.dumps(line, sort_keys=True) + "\n")
            prev = line["line_sha256"]
            written.append(line)
    return written


def verify_manifest(path) -> dict:
    """Recompute the chain and every file hash. A file that is gone or changed, or a broken chain, is reported, never repaired."""
    entries = read_manifest(path)
    prev, broken, missing, changed = GENESIS, [], [], []
    for k, entry in enumerate(entries):
        if entry.get("prev") != prev or _chain_hash(entry) != entry.get("line_sha256"):
            broken.append(k)
        prev = entry.get("line_sha256")
        file = Path(entry["path"])
        if not file.exists():
            missing.append(entry["path"])
        elif sha256_file(file) != entry["sha256"]:
            changed.append(entry["path"])
    return {"entries": len(entries), "chain_ok": not broken, "broken_lines": broken, "missing": missing, "changed": changed,
            "head": prev}


def _rel(path) -> str:
    """The path as written in the manifest: relative to the working folder when it lies inside it, as given otherwise."""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def capture(fresh_dir=FRESH_DIR, out_dir=OUT_DIR, now=None) -> dict:
    """List every saved morning window (and its model record) not yet in the manifest. A listed file whose content has since
    changed is reported and left as it is."""
    manifest = Path(out_dir) / "manifest.jsonl"
    known = {e["path"]: e["sha256"] for e in read_manifest(manifest)}
    stamp = (now or pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    new, changed = [], []
    for file in sorted(Path(fresh_dir).glob("*_run*_*")):
        rel = _rel(file)
        digest = sha256_file(file)
        if rel in known:
            if known[rel] != digest:
                changed.append(rel)
            continue
        new.append({"kind": "morning_window" if file.suffix == ".csv" else "model_record", "path": rel, "sha256": digest,
                    "bytes": file.stat().st_size, "file_mtime_utc": pd.Timestamp(file.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "listed_at_utc": stamp})
    manifest_append(manifest, new)
    return {"added": len(new), "changed_since_listed": changed, "entries": len(known) + len(new)}


def write_record(out_dir, record: dict) -> Path:
    """One file per issue, opened for exclusive creation: a second write of the same issue fails instead of replacing it."""
    folder = Path(out_dir) / "issued"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(record["issued_at_utc"]).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{record['delivery_day']}_run{int(record['morning_run_id']):06d}_{stamp}.json"
    with open(path, "x") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    return path


# --- the daily issue --------------------------------------------------------------------------------------------------------

def read_only(db) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)


def premarket_versions(conn, day: str) -> pd.DataFrame:
    """honest_v2 versions for D whose morning run started between 10:05 and 11:00 Paris on D-1 (the rule of the S2 shadow)."""
    versions = pd.read_sql_query(
        "SELECT f.forecast_id, f.run_id, f.issued_at_utc, f.model, r.started_at_utc FROM forecasts f JOIN runs r ON r.run_id = f.run_id"
        " WHERE f.delivery_day = ? AND f.feature_set = 'honest_v2' ORDER BY f.issued_at_utc", conn, params=(day,))
    if versions.empty:
        return versions
    started = pd.to_datetime(versions["started_at_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    day_before = (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
    keep = (started.dt.date == day_before) & (started.dt.time >= PREMARKET_WINDOW[0]) & (started.dt.time < PREMARKET_WINDOW[1])
    return versions[keep.to_numpy()].reset_index(drop=True)


def forecast_curve(conn, forecast_id: int) -> pd.Series:
    rows = pd.read_sql_query("SELECT timestamp_utc, forecast FROM forecast_values WHERE forecast_id = ? ORDER BY timestamp_utc", conn,
                             params=(int(forecast_id),))
    return pd.Series(rows["forecast"].to_numpy(), index=pd.to_datetime(rows["timestamp_utc"], utc=True))


def window_file(fresh_dir, run_id: int) -> Path | None:
    hits = sorted(Path(fresh_dir).glob(f"*_run{int(run_id):06d}_*.csv"))
    return hits[-1] if hits else None


def versions_used() -> dict:
    import lightgbm
    import scipy
    import sklearn

    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__, "lightgbm": lightgbm.__version__}


def code_sha256() -> str:
    """One hash over every source file of thermo_fr.forecast as it is imported now, in name order. Unlike the commit, it changes
    when a file was edited without a commit, so a record shows which code really ran."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))  # the same hash on Windows and Linux checkouts
    return h.hexdigest()


def code_commit(repo=Path(".")) -> str | None:
    """The checked-out commit, read from .git without running git (no lock files on the laptop's folders)."""
    try:
        git = Path(repo) / ".git"
        if git.is_file():  # a worktree: "gitdir: <path>"
            git = Path(git.read_text().split(":", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ")[0]
    except OSError:
        pass
    return None


def issue(day: str, db=Path("data/forecast.db"), inputs_path=Path("data/forecast/inputs.csv"), fresh_dir=FRESH_DIR, out_dir=OUT_DIR,
          repo=Path("."), clock=None, upto_run_id: int | None = None) -> dict:
    """Issue L0, P, Q, S1 and S1-Q next to the latest pre-market honest_v2 version for D and write the record. Never raises for a
    missing input: the fallback is written and flagged. `upto_run_id` (replays only) restricts to runs up to that id."""
    from .shadow import comparator_model, live_history

    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    started = time.time()
    conn = read_only(db)
    try:
        versions = premarket_versions(conn, day)
        if upto_run_id is not None and len(versions):
            versions = versions[versions["run_id"] <= upto_run_id]
        if versions.empty:
            return {"delivery_day": day, "status": "lost", "reason": "no pre-market honest_v2 version (morning run late, failed or missing)"}
        honest = versions.iloc[-1]
        g = forecast_curve(conn, int(honest["forecast_id"]))
    finally:
        conn.close()
    run_id = int(honest["run_id"])
    if sorted((Path(out_dir) / "issued").glob(f"{day}_run{run_id:06d}_*.json")):
        return {"delivery_day": day, "status": "already issued", "morning_run_id": run_id}
    window = window_file(fresh_dir, run_id)
    model = comparator_model(fresh_dir, run_id) or {}
    record = {"delivery_day": day, "morning_run_id": run_id, "honest_forecast_id": int(honest["forecast_id"]),
              "honest_issued_at_utc": str(honest["issued_at_utc"]), "honest_model": str(honest["model"]),
              "g_fallback": bool(str(honest["model"]).endswith(":fallback")), "model_sha256": model.get("sha256"),
              "window_file": window.name if window else None, "window_sha256": sha256_file(window) if window else None,
              "code_commit": code_commit(repo), "code_sha256": code_sha256(), "versions": versions_used(), "flags": [], "kept": {}, "windows": {}}
    curves = {"g": g}
    try:
        history, record["proxy_gaps_filled"] = fill_proxy_gaps(live_history(inputs_path, fresh_dir, upto_run_id=run_id))
        result = q_live(history, day)
        record["tau_utc"] = result["tau_utc"]
        for name in CANDIDATES:
            curve = result[name]["hourly"].reindex(g.index)
            record["kept"][name] = {str(w): k for w, k in result[name]["kept"].items()}
            record["windows"][name] = {str(w): [None if pd.isna(v) else float(v) for v in s.to_numpy()] for w, s in result[name]["windows"].items()}
            if not curve.isna().any():
                curves[name] = curve
    except Exception as exc:  # noqa: BLE001  any failure is a fallback day, written down and counted
        record["error"] = f"{exc.__class__.__name__}: {exc}"
    if "L0" not in curves:
        record["flags"].append("LEAR fallback")
        curves["S1"] = curves["S1_Q"] = g.copy()
    else:
        curves["S1"] = (g + curves["L0"]) / 2
        if "Q" not in curves:
            record["flags"].append("Q fallback")
            curves["S1_Q"] = curves["S1"].copy()
        else:
            curves["S1_Q"] = (g + curves["Q"]) / 2
    if record["g_fallback"]:
        record["flags"].append("g fallback")
    published = Path(".publish-worktree/published/model/honest_v2.txt")  # origin/main as the 14:30 market publish last fetched it
    record["published_model_sha256"] = sha256_file(published) if published.exists() else None
    if record["published_model_sha256"] and record["model_sha256"] and not record["g_fallback"] and record["published_model_sha256"] != record["model_sha256"]:
        record["flags"].append("g file differs from the published file")  # scored, and reported: the laptop's checkout is behind a refit
    issued = clock()
    record["issued_at_utc"] = pd.Timestamp(issued).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    record["before_window"] = bool(pd.Timestamp(issued) < premarket_cutoff(day))
    if not record["before_window"]:
        record["flags"].append("late")
    record["seconds"] = round(time.time() - started, 1)
    record["timestamps_utc"] = [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in g.index]
    record["forecasts"] = {name: [float(v) for v in curve.reindex(g.index).to_numpy()] for name, curve in curves.items()}
    path = write_record(out_dir, record)
    capture(fresh_dir, out_dir, now=pd.Timestamp(issued))
    manifest_append(Path(out_dir) / "manifest.jsonl", [{"kind": "forward_record", "path": _rel(path), "sha256": sha256_file(path),
                                                         "bytes": path.stat().st_size, "listed_at_utc": record["issued_at_utc"]}])
    return {"delivery_day": day, "status": "issued", "flags": record["flags"], "record": path.as_posix(), "seconds": record["seconds"]}


# --- counts, and the statistics at the end ------------------------------------------------------------------------------------

def expected_days(first=FIRST_DAY, last=LAST_DAY) -> pd.DatetimeIndex:
    days = pd.date_range(first, last, freq="D")
    return days[days.dayofweek.isin(EXPECTED_WEEKDAYS)]


def read_records(out_dir=OUT_DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "issued").glob("*.json"))]


def chosen_records(records: list[dict]) -> dict:
    """Per delivery day, the last record written before the market window; if none, the last late one."""
    by_day: dict[str, dict] = {}
    for r in sorted(records, key=lambda r: r["issued_at_utc"]):
        current = by_day.get(r["delivery_day"])
        if current is None or r["before_window"] or not current["before_window"]:
            by_day[r["delivery_day"]] = r
    return by_day


def status(out_dir=OUT_DIR, today=None, first=FIRST_DAY, last=LAST_DAY) -> dict:
    """Counts only. No error of any forecast is computed here."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)).normalize()
    due = [d for d in expected_days(first, last) if d <= today + pd.Timedelta(days=1)]
    chosen = chosen_records(read_records(out_dir))
    counts = {"expected_so_far": len(due), "issued_before_window": 0, "late": 0, "lost": 0, "Q fallback": 0, "LEAR fallback": 0, "g fallback": 0}
    lost_days, late_days = [], []
    for d in due:
        r = chosen.get(f"{d:%Y-%m-%d}")
        if r is None:
            counts["lost"] += 1
            lost_days.append(f"{d:%Y-%m-%d}")
        elif not r["before_window"]:
            counts["late"] += 1
            late_days.append(f"{d:%Y-%m-%d}")
        else:
            counts["issued_before_window"] += 1
            for flag in ("Q fallback", "LEAR fallback", "g fallback"):
                counts[flag] += int(flag in r["flags"])
    return {**counts, "lost_days": lost_days, "late_days": late_days, "expected_in_all": len(expected_days(first, last))}


def frame_from_records(records: dict, actual: pd.Series) -> pd.DataFrame:
    """Hourly frame of the chosen pre-market records with the auction result: one row per hour, strict by construction."""
    parts = []
    for day, r in records.items():
        if not r["before_window"]:
            continue
        index = pd.to_datetime(r["timestamps_utc"], utc=True)
        part = pd.DataFrame({name: values for name, values in r["forecasts"].items()}, index=index)
        part["delivery_day"] = pd.Timestamp(day)
        part["q_issued"] = "Q" in r["forecasts"] and "Q fallback" not in r["flags"] and "LEAR fallback" not in r["flags"]
        part["actual"] = actual.reindex(index).to_numpy()
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts).sort_index()
    complete = frame.groupby("delivery_day")["actual"].transform(lambda s: s.notna().all())
    frame = frame[complete.to_numpy()].copy()
    frame["strict"] = True
    return frame


def one_sided(frame: pd.DataFrame, column: str, baseline: str, block: int = BLOCK) -> dict:
    diff = (yardstick.daily_abs_error(frame, column) - yardstick.daily_abs_error(frame, baseline)).dropna().to_numpy()
    low, high = yardstick.block_bootstrap_mean(diff, block=block, resamples=RESAMPLES, seed=SEED, level=1 - 2 * (1 - LEVEL))
    return {"days": int(len(diff)), "mean": float(diff.mean()), "lower_one_sided_95": float(low), "upper_one_sided_95": float(high), "block": block}


def outcome(test: dict, scored_days: int, q_share: float) -> str:
    if scored_days < MIN_SCORED_DAYS or q_share < MIN_Q_SHARE:
        return "inconclusive"
    if test["upper_one_sided_95"] < 0:
        return "supported"
    if test["lower_one_sided_95"] > 0:
        return "contradicted"
    return "inconclusive"


def score(out_dir=OUT_DIR, db=Path("data/forecast.db"), today=None, early_look: str | None = None, first=FIRST_DAY, last=LAST_DAY) -> dict:
    """The statistics of the pre-registration, once, from STATS_FROM. Before that date it refuses, unless `early_look` gives a
    reason, which is appended to looks.jsonl before anything is computed."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)).normalize()
    if today < STATS_FROM:
        if not early_look:
            raise PermissionError(f"the statistics are computed once, from {STATS_FROM:%Y-%m-%d}; an earlier look must be declared with --early-look")
        looks = Path(out_dir) / "looks.jsonl"
        looks.parent.mkdir(parents=True, exist_ok=True)
        with open(looks, "a") as f:
            f.write(json.dumps({"date": f"{today:%Y-%m-%d}", "reason": early_look}) + "\n")
    conn = read_only(db)
    try:
        rows = pd.read_sql_query("SELECT timestamp_utc, price FROM actuals", conn)
    finally:
        conn.close()
    actual = pd.Series(rows["price"].to_numpy(), index=pd.to_datetime(rows["timestamp_utc"], utc=True))
    wanted = {f"{d:%Y-%m-%d}" for d in expected_days(first, last)}
    chosen = {d: r for d, r in chosen_records(read_records(out_dir)).items() if d in wanted}
    frame = frame_from_records(chosen, actual)
    out = {"counts": status(out_dir, today=min(today, last), first=first, last=last)}
    if frame.empty:
        return {**out, "outcome": "inconclusive", "reason": "no scored day"}
    scored = int(frame["delivery_day"].nunique())
    q_days = int(frame.loc[frame["q_issued"], "delivery_day"].nunique())
    own = frame[frame["q_issued"]]
    out.update({"scored_days": scored, "scored_hours": int(len(frame)), "q_issued_days": q_days,
                "operational": {name: yardstick.mae(frame, name) for name in ("g", "S1", "S1_Q")},
                "candidate_only": {name: yardstick.mae(own, name) for name in ("g", "L0", "Q", "S1", "S1_Q") if name in own and len(own)}})
    out["operational"]["relative_reduction_S1_Q_vs_S1"] = 1 - out["operational"]["S1_Q"] / out["operational"]["S1"]
    out["main"] = one_sided(frame, "S1_Q", "S1")
    out["main_sensitivity"] = [one_sided(frame, "S1_Q", "S1", block=b) for b in SENSITIVITY_BLOCKS]
    out["deployment"] = one_sided(frame, "S1_Q", "g")
    if len(own) and "Q" in own:
        out["diagnostic_Q_vs_L0"] = one_sided(own, "Q", "L0")
    out["outcome"] = outcome(out["main"], scored, q_days / scored)
    out["deployment_outcome"] = outcome(out["deployment"], scored, q_days / scored)
    return out


# --- command line -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    from .jobs import next_delivery_day

    parser = argparse.ArgumentParser(description="Forward live log of Q (docs/prereg/forward_q_prereg.md)")
    parser.add_argument("stage", choices=["capture", "issue", "status", "verify", "score"])
    parser.add_argument("--date", default=None, help="delivery day, default tomorrow")
    parser.add_argument("--db", default="data/forecast.db")
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--fresh", default=str(FRESH_DIR))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--repo", default=".", help="the checkout whose commit is written into each record")
    parser.add_argument("--early-look", default=None, help="reason for computing the statistics before the end date; it is recorded")
    args = parser.parse_args(argv)
    repo = Path(args.repo)
    if args.stage == "capture":
        print(json.dumps(capture(Path(args.fresh), Path(args.out)), indent=1))
    elif args.stage == "issue":
        day = args.date or next_delivery_day(None)
        print(json.dumps(issue(day, db=args.db, inputs_path=Path(args.data), fresh_dir=Path(args.fresh), out_dir=Path(args.out), repo=repo), indent=1))
    elif args.stage == "status":
        print(json.dumps(status(args.out), indent=1))
    elif args.stage == "verify":
        print(json.dumps(verify_manifest(Path(args.out) / "manifest.jsonl"), indent=1))
    elif args.stage == "score":
        print(json.dumps(score(args.out, db=args.db, early_look=args.early_look), indent=1, default=str))


if __name__ == "__main__":
    main()
