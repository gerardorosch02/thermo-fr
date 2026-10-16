"""Laptop-only live shadow of the model-sprint candidate S2 (sprint pre-registration, section 6). Never published.

S2 is the laptop's own pre-market honest_v2 version for delivery day D combined with a LEAR forecast issued from the same
morning's inputs, with online weights (inverse 14-day exponentially weighted daily MAE of settled Tuesday-to-Saturday
days). S1, the equal average, and LEAR alone are stored next to it as diagnostics.

How it runs on the laptop:
- `morning-run --save-shadow-window` (the laptop's scheduled morning runs only) writes the run's freshly fetched inputs
  window, D-10 to D+1, to data/shadow/fresh/ (git-ignored). That is the only change to the production path.
- `shadow-run` (weekdays 09:30 and 10:00 London, 10:30 and 11:00 Paris) takes the latest honest_v2 version for D whose
  morning run started in the pre-market window (10:05 to 11:00 Paris on D-1), builds LEAR's history from
  data/forecast/inputs.csv plus every saved window of runs up to that one (later windows win), blanks every value whose
  publication time is after 10:05 Paris on D-1 by the timing rules (`as_known`), fits LEAR on its four windows ending D-1,
  and stores S2, S1 and LEAR with the wall-clock issue time. If LEAR cannot be issued (an input missing for D, a failed fit),
  S2 and S1 are stored as copies of the honest_v2 curve with fallback = 1, which counts against the 10% fallback limit.
- `shadow-settle` (daily 14:10 London, after the settle) scores the stored versions against the auction results, records the
  daily errors that update the weights, and, once 40 shadow days are settled, computes the pre-registered statistics once and
  stores them. Before day 40 it reports counts only.
- `shadow-init` (once, before the first live day) stores the weight state: the daily errors of the two components'
  walk-forward forecasts for 2026-09-01 to 2026-10-13 (honest_v2 refitted monthly, LEAR daily), as weight state only: they
  are never printed, compared or shown.

A shadow day is a delivery day from 2026-10-14 whose S2 version and its paired honest_v2 version were both issued before
11:15 Paris on D-1; a version issued at or after 11:15 is kept but does not count. Days on which the morning run produced no
pre-market honest_v2 version are not shadow days and are counted separately.

Everything is written to the local SQLite store (tables shadow_*) and data/shadow/; nothing goes to published/, nothing is
committed, pushed or dispatched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .combine import TUESDAY_TO_SATURDAY, combine, inverse_mae_weights
from .inputs import INPUT_COLUMNS, load_inputs
from .lear import LEAR_WINDOWS, lear_walk_forward
from .market import premarket_cutoff
from .schedule import PREMARKET_WINDOW
from .shape import block_bootstrap_mean
from .store import Store, iso, utc_now
from .timing import (NUCLEAR_PUBLICATION_LAG_HOURS, generation_proxy_issue, load_forecast_issue, premarket_issue_for,
                     price_lag_issue, weather_issued_issue, weather_proxy_issue)

log = logging.getLogger("thermo_fr.shadow")

SHADOW_START = "2026-10-14"
SHADOW_DAYS = 40
STATE_START, STATE_END = "2026-09-01", "2026-10-13"
FRESH_DIR = Path("data/shadow/fresh")
HALF_LIFE, MIN_DAYS = 14.0, 14
FALLBACK_LIMIT = 0.10
ADOPT_UPPER = 1.0
COMPARATOR_SET = "honest_v2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT, delivery_day TEXT NOT NULL, model TEXT NOT NULL, issued_at_utc TEXT NOT NULL,
    shadow_run_id INTEGER, morning_run_id INTEGER, honest_forecast_id INTEGER NOT NULL, fallback INTEGER NOT NULL DEFAULT 0,
    reissued INTEGER NOT NULL DEFAULT 0, before_window INTEGER NOT NULL, w_lear REAL, message TEXT);
CREATE TABLE IF NOT EXISTS shadow_values (
    version_id INTEGER NOT NULL, timestamp_utc TEXT NOT NULL, hour INTEGER NOT NULL, forecast REAL NOT NULL,
    PRIMARY KEY (version_id, timestamp_utc));
CREATE TABLE IF NOT EXISTS shadow_errors (
    delivery_day TEXT NOT NULL, source TEXT NOT NULL, mae_honest_v2 REAL NOT NULL, mae_lear REAL NOT NULL, recorded_at_utc TEXT NOT NULL,
    PRIMARY KEY (delivery_day, source));
CREATE TABLE IF NOT EXISTS shadow_scores (
    version_id INTEGER PRIMARY KEY, delivery_day TEXT NOT NULL, model TEXT NOT NULL, hours INTEGER NOT NULL, mae REAL NOT NULL,
    settled_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shadow_result (
    computed_at_utc TEXT NOT NULL, first_day TEXT NOT NULL, last_day TEXT NOT NULL, result TEXT NOT NULL);
"""


def ensure_schema(store: Store) -> None:
    store.conn.executescript(SCHEMA)
    store.conn.commit()


class ShadowInputError(ValueError):
    """An input LEAR needs for D is missing or unknown at the issue time: S2 falls back to honest_v2."""


# --- the inputs history, gated by publication time -----------------------------------------------------------------------

def save_window(fresh: pd.DataFrame, day: str, run_id: int, started_at, fresh_dir=FRESH_DIR) -> Path:
    """The morning run's freshly fetched inputs window, one file per run (written by morning-run --save-shadow-window)."""
    fresh_dir = Path(fresh_dir)
    fresh_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(started_at).tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    path = fresh_dir / f"{day}_run{int(run_id):06d}_{stamp}.csv"
    fresh.reindex(columns=INPUT_COLUMNS).to_csv(path, index_label="timestamp_utc")
    return path


def saved_windows(fresh_dir=FRESH_DIR, upto_run_id: int | None = None) -> list[tuple[int, Path]]:
    out = []
    for path in sorted(Path(fresh_dir).glob("*_run*_*.csv")):
        try:
            run_id = int(path.stem.split("_run")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if upto_run_id is None or run_id <= upto_run_id:
            out.append((run_id, path))
    return sorted(out)


def live_history(inputs_path, fresh_dir=FRESH_DIR, upto_run_id: int | None = None) -> pd.DataFrame:
    """data/forecast/inputs.csv with every saved morning window up to `upto_run_id` laid over it (later runs win)."""
    table = load_inputs(inputs_path) if Path(inputs_path).exists() else pd.DataFrame(columns=INPUT_COLUMNS)
    table = table.reindex(columns=INPUT_COLUMNS).astype(float)
    for _, path in saved_windows(fresh_dir, upto_run_id):
        window = load_inputs(path).reindex(columns=INPUT_COLUMNS).astype(float)
        table = window.combine_first(table)
    return table.sort_index()


PRICE_COLUMNS = [c for c in INPUT_COLUMNS if c.startswith("price_")]


def publication_times(index: pd.DatetimeIndex, column: str) -> pd.DatetimeIndex | None:
    """When the value of `column` at each valid hour became public, by the timing rules; None for columns LEAR never reads."""
    if column in PRICE_COLUMNS:
        return price_lag_issue(0)(index)  # 13:00 Paris on the day before the price's own delivery day
    if column == "load_fc_mw":
        return load_forecast_issue(index)
    if column in ("temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"):
        return weather_issued_issue(index)
    if column in ("temp_proxy_c", "wind100_proxy_ms", "radiation_proxy_wm2"):
        return weather_proxy_issue(index)
    if column == "nuclear_mw":
        return pd.DatetimeIndex(index) + pd.Timedelta(hours=1 + NUCLEAR_PUBLICATION_LAG_HOURS)
    if column in ("wind_proxy_mw", "solar_proxy_mw"):
        return generation_proxy_issue(index)
    return None


def as_known(table: pd.DataFrame, tau) -> pd.DataFrame:
    """The table as it could be known at tau: every value published after tau blanked, columns LEAR does not read blanked."""
    tau = pd.Timestamp(tau).tz_convert("UTC")
    out = table.copy()
    index = pd.DatetimeIndex(out.index)
    for column in out.columns:
        published = publication_times(index, column)
        if published is None:
            out[column] = np.nan
        else:
            out.loc[np.asarray(published > tau), column] = np.nan
    return out


# --- issuing ----------------------------------------------------------------------------------------------------------------

def premarket_honest_versions(store: Store, day: str) -> pd.DataFrame:
    """honest_v2 versions for D whose morning run started in the pre-market window (10:05 to 11:00 Paris on D-1)."""
    versions = pd.read_sql_query(
        "SELECT f.forecast_id, f.run_id, f.issued_at_utc, f.model, r.started_at_utc FROM forecasts f JOIN runs r ON r.run_id = f.run_id"
        " WHERE f.delivery_day = ? AND f.feature_set = ? ORDER BY f.issued_at_utc", store.conn, params=(day, COMPARATOR_SET))
    if versions.empty:
        return versions
    started = pd.to_datetime(versions["started_at_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    day_before = (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
    clock = started.dt.time
    keep = (started.dt.date == day_before) & (clock >= PREMARKET_WINDOW[0]) & (clock < PREMARKET_WINDOW[1])
    return versions[keep.to_numpy()].reset_index(drop=True)


def lear_live(history: pd.DataFrame, day: str, windows=LEAR_WINDOWS) -> pd.Series:
    """LEAR for every hourly row of D from the history as known at 10:05 Paris on D-1; raises ShadowInputError when it cannot."""
    hours = pd.date_range(pd.Timestamp(day).tz_localize(LOCAL_TZ), (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ),
                          freq="1h", inclusive="left").tz_convert("UTC")
    tau = premarket_issue_for(hours[:1])[0]
    known = as_known(history, tau)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    known = known[known.index < end]
    if known.reindex(hours)["load_fc_mw"].isna().any():
        raise ShadowInputError(f"load forecast for {day} incomplete at the issue time")
    out = lear_walk_forward(known, day, (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), windows=windows,
                            log=lambda *_: None)
    forecast = out["lear"].reindex(hours)
    if forecast.isna().any():
        raise ShadowInputError(f"LEAR could not be issued for every hour of {day} (an input of D or of the lagged days is missing)")
    return forecast


def weight_errors(store: Store) -> pd.DataFrame:
    """Daily MAE of the two components on settled days: the weight state (2026-09-01 to 10-13) and the live shadow days."""
    rows = pd.read_sql_query("SELECT delivery_day, source, mae_honest_v2, mae_lear FROM shadow_errors", store.conn)
    if rows.empty:
        return pd.DataFrame(columns=[COMPARATOR_SET, "lear"], index=pd.DatetimeIndex([]), dtype=float)
    rows = rows.sort_values("source").drop_duplicates("delivery_day", keep="last")  # 'live' after 'state' for any shared day
    frame = rows.set_index(pd.to_datetime(rows["delivery_day"]))[["mae_honest_v2", "mae_lear"]]
    frame.columns = [COMPARATOR_SET, "lear"]
    return frame.sort_index()


def _store_version(store: Store, day: str, model: str, issued_at, shadow_run_id, morning_run_id, honest_id, curve: pd.Series,
                   fallback: bool, before_window: bool, w_lear, message: str = "") -> int:
    cur = store.conn.execute(
        "INSERT INTO shadow_versions (delivery_day, model, issued_at_utc, shadow_run_id, morning_run_id, honest_forecast_id, fallback, reissued,"
        " before_window, w_lear, message) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
        (day, model, iso(issued_at), shadow_run_id, morning_run_id, int(honest_id), int(fallback), int(before_window),
         None if w_lear is None else float(w_lear), message))
    version_id = int(cur.lastrowid)
    local = curve.index.tz_convert(LOCAL_TZ)
    store.conn.executemany("INSERT INTO shadow_values (version_id, timestamp_utc, hour, forecast) VALUES (?, ?, ?, ?)",
                           [(version_id, iso(ts), int(h), float(v)) for ts, h, v in zip(curve.index, local.hour, curve.to_numpy())])
    store.conn.commit()
    return version_id


def shadow_run(store: Store, delivery_day: str | None = None, kind: str = "scheduled", inputs_path=Path("data/forecast/inputs.csv"),
               fresh_dir=FRESH_DIR, windows=LEAR_WINDOWS, now=None, clock=None) -> dict:
    """Issue S2, S1 and LEAR for D next to the latest pre-market honest_v2 version that has none yet. Never raises for a
    missing input: the fallback is stored and counted. `clock` returns the wall-clock issue time (tests replace it)."""
    from .jobs import next_delivery_day

    ensure_schema(store)
    day = delivery_day or next_delivery_day(now)
    run_id = store.start_run("shadow-run", kind, day, now=now)
    summary = {"run_id": run_id, "delivery_day": day, "versions": {}}
    candidates = premarket_honest_versions(store, day)
    done = {r[0] for r in store.conn.execute("SELECT honest_forecast_id FROM shadow_versions WHERE delivery_day = ? AND model = 'S2'", (day,))}
    candidates = candidates[~candidates["forecast_id"].isin(done)] if len(candidates) else candidates
    if candidates.empty:
        message = "no pre-market honest_v2 version without a shadow version yet" if done else "no pre-market honest_v2 version for this day"
        store.finish_run(run_id, "skipped", message, now=now)
        summary["status"] = "skipped"
        summary["message"] = message
        log.info("shadow-run %s: %s", day, message)
        return summary
    honest = candidates.iloc[-1]
    honest_curve = store.forecast_curve(int(honest["forecast_id"]))["forecast"]
    fallback, message, lear = False, "", None
    try:
        history = live_history(inputs_path, fresh_dir, upto_run_id=int(honest["run_id"]))
        lear = lear_live(history, day, windows=windows).reindex(honest_curve.index)
        if lear.isna().any():
            raise ShadowInputError("LEAR hours do not match the honest_v2 curve")
    except Exception as exc:  # noqa: BLE001  any failure is a fallback day, logged and counted
        fallback, message = True, f"{exc.__class__.__name__}: {exc}"
        log.warning("shadow-run %s: fallback to honest_v2: %s", day, message)
    errors = weight_errors(store)
    weights = inverse_mae_weights(errors, [pd.Timestamp(day)], half_life=HALF_LIFE, min_days=MIN_DAYS, update_weekdays=TUESDAY_TO_SATURDAY)
    w_lear = float(weights["lear"].iloc[0]) if "lear" in weights else 0.5
    if fallback:
        s2 = s1 = honest_curve.copy()
    else:
        frame = pd.DataFrame({"delivery_day": pd.Timestamp(day), COMPARATOR_SET: honest_curve, "lear": lear})
        s1 = combine(frame, None, [COMPARATOR_SET, "lear"])
        s2 = combine(frame, pd.DataFrame({COMPARATOR_SET: [1 - w_lear], "lear": [w_lear]}, index=[pd.Timestamp(day)]), [COMPARATOR_SET, "lear"])
    issued = (clock or (lambda: utc_now(now)))()
    before = bool(pd.Timestamp(issued) < premarket_cutoff(day)) and bool(pd.Timestamp(honest["issued_at_utc"]) < premarket_cutoff(day))
    ids = {}
    for model, curve in (("S2", s2), ("S1", s1)) + ((("LEAR", lear),) if not fallback else ()):
        ids[model] = _store_version(store, day, model, issued, run_id, int(honest["run_id"]), int(honest["forecast_id"]), curve,
                                    fallback, before, w_lear if model == "S2" else None, message)
    status = "fallback" if fallback else "ok"
    note = (message + "; " if message else "") + ("" if before else "issued at or after 11:15 Paris on D-1: does not count")
    store.finish_run(run_id, status, note, now=now)
    summary.update(status=status, versions=ids, before_window=before, w_lear=w_lear, message=note)
    log.info("shadow-run %s: %s, versions %s, LEAR weight %.3f, before the window: %s", day, status, ids, w_lear, before)
    return summary


# --- settling and the statistics at day 40 ---------------------------------------------------------------------------------

def counted_versions(store: Store) -> pd.DataFrame:
    """Per delivery day from SHADOW_START, the latest S2 version issued before the window (with its honest_v2 version)."""
    versions = pd.read_sql_query("SELECT * FROM shadow_versions WHERE model = 'S2' AND before_window = 1 AND delivery_day >= ? ORDER BY issued_at_utc",
                                 store.conn, params=(SHADOW_START,))
    return versions.groupby("delivery_day").tail(1).sort_values("delivery_day").reset_index(drop=True)


def _mae(curve: pd.Series, actual: pd.Series) -> tuple[float, int] | None:
    joined = pd.concat([curve.rename("f"), actual.rename("a")], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    return float((joined["f"] - joined["a"]).abs().mean()), int(len(joined))


def version_curve(store: Store, version_id: int) -> pd.Series:
    frame = pd.read_sql_query("SELECT timestamp_utc, forecast FROM shadow_values WHERE version_id = ? ORDER BY timestamp_utc",
                              store.conn, params=(int(version_id),))
    return pd.Series(frame["forecast"].to_numpy(), index=pd.to_datetime(frame["timestamp_utc"], utc=True))


def shadow_settle(store: Store, kind: str = "scheduled", now=None, log_fn=print) -> dict:
    """Score unscored shadow versions, record the live daily errors that update the weights, and at day 40 compute the result."""
    ensure_schema(store)
    run_id = store.start_run("shadow-settle", kind, None, now=now)
    settled = iso(utc_now(now))
    scored = 0
    unscored = pd.read_sql_query("SELECT v.version_id, v.delivery_day, v.model FROM shadow_versions v LEFT JOIN shadow_scores s USING (version_id)"
                                 " WHERE s.version_id IS NULL", store.conn)
    for row in unscored.itertuples():
        actual = store.actuals_for(row.delivery_day)
        result = _mae(version_curve(store, row.version_id), actual) if len(actual) else None
        if result is None:
            continue
        store.conn.execute("INSERT OR REPLACE INTO shadow_scores (version_id, delivery_day, model, hours, mae, settled_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                           (int(row.version_id), row.delivery_day, row.model, result[1], result[0], settled))
        scored += 1
    store.conn.commit()
    # live weight updates: the honest_v2 version and LEAR of every counted, non-fallback day with a settled result
    for v in counted_versions(store).itertuples():
        if v.fallback:
            continue
        actual = store.actuals_for(v.delivery_day)
        lear_id = store.conn.execute("SELECT version_id FROM shadow_versions WHERE model = 'LEAR' AND shadow_run_id = ? AND delivery_day = ?",
                                     (v.shadow_run_id, v.delivery_day)).fetchone()
        if not len(actual) or lear_id is None:
            continue
        honest = _mae(store.forecast_curve(int(v.honest_forecast_id))["forecast"], actual)
        lear = _mae(version_curve(store, lear_id[0]), actual)
        if honest and lear:
            store.conn.execute("INSERT OR REPLACE INTO shadow_errors (delivery_day, source, mae_honest_v2, mae_lear, recorded_at_utc) VALUES (?, 'live', ?, ?, ?)",
                               (v.delivery_day, honest[0], lear[0], settled))
    store.conn.commit()
    days = settled_shadow_days(store)
    summary = {"run_id": run_id, "scored": scored, "settled_shadow_days": len(days), "fallback_days": int(days["fallback"].sum()) if len(days) else 0}
    existing = store.conn.execute("SELECT result FROM shadow_result").fetchone()
    if existing is None and len(days) >= SHADOW_DAYS:
        result = shadow_statistics(days.head(SHADOW_DAYS))
        store.conn.execute("INSERT INTO shadow_result (computed_at_utc, first_day, last_day, result) VALUES (?, ?, ?, ?)",
                           (settled, result["first_day"], result["last_day"], json.dumps(result)))
        store.conn.commit()
        summary["result"] = result
        log_fn(json.dumps(result, indent=2))
    elif existing is not None:
        summary["result"] = json.loads(existing[0])
        log_fn("shadow result already computed (day 40 reached); nothing recomputed")
    else:
        log_fn(f"shadow: {len(days)} of {SHADOW_DAYS} settled shadow days ({summary['fallback_days']} fallback); statistics at day {SHADOW_DAYS} only")
    store.finish_run(run_id, "ok", f"{len(days)} of {SHADOW_DAYS} shadow days settled", now=now)
    return summary


def settled_shadow_days(store: Store) -> pd.DataFrame:
    """Counted days with S2 and honest_v2 both settled: daily MAE of S2, of the honest_v2 version, and of S1."""
    rows = []
    for v in counted_versions(store).itertuples():
        actual = store.actuals_for(v.delivery_day)
        if not len(actual):
            continue
        s2 = _mae(version_curve(store, v.version_id), actual)
        honest = _mae(store.forecast_curve(int(v.honest_forecast_id))["forecast"], actual)
        s1_id = store.conn.execute("SELECT version_id FROM shadow_versions WHERE model = 'S1' AND shadow_run_id = ? AND delivery_day = ?",
                                   (v.shadow_run_id, v.delivery_day)).fetchone()
        s1 = _mae(version_curve(store, s1_id[0]), actual) if s1_id else None
        if s2 and honest:
            rows.append({"delivery_day": v.delivery_day, "mae_s2": s2[0], "mae_honest_v2": honest[0], "mae_s1": s1[0] if s1 else np.nan,
                         "fallback": int(v.fallback)})
    return pd.DataFrame(rows, columns=["delivery_day", "mae_s2", "mae_honest_v2", "mae_s1", "fallback"])


def shadow_statistics(days: pd.DataFrame) -> dict:
    """The pre-registered adoption rule on exactly the first 40 settled shadow days."""
    diff = (days["mae_s2"] - days["mae_honest_v2"]).to_numpy()
    low, high = block_bootstrap_mean(diff, block=7, resamples=5000, seed=0, level=0.90)
    s1 = (days["mae_s1"] - days["mae_honest_v2"]).to_numpy()
    fallback_rate = float(days["fallback"].mean())
    adopt = bool(diff.mean() <= 0 and high <= ADOPT_UPPER and fallback_rate <= FALLBACK_LIMIT)
    return {"days": int(len(days)), "first_day": str(days["delivery_day"].iloc[0]), "last_day": str(days["delivery_day"].iloc[-1]),
            "mean_difference_s2_minus_honest_v2": round(float(diff.mean()), 3), "ci90": [round(low, 3), round(high, 3)],
            "fallback_rate": round(fallback_rate, 3), "s1_minus_honest_v2_diagnostic": round(float(np.nanmean(s1)), 3),
            "mae_s2": round(float(days["mae_s2"].mean()), 3), "mae_honest_v2": round(float(days["mae_honest_v2"].mean()), 3),
            "adopted": adopt,
            "rule": "adopted if the mean daily MAE difference is at most zero, the upper end of its 90% moving-block interval at most "
                    "+1.0 EUR/MWh, and fallbacks at most 10%; stopped by any look-ahead or train/serve fault"}


# --- the weight state ---------------------------------------------------------------------------------------------------------

def shadow_init(store: Store, inputs_path=Path("data/forecast/inputs.csv"), fresh_dir=FRESH_DIR, start: str = STATE_START,
                end: str = STATE_END, windows=LEAR_WINDOWS, log_fn=print) -> dict:
    """Store the S2 weight state: the daily MAE of the honest_v2 walk-forward (monthly refits) and LEAR (daily) on [start, end]
    for the days whose inputs and auction results exist. The errors are stored as weight state only and never printed."""
    from .backtest import walk_forward
    from .features import build_features

    ensure_schema(store)
    history = live_history(inputs_path, fresh_dir)
    actual = pd.concat([store.actuals_for(d) for d in pd.date_range(start, end).strftime("%Y-%m-%d")])
    price = history["price_eur_mwh"].copy()
    if len(actual):
        price = actual.combine_first(price)  # the settled auction results the store holds, then the inputs history
    history = history.assign(price_eur_mwh=price)
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    month_start = pd.Timestamp(start).to_period("M").start_time.strftime("%Y-%m-%d")  # walk_forward refits at month starts
    honest = walk_forward(build_features(history, COMPARATOR_SET), month_start, end_excl, models=("gbm",), log=lambda *_: None).predictions
    honest = honest[pd.to_datetime(honest["delivery_day"]) >= pd.Timestamp(start)]
    lear = lear_walk_forward(history, start, end_excl, windows=windows, log=lambda *_: None)
    frame = honest[["delivery_day", "actual", "gbm"]].join(lear[["lear"]])
    frame = frame.dropna(subset=["actual"])
    daily = frame.assign(e_h=(frame["gbm"] - frame["actual"]).abs(), e_l=(frame["lear"] - frame["actual"]).abs()).groupby("delivery_day")
    table = daily.agg(mae_honest_v2=("e_h", "mean"), mae_lear=("e_l", "mean"), n=("e_l", "count"), hours=("actual", "size"))
    table = table[table["n"] == table["hours"]]  # complete days only
    recorded = iso(utc_now())
    store.conn.executemany("INSERT OR REPLACE INTO shadow_errors (delivery_day, source, mae_honest_v2, mae_lear, recorded_at_utc) VALUES (?, 'state', ?, ?, ?)",
                           [(d.strftime("%Y-%m-%d"), float(r.mae_honest_v2), float(r.mae_lear), recorded) for d, r in table.iterrows()])
    store.conn.commit()
    eligible = int(table.index.dayofweek.isin(TUESDAY_TO_SATURDAY).sum())
    log_fn(f"weight state stored for {len(table)} days from {table.index.min():%Y-%m-%d} to {table.index.max():%Y-%m-%d} "
           f"({eligible} Tuesday-to-Saturday days update the weights); errors are not shown")
    return {"days": int(len(table)), "eligible_days": eligible}
