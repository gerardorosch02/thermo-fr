"""The January rebuild of the test of W: the frame and the schedule the scoring code reads, and the stop conditions.

DRAFT of 2026-11-22, changed on 2026-11-23 to follow amendment 6 and the exception to stop condition 4. Not yet run on any day
of the test window. The choices it makes where the frozen documents leave room are listed in
docs/prereg/spread_w_rebuild_design_note.md and need the owner's word before the read date. Nothing here computes an error
against a price, and nothing here changes W, the comparators, the frozen scoring file (spread_w_test.py) or the logger. The
read itself is done by rebuild/spread_w_score.py.

What it does (docs/prereg/spread_w_prereg.md, spread_w_refetch_spec.md, amendments 1 to 5):

  1. Joins the refetch to the stored files. Everything to 2026-10-10 stays as stored (rule 1). After it, the French inputs
     come from the refetched file, the German prices and ENTSO-E forecasts from the refetched XML, the weather points from
     the refetched point files where the hashed files hold none.
  2. Gives every delivery day of the window one provenance class: saved, rebuilt_french_saved or rebuilt.
  3. Forecasts every day with the frozen functions (spread_eval.fit_months, spread_w.fit_w, spread_hw.fit_comparator). On a
     day whose French window was saved before 10:05 Paris on its eve, the saved load forecast and nuclear output replace
     the refetched ones in the inputs of that day only; as training rows the refetched values are used (rule 2).
  4. On a saved day the scored forecast is the one in the logger's record, never a recomputed one.
  5. Builds the frame of eligible hours and the schedule of all hours, with the reason every other hour is left out. An hour
     is eligible with a realised spread from ENTSO-E's refetch, the German proxies and a forecast from all seven models; a
     missing spread of the previous day does not exclude it (amendment 6).
  6. Runs the six stop conditions and writes stop_checks.json. If one fails, spread_w_score.read() refuses. Stop condition 6
     records whether the floor of 95 days and 2,280 hours is met. Under the floor the read still happens, marked "not
     readable" (amendment 6, section 3), so the floor does not stop the run. Stop condition 4 is checked against the
     reconstructions of 2026-11-23, under the exception the owner accepted that day.
  7. Writes a report without any price in it: days by class, days with hours from the second source, days issued without
     yesterday's spread, saved against refetched inputs (rule 5), saved against rebuilt input fingerprints (amendment 3)
     and saved against rebuilt forecasts (rule 6).

    python -m thermo_fr.rebuild.spread_w_rebuild run    --out <folder> --static ... --french ... --german-raw ... --wind ... --solar ... --live ... --kept ...
    python -m thermo_fr.rebuild.spread_w_rebuild replay --out <folder> --first 2026-09-01 --last 2026-10-10 ...   pre-window days only

`run` assembles the frame of the test window, so it refuses before the read date. `replay` does the same work on days to
2026-10-10, which are design data, and is how this code is tested.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..forecast import forward_q as fq
from ..forecast import spread_eval as se
from ..forecast import spread_hw as shw
from ..forecast import spread_live as lsl
from ..forecast import spread_w as sw
from ..forecast import spread_w_test as swt
from ..forecast.coverage import input_coverage
from ..forecast.inputs import INPUT_COLUMNS, load_inputs
from ..forecast.timing import delivery_days
from . import spread_w_score as score

STORED_LAST = sw.DESIGN_LAST  # 2026-10-10: the last delivery day of the stored files. Nothing to this day is ever replaced
OVERLAP = ("2026-09-11", "2026-10-10")  # stop condition 1 compares the refetch with the stored raw files on these days
SAVED_MODELS = lsl.MODELS  # H, HW, G, GK, GM, W: what a record holds. C is rebuilt on every day
ALL_MODELS = ("H", "HW", "G", "GK", "GM", "W", "C")
SCRAMBLE_DAYS = ("2026-10-11", "2026-10-25", "2026-11-11", "2026-12-25", "2027-01-18")  # stop condition 5
# Stop condition 4 under the exception of 2026-11-23: the stored forecasts are the reconstructions made that day, by file hash.
# Only the file that holds all seven models on the 21,430 hours is accepted, so that HW, GK and GM are always checked too.
RECONSTRUCTION_FILE = score.RECONSTRUCTION_FILE
EXCEPTION = score.EXCEPTION
# The French inputs whose 10:05 version the source does not keep (refetch specification, table "Input by input"): for these
# the window saved on the eve is used. Prices are final and weather comes from the archive, so those are refetched.
WINDOW_COLUMNS = ("load_fc_mw", "nuclear_mw")
WEATHER_COLUMNS = ("temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2")  # compared with the saved window, never taken from it
REVISION_SHARE, REVISION_SIZE = 0.05, 0.01  # rule 5: more than 5% of hours differing by more than 1% is stated beside the result
NOT_ISSUED, NOT_STRICT = score.NOT_ISSUED, score.NOT_STRICT  # why an hour has no issued forecast
EXTRA = pd.Timedelta(seconds=1)  # the rows that carry saved French inputs sit one second after the hour they belong to


@dataclass
class Sources:
    static: Path  # the logger's static folder: the stored files of the first test and the hashed point files
    french: Path  # the refetched French inputs file (forecast-fetch written to a new file)
    german_raw: Path  # folder of the refetched ENTSO-E XML for DE-LU
    wind: Path  # refetched German wind points
    solar: Path  # refetched German solar points
    live: Path | None = None  # the logger's live folder, copied as it is: records, manifest, its own French windows
    production_windows: Path | None = None  # the forward logger's saved French windows (data/shadow/fresh)


# ---- the join ----------------------------------------------------------------------------------------------------------------

def _days(obj) -> np.ndarray:
    return np.asarray(delivery_days(obj.index))


def _float_inputs(path: Path) -> pd.DataFrame:
    return load_inputs(Path(path)).reindex(columns=INPUT_COLUMNS).astype(float)


def join(src: Sources, last: str) -> tuple[se.Data, dict]:
    """Stored values to 2026-10-10, refetched values after it, nothing after `last`. The French table comes back with the
    proxy gaps filled, as in every evaluation."""
    static = Path(src.static)
    stored = se.load_stored(static / "inputs.csv", static / "neighbour_fundamentals.csv", static / "neighbour_sequences.csv.gz", static / "german_wind_points.csv.gz", static / "german_solar_points.csv.gz")
    data = se.extend(stored, se.parse_raw(static / "raw_first_test"), last=STORED_LAST)  # the German side as the first test read it
    new = lsl.parse_german_folder(Path(src.german_raw))

    def after(obj):
        days = _days(obj)
        return obj[(days > pd.Timestamp(STORED_LAST)) & (days <= pd.Timestamp(last))]

    de = pd.concat([data.de, after(new["de"])]).sort_index()
    tso = pd.concat([data.tso, after(new["tso"])]).sort_index()
    if not (de.index.is_unique and tso.index.is_unique):
        raise ValueError("an hour appears twice in the joined German tables")
    # As in the logger: on days the stored files do not hold, a row with no wind forecast at all is removed whole, so that its
    # lag comes out missing and not as 0 MW (the frozen feature code adds onshore and offshore with a missing value read as 0).
    no_wind = (_days(tso) >= pd.Timestamp(lsl.LIVE_DATA_FROM)) & tso["onshore"].isna().to_numpy() & tso["offshore"].isna().to_numpy()
    tso = tso[~no_wind]

    kept, refetched = _float_inputs(static / "inputs.csv"), _float_inputs(src.french)
    french = pd.concat([kept[_days(kept) <= pd.Timestamp(STORED_LAST)], refetched[_days(refetched) > pd.Timestamp(STORED_LAST)]]).sort_index()
    if not french.index.is_unique:
        raise ValueError("an hour appears twice in the joined French table")
    french = french.reindex(pd.date_range(french.index.min(), max(french.index.max(), fq.day_hours(last)[-1]), freq="h"))
    french.index.name = "timestamp_utc"
    filled, gaps = fq.fill_proxy_gaps(french)

    points = {}
    for kind, held, path in (("wind", stored.wind_pts, src.wind), ("solar", stored.solar_pts, src.solar)):
        fresh = se._hourly(path).reindex(columns=held.columns)
        points[kind] = held.combine_first(fresh).sort_index().asfreq("h")  # a hashed value is kept; the refetch adds the hours after it
    info = {"last": last, "german_rows_without_a_wind_forecast_removed": int(no_wind.sum()), "french_proxy_gaps_filled": json.loads(json.dumps(gaps, default=str)),
            "refetched_german_price_hours": int(after(new["de"]).notna().sum()), "refetched_french_rows": int((_days(refetched) > pd.Timestamp(STORED_LAST)).sum())}
    return replace(data, french=se.cut(filled, last), de=de, tso=tso, wind_pts=points["wind"], solar_pts=points["solar"], last=last), info


# ---- provenance --------------------------------------------------------------------------------------------------------------

def read_records(src: Sources, first: str, last: str) -> tuple[dict, dict]:
    """The logger's records of the window that count as saved, by delivery day, and why any other record does not count.
    A record counts if the manifest is whole, it is not a dry run, it was written before the auction, and it holds a number
    for every model at every hour of its day."""
    if src.live is None or not (Path(src.live) / "records" / "issued").exists():
        return {}, {"manifest": None, "not_counted": {}}
    paths = lsl.Paths(root=Path(src.live).parent, live_dir=Path(src.live))
    manifest = lsl.verify(paths)
    if not manifest["ok"]:
        raise ValueError(f"the manifest of the saved records is not whole, so no record is trusted: {manifest}")
    good, unreadable = lsl.issued_records(paths)
    saved, refused = {}, {name: "cannot be read" for name in unreadable}
    for record in good:
        day = pd.Timestamp(record["delivery_day"])
        if not pd.Timestamp(first) <= day <= pd.Timestamp(last):
            continue
        hours = [lsl.iso(t) for t in fq.day_hours(day)]
        why = ("a dry run" if record.get("dry_run") else "not written before the auction" if not record.get("saved_before_auction")
               else "its hours are not the hours of its day" if record.get("timestamps_utc") != hours
               else "a forecast is missing" if any(len(record["forecasts"].get(m) or []) != len(hours) or any(v is None for v in record["forecasts"][m]) for m in SAVED_MODELS) else None)
        if why:
            refused[record["delivery_day"]] = why
        else:
            saved[day] = record
    return saved, {"manifest": manifest, "not_counted": refused}


def eve_window(src: Sources, day: pd.Timestamp) -> Path | None:
    """The latest French window saved on the eve of `day`, for `day`, at or before 10:05 Paris. Production windows are
    stamped with the start of their run, the logger's own with the moment they were written."""
    gate = lsl.gates(day)
    eve = (day - pd.Timedelta(days=1)).tz_localize(lsl.PARIS).tz_convert("UTC")
    found = []
    for folder in (src.production_windows, Path(src.live) / "french_windows" if src.live else None):
        for path in sorted(Path(folder).glob(f"{day:%Y-%m-%d}_*.csv")) if folder and Path(folder).exists() else []:
            try:
                issued = pd.Timestamp(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            issued = issued.tz_localize("UTC") if issued.tzinfo is None else issued.tz_convert("UTC")
            if eve <= issued <= gate["cutoff"]:
                found.append((issued, path.name, path))
    return max(found)[2] if found else None


def classify(src: Sources, first: str, last: str) -> tuple[dict, dict, dict, dict]:
    """{day: class}, the saved records, {day: eve window} and the notes on records that do not count."""
    saved, notes = read_records(src, first, last)
    windows = {day: eve_window(src, day) for day in pd.date_range(first, last, freq="D")}
    windows = {day: path for day, path in windows.items() if path is not None}
    classes = {day: "saved" if day in saved else "rebuilt_french_saved" if day in windows else "rebuilt" for day in pd.date_range(first, last, freq="D")}
    return classes, saved, windows, notes


# ---- forecasts -----------------------------------------------------------------------------------------------------------------

def with_window(french: pd.DataFrame, window: Path, day: pd.Timestamp) -> pd.DataFrame:
    """The French table with the saved load forecast and nuclear output of the window laid over it, to the end of `day`.
    A value the window does not hold stays as refetched."""
    saved = _float_inputs(window)
    saved = saved[_days(saved) <= day]
    out = french.copy()
    rows = saved.index.intersection(out.index)
    for column in WINDOW_COLUMNS:
        values = saved.loc[rows, column]
        out.loc[values.dropna().index, column] = values.dropna()
    return out


def augmented(feat: dict, overrides: dict) -> dict:
    """The feature tables with extra rows for the days that have saved French inputs. An extra row has the inputs built from
    the saved window, sits in its own delivery day and has no target, so it is forecast by the model of its block and is
    never a training row. The rows of the refetch stay as they are, so every fit is exactly the fit without the extra rows."""
    if not overrides:
        return feat
    frames, sets = [feat["frame"]], {name: [table] for name, table in feat["sets"].items()}
    for day, tables in sorted(overrides.items()):
        rows = feat["frame"].index[np.asarray(feat["frame"]["delivery_day"] == day)]
        extra = feat["frame"].loc[rows].copy()
        extra[["fr", "de", "spread"]] = np.nan
        extra.index = rows + EXTRA
        frames.append(extra)
        for name in sets:
            part = tables[name].reindex(rows).copy()
            part.index = rows + EXTRA
            sets[name].append(part)
    frame = pd.concat(frames)
    return {**feat, "frame": frame, "y": frame["spread"], "sets": {name: pd.concat(parts) for name, parts in sets.items()}}


def forecast_all(feat: dict, first: str, last: str, log=lambda *_: None) -> pd.DataFrame:
    """Every model's walk-forward forecasts for first to last, by the frozen functions."""
    out = se.fit_months(feat, se.months_of(first, last), models=("H", "G", "C"), log=log)
    out["W"] = sw.fit_w(feat, first, last, log=log)
    for name in ("HW", "GK", "GM"):
        out[name] = shw.fit_comparator(feat, name, first, last, log=log)
    return out[list(ALL_MODELS)]


def override_tables(data: se.Data, windows: dict, classes: dict, log=lambda *_: None) -> dict:
    """{day: {input set: rows of the day}} for the days rebuilt with saved French inputs."""
    out = {}
    for day, path in sorted(windows.items()):
        if classes[day] != "rebuilt_french_saved":
            continue
        feat = se.features(replace(data, french=with_window(data.french, path, day)))
        rows = np.asarray(feat["frame"]["delivery_day"] == day)
        out[day] = {name: table.loc[rows] for name, table in feat["sets"].items()}
        log(f"inputs of {day:%Y-%m-%d} built with the window {path.name}")
    return out


# ---- the frame and the schedule --------------------------------------------------------------------------------------------------

def assemble(feat: dict, pred: pd.DataFrame, overrides: dict, classes: dict, saved: dict, first: str, last: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The frame of scored hours and the schedule of every hour of the window. `pred` is indexed like the augmented frame."""
    frame = feat["frame"]
    window = np.asarray((frame["delivery_day"] >= pd.Timestamp(first)) & (frame["delivery_day"] <= pd.Timestamp(last)))
    hours = frame.index[window]
    days = frame.loc[hours, "delivery_day"]
    table = pd.DataFrame({"delivery_day": days, "hour": frame.loc[hours, "hour"], "dow": frame.loc[hours, "dow"], "holiday": frame.loc[hours, "holiday"],
                          "spread": feat["y"].loc[hours], "de_holiday": feat["sets"]["G"].loc[hours, "de_holiday"],
                          "wind_px": feat["px"].loc[hours, "wind"], "solar_px": feat["px"].loc[hours, "solar"], "yesterday": feat["yesterday"].loc[hours]})
    table["provenance"] = days.map(classes).to_numpy()
    rebuilt = pred.reindex(hours)  # every model from the refetch alone: scored on a rebuilt day, compared on the others
    forecasts = rebuilt.copy()
    missing = feat["sets"]["G"].loc[hours].isna().sum(axis=1).astype(int)
    notes = {"saved_days_with_an_input_missing_on_some_hours_only": []}
    for day, tables in overrides.items():
        rows = hours[np.asarray(days == day)]
        forecasts.loc[rows, list(ALL_MODELS)] = pred.reindex(rows + EXTRA).to_numpy()
        missing.loc[rows] = tables["G"].reindex(rows).isna().sum(axis=1).astype(int)
    for day, record in saved.items():
        rows = pd.DatetimeIndex(pd.to_datetime(record["timestamps_utc"], utc=True))
        if not rows.equals(hours[np.asarray(days == day)]):
            raise ValueError(f"the record of {day:%Y-%m-%d} does not hold the hours of its day")
        for name in SAVED_MODELS:  # the saved forecast is the one scored; C is not attainable and stays rebuilt
            forecasts.loc[rows, name] = np.asarray(record["forecasts"][name], dtype=float)
        gaps = record["evidence"]["input_hours_missing_for_D"]
        if any(n != len(rows) for n in gaps.values()):  # the record counts hours by input, not inputs by hour: an input missing on some hours is counted on all
            notes["saved_days_with_an_input_missing_on_some_hours_only"].append(f"{day:%Y-%m-%d}")
        missing.loc[rows] = len(gaps)
    strict = frame.loc[hours, "weather_point_in_time"].astype(bool)
    is_saved = table["provenance"] == "saved"
    has_all = forecasts[list(ALL_MODELS)].notna().all(axis=1)  # C too: an hour needs a forecast from each of the seven
    issued = has_all & (is_saved | strict)  # a rebuilt hour whose weather is not the forecast issued two days ahead was never a forecast of 10:05
    no_fr, no_de = frame.loc[hours, "fr"].isna(), frame.loc[hours, "de"].isna()  # realised prices, from ENTSO-E's refetch only
    no_spread = table["spread"].isna()
    if (no_spread != (no_fr | no_de)).any():
        raise ValueError("the realised spread is not missing exactly where a realised price is")
    price = np.where(no_fr & no_de, score.PRICE_REASONS[2], np.where(no_fr, score.PRICE_REASONS[0], np.where(no_de, score.PRICE_REASONS[1], "")))
    no_proxy = (table["wind_px"].isna() | table["solar_px"].isna()).to_numpy()
    unissued = np.where(issued.to_numpy(), "", np.where(has_all.to_numpy(), NOT_STRICT, NOT_ISSUED))
    # Every cause that applies is written, so a missing price is named even on an hour that had no forecast either.
    reason = pd.Series([score.reason_text([part for part in (u, p, score.NO_PROXY if x else "") if part]) for u, p, x in zip(unissued, price, no_proxy)], index=hours)
    schedule = pd.DataFrame({"delivery_day": days, "provenance": table["provenance"], "issued": issued.astype(bool), "inputs_missing": missing, "unscored_reason": reason})
    scored = issued & (reason == "")
    out = pd.concat([table.loc[scored, ["delivery_day", "hour", "dow", "holiday", "spread", "de_holiday", "wind_px", "solar_px", "yesterday"]], forecasts.loc[scored, list(ALL_MODELS)],
                     table.loc[scored, ["provenance"]]], axis=1)
    notes["rebuilt"] = rebuilt
    return out, schedule, notes


# ---- the stop conditions -----------------------------------------------------------------------------------------------------------

def overlap_check(src: Sources) -> dict:
    """Stop condition 1: the refetched German series against the stored raw files of the first test, 11 September to 10 October."""
    kept, new = se.parse_raw(Path(src.static) / "raw_first_test"), lsl.parse_german_folder(Path(src.german_raw))
    out = {}

    def one(name, a, b, tolerance, share):
        days = _days(a)
        a = a[(days >= pd.Timestamp(OVERLAP[0])) & (days <= pd.Timestamp(OVERLAP[1]))].dropna()
        b = b.reindex(a.index)
        close = ((a - b).abs() <= tolerance) & b.notna()
        out[name] = {"hours": int(len(a)), "missing_in_refetch": int(b.isna().sum()), "share_equal": float(close.mean()) if len(a) else None,
                     "largest_difference": float((a - b).abs().max()) if len(a) else None, "required_share": share, "pass": bool(len(a) > 0 and close.mean() >= share)}

    one("German coupled price", kept["de"], new["de"], se.PRICE_MATCH, se.PRICE_SHARE)
    for column in se.TSO_COLUMNS:
        one(f"German {column} forecast", kept["tso"][column], new["tso"][column], se.MW_MATCH, se.MW_SHARE)
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def french_check(src: Sources, data: se.Data) -> dict:
    """Stop condition 2: do the refetched French inputs give the stored H inputs to 2026-10-10, and do the refetched points
    equal the hashed files where both hold a value? The stored values are used whatever this finds; it is reported."""
    stored = replace(data, french=se.cut(data.french, STORED_LAST), de=se.cut(data.de, STORED_LAST), tso=se.cut(data.tso, STORED_LAST), last=STORED_LAST)
    fresh, _ = fq.fill_proxy_gaps(_float_inputs(src.french))
    a, b = se.features(stored)["sets"]["H"], se.features(replace(stored, french=se.cut(fresh, STORED_LAST)))["sets"]["H"]
    b = b.reindex(a.index)
    columns = {}
    for column in a.columns:
        x, y = a[column].to_numpy(float), b[column].to_numpy(float)
        held = ~np.isnan(x)
        with np.errstate(invalid="ignore"):
            apart = held & (np.isnan(y) | (np.abs(x - y) > lsl.SCRAMBLE_TOLERANCE * np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))))
        if apart.any():
            gaps = np.abs(x - y)[apart & ~np.isnan(y)]
            columns[column] = {"hours": int(apart.sum()), "missing_in_refetch": int((held & np.isnan(y)).sum()), "largest_difference": float(gaps.max()) if len(gaps) else None}
    points = {}
    static = Path(src.static)
    for kind, path in (("wind", src.wind), ("solar", src.solar)):
        held, fresh_points = se._hourly(static / f"german_{kind}_points.csv.gz"), se._hourly(path)
        fresh_points = fresh_points.reindex(index=held.index, columns=held.columns)
        both = held.notna() & fresh_points.notna()
        apart = both & ((held - fresh_points).abs() > 1e-9)
        points[kind] = {"values_both_hold": int(both.to_numpy().sum()), "values_that_differ": int(apart.to_numpy().sum()),
                        "largest_difference": float((held - fresh_points).abs().where(both).max().max()) if both.to_numpy().any() else None,
                        "hashed_values_missing_in_refetch": int((held.notna() & fresh_points.isna()).to_numpy().sum())}
    return {"hours_compared": int(len(a)), "H_inputs_that_differ": columns, "points": points, "stored_values_used_to": STORED_LAST,
            "differences_found": bool(columns or any(p["values_that_differ"] for p in points.values())), "pass": True}


def coverage_check(feat: dict, first: str, last: str, accepted: dict | None = None) -> dict:
    """Stop condition 3: no input missing on a whole day of the window unless it was missing in the same way a year earlier.
    A hole named in an amendment (`accepted`: {input: [days]}) no longer stops the run; every other new hole does."""
    frame = feat["frame"]
    table = pd.concat([feat["sets"]["C"], frame[["fr", "de"]]], axis=1)
    year = pd.DateOffset(years=1)
    earlier = input_coverage(table, frame["delivery_day"], pd.Timestamp(first) - year, pd.Timestamp(last) - year)
    days = pd.to_datetime(frame["delivery_day"])
    holes = {}
    for column in table.columns:
        inside = table[column].notna()[np.asarray((days >= pd.Timestamp(first)) & (days <= pd.Timestamp(last)))]
        any_value = inside.groupby(days[inside.index]).any()
        missing = [f"{d:%Y-%m-%d}" for d in any_value.index[~any_value]]
        if missing:
            holes[column] = missing
    before = {k: int(v) for k, v in earlier["days_fully_missing"].items() if v}
    accepted = accepted or {}
    new = {column: [d for d in missing if d not in set(accepted.get(column, []))] for column, missing in holes.items() if column not in before}
    new = {column: missing for column, missing in new.items() if missing}
    return {"inputs_with_a_fully_missing_day": holes, "the_same_inputs_a_year_earlier": before, "accepted_by_an_amendment": accepted, "new_holes_not_accepted": new, "pass": not new}


def run_selftest(static: Path, log=lambda *_: None) -> dict:
    """The logger's self-test, run here and now: does this workspace give the forecasts of the reference machine? It reads
    the static files only and involves no price of the test window. Nothing is taken on trust from a file."""
    static = Path(static)
    if static.name != "static":
        return {"pass": False, "error": "the static folder must be named `static`, as the logger expects"}
    try:
        result = lsl.selftest(lsl.Paths(root=static.parent), log=log)
    except Exception as exc:  # a workspace that cannot run it has not passed it
        return {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"pass": bool(result.get("pass")), "verdict": result.get("verdict"), "code_matches_reference": result.get("code_matches_reference"), "run_at_utc": lsl.iso(lsl.utc_now())}


def reproduce_check(pred: pd.DataFrame, kept_folder: Path | None, machine: dict | None = None) -> dict:
    """Stop condition 4, under the exception of 2026-11-23 (docs/prereg/spread_w_stop_condition_4_exception.md): the rebuild
    reproduces the reconstructions of that day on 2024-05-01 to 2026-10-10 to 1e-9 for H, G, C and W, in a workspace that
    passes the logger's self-test on the day. HW, GK and GM are checked the same way where the file holds them."""
    machine = machine or {"pass": False, "error": "the self-test was not run"}
    path = Path(kept_folder) / RECONSTRUCTION_FILE if kept_folder is not None else None
    if path is None or not path.exists():
        return {"basis": EXCEPTION, "pass": False, "error": f"{RECONSTRUCTION_FILE} of the reconstructions of 2026-11-23 was not given"}
    models, required = ALL_MODELS, score.STOP_4_REQUIRED
    kept = pd.read_csv(path, index_col=0, float_precision="round_trip")
    kept.index = pd.to_datetime(kept.index, utc=True)
    absent = [m for m in models if m not in kept.columns]
    if absent:
        return {"basis": EXCEPTION, "pass": False, "error": f"the file of stored forecasts lacks {absent}"}
    common = kept.index.intersection(pred.index)
    mine, theirs = pred.reindex(index=common, columns=list(models)), kept.loc[common, list(models)]
    compared = {m: int((mine[m].notna() & theirs[m].notna()).sum()) for m in models}  # a missing forecast is a failure, never a skipped hour
    largest = {m: float((mine[m] - theirs[m]).abs().max()) if compared[m] else None for m in models}
    whole = {m: bool(len(kept) > 0 and compared[m] == len(kept)) for m in models}
    within = {m: bool(whole[m] and largest[m] <= se.TOLERANCE) for m in models}
    sha = se.sha256(path)
    return {"basis": EXCEPTION, "file": path.name, "sha256": sha, "is_the_reconstruction_of_2026_11_23": bool(sha == score.RECONSTRUCTION_SHA256), "self_test_of_this_workspace": machine,
            "models": list(models), "stored_hours": int(len(kept)), "hours_compared": int(len(common)), "values_compared": compared, "largest_difference": largest, "within_tolerance": within,
            "tolerance": se.TOLERANCE, "required": list(required), "reported_not_required": [m for m in models if m not in required],
            "pass": bool(sha == score.RECONSTRUCTION_SHA256 and machine.get("pass") is True and all(within[m] for m in required))}


def scramble_checks(data: se.Data, days, log=print) -> dict:
    """Stop condition 5: inputs identical and the forecasts of H, G and W unchanged to 1e-9 under the scramble, while C moves."""
    monthly, weekly = se.scramble_check(data, days, log=log), sw.scramble_check(data, days, log=log)
    out = {}
    for day in days:
        out[day] = {"H_and_G": monthly[day], "W": weekly[day], "C_moved": bool(monthly[day]["C_moved_by"] > se.TOLERANCE)}
        out[day]["pass"] = bool(monthly[day]["pass"] and weekly[day]["pass"] and out[day]["C_moved"])
    out["pass"] = all(v["pass"] for v in out.values())
    return out


# ---- the report without prices ---------------------------------------------------------------------------------------------------

def revisions(data: se.Data, windows: dict) -> dict:
    """Rule 5 of the refetch specification: for every day with a saved French window, the saved value against the refetched
    one. No price is involved."""
    totals = {}
    for day, path in sorted(windows.items()):
        saved = _float_inputs(path)
        local = saved.index.tz_convert(lsl.PARIS)
        on_day = _days(saved) == day
        before = ((_days(saved) == day - pd.Timedelta(days=2)) | ((_days(saved) == day - pd.Timedelta(days=1)) & (local.hour < 8)))  # nuclear output is used to 08:00 Paris on the eve
        groups = {"load_fc_mw": (on_day, ["load_fc_mw"]), "nuclear_mw": (before, ["nuclear_mw"]),
                  "weather": (on_day, [c for c in saved.columns if c in WEATHER_COLUMNS or c.startswith("wind_pt_") or c.startswith("solar_pt_")])}
        for name, (rows, columns) in groups.items():
            a = saved.loc[rows, columns]
            b = data.french.reindex(index=a.index, columns=columns)
            both = (a.notna() & b.notna()).to_numpy()
            gap, size = (a - b).abs().to_numpy(), np.maximum(a.abs().to_numpy(), b.abs().to_numpy())
            differ = both & (gap > 1e-9 * np.maximum(1.0, size))
            large = both & (gap > REVISION_SIZE * size)
            t = totals.setdefault(name, {"days": 0, "values_compared": 0, "values_that_differ": 0, "values_that_differ_by_more_than_1_percent": 0, "largest_difference": 0.0,
                                         "saved_but_missing_in_refetch": 0, "refetched_but_missing_in_window": 0})
            t["days"] += 1
            t["values_compared"] += int(both.sum())
            t["values_that_differ"] += int(differ.sum())
            t["values_that_differ_by_more_than_1_percent"] += int(large.sum())
            t["largest_difference"] = max(t["largest_difference"], float(gap[both].max()) if both.any() else 0.0)
            t["saved_but_missing_in_refetch"] += int((a.notna() & b.isna()).to_numpy().sum())
            t["refetched_but_missing_in_window"] += int((a.isna() & b.notna()).to_numpy().sum())
    for name, t in totals.items():
        t["share_that_differ"] = t["values_that_differ"] / t["values_compared"] if t["values_compared"] else None
        t["share_by_more_than_1_percent"] = t["values_that_differ_by_more_than_1_percent"] / t["values_compared"] if t["values_compared"] else None
        t["stated_beside_the_result"] = bool(name in WINDOW_COLUMNS and t["values_compared"] and t["share_by_more_than_1_percent"] > REVISION_SHARE)
    return totals


def saved_days_report(feat: dict, saved: dict, rebuilt: pd.DataFrame) -> dict:
    """What the records say about themselves, the saved inputs against the rebuilt inputs by fingerprint (amendment 3), and
    the saved forecasts against the rebuilt ones (rule 6). A difference between two forecasts is not an error against a price."""
    out = {"days": len(saved), "with_every_fetch_in_by_10_05_paris": 0, "with_hours_from_the_second_source": {}, "issued_without_yesterdays_spread_on_some_hour": [],
           "with_an_input_left_missing": {}, "by_code_hash": {}, "inputs_that_differ_from_the_rebuild": {}, "forecast_difference_from_the_rebuild": {}}
    gaps = {name: [] for name in SAVED_MODELS}
    for day, record in sorted(saved.items()):
        label = f"{day:%Y-%m-%d}"
        out["with_every_fetch_in_by_10_05_paris"] += bool(record.get("all_fetches_finished_by_cutoff"))
        out["by_code_hash"][record["code_sha256"]] = out["by_code_hash"].get(record["code_sha256"], 0) + 1
        taken = ((record.get("inputs") or {}).get(lsl.SECOND_SOURCE) or {}).get("hours_taken_from_it", 0)
        if taken:
            out["with_hours_from_the_second_source"][label] = int(taken)
        evidence = record["evidence"]
        if evidence["yesterdays_spread_present_hours"] < evidence["hours"]:
            out["issued_without_yesterdays_spread_on_some_hour"].append(label)
        if evidence["input_hours_missing_for_D"]:
            out["with_an_input_left_missing"][label] = evidence["input_hours_missing_for_D"]
        rows = np.asarray(feat["frame"]["delivery_day"] == day)
        if "fingerprint" in record:
            differ = lsl.compare_fingerprints(lsl.fingerprint(feat["sets"]["G"].loc[rows]), record["fingerprint"]["inputs_of_D"])
            if differ:
                out["inputs_that_differ_from_the_rebuild"][label] = sorted(differ)
        hours = feat["frame"].index[rows]
        for name in SAVED_MODELS:
            gaps[name].append(np.abs(np.asarray(record["forecasts"][name], dtype=float) - rebuilt.loc[hours, name].to_numpy()))
    for name, parts in gaps.items():
        if parts:
            values = np.concatenate(parts)
            out["forecast_difference_from_the_rebuild"][name] = {"largest": float(np.nanmax(values)), "mean": float(np.nanmean(values))}
    return out


# ---- the run -----------------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def rebuild(src: Sources, out: Path, first: str, last: str, kept: Path | None, accepted: dict | None = None, scramble_days=SCRAMBLE_DAYS,
            reproduce_from: str = se.DEV_FIRST, log=print) -> dict:
    """Everything, in the order of the plan. Returns the stop-condition record. No error against a realised spread is computed."""
    if pd.Timestamp(last) > pd.Timestamp(STORED_LAST) and swt._today() < pd.Timestamp(swt.READ_FROM):
        raise SystemExit(f"sealed until {swt.READ_FROM}: the frame of the window is not assembled before the read date")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"test": swt.TEST_NAME, "window": [first, last], "all_pass": False, "refetch": {name: (se.sha256(path) if Path(path).is_file() else lsl.folder_sha256(path))
                                                                                                 for name, path in (("french", src.french), ("german_raw", src.german_raw), ("wind", src.wind), ("solar", src.solar))}}

    def stop(name: str) -> dict:
        _write(out / "stop_checks.json", checks)
        log(f"STOP: {name}. The cause goes into an amendment before anything else is read")
        return checks

    checks["1_german_overlap"] = overlap_check(src)
    if not checks["1_german_overlap"]["pass"]:
        return stop("the refetched German series do not match the stored raw files")
    data, checks["join"] = join(src, last)
    checks["2_french_and_points"] = french_check(src, data)
    feat = se.features(data)
    checks["3_coverage"] = coverage_check(feat, first, last, accepted)
    if not checks["3_coverage"]["pass"]:
        return stop("an input is missing on a whole day of the window")
    classes, saved, windows, notes = classify(src, first, last)
    overrides = override_tables(data, windows, classes, log=log)
    wide = augmented(feat, overrides)
    pred = forecast_all(wide, reproduce_from, last, log=log)
    checks[score.STOP_4] = reproduce_check(pred, kept, run_selftest(src.static))
    checks["5_scramble"] = scramble_checks(data, list(scramble_days), log=log)
    frame, schedule, extra = assemble(feat, pred, overrides, classes, saved, first, last)
    # Stop condition 6 as amendment 6 leaves it: the floor is recorded here and applied by the scoring code, which marks a
    # read under it "not readable". What stops the run is a frame or a schedule the scoring code would refuse, found now and
    # not on the read date. Neither check computes an error.
    scheduled = pd.date_range(first, last, freq="D")
    try:
        score.validate(frame, first, last)
        score.check_schedule(frame, schedule, first, last, expected_hours=int(sum(len(fq.day_hours(day)) for day in scheduled)))
        refused = None
    except ValueError as exc:
        refused = str(exc)
    left_out = schedule.loc[~schedule.index.isin(frame.index), "unscored_reason"]
    checks["6_eligible_hours"] = {"days": int(frame["delivery_day"].nunique()), "hours": int(len(frame)), "scheduled_days": int(len(scheduled)), "scheduled_hours": int(len(schedule)),
                                  "floor": {"least_days": score.FLOOR_DAYS, "least_hours": score.FLOOR_HOURS, "met": bool(frame["delivery_day"].nunique() >= score.FLOOR_DAYS and len(frame) >= score.FLOOR_HOURS)},
                                  "days_with_no_eligible_hour": sorted(f"{d:%Y-%m-%d}" for d in set(schedule["delivery_day"]) - set(frame["delivery_day"])),
                                  "hours_left_out_by_reason": {k: int(v) for k, v in left_out.value_counts().sort_index().items()},
                                  "eligible_hours_without_yesterdays_spread": int(frame["yesterday"].isna().sum()),
                                  "refused_by_the_scoring_code": refused, "pass": refused is None}
    frame.to_csv(out / "test_predictions.csv.gz")  # working files, never published
    schedule.to_csv(out / "schedule.csv")
    checks["test_predictions_sha256"], checks["schedule_sha256"] = se.sha256(out / "test_predictions.csv.gz"), se.sha256(out / "schedule.csv")
    report = {"window": [first, last], "days_by_class": {k: int(sum(v == k for v in classes.values())) for k in swt.PROVENANCE}, "records": notes,
              "french_windows_saved_on_the_eve": {f"{d:%Y-%m-%d}": p.name for d, p in sorted(windows.items())},
              "saved_days": saved_days_report(feat, saved, extra["rebuilt"]), "saved_against_refetched_french_inputs": revisions(data, windows),
              "saved_days_with_an_input_missing_on_some_hours_only": extra["saved_days_with_an_input_missing_on_some_hours_only"],
              "note": "No error against any price is computed here. The forecasts of saved days are the saved ones; C is rebuilt on every day."}
    _write(out / "rebuild_report.json", report)
    checks["all_pass"] = all(checks[k]["pass"] for k in ("1_german_overlap", "2_french_and_points", "3_coverage", score.STOP_4, "5_scramble", "6_eligible_hours"))
    _write(out / "stop_checks.json", checks)
    log("stop conditions: " + ("all pass" if checks["all_pass"] else "FAILED, nothing is scored"))
    return checks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild of the test of W: frame, schedule and stop conditions. No error is computed")
    parser.add_argument("stage", choices=["run", "replay"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--french", required=True)
    parser.add_argument("--german-raw", required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--solar", required=True)
    parser.add_argument("--live")
    parser.add_argument("--production-windows")
    parser.add_argument("--kept", help="folder of the stored forecasts (comparator_predictions.csv.gz or w_predictions.csv.gz)")
    parser.add_argument("--accepted-holes", help="JSON {input: [days]} of the holes an amendment has accepted")
    parser.add_argument("--first")
    parser.add_argument("--last")
    args = parser.parse_args(argv)
    src = Sources(Path(args.static), Path(args.french), Path(args.german_raw), Path(args.wind), Path(args.solar), Path(args.live) if args.live else None,
                  Path(args.production_windows) if args.production_windows else None)
    accepted = json.loads(Path(args.accepted_holes).read_text()) if args.accepted_holes else None
    if args.stage == "run":
        rebuild(src, Path(args.out), swt.FIRST, swt.LAST, Path(args.kept) if args.kept else None, accepted)
        return
    if not args.first or not args.last or pd.Timestamp(args.last) > pd.Timestamp(STORED_LAST):
        raise SystemExit(f"a replay needs --first and --last, and ends on {STORED_LAST} at the latest")
    rebuild(src, Path(args.out), args.first, args.last, Path(args.kept) if args.kept else None, accepted, scramble_days=(args.first, args.last))


if __name__ == "__main__":
    main()
