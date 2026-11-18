"""The checks that come before the forward test of Q (docs/prereg/forward_q_prereg.md, section 3).

    python -m thermo_fr.forecast.forward_q_checks same-path --out DIR --frame FRENCH_FRAME
    python -m thermo_fr.forecast.forward_q_checks scramble  --out DIR
    python -m thermo_fr.forecast.forward_q_checks replay    --out DIR --db STORE --fresh WINDOWS [--shard K/N]   # unscored: no price is read
    python -m thermo_fr.forecast.forward_q_checks vectors   --out DIR --fresh WINDOWS                           # load forecast revision vectors
    python -m thermo_fr.forecast.forward_q_checks gbm       --out DIR --frame FRENCH_FRAME                      # honest_v2 at production settings
    python -m thermo_fr.forecast.forward_q_checks revision  --out DIR --frame FRENCH_FRAME [--shard K/N]        # L0 and Q refitted on revised load
    python -m thermo_fr.forecast.forward_q_checks summary   --out DIR --frame FRENCH_FRAME
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from . import forward_q as fq
from . import french_rl, lear, lear_recipe, yardstick
from . import neighbour_rl as nr
from .features import build_features
from .models import GBM_PARAMS, fit_predict
from .refit import REFIT_PARAMS
from .timing import delivery_days

SAME_PATH_DAYS = french_rl.SAME_PATH_DAYS
SCRAMBLE_DAYS = ("2024-10-27", "2025-03-30", "2026-03-29")
TOLERANCE = 1e-9
REPLAY_FROM = "2026-10-20"
REPLAY_TOLERANCE = 0.01
REVISION_STEP, REVISION_SEED = 13, 0
STORED = {"L0": "lear_w{w}", "P": "P_w{w}", "Q": "Q_w{w}"}


def _stored(frame: pd.DataFrame, day, name: str, w: int) -> pd.Series:
    rows = frame[frame["delivery_day"] == pd.Timestamp(day)]
    return rows.groupby("hour")[STORED[name].format(w=w)].first()


def same_path(hourly: pd.DataFrame, frame: pd.DataFrame, days=SAME_PATH_DAYS, log=print) -> dict:
    base = lear_recipe.selection_inputs(hourly)
    result = {"days": {}, "max_abs_difference": 0.0}
    for d in days:
        live = fq.q_live(base, d)
        worst = {}
        for name in fq.CANDIDATES:
            diffs = [float((live[name]["windows"][w].reindex(_stored(frame, d, name, w).index) - _stored(frame, d, name, w)).abs().max()) for w in nr.WINDOWS]
            worst[name] = max(diffs)
        result["days"][d] = worst
        result["max_abs_difference"] = max(result["max_abs_difference"], *worst.values())
        log(f"{d}: largest difference from the stored window forecasts {max(worst.values()):.3e} {worst}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    return result


def scramble(hourly: pd.DataFrame, days=SCRAMBLE_DAYS, log=print) -> dict:
    base = lear_recipe.selection_inputs(hourly)
    result = {"days": {}, "max_abs_difference": 0.0}
    for k, d in enumerate(days):
        day = pd.Timestamp(d)
        noisy = lear_recipe.scramble_from(base, day, seed=k)
        a, b = noisy.to_numpy(dtype=float), base.to_numpy(dtype=float)
        changed = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
        before, after = fq.q_live(base, d), fq.q_live(noisy, d)
        worst = {name: max(float((before[name]["windows"][w] - after[name]["windows"][w]).abs().max()) for w in nr.WINDOWS) for name in fq.CANDIDATES}
        result["days"][d] = {"scrambled_input_values": changed, **worst}
        result["max_abs_difference"] = max(result["max_abs_difference"], *worst.values())
        log(f"{d}: {changed} input values scrambled, largest change {max(worst.values()):.3e}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    return result


def replay(db, inputs_path, fresh_dir, out: Path, shard=(0, 1), first: str = REPLAY_FROM, log=print) -> dict:
    """L0 through q_live from the saved windows of each pre-market run, against the live LEAR the S2 shadow stored for the same
    run. Forecasts only: the table of auction prices is never opened here."""
    from .shadow import live_history

    conn = fq.read_only(db)
    versions = pd.read_sql_query("SELECT version_id, delivery_day, morning_run_id, issued_at_utc FROM shadow_versions WHERE model = 'LEAR' AND before_window = 1"
                                 " AND fallback = 0 AND delivery_day >= ? ORDER BY delivery_day, version_id", conn, params=(first,))
    versions = versions.groupby("delivery_day").last().reset_index()
    folder = Path(out) / "replay"
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in versions.iterrows():
        path = folder / f"{v['delivery_day']}.json"
        if k % shard[1] != shard[0] or path.exists():
            continue
        t0 = time.time()
        stored = pd.read_sql_query("SELECT timestamp_utc, forecast FROM shadow_values WHERE version_id = ? ORDER BY timestamp_utc", conn, params=(int(v["version_id"]),))
        stored = pd.Series(stored["forecast"].to_numpy(), index=pd.to_datetime(stored["timestamp_utc"], utc=True))
        record = {"delivery_day": v["delivery_day"], "morning_run_id": int(v["morning_run_id"]), "live_issued_at_utc": v["issued_at_utc"]}
        try:
            history = live_history(inputs_path, fresh_dir, upto_run_id=int(v["morning_run_id"]))
            live = fq.q_live(history, v["delivery_day"])
            l0 = live["L0"]["hourly"].reindex(stored.index)
            record.update({"hours": int(len(stored)), "l0_max_abs_difference": float((l0 - stored).abs().max()),
                           "l0_mean_abs_difference": float((l0 - stored).abs().mean()),
                           "complete": {name: bool(live[name]["hourly"].reindex(stored.index).notna().all()) for name in fq.CANDIDATES},
                           "kept": {name: {str(w): c for w, c in live[name]["kept"].items()} for name in ("P", "Q")},
                           "forecasts": {name: [float(x) for x in live[name]["hourly"].reindex(stored.index).to_numpy()] for name in fq.CANDIDATES}})
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{exc.__class__.__name__}: {exc}"
        record["seconds"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(record, indent=1))
        log(f"{v['delivery_day']}: {record.get('error') or 'L0 differs from the live LEAR by at most %.2e' % record['l0_max_abs_difference']} ({record['seconds']} s)", flush=True)
    conn.close()
    return {"days": int(len(versions))}


def replay_summary(out: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted((Path(out) / "replay").glob("*.json"))]
    good = [r for r in records if "error" not in r]
    return {"days": len(records), "errors": [{"day": r["delivery_day"], "error": r["error"]} for r in records if "error" in r],
            "first_day": records[0]["delivery_day"] if records else None, "last_day": records[-1]["delivery_day"] if records else None,
            "l0_max_abs_difference": max((r["l0_max_abs_difference"] for r in good), default=None),
            "l0_mean_abs_difference": float(np.mean([r["l0_mean_abs_difference"] for r in good])) if good else None,
            "days_with_L0_P_Q_complete": int(sum(all(r["complete"].values()) for r in good)),
            "q_kept_min_of_24": min((min(v for v in r["kept"]["Q"].values() if v is not None) for r in good), default=None),
            "seconds_median": float(np.median([r["seconds"] for r in records])) if records else None,
            "passes": bool(records and len(good) == len(records) and all(all(r["complete"].values()) for r in good)
                           and max(r["l0_max_abs_difference"] for r in good) <= REPLAY_TOLERANCE)}


def revision_vectors(fresh_dir) -> pd.DataFrame:
    """One row per delivery day with a window saved between 10:05 and 11:00 Paris on D-1: the ratio of the load forecast of D in
    the last such window to the last value seen in any later window, by Paris clock hour. Days with a clock change are left out."""
    seen = []
    for path in sorted(Path(fresh_dir).glob("*_run*_*.csv")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_run(\d+)_(\d{8}T\d{6}Z)\.csv", path.name)
        table = pd.read_csv(path, index_col=0, usecols=["timestamp_utc", "load_fc_mw"])
        table.index = pd.to_datetime(table.index, utc=True)
        s = table["load_fc_mw"].dropna()
        seen.append(pd.DataFrame({"t": s.index, "v": s.to_numpy(), "stamp": pd.Timestamp(m.group(3)), "target": pd.Timestamp(m.group(1))}))
    seen = pd.concat(seen, ignore_index=True)
    final = seen.sort_values("stamp").groupby("t").agg(final=("v", "last"), last_stamp=("stamp", "last"))
    seen["day"] = delivery_days(pd.DatetimeIndex(seen["t"]))
    clock = pd.DatetimeIndex(seen["stamp"]).tz_convert(LOCAL_TZ)
    minutes = clock.hour * 60 + clock.minute
    pre = seen[(seen["day"] == seen["target"]) & (minutes >= 10 * 60 + 5) & (minutes < 11 * 60) & (clock.normalize().tz_localize(None) == seen["target"] - pd.Timedelta(days=1))]
    pre = pre.sort_values("stamp").groupby("t").last().join(final)
    pre = pre[pre["last_stamp"] > pre["stamp"]]
    pre["hour"] = pd.DatetimeIndex(pre.index).tz_convert(LOCAL_TZ).hour
    pre["ratio"] = pre["v"] / pre["final"]
    table = pre.pivot_table(index="day", columns="hour", values="ratio", aggfunc="first")
    return table[table.notna().sum(axis=1) == 24].reindex(columns=range(24))


def revision_plan(frame: pd.DataFrame, vectors: pd.DataFrame) -> pd.DataFrame:
    """Every REVISION_STEP-th rule day, starting with the first, with the revision vector drawn for it (seed REVISION_SEED)."""
    rule = frame[(frame["delivery_day"] >= french_rl.RULE_START) & frame["strict"].astype(bool)]
    days = pd.DatetimeIndex(sorted(rule["delivery_day"].unique()))[::REVISION_STEP]
    draw = np.random.default_rng(REVISION_SEED).integers(0, len(vectors), size=len(days))
    return pd.DataFrame({"day": days, "vector": draw, "vector_day": vectors.index[draw]})


def revised_inputs(base: pd.DataFrame, plan: pd.DataFrame, vectors: pd.DataFrame, only=None) -> pd.DataFrame:
    """The inputs with the load forecast of each planned day (or of `only`) multiplied by its vector, D only."""
    out = base.copy()
    days = delivery_days(out.index)
    hours = out.index.tz_convert(LOCAL_TZ).hour
    for _, row in plan.iterrows():
        if only is not None and row["day"] != pd.Timestamp(only):
            continue
        mask = np.asarray(days == row["day"])
        out.loc[mask, "load_fc_mw"] = out.loc[mask, "load_fc_mw"].to_numpy() * vectors.iloc[int(row["vector"])].reindex(hours[mask]).to_numpy()
    return out


def gbm_production(hourly: pd.DataFrame, out: Path, plan: pd.DataFrame | None, vectors: pd.DataFrame | None, log=print) -> pd.DataFrame:
    """honest_v2 refitted month by month with the production settings (300 trees of 31 leaves): its forecast on every row, and on
    the planned days its forecast from the revised load. The models are fitted on the unrevised inputs only."""
    base = lear_recipe.selection_inputs(hourly)
    table = build_features(base, "honest_v2")
    revised = build_features(revised_inputs(base, plan, vectors), "honest_v2") if plan is not None else None
    params = {**GBM_PARAMS, **REFIT_PARAMS}
    days = table.info["delivery_day"]
    folder = Path(out) / "gbm_prod"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for start in pd.date_range(pd.Timestamp(lear_recipe.WALK_START), pd.Timestamp(lear_recipe.TEST_END), freq="MS", inclusive="left"):
        path = folder / f"honest_v2_prod_{start:%Y-%m}.csv"
        if not path.exists():
            train, test = np.asarray(days < start), np.asarray((days >= start) & (days < start + pd.DateOffset(months=1)))
            from .models import make_model

            keep = table.y[train].notna()
            model = make_model("gbm", params)
            model.fit(table.X[train][keep], table.y[train][keep])
            part = table.info.loc[test, ["delivery_day", "hour"]].copy()
            part["honest_v2_prod"] = np.asarray(model.predict(table.X[test]), dtype=float)
            if revised is not None:
                part["honest_v2_prod_revised"] = np.asarray(model.predict(revised.X.loc[table.X.index[test]]), dtype=float)
            tmp = path.with_suffix(".tmp")
            part.to_csv(tmp, index_label="timestamp_utc")
            tmp.replace(path)
            log(f"  honest_v2 at production settings: {start:%Y-%m} trained on {int(keep.sum()):,} hours", flush=True)
        frames.append(lear_recipe.read_frame(path))
    return pd.concat(frames).sort_index()


def revision_lear(hourly: pd.DataFrame, plan: pd.DataFrame, vectors: pd.DataFrame, out: Path, shard=(0, 1), log=print) -> None:
    base = lear_recipe.selection_inputs(hourly)
    folder = Path(out) / "revision"
    folder.mkdir(parents=True, exist_ok=True)
    for k, row in plan.iterrows():
        path = folder / f"{row['day']:%Y-%m-%d}.csv"
        if k % shard[1] != shard[0] or path.exists():
            continue
        t0 = time.time()
        live = fq.q_live(revised_inputs(base, plan, vectors, only=row["day"]), row["day"], candidates=("L0", "Q"))
        part = pd.DataFrame({"L0_revised": live["L0"]["hourly"], "Q_revised": live["Q"]["hourly"]})
        tmp = path.with_suffix(".tmp")
        part.to_csv(tmp, index_label="timestamp_utc")
        tmp.replace(path)
        log(f"{row['day']:%Y-%m-%d}: L0 and Q refitted on the revised load forecast ({time.time() - t0:.0f} s)", flush=True)


def _pair(frame: pd.DataFrame, a: str, b: str) -> dict:
    p = yardstick.paired_daily(frame, a, b, one_sided_level=0.95)
    return {"mae": yardstick.mae(frame, a), "baseline_mae": yardstick.mae(frame, b), "gain": yardstick.mae(frame, b) - yardstick.mae(frame, a), **p}


def summary(out: Path, frame: pd.DataFrame) -> dict:
    result = {}
    for name in ("same_path", "scramble"):
        path = Path(out) / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    result["replay"] = replay_summary(out)
    gbm_files = sorted((Path(out) / "gbm_prod").glob("*.csv"))
    if gbm_files:
        gbm = pd.concat([lear_recipe.read_frame(p) for p in gbm_files]).sort_index()
        frame = frame.join(gbm.drop(columns=["delivery_day", "hour"]))
        frame["S1_prod"] = (frame["honest_v2_prod"] + frame["L0"]) / 2
        frame["S1_Q_prod"] = (frame["honest_v2_prod"] + frame["Q"]) / 2
        rule = frame[frame["delivery_day"] >= french_rl.RULE_START]
        result["production_settings"] = {
            "all_864_days": {c: yardstick.mae(frame, c) for c in ("honest_v2", "honest_v2_prod", "S1", "S1_prod")},
            "rule_763_days": {"honest_v2": yardstick.mae(rule, "honest_v2"), "honest_v2_prod": yardstick.mae(rule, "honest_v2_prod"),
                              "research_settings_S1_Q_vs_S1": _pair(rule, "S1_Q", "S1"), "production_settings_S1_Q_vs_S1": _pair(rule, "S1_Q_prod", "S1_prod"),
                              "production_settings_S1_Q_vs_honest_v2": _pair(rule, "S1_Q_prod", "honest_v2_prod"),
                              "production_vs_research_honest_v2": _pair(rule, "honest_v2_prod", "honest_v2")}}
        files = sorted((Path(out) / "revision").glob("*.csv"))
        if files and "honest_v2_prod_revised" in frame:
            lear_rev = pd.concat([_read_plain(p) for p in files]).sort_index()
            rows = frame.join(lear_rev, how="inner")
            rows["S1_prod_revised"] = (rows["honest_v2_prod_revised"] + rows["L0_revised"]) / 2
            rows["S1_Q_prod_revised"] = (rows["honest_v2_prod_revised"] + rows["Q_revised"]) / 2
            rows = rows[rows["strict"].astype(bool)]
            m = lambda c: yardstick.mae(rows, c)  # noqa: E731
            result["load_revision"] = {
                "days": int(rows["delivery_day"].nunique()), "hours": int(len(rows)),
                "archive_load": {"L0": m("L0"), "Q": m("Q"), "honest_v2_prod": m("honest_v2_prod"), "S1_prod": m("S1_prod"), "S1_Q_prod": m("S1_Q_prod"),
                                 "gain_Q_over_L0": m("L0") - m("Q"), "gain_S1_Q_over_S1": m("S1_prod") - m("S1_Q_prod")},
                "revised_load": {"L0": m("L0_revised"), "Q": m("Q_revised"), "honest_v2_prod": m("honest_v2_prod_revised"), "S1_prod": m("S1_prod_revised"),
                                 "S1_Q_prod": m("S1_Q_prod_revised"), "gain_Q_over_L0": m("L0_revised") - m("Q_revised"),
                                 "gain_S1_Q_over_S1": m("S1_prod_revised") - m("S1_Q_prod_revised")},
                "mean_abs_change_of_forecast": {c: float((rows[f"{c}_revised"] - rows[c]).abs().mean()) for c in ("L0", "Q", "honest_v2_prod")},
                "paired_revised_minus_archive": {c: yardstick.paired_daily(rows, f"{c}_revised", c) for c in ("L0", "Q", "S1_prod", "S1_Q_prod")}}
    return result


def _read_plain(path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def main(argv=None) -> None:
    from .inputs import load_inputs

    parser = argparse.ArgumentParser(description="Checks before the forward test of Q (docs/prereg/forward_q_prereg.md)")
    parser.add_argument("stage", choices=["same-path", "scramble", "replay", "vectors", "gbm", "revision", "summary"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--frame", default=None, help="frame.csv of the French residual load run")
    parser.add_argument("--db", default="data/forecast.db")
    parser.add_argument("--fresh", default="data/shadow/fresh")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", type=lear_recipe._shard)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = nr.scored_frame(args.frame) if args.frame else None
    if args.stage == "same-path":
        result = same_path(load_inputs(Path(args.data)), frame)
        (out / "same_path.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes")}))
    elif args.stage == "scramble":
        result = scramble(load_inputs(Path(args.data)))
        (out / "scramble.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes")}))
    elif args.stage == "replay":
        print(replay(args.db, Path(args.data), Path(args.fresh), out, args.shard))
    elif args.stage == "vectors":
        vectors = revision_vectors(args.fresh)
        vectors.to_csv(out / "revision_vectors.csv", index_label="day")
        print(f"{len(vectors)} revision vectors; mean absolute revision {100 * float((vectors - 1).abs().mean().mean()):.2f}% of the load forecast")
    elif args.stage in ("gbm", "revision", "summary"):
        vectors = pd.read_csv(out / "revision_vectors.csv", index_col=0, parse_dates=[0])
        vectors.columns = [int(c) for c in vectors.columns]
        plan = revision_plan(frame, vectors)
        plan.to_csv(out / "revision_plan.csv", index=False)
        if args.stage == "gbm":
            gbm_production(load_inputs(Path(args.data)), out, plan, vectors)
        elif args.stage == "revision":
            revision_lear(load_inputs(Path(args.data)), plan, vectors, out, args.shard)
        else:
            result = summary(out, frame)
            (out / "checks.json").write_text(json.dumps(result, indent=1, default=str))
            print(json.dumps({k: (v if k != "replay" else v) for k, v in result.items() if k in ("replay", "production_settings", "load_revision")}, indent=1, default=str))


if __name__ == "__main__":
    main()
