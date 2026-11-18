"""How much the German and Spanish price series matter (docs/prereg/price_seq_prereg.md), stage one: measurement.

ENTSO-E returns more than one day-ahead price series for DE-LU (two since 2021) and for Spain (four since September 2025).
The stored inputs hold the wrong German series and, since October 2025, a blend. The corrected inputs replace the two columns
by the coupled auction: sequence 1 for DE-LU, the unnumbered series for Spain.

    python -m thermo_fr.forecast.price_seq inputs  --out DIR --sequences FILE      # the corrected inputs and the stop checks
    python -m thermo_fr.forecast.price_seq gbm     --out DIR                        # honest_v2 on corrected inputs, both settings
    python -m thermo_fr.forecast.price_seq lear    --out DIR [--shard K/N]          # L0 and Q on every fourth development day
    python -m thermo_fr.forecast.price_seq summary --out DIR --frame FRENCH_FRAME --gbm-prod DIR

FILE is a table of hourly means per zone, resolution and sequence, built from the cached raw files on the laptop (columns
<zone>__<resolution>__<sequence>, hourly UTC). It is not in the repository.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import forward_q as fq
from . import french_rl, lear_recipe, yardstick
from . import neighbour_rl as nr
from .features import build_features
from .models import GBM_PARAMS, make_model
from .refit import REFIT_PARAMS

STEP = 4  # every fourth development day, starting with the first
COLUMNS = {"price_de_lu_eur_mwh": ("de_lu__PT60M__1", "de_lu__PT15M__1"), "price_es_eur_mwh": ("es__PT60M__None", "es__PT15M__None")}
SETTINGS = {"research": dict(GBM_PARAMS), "production": {**GBM_PARAMS, **REFIT_PARAMS}}
SMALL_LIMIT = 0.10
TOLERANCE = 1e-9


def read_sequences(path) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    table.index = pd.to_datetime(table.index, utc=True)
    return table


def coupled_series(sequences: pd.DataFrame) -> pd.DataFrame:
    """The coupled day-ahead price of DE-LU and Spain, hourly: the hourly series while it lasts, then the quarter-hourly one."""
    return pd.DataFrame({column: sequences[hourly].combine_first(sequences[quarter]) for column, (hourly, quarter) in COLUMNS.items()})


def corrected_inputs(base: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    """The inputs with the two price columns replaced on every hour the raw files cover. Nothing else changes."""
    out = base.copy()
    coupled = coupled_series(sequences).reindex(out.index)
    for column in COLUMNS:
        have = coupled[column].notna()
        out.loc[have, column] = coupled.loc[have, column]
    return out


def sampled_days(frame: pd.DataFrame) -> pd.DatetimeIndex:
    rule = frame[(frame["delivery_day"] >= french_rl.RULE_START) & frame["strict"].astype(bool)]
    return pd.DatetimeIndex(sorted(rule["delivery_day"].unique()))[::STEP]


def stop_checks(base: pd.DataFrame, corrected: pd.DataFrame, frame: pd.DataFrame) -> dict:
    a, b = base.to_numpy(dtype=float), corrected.to_numpy(dtype=float)
    differs = pd.Series(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum(axis=0), index=base.columns)
    changed = {c: int(v) for c, v in differs.items() if v}
    day = sampled_days(frame)[0]
    live = fq.q_live(lear_recipe.selection_inputs(base), day, candidates=("L0", "Q"))
    rows = frame[frame["delivery_day"] == day]
    worst = 0.0
    for name, column in (("L0", "lear_w{w}"), ("Q", "Q_w{w}")):
        for w in nr.WINDOWS:
            stored = rows.groupby("hour")[column.format(w=w)].first()
            worst = max(worst, float((live[name]["windows"][w].reindex(stored.index) - stored).abs().max()))
    reproduction = nr.reproduction(frame)
    return {"columns_changed": changed, "only_the_two_columns": bool(set(changed) <= set(COLUMNS)), "first_sampled_day": f"{day:%Y-%m-%d}",
            "stored_inputs_same_path_max_abs_difference": worst, "reference_figures": {k: reproduction[k]["printed"] for k in nr.REPRODUCE},
            "passes": bool(set(changed) <= set(COLUMNS) and worst <= TOLERANCE and reproduction["ok"])}


def gbm_corrected(corrected: pd.DataFrame, out: Path, log=print) -> None:
    """honest_v2 month by month on the corrected inputs, at research and at production settings (one file per month and setting)."""
    table = build_features(lear_recipe.selection_inputs(corrected), "honest_v2")
    days = table.info["delivery_day"]
    folder = Path(out) / "gbm"
    folder.mkdir(parents=True, exist_ok=True)
    for start in pd.date_range(pd.Timestamp(lear_recipe.WALK_START), pd.Timestamp(lear_recipe.TEST_END), freq="MS", inclusive="left"):
        path = folder / f"honest_v2_corrected_{start:%Y-%m}.csv"
        if path.exists():
            continue
        train, test = np.asarray(days < start), np.asarray((days >= start) & (days < start + pd.DateOffset(months=1)))
        keep = table.y[train].notna()
        part = table.info.loc[test, ["delivery_day", "hour"]].copy()
        for name, params in SETTINGS.items():
            model = make_model("gbm", params)
            model.fit(table.X[train][keep], table.y[train][keep])
            part[f"honest_v2_{name}_corrected"] = np.asarray(model.predict(table.X[test]), dtype=float)
        tmp = path.with_suffix(".tmp")
        part.to_csv(tmp, index_label="timestamp_utc")
        tmp.replace(path)
        log(f"  honest_v2 on corrected inputs: {start:%Y-%m} trained on {int(keep.sum()):,} hours", flush=True)


def lear_corrected(corrected: pd.DataFrame, frame: pd.DataFrame, out: Path, shard=(0, 1), log=print) -> None:
    base = lear_recipe.selection_inputs(corrected)
    folder = Path(out) / "lear"
    folder.mkdir(parents=True, exist_ok=True)
    for k, day in enumerate(sampled_days(frame)):
        path = folder / f"{day:%Y-%m-%d}.csv"
        if k % shard[1] != shard[0] or path.exists():
            continue
        t0 = time.time()
        live = fq.q_live(base, day, candidates=("L0", "Q"))
        part = pd.DataFrame({"L0_corrected": live["L0"]["hourly"], "Q_corrected": live["Q"]["hourly"]})
        tmp = path.with_suffix(".tmp")
        part.to_csv(tmp, index_label="timestamp_utc")
        tmp.replace(path)
        log(f"{day:%Y-%m-%d}: L0 and Q on corrected inputs ({time.time() - t0:.0f} s)", flush=True)


def _read(path) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    table.index = pd.to_datetime(table.index, utc=True)
    return table


def _paired(frame: pd.DataFrame, a: str, b: str) -> dict:
    p = yardstick.paired_daily(frame, a, b, one_sided_level=0.95)
    return {"mae": yardstick.mae(frame, a), "against": yardstick.mae(frame, b), "mean": p["mean"], "ci95": p["ci95"],
            "upper_one_sided_95": p["upper_one_sided"], "share_days_better": p["share_days_better"], "days": p["days"]}


def summary(out: Path, frame: pd.DataFrame, gbm_prod: Path) -> dict:
    result = {"checks": json.loads((Path(out) / "stop_checks.json").read_text())}
    gbm = pd.concat([_read(p) for p in sorted((Path(out) / "gbm").glob("*.csv"))]).sort_index().drop(columns=["delivery_day", "hour"])
    prod = pd.concat([_read(p) for p in sorted(Path(gbm_prod).glob("*.csv"))]).sort_index()[["honest_v2_prod"]]
    frame = frame.join(gbm).join(prod)
    rule = frame[frame["delivery_day"] >= french_rl.RULE_START]
    result["honest_v2_all_763_days"] = {
        "research": _paired(rule, "honest_v2_research_corrected", "honest_v2"),
        "production": _paired(rule, "honest_v2_production_corrected", "honest_v2_prod")}
    lear_files = sorted((Path(out) / "lear").glob("*.csv"))
    rows = frame.join(pd.concat([_read(p) for p in lear_files]).sort_index(), how="inner")
    rows = rows[rows["strict"].astype(bool)].copy()
    rows["S1_stored"] = (rows["honest_v2_prod"] + rows["L0"]) / 2
    rows["S1_Q_stored"] = (rows["honest_v2_prod"] + rows["Q"]) / 2
    rows["S1_corrected"] = (rows["honest_v2_production_corrected"] + rows["L0_corrected"]) / 2
    rows["S1_Q_corrected"] = (rows["honest_v2_production_corrected"] + rows["Q_corrected"]) / 2
    m = lambda c: yardstick.mae(rows, c)  # noqa: E731
    gain_stored, gain_corrected = m("S1_stored") - m("S1_Q_stored"), m("S1_corrected") - m("S1_Q_corrected")
    main = _paired(rows, "S1_Q_corrected", "S1_corrected")
    result["sampled_days"] = {
        "days": int(rows["delivery_day"].nunique()), "hours": int(len(rows)), "expected_days": int(len(sampled_days(frame))),
        "stored": {"L0": m("L0"), "Q": m("Q"), "honest_v2_production": m("honest_v2_prod"), "S1": m("S1_stored"), "S1_Q": m("S1_Q_stored"),
                   "gain_Q_over_L0": m("L0") - m("Q"), "gain_S1_Q_over_S1": gain_stored},
        "corrected": {"L0": m("L0_corrected"), "Q": m("Q_corrected"), "honest_v2_production": m("honest_v2_production_corrected"), "S1": m("S1_corrected"),
                      "S1_Q": m("S1_Q_corrected"), "gain_Q_over_L0": m("L0_corrected") - m("Q_corrected"), "gain_S1_Q_over_S1": gain_corrected},
        "corrected_minus_stored": {name: _paired(rows, f"{name}_corrected", stored) for name, stored in
                                   (("L0", "L0"), ("Q", "Q"), ("honest_v2_production", "honest_v2_prod"), ("S1", "S1_stored"), ("S1_Q", "S1_Q_stored"))},
        "S1_Q_against_S1_on_corrected": main, "S1_Q_against_S1_on_stored": _paired(rows, "S1_Q_stored", "S1_stored"),
        "Q_against_L0_on_corrected": _paired(rows, "Q_corrected", "L0_corrected"),
        "mean_abs_change_of_forecast": {"L0": float((rows["L0_corrected"] - rows["L0"]).abs().mean()), "Q": float((rows["Q_corrected"] - rows["Q"]).abs().mean()),
                                        "honest_v2_production": float((rows["honest_v2_production_corrected"] - rows["honest_v2_prod"]).abs().mean())}}
    conditions = {"1_S1_Q_beats_S1_on_corrected_inputs_one_sided_95": bool(gain_corrected > 0 and main["upper_one_sided_95"] < 0),
                  "2_gain_moves_by_less_than_0.10": bool(abs(gain_corrected - gain_stored) < SMALL_LIMIT)}
    result["rule"] = {"conditions": conditions, "gain_stored": gain_stored, "gain_corrected": gain_corrected, "change_of_gain": gain_corrected - gain_stored,
                      "verdict": "small: the forward test starts on 2026-12-01 as registered" if all(conditions.values()) else "large: the owner decides before 2026-12-01"}
    result["worth_its_own_registration"] = {
        "S1_improves_by_0.25_or_more_on_sampled_days": bool(m("S1_stored") - m("S1_corrected") >= 0.25),
        "honest_v2_production_improves_by_0.25_or_more_on_763_days": bool(-result["honest_v2_all_763_days"]["production"]["mean"] >= 0.25)}
    return result


def main(argv=None) -> None:
    from .inputs import load_inputs

    parser = argparse.ArgumentParser(description="German and Spanish price series, stage one (docs/prereg/price_seq_prereg.md)")
    parser.add_argument("stage", choices=["inputs", "gbm", "lear", "summary"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--frame", default=None, help="frame.csv of the French residual load run")
    parser.add_argument("--gbm-prod", default=None, help="folder of honest_v2 at production settings on the stored inputs (forward-q checks)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", type=lear_recipe._shard)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = nr.scored_frame(args.frame) if args.frame else None
    if args.stage == "inputs":
        base = load_inputs(Path(args.data))
        corrected = corrected_inputs(base, read_sequences(args.sequences))
        corrected.to_csv(out / "inputs_corrected.csv", index_label="timestamp_utc")
        checks = stop_checks(base, corrected, frame)
        (out / "stop_checks.json").write_text(json.dumps(checks, indent=1))
        print(json.dumps(checks, indent=1))
    else:
        corrected = load_inputs(out / "inputs_corrected.csv")
        if args.stage == "gbm":
            gbm_corrected(corrected, out)
        elif args.stage == "lear":
            lear_corrected(corrected, frame, out, args.shard)
        else:
            result = summary(out, frame, Path(args.gbm_prod))
            (out / "results.json").write_text(json.dumps(result, indent=1, default=str))
            print(json.dumps({k: result[k] for k in ("rule", "worth_its_own_registration")}, indent=1))


if __name__ == "__main__":
    main()
