"""Comparators of W added by amendment 1 of docs/prereg/spread_w_prereg.md. spread_w.py is not touched.

  HW  the matched French-only comparator: H's inputs with W's training, a refit every seventh day on W's grid and W's
      weights of half-life 180 days. It is fitted on the same training hours as W and scored on the same hours. The only
      difference from W is that the German block is left out. W against HW is the primary test of whether the German
      inputs add forecast value.
  GK  G's inputs, weekly refits, no weights. Reported only: one of W's two changes on its own.
  GM  G's inputs, monthly refits, weights of half-life 180 days. Reported only: the other change on its own.

    python -m thermo_fr.forecast.spread_hw design   --out <folder> --raw <xml> --kept <folder> --w <w_predictions.csv.gz> --sequences ... --wind ... --solar ...
    python -m thermo_fr.forecast.spread_hw scramble --out <folder> --raw <xml> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import spread_eval as se
from . import spread_w as sw

COMPARATORS = {  # name: (input set, weekly refits on W's grid, half-life in days or None)
    "HW": ("H", True, sw.HALF_LIFE_DAYS),
    "GK": ("G", True, None),
    "GM": ("G", False, sw.HALF_LIFE_DAYS),
}


def blocks(first: str, last: str, weekly: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(refit day, first day after the block) for every block that covers first to last."""
    if weekly:
        return [(d, d + pd.Timedelta(days=sw.REFIT_EVERY_DAYS)) for d in sw.refit_days(first, last)]
    return [(d, d + pd.DateOffset(months=1)) for d in se.months_of(first, last)]


def fit_comparator(feat: dict, name: str, first: str, last: str, log=lambda *_: None) -> pd.Series:
    """Walk-forward forecasts of one comparator. Training rows are W's: every hour of an earlier day with a realised spread."""
    inputs, weekly, half_life = COMPARATORS[name]
    frame, y, X = feat["frame"], feat["y"], feat["sets"][inputs]
    pred = pd.Series(np.nan, index=frame.index, name=name)
    for start, end in blocks(first, last, weekly):
        if start < feat["first_month"]:
            raise ValueError(f"{start:%Y-%m-%d} is before the German proxies exist")
        train = np.asarray((frame["delivery_day"] < start) & y.notna())
        block = np.asarray((frame["delivery_day"] >= start) & (frame["delivery_day"] < end) & (frame["delivery_day"] >= pd.Timestamp(first)) & (frame["delivery_day"] <= pd.Timestamp(last)))
        if not block.any():
            continue
        weight = None if half_life is None else np.power(0.5, (start - frame["delivery_day"][train]).dt.days.to_numpy() / half_life)
        model = lgb.LGBMRegressor(**se.PARAMS).fit(X[train], y[train], sample_weight=weight)
        pred[block] = model.predict(X[block])
        log(f"{name} fitted {start:%Y-%m-%d}")
    return pred


def scramble_check(data: se.Data, days=sw.SCRAMBLE_DAYS, names=("HW",), log=print) -> dict:
    """Each comparator's forecasts of a day from the data as they are and with everything unknown at the gate scrambled."""
    out, feat = {}, se.features(data)
    for day in days:
        rows = np.asarray(feat["frame"]["delivery_day"] == pd.Timestamp(day))
        feat_s = se.features(se.scrambled(data, day))
        rows_s = np.asarray(feat_s["frame"]["delivery_day"] == pd.Timestamp(day))
        out[day] = {"hours": int(rows.sum())}
        for name in names:
            a, b = fit_comparator(feat, name, day, day)[rows], fit_comparator(feat_s, name, day, day)[rows_s]
            inputs = COMPARATORS[name][0]
            same = bool(feat["sets"][inputs].loc[rows].equals(feat_s["sets"][inputs].loc[rows_s]))
            diff = float((a - b.reindex(a.index)).abs().max())
            out[day][name] = {"largest_difference": diff, "inputs_identical": same, "pass": bool(rows.sum() >= 23 and a.notna().all() and diff <= se.TOLERANCE and same)}
            log(f"scramble {day}: {name} differs by at most {diff:.3g} over {int(rows.sum())} hours, inputs identical: {same}")
    out["pass"] = all(v[name]["pass"] for v in out.values() for name in names)
    return out


def compare(frame: pd.DataFrame, candidate: str, base: str) -> dict:
    """The gain rule's pieces for candidate against base (se.decide compares its column G with its column H)."""
    swapped = frame.drop(columns=["H", "G"]).assign(H=frame[base], G=frame[candidate])
    d = se.decide(swapped)
    return {"gain": d["gain"], "paired": d["paired"], "slices": {k: v["G_minus_H"] for k, v in d["slices"].items()}, "slice_days": {k: v["days"] for k, v in d["slices"].items()},
            "conditions": d["conditions"], "verdict_if_this_were_a_test": d["verdict"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Comparators of W (amendment 1): design figures and the scramble check")
    parser.add_argument("stage", choices=["design", "scramble"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs", default="data/forecast/inputs.csv")
    parser.add_argument("--neighbours", default="data/forecast/neighbour_fundamentals.csv")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--solar", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--w", help="w_predictions.csv.gz of spread_w design (H, G, C, W on the backtest and the opened window)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = se.load_stored(args.inputs, args.neighbours, args.sequences, args.wind, args.solar)
    data = se.extend(stored, se.parse_raw(Path(args.raw)), last=sw.DESIGN_LAST)
    if args.stage == "scramble":
        (out / "hw_scramble.json").write_text(json.dumps(scramble_check(data), indent=1))
        return
    feat = se.features(data)
    kept = pd.read_csv(args.w, index_col=0, parse_dates=["delivery_day"])
    kept.index = pd.to_datetime(kept.index, utc=True)
    for name in COMPARATORS:
        kept[name] = fit_comparator(feat, name, se.DEV_FIRST, sw.DESIGN_LAST, log=print).reindex(kept.index)
        assert kept[name].notna().all(), name
    kept.to_csv(out / "comparator_predictions.csv.gz")  # working file, never published
    result = {"predictions_sha256": se.sha256(out / "comparator_predictions.csv.gz"), "comparators": {k: {"inputs": v[0], "weekly": v[1], "half_life_days": v[2]} for k, v in COMPARATORS.items()}}
    for label, part in (("backtest 2024-05-01 to 2026-06-30", kept[kept["delivery_day"] <= pd.Timestamp(se.DEV_LAST)]), ("opened window 2026-07-01 to 2026-10-10", kept[kept["delivery_day"] > pd.Timestamp(se.DEV_LAST)])):
        result[label] = {"hours": int(len(part)), "days": int(part["delivery_day"].nunique()),
                         "mae": {m: float((part[m] - part["spread"]).abs().mean()) for m in ("yesterday", "H", "HW", "G", "GK", "GM", "W", "C")},
                         "bias": {m: float((part[m] - part["spread"]).mean()) for m in ("H", "HW", "G", "W")},
                         "W against HW": compare(part, "W", "HW"), "W against H": compare(part, "W", "H"), "W against G": compare(part, "W", "G"), "HW against H": compare(part, "HW", "H")}
    (out / "comparator_design.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v["mae"] for k, v in result.items() if isinstance(v, dict) and "mae" in v}, indent=1))


if __name__ == "__main__":
    main()
