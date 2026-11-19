"""W: the France minus Germany spread forecast redesigned after the out-of-sample test of 2026-11-19.

W reads exactly the inputs of G (thermo_fr.forecast.spread_eval: the spread's own past, the calendar, the French inputs of
the public model and the German block). Two things change, both aimed at a market whose level can break as it did in July
2026, when the French discount to Germany closed from about 43 to about 10 EUR/MWh within a month:
  refit      every seventh day, counted from 2024-05-01, instead of once a month
  weights    each training day counts 0.5 ** (age in days / 180), so the last half year carries about as much as all the
             history before it
Everything else is G: 300 trees of 31 leaves, the level of the spread as the target, every earlier day from 2021 in the fit.

The window 2026-07-01 to 2026-10-10 was opened for diagnosis with the owner's approval and W was designed with it in view.
W's figures on that window and on the backtest are design figures. W is judged only on delivery days after 2026-10-10
(docs/prereg/spread_w_prereg.md).

    python -m thermo_fr.forecast.spread_w design   --out <folder> --raw <xml> --sequences ... --wind ... --solar ...
    python -m thermo_fr.forecast.spread_w scramble --out <folder> --raw <xml> --sequences ... --wind ... --solar ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import spread_eval as se

HALF_LIFE_DAYS = 180
REFIT_EVERY_DAYS = 7
GRID_ANCHOR = "2024-05-01"  # refits fall on this day and every seventh day after it
DESIGN_LAST = "2026-10-10"  # the last delivery day W's design could see
SCRAMBLE_DAYS = ("2025-10-26", "2026-07-01", "2026-08-26")  # a 25-hour day, the first day after the break, a reconstructed-proxy day


def refit_days(first: str, last: str) -> pd.DatetimeIndex:
    """The refit days whose seven-day blocks cover first to last."""
    grid = pd.date_range(GRID_ANCHOR, pd.Timestamp(last), freq=f"{REFIT_EVERY_DAYS}D")
    return grid[grid + pd.Timedelta(days=REFIT_EVERY_DAYS) > pd.Timestamp(first)]


def weights(training_days: pd.Series, refit_day: pd.Timestamp) -> np.ndarray:
    """0.5 ** (age / half-life), age counted in whole days back from the refit day."""
    age = (pd.Timestamp(refit_day) - training_days).dt.days.to_numpy()
    if (age <= 0).any():
        raise ValueError("a training day is not before the refit day")
    return np.power(0.5, age / HALF_LIFE_DAYS)


def fit_w(feat: dict, first: str, last: str, log=lambda *_: None) -> pd.Series:
    """Walk-forward forecasts of W for delivery days first to last. The model of a block is fitted on days before the block."""
    frame, y, X = feat["frame"], feat["y"], feat["sets"]["G"]
    pred = pd.Series(np.nan, index=frame.index, name="W")
    for start in refit_days(first, last):
        if start < feat["first_month"]:
            raise ValueError(f"{start:%Y-%m-%d} is before the German proxies exist")
        train = np.asarray((frame["delivery_day"] < start) & y.notna())
        block = np.asarray((frame["delivery_day"] >= start) & (frame["delivery_day"] < start + pd.Timedelta(days=REFIT_EVERY_DAYS))
                           & (frame["delivery_day"] >= pd.Timestamp(first)) & (frame["delivery_day"] <= pd.Timestamp(last)))
        if not block.any():
            continue
        model = lgb.LGBMRegressor(**se.PARAMS).fit(X[train], y[train], sample_weight=weights(frame["delivery_day"][train], start))
        pred[block] = model.predict(X[block])
        log(f"fitted {start:%Y-%m-%d}")
    return pred


def scramble_check(data: se.Data, days=SCRAMBLE_DAYS, log=print) -> dict:
    """W's forecasts of each day from the data as they are and with everything unknown at the gate replaced by noise."""
    out, feat = {}, se.features(data)
    for day in days:
        rows = np.asarray(feat["frame"]["delivery_day"] == pd.Timestamp(day))
        a = fit_w(feat, day, day)[rows]
        feat_s = se.features(se.scrambled(data, day))
        rows_s = np.asarray(feat_s["frame"]["delivery_day"] == pd.Timestamp(day))
        b = fit_w(feat_s, day, day)[rows_s]
        same = bool(feat["sets"]["G"].loc[rows].equals(feat_s["sets"]["G"].loc[rows_s]))
        diff = float((a - b.reindex(a.index)).abs().max())
        out[day] = {"hours": int(rows.sum()), "largest_difference": diff, "inputs_identical": same, "pass": bool(rows.sum() >= 23 and a.notna().all() and diff <= se.TOLERANCE and same)}
        log(f"scramble {day}: W differs by at most {diff:.3g} over {int(rows.sum())} hours, inputs identical: {same}")
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def design_figures(frame: pd.DataFrame) -> dict:
    """Errors of H, G and W on a scored frame, with the gain rule applied to W against each. Design figures, not a test."""
    against = {}
    for base in ("H", "G"):
        swapped = frame.drop(columns=["H", "G"]).assign(H=frame[base], G=frame["W"])  # se.decide compares the column G with the column H
        d = se.decide(swapped)
        against[base] = {"gain": d["gain"], "paired": d["paired"], "slices": {k: v["G_minus_H"] for k, v in d["slices"].items()}, "conditions": d["conditions"], "verdict_if_this_were_a_test": d["verdict"]}
    return {"hours": int(len(frame)), "days": int(frame["delivery_day"].nunique()), "mae": {m: float((frame[m] - frame["spread"]).abs().mean()) for m in ("yesterday", "H", "G", "W", "C")},
            "bias": {m: float((frame[m] - frame["spread"]).mean()) for m in ("H", "G", "W")}, "W_against": against}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="W, the redesigned spread forecast: design figures and the scramble check")
    parser.add_argument("stage", choices=["design", "scramble"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs", default="data/forecast/inputs.csv")
    parser.add_argument("--neighbours", default="data/forecast/neighbour_fundamentals.csv")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--solar", required=True)
    parser.add_argument("--raw", required=True, help="the raw ENTSO-E files of the spread test (1 June to 10 October 2026)")
    parser.add_argument("--kept", required=True, help="folder holding dev_predictions.csv.gz and test_predictions.csv.gz of spread_eval")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = se.load_stored(args.inputs, args.neighbours, args.sequences, args.wind, args.solar)
    data = se.extend(stored, se.parse_raw(Path(args.raw)), last=DESIGN_LAST)
    if args.stage == "scramble":
        result = scramble_check(data)
        (out / "w_scramble.json").write_text(json.dumps(result, indent=1))
        return
    feat = se.features(data)
    w = fit_w(feat, se.DEV_FIRST, DESIGN_LAST, log=print)
    result = {"half_life_days": HALF_LIFE_DAYS, "refit_every_days": REFIT_EVERY_DAYS, "grid_anchor": GRID_ANCHOR}
    frames = []
    for name, file in (("backtest 2024-05-01 to 2026-06-30", "dev_predictions.csv.gz"), ("opened window 2026-07-01 to 2026-10-10", "test_predictions.csv.gz")):
        kept = pd.read_csv(Path(args.kept) / file, index_col=0, parse_dates=["delivery_day"])
        kept.index = pd.to_datetime(kept.index, utc=True)
        kept["W"] = w.reindex(kept.index)
        assert kept["W"].notna().all()
        result[name] = design_figures(kept)
        frames.append(kept)
    pd.concat(frames).to_csv(out / "w_predictions.csv.gz")  # working file, never published
    result["predictions_sha256"] = se.sha256(out / "w_predictions.csv.gz")
    (out / "w_design.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: (v["mae"] if isinstance(v, dict) else v) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
