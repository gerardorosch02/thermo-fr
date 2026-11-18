"""Build published/research.json, the aggregates behind the dashboard's Research page.

The page compares the public tree model (honest_v2 at the settings of the stored file) with a LEAR and with their equal
average (S1) on the selection window of the experiment log, and shows the live record of the same three from the shadow
run on the owner's machine. Everything written here is an aggregate: errors per month, per hour, per slice and per price
level, daily errors and daily mean prices of the live days. No hourly price and no input row is written.

The inputs are files that are not in the repository (stored backtest forecasts and an extract of the local store), so
this script is run by hand when the research figures change, and its output is committed:

    python scripts/build_research.py --frame work/french-rl/frame.csv --gbm-prod work/forward-q/gbm_prod \
        --shadow-days shadow_days.csv --shadow-hourly shadow_hourly.csv --risk-band docs/prereg/risk_band_results.json \
        --experiments docs/experiments.md --prereg-dir docs/prereg --out published/research.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from thermo_fr.forecast.shape import block_bootstrap_mean

RULE_START = "2024-05-29"  # first day on which the residual-load LEAR (Q) issues its own forecast
PRICE_EDGES = [-np.inf, 50, 100, 150, 200, 250, np.inf]
PRICE_LABELS = ["under 50", "50 to 100", "100 to 150", "150 to 200", "200 to 250", "250 and over"]
# Measured limits of the backtest, from the results files of the registered checks (docs/prereg on branch foundations):
# forward_q_checks_results.md (load forecast revisions replayed) and price_seq_results.md (coupled German and Spanish prices).
LIMITS = {"load_replay_days": 59, "load_replay_lear": 0.11, "load_replay_tree": 0.02, "price_max_change": 0.12}
# No comparison with the previous day's price is published (owner's decision of 2026-10-12, tests/test_eex_only_benchmark.py).
MODELS = {"tree": "Tree model (public)", "lear": "LEAR", "s1": "Ensemble"}


def mae(frame: pd.DataFrame, column: str) -> float:
    return float((frame[column] - frame["actual"]).abs().mean())


def daily_mae(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame[column] - frame["actual"]).abs().groupby(frame["delivery_day"]).mean()


def paired(frame: pd.DataFrame, candidate: str, baseline: str) -> dict:
    """Mean paired daily difference (candidate minus baseline) with the project's interval: moving blocks of 7 days, 5000 resamples."""
    diff = (daily_mae(frame, candidate) - daily_mae(frame, baseline)).to_numpy()
    low, high = block_bootstrap_mean(diff, block=7, resamples=5000, seed=0, level=0.95)
    return {"days": int(len(diff)), "mean": round(float(diff.mean()), 3), "lo": round(float(low), 3), "hi": round(float(high), 3),
            "share_days_better": round(float((diff < 0).mean()), 3)}


def model_rows(frame: pd.DataFrame, keys=MODELS) -> list[dict]:
    return [{"key": k, "label": label, "mae": round(mae(frame, k), 2)} for k, label in keys.items()]


def load_backtest(frame_path, gbm_prod_dir) -> pd.DataFrame:
    frame = pd.read_csv(frame_path)
    frame = frame[frame["strict"].astype(bool)].copy()
    prod = pd.concat(pd.read_csv(p) for p in sorted(glob.glob(os.path.join(gbm_prod_dir, "honest_v2_prod_*.csv"))))
    frame = frame.merge(prod[["timestamp_utc", "honest_v2_prod"]], on="timestamp_utc", how="left", validate="one_to_one")
    if frame["honest_v2_prod"].isna().any():
        raise ValueError("the production-settings forecasts do not cover every strict hour of the frame")
    frame["tree"], frame["lear"] = frame["honest_v2_prod"], frame["L0"]
    frame["s1"] = (frame["tree"] + frame["lear"]) / 2
    frame["s1_q"] = (frame["tree"] + frame["Q"]) / 2
    frame["s1_research"] = (frame["honest_v2"] + frame["L0"]) / 2
    return frame


def slices(frame: pd.DataFrame) -> list[dict]:
    """The protocol's slices. The windiest and sunniest days and the top hours are picked on outcomes, so they are stress views."""
    day = frame.groupby("delivery_day")[["wind_mw", "solar_mw"]].mean()
    windy = day.index[day["wind_mw"] >= day["wind_mw"].quantile(0.9)]
    sunny = day.index[day["solar_mw"] >= day["solar_mw"].quantile(0.9)]
    masks = {"Weekends": frame["dow"] >= 5, "Public holidays": frame["holiday"] == 1,
             "Windiest tenth of days": frame["delivery_day"].isin(windy), "Sunniest tenth of days": frame["delivery_day"].isin(sunny),
             "Top 5% of hours by price": frame["actual"] >= frame["actual"].quantile(0.95)}
    return [{"slice": name, "hours": int(mask.sum()), "tree": round(mae(frame[mask], "tree"), 2), "lear": round(mae(frame[mask], "lear"), 2),
             "s1": round(mae(frame[mask], "s1"), 2)} for name, mask in masks.items()]


def price_bins(daily_price: pd.Series) -> pd.Series:
    return pd.cut(daily_price, PRICE_EDGES, labels=PRICE_LABELS, right=False)


def backtest_block(frame: pd.DataFrame) -> dict:
    month = frame["delivery_day"].str[:7]
    monthly = []
    for name, group in frame.groupby(month):
        monthly.append({"month": name, "days": int(group["delivery_day"].nunique()), **{k: round(mae(group, k), 2) for k in MODELS}})
    by_hour = [{"hour": int(h), **{k: round(mae(g, k), 2) for k in MODELS}} for h, g in frame.groupby("hour")]
    rule = frame[frame["delivery_day"] >= RULE_START]
    months_better = sum(1 for m in monthly if m["s1"] < m["tree"])
    saved = (daily_mae(frame, "tree") - daily_mae(frame, "s1")).sort_index().cumsum()
    return {
        "first_day": frame["delivery_day"].min(), "last_day": frame["delivery_day"].max(), "days": int(frame["delivery_day"].nunique()),
        "hours": int(len(frame)), "mean_price": round(float(frame["actual"].mean()), 1),
        "models": model_rows(frame),
        "paired_s1_tree": paired(frame, "s1", "tree"), "months": len(monthly), "months_s1_better": months_better,
        "monthly": monthly, "by_hour": by_hour, "cumulative": [{"delivery_day": d, "saved": round(float(v), 1)} for d, v in saved.items()], "slices": slices(frame),
        "research_settings": {"tree": round(mae(frame, "honest_v2"), 2), "s1": round(mae(frame, "s1_research"), 2)},
        "residual_load": {"first_day": RULE_START, "days": int(rule["delivery_day"].nunique()), "tree": round(mae(rule, "tree"), 2),
                          "s1": round(mae(rule, "s1"), 2), "s1_q": round(mae(rule, "s1_q"), 2), "paired": paired(rule, "s1_q", "s1")},
    }


def live_block(days_path, hourly_path) -> tuple[dict, pd.DataFrame]:
    hourly = pd.read_csv(hourly_path).rename(columns={"honest_v2": "tree", "LEAR": "lear", "S1": "s1"})
    days = pd.read_csv(days_path)
    daily = pd.DataFrame({k: daily_mae(hourly, k) for k in MODELS}).round(2)
    span = pd.date_range(hourly["delivery_day"].min(), hourly["delivery_day"].max(), freq="D")
    expected = int(span.dayofweek.isin([1, 2, 3, 4, 5]).sum())  # Tuesday to Saturday, the days the weekday morning runs forecast
    high = hourly[hourly["actual"] >= 200]
    block = {
        "first_day": hourly["delivery_day"].min(), "last_day": hourly["delivery_day"].max(), "days": int(hourly["delivery_day"].nunique()),
        "expected_days": expected,
        "hours": int(len(hourly)), "mean_price": round(float(hourly["actual"].mean()), 1), "fallback_days": int(days["fallback"].sum()),
        "models": model_rows(hourly), "paired_s1_tree": paired(hourly, "s1", "tree"), "paired_lear_tree": paired(hourly, "lear", "tree"),
        "daily": [{"delivery_day": d, **row} for d, row in daily.to_dict("index").items()],
        "above_200": {"hours": int(len(high)), "share": round(len(high) / len(hourly), 3),
                      "bias_lear": round(float((high["lear"] - high["actual"]).mean()), 1),
                      "bias_tree": round(float((high["tree"] - high["actual"]).mean()), 1)},
    }
    return block, hourly


def regime_block(frame: pd.DataFrame, live_hourly: pd.DataFrame) -> dict:
    back = frame.groupby("delivery_day")["actual"].mean()
    live = live_hourly.groupby("delivery_day")["actual"].mean()
    b, l = price_bins(back).value_counts(), price_bins(live).value_counts()
    bins = [{"label": label, "backtest_days": int(b.get(label, 0)), "live_days": int(l.get(label, 0)),
             "backtest_share": round(float(b.get(label, 0) / len(back)), 4), "live_share": round(float(l.get(label, 0) / len(live)), 4)}
            for label in PRICE_LABELS]
    extreme = frame[frame["delivery_day"].isin(back.index[back >= 150])]
    return {"bins": bins, "backtest_days": int(len(back)), "live_days": int(len(live)),
            "backtest_days_150_plus": int((back >= 150).sum()), "live_days_200_plus": int((live >= 200).sum()),
            "backtest_150_plus": {"tree": round(mae(extreme, "tree"), 1), "lear": round(mae(extreme, "lear"), 1),
                                  "bias_lear": round(float((extreme["lear"] - extreme["actual"]).mean()), 1),
                                  "bias_tree": round(float((extreme["tree"] - extreme["actual"]).mean()), 1)} if len(extreme) else None}


def band_block(path) -> dict | None:
    if not path or not Path(path).exists():
        return None
    r = json.loads(Path(path).read_text())
    overall, by_slice, scored = r["S1"]["overall"], r["S1"]["by_slice"], r["scored_set"]
    def row(level):
        x = overall["B"][level]
        return {"level": int(level), "coverage": round(x["coverage"], 4), "width": round(x["mean_width"], 1), "score": round(x["interval_score"], 1)}
    out = {"first_day": scored["first_day"], "last_day": scored["last_day"], "days": scored["days"], "rows": [row("80"), row("95")],
           "challenger_score_80": round(overall["T"]["80"]["interval_score"], 1), "challenger_coverage_80": round(overall["T"]["80"]["coverage"], 4)}
    if "2026 H1" in by_slice:
        out["coverage_80_2026_h1"] = round(by_slice["2026 H1"]["B"]["80"]["coverage"], 4)
    if "holidays" in by_slice:
        out["holiday_coverage_80"] = round(by_slice["holidays"]["B"]["80"]["coverage"], 4)
    return out


def ledger_block(experiments, prereg_dir) -> dict | None:
    if not experiments or not Path(experiments).exists():
        return None
    rows = [line for line in Path(experiments).read_text().splitlines() if line.startswith("| 20")]
    folder = Path(prereg_dir)
    return {"log_rows": len(rows), "registrations": len(list(folder.glob("*_prereg.md"))), "amendments": len(list(folder.glob("*amendment*.md"))),
            "reviews": len(list(folder.glob("*_review.md")))}


def build(args) -> dict:
    frame = load_backtest(args.frame, args.gbm_prod)
    live, live_hourly = live_block(args.shadow_days, args.shadow_hourly)
    return {"built_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "Aggregates only. Backtest: strict rows of the selection window, the tree model at the settings of the public file. "
                    "Live: the shadow run's forecasts issued before 11:15 Paris, scored against the auction.",
            "backtest": backtest_block(frame), "live": live, "regime": regime_block(frame, live_hourly),
            "band": band_block(args.risk_band), "ledger": ledger_block(args.experiments, args.prereg_dir), "limits": LIMITS,
            "forward_test": {"first_day": args.forward_first, "last_day": args.forward_last}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frame", required=True)
    parser.add_argument("--gbm-prod", required=True)
    parser.add_argument("--shadow-days", required=True)
    parser.add_argument("--shadow-hourly", required=True)
    parser.add_argument("--risk-band")
    parser.add_argument("--experiments")
    parser.add_argument("--prereg-dir", default="docs/prereg")
    parser.add_argument("--forward-first", default="2026-12-01", help="first counted delivery day of the forward test of the residual-load step")
    parser.add_argument("--forward-last", default="2027-04-17")
    parser.add_argument("--out", default="published/research.json")
    args = parser.parse_args()
    result = build(args)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
