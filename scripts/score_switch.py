"""The one read of the distance switch (docs/prereg/distance_switch_prereg.md).

Locked until 2027-04-18, like the statistics of the forward test of Q: before that date it refuses, unless an early look is
declared, which is written to looks.jsonl before anything is computed. It reads only what the forward records hold (g, L0,
S1, W, the distance and the branch, all written before any price existed) and the auction prices of the store. P and Q are
never read here.

    python scripts/score_switch.py [--out data/forward] [--db data/forecast.db] [--early-look "reason"]

It lives outside thermo_fr/forecast on purpose: the forward records carry a hash of that folder, and adding the scorer must
not change it. Written and frozen during the burn-in, before the first counted day (2026-12-01).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermo_fr.forecast import forward_q as fq
from thermo_fr.forecast import yardstick

ONE_SIDED = 1 - 0.05 / 2  # two comparisons: W against S1 (primary) and W against g
RULE_TOLERANCE = 1e-9
BRANCHES = ("ensemble", "tree", "fallback")


def rule_holds(record: dict) -> bool:
    """W recomputed from the logged g, L0 and distance equals the logged W, and the logged branch is the one the rule takes."""
    forecasts = record["forecasts"]
    index = pd.RangeIndex(len(forecasts["g"]))
    g = pd.Series(forecasts["g"], index=index, dtype=float)
    l0 = pd.Series(forecasts["L0"], index=index, dtype=float) if "L0" in forecasts else None
    distance = (record.get("distance") or {}).get("distance")
    expected, branch = fq.switch_forecast(g, l0, distance)
    same = bool(np.allclose(expected.to_numpy(), np.asarray(forecasts["W"], dtype=float), rtol=0, atol=RULE_TOLERANCE))
    return same and branch == record.get("switch_branch")


def day_table(chosen: dict) -> pd.DataFrame:
    """One row per delivery day with a record: what the record says about the switch. No price is involved."""
    rows = []
    for day, r in sorted(chosen.items()):
        has_w = "W" in r.get("forecasts", {}) and r.get("switch_branch") in BRANCHES
        rows.append({"delivery_day": day, "before_window": bool(r["before_window"]), "has_switch": has_w,
                     "branch": r.get("switch_branch") if has_w else None, "distance": (r.get("distance") or {}).get("distance"),
                     "rule_holds": rule_holds(r) if has_w else None})
    return pd.DataFrame(rows, columns=["delivery_day", "before_window", "has_switch", "branch", "distance", "rule_holds"]).set_index("delivery_day")


def spells(days: list[str], expected: list[str]) -> list[list[str]]:
    """Maximal runs of consecutive expected delivery days (Saturday to Tuesday is consecutive: Sunday and Monday are not forecast)."""
    position = {d: k for k, d in enumerate(expected)}
    out: list[list[str]] = []
    for d in sorted(days, key=position.__getitem__):
        if out and position[d] - position[out[-1][-1]] == 1:
            out[-1].append(d)
        else:
            out.append([d])
    return out


def paired(frame: pd.DataFrame, candidate: str, base: str) -> dict:
    diff = (yardstick.daily_abs_error(frame, candidate) - yardstick.daily_abs_error(frame, base)).dropna()
    if diff.empty:
        return {"days": 0}
    values = diff.to_numpy()
    lo, hi = yardstick.block_bootstrap_mean(values, block=fq.BLOCK, resamples=fq.RESAMPLES, seed=fq.SEED, level=0.95)
    lower, upper = yardstick.block_bootstrap_mean(values, block=fq.BLOCK, resamples=fq.RESAMPLES, seed=fq.SEED, level=1 - 2 * (1 - ONE_SIDED))
    return {"days": int(len(values)), "candidate_mae": float(yardstick.mae(frame, candidate)), "base_mae": float(yardstick.mae(frame, base)),
            "mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "one_sided_upper": float(upper), "mirrored_lower": float(lower),
            "share_days_better": float((values < 0).mean())}


def interpret(n_spells: int, w_s1: dict, w_s1_without_longest: dict) -> str:
    """The interpretation written in the registration before any day counted."""
    if n_spells == 0:
        return "untested: the switch never activated"
    if n_spells == 1:
        return "one episode: what the switch saved or cost in that spell, nothing about repetition"
    if w_s1["one_sided_upper"] < 0 and w_s1_without_longest.get("days", 0) > 0 and w_s1_without_longest["mean"] < 0:
        return "supported"
    if w_s1["mirrored_lower"] > 0:
        return "contradicted"
    return "undecided"


def evaluate(chosen: dict, actual: pd.Series, expected: list[str]) -> dict:
    """Everything the registration asks for, from the chosen records of the expected days and the auction prices."""
    chosen = {d: r for d, r in chosen.items() if d in set(expected)}
    table = day_table(chosen)
    counts = {"expected": len(expected), "with_a_record": int(len(table)), "lost": len(expected) - int(len(table)),
              "late": int((~table["before_window"]).sum()) if len(table) else 0,
              "without_switch_fields": int((table["before_window"] & ~table["has_switch"]).sum()) if len(table) else 0}
    usable = {d: r for d, r in chosen.items() if r["before_window"] and table.loc[d, "has_switch"]}
    counts["rule_mismatches"] = int(sum(not table.loc[d, "rule_holds"] for d in usable))
    frame = fq.frame_from_records(usable, actual)
    out = {"counts": counts}
    if frame.empty:
        return {**out, "interpretation": "untested: no scored day"}
    frame = frame.dropna(subset=["W", "S1", "g"])
    scored = sorted(f"{d:%Y-%m-%d}" for d in frame["delivery_day"].unique())
    branch = table.loc[scored, "branch"]
    counts.update({"scored": len(scored), **{f"branch_{b}": int((branch == b).sum()) for b in BRANCHES}})
    activated = [d for d in scored if branch[d] == "tree"]
    runs = spells(activated, expected)
    longest = max(runs, key=len) if runs else []
    key = frame["delivery_day"].dt.strftime("%Y-%m-%d")
    out["W_against_S1"] = paired(frame, "W", "S1")
    out["W_against_g"] = paired(frame, "W", "g")
    out["activated"] = {"days": len(activated), "spells": len(runs),
                        "W_against_S1_on_activated_days": paired(frame[key.isin(activated)], "W", "S1"),
                        "by_spell": [{"first": r[0], "last": r[-1], "days": len(r), "mean_distance": float(table.loc[r, "distance"].mean()),
                                      "W_minus_S1": paired(frame[key.isin(r)], "W", "S1").get("mean")} for r in runs]}
    without = paired(frame[~key.isin(longest)], "W", "S1") if longest else out["W_against_S1"]
    out["W_against_S1_without_longest_spell"] = without
    out["interpretation"] = interpret(len(runs), out["W_against_S1"], without)
    return out


def read(out_dir=fq.OUT_DIR, db=Path("data/forecast.db"), today=None, early_look: str | None = None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.now(tz=fq.LOCAL_TZ).tz_localize(None)).normalize()
    if today < fq.STATS_FROM:
        if not early_look:
            raise PermissionError(f"the switch is read once, from {fq.STATS_FROM:%Y-%m-%d}; an earlier look must be declared with --early-look")
        looks = Path(out_dir) / "looks.jsonl"
        looks.parent.mkdir(parents=True, exist_ok=True)
        with open(looks, "a") as f:
            f.write(json.dumps({"date": f"{today:%Y-%m-%d}", "test": "distance switch", "reason": early_look}) + "\n")
    conn = fq.read_only(db)
    try:
        rows = pd.read_sql_query("SELECT timestamp_utc, price FROM actuals", conn)
    finally:
        conn.close()
    actual = pd.Series(rows["price"].to_numpy(), index=pd.to_datetime(rows["timestamp_utc"], utc=True))
    expected = [f"{d:%Y-%m-%d}" for d in fq.expected_days()]
    return evaluate(fq.chosen_records(fq.read_records(out_dir)), actual, expected)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="The one read of the distance switch (docs/prereg/distance_switch_prereg.md)")
    parser.add_argument("--out", default=str(fq.OUT_DIR))
    parser.add_argument("--db", default="data/forecast.db")
    parser.add_argument("--early-look", default=None, help="reason for reading before the end date; it is recorded")
    args = parser.parse_args(argv)
    print(json.dumps(read(args.out, Path(args.db), early_look=args.early_look), indent=1))


if __name__ == "__main__":
    main()
