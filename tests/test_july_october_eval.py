"""The pure pieces of the July to October development evaluation (docs/prereg/july_october_eval_prereg.md)."""

import numpy as np
import pandas as pd

from thermo_fr.forecast import july_october_eval as ev


def test_constants_match_the_frozen_plan():
    assert (ev.FIRST, ev.LAST) == ("2026-07-01", "2026-10-10")
    assert len(ev.all_days()) == ev.EXPECTED_DAYS == 102 and ev.EXPECTED_HOURS == 2448
    assert ev.BANDS == (1.0, 1.6) and ev.HIGH_PRICE == 200.0 and ev.OVERSHOOT == 5.0 and ev.MIN_HIGH_DAYS == 5
    assert ev.COMPARISONS == (("T800", "T300"), ("S1", "T300"), ("S1-Q", "S1"))
    assert abs(ev.ONE_SIDED - (1 - 0.05 / 3)) < 1e-12 and (ev.BLOCK, ev.RESAMPLES, ev.SEED) == (7, 5000, 0)
    assert ev.SETTINGS["T300"]["n_estimators"] == 300 and ev.SETTINGS["T300"]["num_leaves"] == 31
    assert ev.SETTINGS["T800"]["n_estimators"] == 800 and ev.SETTINGS["T800"]["num_leaves"] == 63


def test_spells_are_runs_of_consecutive_days():
    runs = ev.spells(["2026-08-03", "2026-08-01", "2026-08-02", "2026-08-10", "2026-08-12", "2026-08-13"])
    assert runs == [["2026-08-01", "2026-08-02", "2026-08-03"], ["2026-08-10"], ["2026-08-12", "2026-08-13"]]
    assert ev.spells([]) == []


def frame(days, actual, cand, base):
    rows = []
    for d, a, c, b in zip(days, actual, cand, base):
        for h in range(24):
            rows.append({"delivery_day": d, "hour": h, "actual": a, "cand": c, "base": b})
    return pd.DataFrame(rows)


def test_paired_verdicts():
    days = [f"{d:%Y-%m-%d}" for d in pd.date_range("2026-07-01", periods=60)]
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 60)
    better = ev.paired(frame(days, np.zeros(60), 5 + noise, 10 + noise), "cand", "base")
    assert better["verdict"] == "supports" and better["mean"] < 0 and better["share_days_better"] == 1.0
    worse = ev.paired(frame(days, np.zeros(60), 10 + noise, 5 + noise), "cand", "base")
    assert worse["verdict"] == "contradicts"
    same = ev.paired(frame(days, np.zeros(60), 10 + noise, 10 - noise), "cand", "base")
    assert same["verdict"] == "undecided" and same["mirrored_lower"] <= same["lo"] <= same["hi"] <= same["one_sided_upper"]


def test_recurrence_rule():
    def report(days, n_spells, realised, predicted, lear_minus_tree):
        return {"days": days, "spells": n_spells, "realised_change": realised, "L0_predicted_change": predicted, "L0_minus_T300": {"mean": lear_minus_tree}}

    assert ev.recurrence(report(4, 2, -10, -30, 5), report(2, 1, -10, -30, 5)) == "too few days"
    assert ev.recurrence(report(12, 1, -10, -30, 5), report(0, 0, 0, 0, 0)) == "too few days"
    assert ev.recurrence(report(12, 3, -10, -30, 5), report(6, 2, -10, -20, 2)) == "recurs"
    assert ev.recurrence(report(12, 3, -10, -30, 5), report(6, 2, -10, -12, 2)) == "does not recur"  # overshoot under 5 without the longest spell
    assert ev.recurrence(report(12, 3, -10, -30, -1), report(6, 2, -10, -30, 2)) == "does not recur"  # the LEAR does not lose to the tree
