"""The scorer of the distance switch (scripts/score_switch.py), on synthetic records only."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import forward_q as fq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_switch", ROOT / "scripts" / "score_switch.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

EXPECTED = [f"{d:%Y-%m-%d}" for d in fq.expected_days()][:12]  # 1 to 16 December 2026, Tuesday to Saturday


def hours(day):
    return pd.date_range(pd.Timestamp(day, tz="Europe/Paris"), periods=24, freq="h").tz_convert("UTC")


def record(day, distance, lear=100.0, tree=110.0, before_window=True, with_lear=True, with_switch=True):
    index = hours(day)
    g = pd.Series(tree, index=index)
    l0 = pd.Series(lear, index=index) if with_lear else None
    w, branch = fq.switch_forecast(g, l0, distance if with_lear else None)
    forecasts = {"g": list(g), "S1": list((g + l0) / 2) if with_lear else list(g)}
    if with_lear:
        forecasts["L0"] = list(l0)
    r = {"delivery_day": day, "issued_at_utc": f"{pd.Timestamp(day) - pd.Timedelta(hours=14):%Y-%m-%dT%H:%M:%SZ}", "before_window": before_window,
         "flags": [] if with_lear else ["LEAR fallback", "switch fallback"], "timestamps_utc": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in index],
         "forecasts": forecasts}
    if with_switch:
        r["forecasts"]["W"] = list(w)
        r["switch_branch"], r["switch_threshold"] = branch, fq.SWITCH_DISTANCE
        if with_lear:
            r["distance"] = {"distance": distance}
    return r


def prices(days, value=100.0):
    return pd.concat([pd.Series(value, index=hours(d)) for d in days])


def test_boundary_just_below_and_exactly_at_the_threshold():
    below, at = record(EXPECTED[0], 1.5999, lear=60.0), record(EXPECTED[1], 1.6, lear=60.0)
    assert below["switch_branch"] == "ensemble" and below["forecasts"]["W"] == below["forecasts"]["S1"]
    assert at["switch_branch"] == "tree" and at["forecasts"]["W"] == at["forecasts"]["g"]
    assert ss.rule_holds(below) and ss.rule_holds(at)


def test_a_missing_component_is_a_fallback_and_not_an_activation():
    r = record(EXPECTED[0], 2.5, with_lear=False)
    assert r["switch_branch"] == "fallback" and r["forecasts"]["W"] == r["forecasts"]["g"] and ss.rule_holds(r)
    chosen = {EXPECTED[0]: r, EXPECTED[1]: record(EXPECTED[1], 0.3)}
    out = ss.evaluate(chosen, prices(EXPECTED[:2]), EXPECTED)
    assert out["counts"]["branch_fallback"] == 1 and out["activated"]["days"] == 0 and out["interpretation"].startswith("untested")


def test_a_tampered_forecast_is_reported():
    r = record(EXPECTED[0], 0.4)
    r["forecasts"]["W"][3] += 0.5
    assert not ss.rule_holds(r)
    wrong_branch = record(EXPECTED[1], 2.0, lear=60.0)
    wrong_branch["switch_branch"] = "ensemble"
    assert not ss.rule_holds(wrong_branch)
    out = ss.evaluate({EXPECTED[0]: r, EXPECTED[1]: wrong_branch}, prices(EXPECTED[:2]), EXPECTED)
    assert out["counts"]["rule_mismatches"] == 2


def test_spells_cross_the_weekend():
    saturday, tuesday = EXPECTED[4], EXPECTED[5]
    assert pd.Timestamp(saturday).dayofweek == 5 and pd.Timestamp(tuesday).dayofweek == 1
    assert ss.spells([tuesday, saturday, EXPECTED[9]], EXPECTED) == [[saturday, tuesday], [EXPECTED[9]]]
    assert ss.spells([], EXPECTED) == []


def test_the_read_on_two_spells():
    high = {EXPECTED[1], EXPECTED[2], EXPECTED[8]}  # two spells: two days, then one
    chosen = {d: record(d, 2.0 if d in high else 0.3, lear=60.0 if d in high else 100.0) for d in EXPECTED[:10]}
    chosen[EXPECTED[10]] = record(EXPECTED[10], 0.2, before_window=False)  # late: kept, not scored
    chosen[EXPECTED[11]] = record(EXPECTED[11], 0.2, with_switch=False)  # written by the old code: no W
    out = ss.evaluate(chosen, prices(EXPECTED), EXPECTED)
    c = out["counts"]
    assert (c["expected"], c["with_a_record"], c["lost"], c["late"], c["without_switch_fields"], c["scored"]) == (12, 12, 0, 1, 1, 10)
    assert (c["branch_tree"], c["branch_ensemble"], c["branch_fallback"], c["rule_mismatches"]) == (3, 7, 0, 0)
    assert out["activated"]["days"] == 3 and out["activated"]["spells"] == 2
    # on a high day the tree is 10 off and the ensemble 15 off, so W saves 5; elsewhere W is the ensemble
    assert out["activated"]["W_against_S1_on_activated_days"]["mean"] == pytest.approx(-5.0)
    assert out["W_against_S1"]["mean"] == pytest.approx(-1.5) and out["W_against_S1"]["days"] == 10
    assert [s["days"] for s in out["activated"]["by_spell"]] == [2, 1] and out["activated"]["by_spell"][0]["W_minus_S1"] == pytest.approx(-5.0)
    assert out["W_against_S1_without_longest_spell"]["days"] == 8 and out["W_against_S1_without_longest_spell"]["mean"] == pytest.approx(-0.625)
    assert out["W_against_g"]["mean"] == pytest.approx((7 * -5.0 + 3 * 0.0) / 10)  # the ensemble is 5 off on normal days, the tree 10
    assert out["interpretation"] in ("supported", "undecided")


def test_interpretations_written_in_the_registration():
    good, flat, bad = {"one_sided_upper": -0.2, "mirrored_lower": -2.0}, {"one_sided_upper": 0.4, "mirrored_lower": -0.9}, {"one_sided_upper": 2.0, "mirrored_lower": 0.3}
    assert ss.interpret(0, flat, flat).startswith("untested")
    assert ss.interpret(1, good, {"days": 5, "mean": -1.0}).startswith("one episode")
    assert ss.interpret(2, good, {"days": 5, "mean": -1.0}) == "supported"
    assert ss.interpret(2, good, {"days": 5, "mean": 0.1}) == "undecided"  # the gain does not survive without the longest spell
    assert ss.interpret(3, bad, {"days": 5, "mean": 1.0}) == "contradicted"
    assert ss.interpret(2, flat, {"days": 5, "mean": -0.1}) == "undecided"


def test_one_spell_is_one_episode():
    chosen = {d: record(d, 2.0 if d in EXPECTED[3:6] else 0.3, lear=60.0 if d in EXPECTED[3:6] else 100.0) for d in EXPECTED[:8]}
    out = ss.evaluate(chosen, prices(EXPECTED), EXPECTED)
    assert out["activated"]["spells"] == 1 and out["activated"]["days"] == 3 and out["interpretation"].startswith("one episode")


def test_the_read_is_locked_before_the_end_date(tmp_path):
    with pytest.raises(PermissionError, match="read once"):
        ss.read(tmp_path, tmp_path / "none.db", today="2026-12-15")
    assert not (tmp_path / "looks.jsonl").exists()
    assert ss.ONE_SIDED == pytest.approx(0.975) and np.isclose(ss.RULE_TOLERANCE, 1e-9)
