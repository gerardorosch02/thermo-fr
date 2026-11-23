"""Scoring of the test of W under amendment 6, on synthetic frames: the floor, days left out, yesterday's spread on its own
hours, the reasons an hour may be left out, the seal. The frozen scoring file is used as the reference where both apply."""

import json

import numpy as np
import pandas as pd
import pytest

import test_spread_w_test as base
from thermo_fr.forecast import spread_eval as se
from thermo_fr.forecast import spread_w_test as wt
from thermo_fr.rebuild import spread_w_score as sc

NO_FRENCH = f"{sc.NO_SPREAD}: no French price"


def without(f, schedule, days, reason=NO_FRENCH, issued=True):
    """The frame with whole days left out, and the schedule saying why."""
    gone = f["delivery_day"].isin(pd.to_datetime(days))
    schedule = schedule.copy()
    schedule.loc[gone.to_numpy(), "unscored_reason"] = reason
    schedule.loc[gone.to_numpy(), "issued"] = issued
    return f[~gone], schedule


def test_on_a_complete_window_the_decisions_are_those_of_the_frozen_scoring_code():
    f, schedule = base.both()
    new, old = sc.report(f, schedule), wt.report(f, schedule)
    assert new["floor"] == {"eligible_days": 100, "eligible_hours": 2401, "least_days": 95, "least_hours": 2280, "readable": True} and new["status"] == "read"
    for a, b in zip(new["decisions"], old["decisions"]):
        assert a["outcome"] == b["rule"]["verdict"]
        assert {k: a[k] for k in ("candidate", "base", "gain", "paired", "slices", "rule", "second_reading_small_slices_set_aside")} == {k: b[k] for k in ("candidate", "base", "gain", "paired", "slices", "rule", "second_reading_small_slices_set_aside")}
    assert {m: new["mae"][m] for m in sc.COMPARED} == {m: old["mae"][m] for m in sc.COMPARED} and "yesterday" not in new["mae"]
    assert {m: new["against_yesterdays_spread"]["mae"][m] for m in wt.MODELS} == {m: old["mae"][m] for m in wt.MODELS}  # with no hour missing, the subset is the whole frame
    assert new["against_yesterdays_spread"]["direction_right"] == old["direction_right"] and new["against_yesterdays_spread"]["W against yesterday"] == old["reported_only"]["W against yesterday"]
    assert new["excluded"]["hours_left_out"] == 0 and new["excluded"]["days_with_no_eligible_hour"] == {}


def test_days_with_no_eligible_hour_are_left_out_named_and_the_test_is_still_read_above_the_floor():
    f, schedule = without(*base.both(), ["2026-11-20", "2026-11-21", "2026-11-22"])
    with pytest.raises(ValueError, match="97 days"):
        wt.validate(f)  # the frozen code refuses anything but the 100 days
    out = sc.report(f, schedule)
    assert out["floor"]["eligible_days"] == 97 and out["floor"]["eligible_hours"] == 2401 - 72 and out["floor"]["readable"]
    assert all(d["outcome"] in ("passes", "fails", "contradicts") for d in out["decisions"]) and out["decisions"][0]["paired"]["days"] == 97
    assert out["excluded"]["days_with_no_eligible_hour"] == {d: [NO_FRENCH] for d in ("2026-11-20", "2026-11-21", "2026-11-22")}
    assert out["excluded"]["by_reason"] == {NO_FRENCH: 72} and len(out["excluded"]["hours_by_day_and_reason_utc"]["2026-11-21"][NO_FRENCH]) == 24
    assert out["availability"]["all"]["scheduled_hours"] == 2401 and out["availability"]["all"]["eligible_hours"] == 2329


def test_under_95_days_no_decision_is_drawn_and_the_figures_are_marked_not_decisive():
    f, schedule = without(*base.both(), pd.date_range("2026-11-20", periods=6))
    out = sc.report(f, schedule)
    assert out["floor"]["eligible_days"] == 94 and not out["floor"]["readable"] and out["status"].startswith("below the floor")
    assert [d["outcome"] for d in out["decisions"]] == [sc.NOT_READABLE] * 3 and not any(d["decisive"] for d in out["decisions"])
    for d in out["decisions"]:
        for reading in (d["rule"], d["second_reading_small_slices_set_aside"]):
            assert reading["verdict"] == sc.NOT_READABLE and reading["conditions"] is None  # no verdict can be read off the three conditions
    assert out["decisions"][0]["gain"] > 0 and out["mae"]["hours"] == 2401 - 144  # the figures are still there


def test_the_floor_is_exactly_95_days_and_2280_hours_both():
    f, schedule = base.both()
    five = f["delivery_day"].isin(pd.date_range("2026-11-20", periods=5))
    schedule.loc[five.to_numpy(), "unscored_reason"] = NO_FRENCH
    g = f[~five]  # 95 days, 2,281 hours
    assert sc.floor(g) == {"eligible_days": 95, "eligible_hours": 2281, "least_days": 95, "least_hours": 2280, "readable": True}
    assert sc.floor(g.iloc[1:])["eligible_hours"] == 2280 and sc.floor(g.iloc[1:])["readable"]
    assert sc.floor(g.iloc[2:])["eligible_hours"] == 2279 and not sc.floor(g.iloc[2:])["readable"]
    six = f[~f["delivery_day"].isin(pd.date_range("2026-11-20", periods=6))]
    assert sc.floor(six)["eligible_days"] == 94 and not sc.floor(six)["readable"]
    assert sc.report(g, schedule)["status"] == "read"


def test_the_hours_floor_binds_even_when_enough_days_are_eligible():
    f, schedule = base.both()
    drop = f["delivery_day"].isin(pd.date_range("2026-11-01", periods=12)) & (f["hour"] < 12)  # 12 days lose half their hours: 100 days, 2,257 hours
    schedule.loc[drop.to_numpy(), "unscored_reason"] = sc.NO_PROXY
    out = sc.report(f[~drop], schedule)
    assert out["floor"]["eligible_days"] == 100 and out["floor"]["eligible_hours"] == 2401 - 144 and not out["floor"]["readable"]
    assert out["decisions"][0]["outcome"] == sc.NOT_READABLE


def test_an_hour_without_yesterdays_spread_stays_in_the_decisions_and_yesterday_is_scored_on_its_own_hours():
    f, schedule = base.both()
    full = sc.report(f, schedule)
    g = f.copy()
    day = (g["delivery_day"] == pd.Timestamp("2026-11-22")) | ((g["delivery_day"] == pd.Timestamp("2026-12-05")) & (g["hour"] < 9))  # a whole day and part of another
    g.loc[day, "yesterday"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        wt.validate(g)  # the frozen code would have dropped these hours for every model
    out = sc.report(g, schedule)
    assert out["floor"]["eligible_days"] == 100 and out["floor"]["eligible_hours"] == 2401
    assert out["decisions"] == full["decisions"]  # the decisions never see yesterday's spread
    assert out["mae"] == full["mae"] and out["by_month"] == full["by_month"] and "yesterday" not in out["mae"]
    sub = g[~day]
    against = out["against_yesterdays_spread"]
    assert against["hours"] == 2401 - 33 and against["days"] == 99 and against["mae"]["hours"] == 2401 - 33
    for name in ("yesterday", "W", "H"):  # both sides on the same subset, overall, by month and by class
        assert against["mae"][name] == pytest.approx((sub[name] - sub["spread"]).abs().mean())
        november = sub[sub["delivery_day"].dt.month == 11]
        assert against["by_month"]["2026-11"][name] == pytest.approx((november[name] - november["spread"]).abs().mean())
        saved = sub[sub["provenance"] == "saved"]
        assert against["by_class"]["saved"][name] == pytest.approx((saved[name] - saved["spread"]).abs().mean())
        assert out["availability"]["saved"]["mae_on_hours_with_yesterdays_spread"][name] == against["by_class"]["saved"][name]
    assert against["by_month"]["2026-11"]["hours"] == 30 * 24 - 24 and against["W against yesterday"]["days"] == 99
    daily = sub.groupby("delivery_day")[["spread", "yesterday", "W"]].mean()
    assert against["base_spread_mae"]["W"] == pytest.approx((daily["W"] - daily["spread"]).abs().mean())
    move = daily["spread"] - daily["yesterday"]
    big = move.abs() > se.MOVE
    assert against["days_with_move_above_5"] == int(big.sum())
    assert against["direction_right"]["W"] == pytest.approx((np.sign(daily.loc[big, "W"] - daily.loc[big, "yesterday"]) == np.sign(move[big])).mean())
    assert out["excluded"]["eligible_hours_without_yesterdays_spread"] == 33
    assert out["excluded"]["days_with_an_eligible_hour_without_yesterdays_spread"] == ["2026-11-22", "2026-12-05"]


def test_every_hour_left_out_carries_a_reason_of_the_amendment_that_fits_whether_it_was_issued():
    f, schedule = base.both()
    german, both_prices = sc.PRICE_REASONS[1], sc.PRICE_REASONS[2]
    refused_when_issued = ("no spread of the previous day at that hour", "something else", sc.NO_SPREAD, f"{sc.NO_SPREAD}:", f"{sc.NO_SPREAD}: banana", sc.NOT_ISSUED, sc.NOT_STRICT,
                           f"{NO_FRENCH}; {german}", f"{NO_FRENCH}; {NO_FRENCH}")
    for reason in refused_when_issued:
        with pytest.raises(ValueError, match="does not allow"):
            sc.report(*without(f, schedule, ["2026-11-21"], reason=reason))
    with pytest.raises(ValueError, match="without a reason"):
        sc.report(*without(f, schedule, ["2026-11-21"], reason=""))
    for reason in (NO_FRENCH, german, both_prices, sc.NO_PROXY, f"{german}; {sc.NO_PROXY}"):
        assert sc.report(*without(f, schedule, ["2026-11-21"], reason=reason))["floor"]["eligible_days"] == 99
    # an hour with no issued forecast must say so, and nothing else may stand in for that
    for reason in ("banana", "no spread of the previous day at that hour", NO_FRENCH, sc.NO_PROXY, f"{sc.NOT_ISSUED}; {sc.NOT_STRICT}"):
        with pytest.raises(ValueError, match="does not allow"):
            sc.report(*without(f, schedule, ["2026-11-21"], reason=reason, issued=False))
    joined = f"{sc.NOT_STRICT}; {NO_FRENCH}"
    out = sc.report(*without(f, schedule, ["2026-11-21"], reason=joined, issued=False))
    assert out["excluded"]["by_reason"] == {joined: 24} and out["excluded"]["hours_by_single_reason"] == {NO_FRENCH: 24, sc.NOT_STRICT: 24}  # the missing price is named too
    assert sc.report(*without(f, schedule, ["2026-11-21"], reason=sc.NOT_ISSUED, issued=False))["excluded"]["by_reason"] == {sc.NOT_ISSUED: 24}


def test_a_frame_is_refused_when_a_model_is_missing_or_infinite_or_a_day_is_outside_the_window():
    f, schedule = base.both()
    g = f.copy()
    g.iloc[5, g.columns.get_loc("GK")] = np.nan
    with pytest.raises(ValueError, match="scored on different hours"):
        sc.report(g, schedule)
    g = f.copy()
    g.iloc[5, g.columns.get_loc("W")] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        sc.report(g, schedule)
    g = f.copy()
    g["yesterday"] = g["yesterday"].astype(str)
    with pytest.raises(ValueError, match="not a number"):
        sc.report(g, schedule)
    with pytest.raises(ValueError, match="registered days"):
        sc.report(f, schedule.drop(index=f.index[10]))  # the schedule must hold every hour of the window
    late = f[f["delivery_day"] >= pd.Timestamp("2027-01-01")].copy()  # January only, so no clock change is crossed
    late.index = late.index + pd.Timedelta(days=7)  # the last week now falls after the window, with consistent time stamps
    late["delivery_day"] = late["delivery_day"] + pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="outside the registered window"):
        sc.validate(late)
    sc.validate(late, last="2027-01-25")  # and it is the window check that refused it


def _checks(folder, **changes):
    fourth = {"pass": True, "basis": sc.EXCEPTION, "file": "comparator_predictions.csv.gz", "sha256": "6d6a", "largest_difference": {"W": 5.7e-14}, "self_test_of_this_workspace": {"pass": True}}
    checks = {"test": wt.TEST_NAME, "window": [wt.FIRST, wt.LAST], "all_pass": True, "test_predictions_sha256": se.sha256(folder / "frame.csv.gz"), "schedule_sha256": se.sha256(folder / "schedule.csv"),
              sc.STOP_4: fourth, **changes}
    (folder / "checks.json").write_text(json.dumps(checks))


def test_the_read_is_sealed_bound_to_the_checked_files_and_states_the_exception(tmp_path, monkeypatch):
    f, schedule = base.both()
    f.loc[f["delivery_day"] == pd.Timestamp("2026-11-22"), "yesterday"] = np.nan  # a frame with yesterday's spread missing goes through a file and back
    f.to_csv(tmp_path / "frame.csv.gz")
    schedule.to_csv(tmp_path / "schedule.csv")
    _checks(tmp_path)
    args = (tmp_path / "frame.csv.gz", tmp_path / "schedule.csv", tmp_path / "checks.json")
    monkeypatch.setattr(wt, "_today", lambda: pd.Timestamp("2027-01-18"))
    with pytest.raises(SystemExit, match="sealed"):
        sc.read(*args)
    monkeypatch.setattr(wt, "_today", lambda: pd.Timestamp("2027-01-19"))
    out = sc.read(*args)
    assert out["floor"]["readable"] and out["excluded"]["eligible_hours_without_yesterdays_spread"] == 24
    assert out["decisions"] == sc.report(f, schedule)["decisions"]  # the file gives back the numbers to the last bit
    assert sc.EXCEPTION in out["stop_condition_4"]["statement"] and out["stop_condition_4"]["basis"] == sc.EXCEPTION and out["stop_condition_4"]["largest_difference"] == {"W": 5.7e-14}
    for changes, message in (({"all_pass": False}, "stop condition"), ({"schedule_sha256": "0"}, "not the ones"), ({"test_predictions_sha256": "0"}, "not the ones"),
                             ({sc.STOP_4: {"pass": True, "basis": "met"}}, "exception"), ({sc.STOP_4: {"pass": False, "basis": sc.EXCEPTION}}, "exception"), ({"window": ["2026-10-11", "2027-01-19"]}, "this test and window")):
        _checks(tmp_path, **changes)
        with pytest.raises(SystemExit, match=message):
            sc.read(*args)
    _checks(tmp_path)
    monkeypatch.setattr(sc, "FROZEN_SCORING_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the registered file"):
        sc.read(*args)


def test_the_frozen_scoring_file_is_the_registered_one():
    assert se.sha256(wt.__file__) == sc.FROZEN_SCORING_SHA256
