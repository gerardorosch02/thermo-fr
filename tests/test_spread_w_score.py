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
    assert {m: new["mae"][m] for m in sc.COMPARED} == {m: old["mae"][m] for m in sc.COMPARED} and new["mae"]["yesterday"] == old["mae"]["yesterday"]
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
    assert [d["outcome"] for d in out["decisions"]] == [sc.NOT_READABLE] * 3
    assert all(d["rule"]["verdict"] == sc.NOT_READABLE and d["second_reading_small_slices_set_aside"]["verdict"] == sc.NOT_READABLE for d in out["decisions"])
    assert out["decisions"][0]["gain"] > 0 and out["mae"]["hours"] == 2401 - 144  # the figures are still there


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
    day = g["delivery_day"] == pd.Timestamp("2026-11-22")
    g.loc[day, "yesterday"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        wt.validate(g)  # the frozen code would have dropped the day for every model
    out = sc.report(g, schedule)
    assert out["floor"]["eligible_days"] == 100 and out["floor"]["eligible_hours"] == 2401
    assert out["decisions"] == full["decisions"]  # the decisions never see yesterday's spread
    assert {m: out["mae"][m] for m in sc.COMPARED} == {m: full["mae"][m] for m in sc.COMPARED}
    sub = g[~day]
    against = out["against_yesterdays_spread"]
    assert against["hours"] == 2401 - 24 and against["days"] == 99 and out["mae"]["hours_with_yesterdays_spread"] == 2401 - 24
    assert out["mae"]["yesterday"] == pytest.approx((sub["yesterday"] - sub["spread"]).abs().mean())
    assert against["mae"]["W"] == pytest.approx((sub["W"] - sub["spread"]).abs().mean())  # both sides on the same subset
    assert against["W against yesterday"]["days"] == 99
    assert out["excluded"]["eligible_hours_without_yesterdays_spread"] == 24 and out["excluded"]["days_with_an_eligible_hour_without_yesterdays_spread"] == ["2026-11-22"]


def test_an_issued_hour_can_be_left_out_only_for_a_reason_of_the_amendment():
    f, schedule = base.both()
    for reason in ("no spread of the previous day at that hour", "something else"):
        with pytest.raises(ValueError, match="does not allow"):
            sc.report(*without(f, schedule, ["2026-11-21"], reason=reason))
    with pytest.raises(ValueError, match="without a reason"):
        sc.report(*without(f, schedule, ["2026-11-21"], reason=""))
    for reason in (sc.NO_SPREAD, sc.NO_PROXY, f"{sc.NO_SPREAD}: no German price"):
        assert sc.report(*without(f, schedule, ["2026-11-21"], reason=reason))["floor"]["eligible_days"] == 99
    out = sc.report(*without(f, schedule, ["2026-11-21"], reason="no forecast issued", issued=False))  # an hour with no forecast needs no registered reason
    assert out["excluded"]["by_reason"] == {"no forecast issued": 24}


def test_a_frame_is_refused_when_a_model_is_missing_an_hour_or_a_day_is_outside_the_window():
    f, schedule = base.both()
    g = f.copy()
    g.iloc[5, g.columns.get_loc("GK")] = np.nan
    with pytest.raises(ValueError, match="scored on different hours"):
        sc.report(g, schedule)
    with pytest.raises(ValueError, match="registered days"):
        sc.report(f, schedule.drop(index=f.index[10]))  # the schedule must hold every hour of the window
    late = f.copy()
    late["delivery_day"] = late["delivery_day"] + pd.Timedelta(days=400)
    with pytest.raises(ValueError):
        sc.validate(late)


def test_the_read_is_sealed_until_the_date_and_bound_to_the_checked_forecasts(tmp_path, monkeypatch):
    f, schedule = base.both()
    f.to_csv(tmp_path / "frame.csv.gz")
    schedule.to_csv(tmp_path / "schedule.csv")
    checks = {"test": wt.TEST_NAME, "window": [wt.FIRST, wt.LAST], "all_pass": True, "test_predictions_sha256": se.sha256(tmp_path / "frame.csv.gz"), "schedule_sha256": se.sha256(tmp_path / "schedule.csv")}
    (tmp_path / "checks.json").write_text(json.dumps(checks))
    args = (tmp_path / "frame.csv.gz", tmp_path / "schedule.csv", tmp_path / "checks.json")
    monkeypatch.setattr(wt, "_today", lambda: pd.Timestamp("2027-01-18"))
    with pytest.raises(SystemExit, match="sealed"):
        sc.read(*args)
    monkeypatch.setattr(wt, "_today", lambda: pd.Timestamp("2027-01-19"))
    assert sc.read(*args)["floor"]["readable"]
    (tmp_path / "checks.json").write_text(json.dumps({**checks, "all_pass": False}))
    with pytest.raises(SystemExit, match="stop condition"):
        sc.read(*args)
    (tmp_path / "checks.json").write_text(json.dumps({**checks, "schedule_sha256": "0"}))
    with pytest.raises(SystemExit, match="not the ones"):
        sc.read(*args)
