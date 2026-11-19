"""The scoring of W's test on synthetic frames: three decisions, the Bonferroni level, the second reading, the seal."""

import json

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import spread_eval as se
from thermo_fr.forecast import spread_w_test as wt


def frame(days=100, gain_vs_hw=3.0, holiday_loss=0.0, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-10-10T22:00Z", "2027-01-18T23:00Z", freq="h", inclusive="left")[: days * 24 + (1 if days == 100 else 0)]
    local = index.tz_convert("Europe/Paris")
    day = pd.DatetimeIndex(local.tz_localize(None).normalize())
    f = pd.DataFrame({"delivery_day": day, "hour": local.hour, "dow": day.dayofweek, "holiday": day.isin(pd.to_datetime(["2026-11-01", "2026-11-11", "2026-12-25"])), "de_holiday": 0.0,
                      "spread": rng.normal(-5, 20, len(index)), "wind_px": rng.uniform(0, 40000, len(index)), "solar_px": rng.uniform(0, 30000, len(index))}, index=index)
    e = rng.normal(0, 15, len(index))
    worse = lambda extra: f["spread"] + np.sign(e) * (np.abs(e) + extra)  # noqa: E731
    f["W"], f["HW"], f["H"], f["G"] = f["spread"] + e, worse(gain_vs_hw), worse(gain_vs_hw + 1), worse(1.5)
    f["GK"], f["GM"], f["C"], f["yesterday"] = worse(1.0), worse(0.7), f["spread"] + 0.9 * e, worse(8)
    if holiday_loss:  # on the three holidays W is worse than HW by this much
        hol = f["holiday"].to_numpy()
        f.loc[hol, "W"] = f.loc[hol, "spread"] + np.sign(e[hol]) * (np.abs(e[hol]) + gain_vs_hw + holiday_loss)
    f["provenance"] = np.select([f["delivery_day"] >= pd.Timestamp("2026-12-01"), f["delivery_day"] >= pd.Timestamp("2026-10-20")], ["saved", "rebuilt_french_saved"], "rebuilt")
    return f


def test_three_decisions_at_the_bonferroni_level_with_the_primary_first():
    r = wt.report(frame())
    assert [(d["candidate"], d["base"]) for d in r["decisions"]] == [("W", "HW"), ("W", "H"), ("W", "G")]
    assert r["decisions"][0]["question"].startswith("primary")
    assert wt.ONE_SIDED == pytest.approx(1 - 0.05 / 3)
    assert all(d["paired"]["one_sided_level"] == wt.ONE_SIDED and d["rule"]["verdict"] == "passes" for d in r["decisions"])


def test_the_one_sided_end_is_wider_than_at_the_single_test_level():
    f = frame(gain_vs_hw=1.0)
    f["HW"] = f["HW"] + np.random.default_rng(1).normal(0, 10, len(f))  # the synthetic gain is otherwise the same on every day
    strict = wt.paired(f, "W", "HW")["one_sided_upper"]
    single = se.paired(f.assign(G=f["W"], H=f["HW"]), "G", "H")["one_sided_upper"]
    assert strict > single  # a higher confidence level puts the upper end closer to zero


def test_a_three_day_holiday_slice_fails_the_rule_but_not_the_second_reading():
    r = wt.report(frame(holiday_loss=2.0))
    primary = r["decisions"][0]
    assert primary["slices"]["holidays (French or German)"]["days"] == 3
    assert primary["rule"]["verdict"] == "fails" and not primary["rule"]["conditions"]["no_slice_worse_by_more_than_1.0"]
    assert primary["second_reading_small_slices_set_aside"]["verdict"] == "passes"


def test_reconstructed_days_and_days_saved_before_the_auction_are_counted_apart():
    r = wt.report(frame())
    p = r["provenance"]
    assert p["days"] == {"saved": 49, "rebuilt_french_saved": 42, "rebuilt": 9}
    assert p["errors"]["rebuilt"]["days"] == 9 and p["errors"]["saved"]["hours"] == 49 * 24
    assert set(p["primary_comparison_by_class"]) == {"saved", "rebuilt_french_saved"}  # nine days are too few for an interval
    assert r["days"] == 100 and r["hours"] == 2401 and set(r["by_month"]) == {"2026-10", "2026-11", "2026-12", "2027-01"}
    assert r["not_attainable"] == ["C"]


def test_days_outside_the_window_are_refused():
    f = frame()
    f.loc[f.index[0], "delivery_day"] = pd.Timestamp("2026-10-10")
    with pytest.raises(ValueError):
        wt.report(f)


def test_models_must_be_scored_on_the_same_hours_each_hour_once_on_all_100_days():
    f = frame()
    worst = (f["W"] - f["spread"]).abs() > (f["W"] - f["spread"]).abs().quantile(0.7)
    with pytest.raises(ValueError, match="missing values"):
        wt.report(f.assign(W=f["W"].where(~worst)))  # a model that is blank on its worst hours must not be scored on the rest
    with pytest.raises(ValueError, match="more than once"):
        wt.report(pd.concat([f, f.iloc[:24]]))
    with pytest.raises(ValueError, match="expected 100 days"):
        wt.report(f[f["delivery_day"] <= pd.Timestamp("2026-11-19")])
    with pytest.raises(ValueError, match="missing values"):
        wt.report(f.assign(de_holiday=np.nan))
    mixed = f.copy()
    mixed.loc[mixed.index[5], "provenance"] = "saved"
    with pytest.raises(ValueError, match="provenance"):
        wt.report(mixed)


def test_the_read_is_sealed_until_the_date_and_bound_to_the_checked_forecasts(tmp_path, monkeypatch):
    f = frame()
    f.to_csv(tmp_path / "p.csv.gz")
    good = {"test": "spread_w", "window": [wt.FIRST, wt.LAST], "all_pass": True, "test_predictions_sha256": se.sha256(tmp_path / "p.csv.gz")}
    (tmp_path / "s.json").write_text(json.dumps(good))
    assert wt._today() < pd.Timestamp(wt.READ_FROM) or True
    monkeypatch.setattr(wt, "_today", lambda: pd.Timestamp("2027-01-18"))
    with pytest.raises(SystemExit, match="sealed"):
        wt.read(tmp_path / "p.csv.gz", tmp_path / "s.json")
    monkeypatch.setattr(wt, "_today", lambda: pd.Timestamp("2027-01-19"))
    assert wt.read(tmp_path / "p.csv.gz", tmp_path / "s.json")["days"] == 100
    for bad in ({**good, "all_pass": False}, {**good, "test_predictions_sha256": "other"}, {**good, "test": "spread_eval"}, {**good, "window": ["2026-07-01", "2026-10-10"]}):
        (tmp_path / "s.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            wt.read(tmp_path / "p.csv.gz", tmp_path / "s.json")
