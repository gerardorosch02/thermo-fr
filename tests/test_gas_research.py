"""The ttf-gas research measurement (addendum v5, section 3): research sets outside GATED_SETS, the idealised timing rule
gas_settle_ideal, the next-contract substitution, the refusals of every live path, and the construction, perturbation and
hard gate checks against the ideal rule. Hand-built data only."""

import json

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_gas import CASES, CAL, eua, settlements, value
from thermo_fr.dashboard import data as dashboard_data
from thermo_fr.forecast import publish, timing
from thermo_fr.forecast.backtest import run_backtest, walk_forward
from thermo_fr.forecast.day import forecast_day
from thermo_fr.forecast.features import (GAS, GATED_SETS, HARD_GATED_SETS, RESEARCH_LABEL, RESEARCH_SETS, build_features,
                                         feature_timings, refuse_research)
from thermo_fr.forecast.gas import GasInputs, issue_time, known_from_ideal, load_dbnomics_json
from thermo_fr.forecast.jobs import morning_run
from thermo_fr.forecast.refit import refit_model, refit_sets
from thermo_fr.forecast.store import Store
from thermo_fr.forecast.timing import LookaheadError, check_point_in_time


def ideal(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["known_from"] = known_from_ideal(out["trade_date"]).to_numpy()
    return out


def research_inputs(frame=None, **kwargs) -> GasInputs:
    return GasInputs(ideal(settlements(**kwargs) if frame is None else frame), CAL, eua=eua(), mode="ideal", substitute_next=True)


@pytest.fixture(scope="module")
def full():
    return research_inputs()


# --- the sets and the rule --------------------------------------------------------------------------------------------------

def test_research_sets_are_outside_the_gated_sets_but_hard_gated_in_the_backtest():
    assert set(RESEARCH_SETS) == {"research_ttf_ideal_chg", "research_ttf_ideal_rel"}
    assert not set(RESEARCH_SETS) & set(GATED_SETS) and set(RESEARCH_SETS) <= set(HARD_GATED_SETS)
    for name in RESEARCH_SETS:
        timings = feature_timings(name)
        assert all(timings[f] == "gas_settle_ideal" for f in GAS)
    assert "not a forecast model" in RESEARCH_LABEL and "upper bound under an idealised timing assumption" in RESEARCH_LABEL


def test_gas_settle_ideal_is_1720_amsterdam_on_the_trade_date():
    assert list(known_from_ideal(["2026-07-15", "2026-01-15"])) == [pd.Timestamp("2026-07-15T15:20Z"), pd.Timestamp("2026-01-15T16:20Z")]
    idx = pd.date_range("2026-07-15", periods=24, freq="1h", tz="Europe/Paris").tz_convert("UTC")
    assert (timing.issue_times(idx, "gas_settle_ideal") == pd.Timestamp("2026-07-13T15:20Z")).all()  # 17:20 on D-2
    year = pd.date_range("2024-01-01", "2026-06-30", freq="1h", tz="UTC")
    rules = {f: "gas_settle_ideal" for f in GAS}
    check_point_in_time(year, rules)
    check_point_in_time(year, rules, deadline="premarket")
    # the settlement of D-1 itself would be known only at 17:20 on D-1, after the gate
    assert (timing.issue_times(idx, "gas_settle_ideal") + pd.Timedelta(days=1) > timing.gate_for(idx)).all()


# --- refusals ---------------------------------------------------------------------------------------------------------------

def test_every_live_path_refuses_the_research_sets(tmp_path):
    inputs = synthetic_inputs("2024-01-01", "2024-02-01")
    for name in RESEARCH_SETS:
        with pytest.raises(ValueError, match="refit refuses"):
            refit_model(inputs, out_dir=tmp_path, feature_set=name)
        with pytest.raises(ValueError, match="refit refuses"):
            refit_sets(inputs, out_dir=tmp_path, feature_sets=("honest_v2", name))
        with pytest.raises(ValueError, match="morning-run refuses"):
            morning_run(Store(tmp_path / "db.sqlite"), "2024-01-20", feature_sets=(name,), inputs_path=tmp_path / "none.csv")
        with pytest.raises(ValueError, match="forecast refuses"):
            forecast_day("2024-01-20", inputs, feature_set=name)
        for panel in (lambda: dashboard_data.forecast_panel(tmp_path / "db.sqlite", "2024-01-20", name, "2024-01-19"),
                      lambda: dashboard_data.performance_panel(tmp_path / "db.sqlite", name),
                      lambda: dashboard_data.calibration_panel(tmp_path / "db.sqlite", name),
                      lambda: dashboard_data.market_panel(tmp_path / "db.sqlite", name),
                      lambda: dashboard_data.shape_panel(tmp_path / "db.sqlite", name),
                      lambda: dashboard_data.refit_panel(feature_set=name),
                      lambda: dashboard_data.revisions_panel(tmp_path / "db.sqlite", "2024-01-20", name)):
            with pytest.raises(ValueError, match="dashboard refuses"):
                panel()
    assert not set(publish.PUBLIC_FEATURE_SETS) & set(RESEARCH_SETS) and publish.PUBLIC_FEATURE_SET not in RESEARCH_SETS
    with pytest.raises(ValueError, match="publish refuses"):
        refuse_research(publish.PUBLIC_FEATURE_SETS + ("research_ttf_ideal_chg",), "publish")


def test_research_sets_need_the_ideal_inputs_and_v4_sets_refuse_them():
    inputs = synthetic_inputs()
    contracts = [p.strftime("%Y-%m") for p in pd.period_range("2023-07", "2024-07", freq="M")]
    frame = settlements(first="2023-06-01", last="2024-05-31", contracts=contracts)
    dbn = GasInputs(frame, CAL, eua=eua("2023-06-01", "2024-05-31"))
    research = GasInputs(ideal(frame), CAL, eua=eua("2023-06-01", "2024-05-31"), mode="ideal", substitute_next=True)
    with pytest.raises(ValueError, match="idealised gas inputs"):
        build_features(inputs, "research_ttf_ideal_chg", gas=dbn)
    with pytest.raises(ValueError, match="must not use the idealised"):
        build_features(inputs, "honest_v2_gas", gas=research)
    with pytest.raises(ValueError, match="needs the gas data"):
        build_features(inputs, "research_ttf_ideal_rel")
    table = build_features(inputs, "research_ttf_ideal_rel", gas=research)
    assert table.target_offset is not None and {"gas_next_contract", "gas_known_from_t1"} <= set(table.info)
    with pytest.raises(ValueError, match="cannot share a run"):
        run_backtest(inputs, "2024-03-01", "2024-04-01", feature_sets=("honest_v2_gas", "research_ttf_ideal_chg"), gas=research,
                     train_start="2023-01-01")


# --- check 1 against the ideal rule: construction, with and without the substitution ----------------------------------------

@pytest.mark.parametrize("day,t1,r3,r9,contract,age,roll", CASES)
def test_construction_by_hand_under_the_ideal_rule(full, day, t1, r3, r9, contract, age, roll):
    f = full.day_features(day)
    assert not f["gas_missing"] and not f["gas_next_contract"]
    assert (f["gas_t1"], f["gas_r3"], f["gas_r9"], f["gas_contract"], f["gas_age_days"]) == (
        pd.Timestamp(t1), pd.Timestamp(r3), pd.Timestamp(r9), contract, age)
    assert f["gas_chg_lag1_eur"] == pytest.approx(value(contract, t1) - value(contract, r3))
    assert f["gas_known_from_t1"] == known_from_ideal([t1])[0] <= issue_time(day)


def test_next_contract_stands_in_for_a_missing_front_month_day():
    # Oct26 is the front month in mid-September; drop its 16 September settlement
    gas = research_inputs(drop={("2026-10", "2026-09-16")})
    f = gas.day_features("2026-09-18")  # T1 = 16 Sep: Nov26 stands in, no staleness
    assert f["gas_t1"] == pd.Timestamp("2026-09-16") and f["gas_contract"] == "2026-11" and f["gas_next_contract"] and f["gas_age_days"] == 2
    assert f["gas_r3"] == pd.Timestamp("2026-09-15") and f["gas_chg_lag1_eur"] == pytest.approx(value("2026-11", "2026-09-16") - value("2026-11", "2026-09-15"))
    assert f["gas_chg_lag7_eur"] == pytest.approx(value("2026-11", "2026-09-16") - value("2026-11", "2026-09-09"))  # both ends on Nov26
    # without the substitution (the v4 rule), the same gap steps back one trading day in Oct26
    v4 = GasInputs(settlements(drop={("2026-10", "2026-09-16")}), CAL, eua=eua())
    g = v4.day_features("2026-09-18")
    assert g["gas_t1"] == pd.Timestamp("2026-09-15") and g["gas_contract"] == "2026-10" and not g["gas_next_contract"]
    # front and next both missing: one trading day back, where the front month has its value again
    both = research_inputs(drop={("2026-10", "2026-09-16"), ("2026-11", "2026-09-16")})
    h = both.day_features("2026-09-18")
    assert h["gas_t1"] == pd.Timestamp("2026-09-15") and h["gas_contract"] == "2026-10" and not h["gas_next_contract"]


def test_a_long_gap_before_expiry_is_bridged_by_the_next_contract():
    # Mar26 expires 2026-02-26; drop its last 18 trading days, as in the fetch facts (values themselves are synthetic)
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-02-03", "2026-02-26") if CAL.is_trading(d)]
    gas = research_inputs(drop={("2026-03", d) for d in days})
    for day in ("2026-02-10", "2026-02-20", "2026-02-27"):
        f = gas.day_features(day)
        assert not f["gas_missing"] and f["gas_contract"] == "2026-04" and f["gas_next_contract"]
    assert not gas.day_features("2026-03-02")["gas_next_contract"]  # Apr26 is the front month itself from 27 February


# --- check 2 against the ideal rule: perturbation -------------------------------------------------------------------------

@pytest.mark.parametrize("day", ["2026-09-14", "2026-09-16", "2026-09-19", "2026-04-07", "2026-11-01", "2027-01-02"])
def test_perturbation_on_or_after_d1_changes_nothing_under_the_ideal_rule(full, day):
    base = ideal(settlements())
    d = pd.Timestamp(day)
    rng = np.random.default_rng(4)
    perturbed = base.copy()
    later = pd.to_datetime(perturbed["trade_date"]) >= d - pd.Timedelta(days=1)
    perturbed.loc[later, "value"] += rng.uniform(5, 50, int(later.sum()))
    revisions = base[~later].copy()
    revisions["value"] += rng.uniform(5, 50, len(revisions))
    revisions["known_from"] = issue_time(day) + pd.Timedelta(minutes=1)
    got = GasInputs(pd.concat([perturbed, revisions], ignore_index=True), CAL, eua=eua(), mode="ideal", substitute_next=True).day_features(day)
    want = full.day_features(day)
    for k in GAS + ["gas_t1", "gas_r3", "gas_r9", "gas_contract", "gas_next_contract"]:
        assert got[k] == want[k], k


# --- check 3 against the ideal rule: the hard gate --------------------------------------------------------------------------

def test_hard_gate_stops_a_research_backtest(monkeypatch):
    inputs = synthetic_inputs()
    contracts = [p.strftime("%Y-%m") for p in pd.period_range("2023-07", "2024-07", freq="M")]
    gas = GasInputs(ideal(settlements(first="2023-06-01", last="2024-05-31", contracts=contracts)), CAL, eua=eua("2023-06-01", "2024-05-31"),
                    mode="ideal", substitute_next=True)
    table = build_features(inputs, "research_ttf_ideal_chg", gas=gas)
    rows = table.info["delivery_day"] == pd.Timestamp("2024-03-12")
    table.info.loc[rows, "gas_known_from_r9"] = table.info.loc[rows, "gas_issue_time"] + pd.Timedelta(minutes=5)
    with pytest.raises(LookaheadError):
        walk_forward(table, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, train_start="2023-01-01")
    clean = build_features(inputs, "research_ttf_ideal_chg", gas=gas)
    monkeypatch.setitem(timing.TIMINGS, "gas_settle_ideal",
                        timing.InputTiming("gas_settle_ideal", lambda idx: timing._local_clock(timing.delivery_days(idx), -1, 17), "late"))
    with pytest.raises(LookaheadError, match="12:00"):
        walk_forward(clean, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, train_start="2023-01-01")


def test_monthly_only_loader_skips_strips_seasons_and_years(tmp_path):
    docs = [{"series_code": f"D.{i}", "series_name": name, "period": ["2023-01-03", "2023-01-04"], "value": [1.5, "NA"]}
            for i, name in enumerate(["Daily – Dec22", "Daily – Summer23", "Daily – Cal 23", "Daily – May25-Sep25", "Daily – GY 25", "Daily – Feb23"])]
    path = tmp_path / "series_0.json"
    path.write_text(json.dumps({"series": {"docs": docs, "num_found": len(docs)}}))
    (tmp_path / "dataset.json").write_text(json.dumps({"dataset": {}, "_meta": {}}))
    frame = load_dbnomics_json([path, tmp_path / "dataset.json"], monthly_only=True)
    assert sorted(frame["contract"].unique()) == ["2022-12", "2023-02"] and len(frame) == 2
    assert frame.attrs["skipped_series"] == 4 and frame.attrs["monthly_series"] == 2


def test_gas_research_command_end_to_end_on_synthetic_data(tmp_path, monkeypatch, capsys):
    from test_gas_checks import dbnomics_pages
    from thermo_fr import cli
    from thermo_fr.forecast import models

    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    data = tmp_path / "forecast"
    data.mkdir()
    synthetic_inputs().to_csv(data / "inputs.csv", index_label="timestamp_utc")
    contracts = [p.strftime("%Y-%m") for p in pd.period_range("2023-07", "2024-07", freq="M")]
    frame = settlements(first="2023-06-01", last="2024-05-31", contracts=contracts, drop={("2024-04", "2024-03-12")})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    gas_dir = tmp_path / "gas"
    (gas_dir / "dbnomics").mkdir(parents=True)
    for i, page in enumerate(dbnomics_pages(frame)):
        (gas_dir / "dbnomics" / f"p{i}.json").write_text(json.dumps(page))
    pd.DataFrame({"date": ["2026-01-01"]}).to_csv(gas_dir / "ice_closures.csv", index=False)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="selection window only"):
        cli.main(["gas-research", "--data", str(data / "inputs.csv"), "--gas", str(gas_dir), "--out", str(out), "--test-end", "2026-08-01"])
    cli.main(["gas-research", "--data", str(data / "inputs.csv"), "--gas", str(gas_dir), "--out", str(out),
              "--test-start", "2024-02-01", "--test-end", "2024-05-01"])
    text = capsys.readouterr().out
    assert "Research measurement, not a forecast model" in text and "Reading rule:" in text
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["sets"]) == {"honest_v2_t23", "research_ttf_ideal_chg", "research_ttf_ideal_rel"}
    r1 = summary["sets"]["research_ttf_ideal_chg"]
    assert r1["substituted_days"] >= 1 and set(r1["mechanism"]) >= {"age_2", "age_above_2", "top10_gas_change_days"}
    assert 0 <= r1["split_gain"]["gas_total"] <= 1 and set(summary["reading_rule"]["candidates"]) == set(RESEARCH_SETS)
    predictions = pd.read_csv(out / "predictions_research_ttf_ideal_chg.csv")
    assert not set(GAS) & set(predictions.columns) and "gas_next_contract" in predictions
