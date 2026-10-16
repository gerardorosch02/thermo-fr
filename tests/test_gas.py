"""The ttf-gas machinery (forecast/gas.py and its wiring), on hand-built settlement series. No real gas data is used.

The five checks of the pre-registration, section 4:
1. construction: every delivery weekday, Easter, Christmas, a roll, a missing final day, a missing T1 within and beyond the
   tolerance, a missing reference within and beyond it, a value first known after the issue time, a revision;
2. perturbation: settlements dated on or after D-1, or known after the issue time, do not change the features of D;
3. hard gate: the backtest stops (LookaheadError) for a candidate that fails the gate or the pre-market deadline;
4. companion column: every known_from used is at or before the issue time, asserted;
5. reissue equality: a day rebuilt from the snapshot known at its issue time equals its backtest row.
"""

import json

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr import cli
from thermo_fr.forecast import features as features_module
from thermo_fr.forecast import models, timing
from thermo_fr.forecast.backtest import run_backtest, walk_forward
from thermo_fr.forecast.features import GAS, GAS_SRMC, HONEST_V2, build_features
from thermo_fr.forecast.gas import (
    Calendars,
    GasDataError,
    GasInputs,
    assert_known_before_issue,
    default_ice_closures,
    issue_time,
    load_gas_dir,
)
from thermo_fr.forecast.models import fit_predict
from thermo_fr.forecast.timing import InputTiming, LookaheadError

BASE = pd.Timestamp("2020-01-01")
CAL = Calendars.build(default_ice_closures(range(2022, 2028)), years=range(2022, 2028))


def value(contract: str, day) -> float:
    """A settlement that encodes its contract and its date: 20 + months since Jan 2020 / 10 + 0.01 per calendar day."""
    months = (pd.Period(contract, "M") - pd.Period("2020-01", "M")).n
    return 20 + months / 10 + 0.01 * (pd.Timestamp(day) - BASE).days


def known(day) -> pd.Timestamp:
    """The next calendar morning at 06:00 UTC: after the settlement, before any issue time that may use it."""
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1, hours=6)


def settlements(first="2026-01-02", last="2027-01-29", contracts=None, drop=(), extra=()) -> pd.DataFrame:
    """Every listed contract on every ICE Endex trading day from `first` to its expiry (or `last`)."""
    contracts = contracts or [p.strftime("%Y-%m") for p in pd.period_range("2026-02", "2027-04", freq="M")]
    rows = []
    days = pd.bdate_range(first, last)
    for c in contracts:
        expiry = pd.Timestamp(CAL.expiry(c))
        for d in days:
            if d > expiry or not CAL.is_trading(d) or (c, d.strftime("%Y-%m-%d")) in drop:
                continue
            rows.append({"contract": c, "trade_date": d.strftime("%Y-%m-%d"), "value": value(c, d), "known_from": known(d)})
    rows += list(extra)
    return pd.DataFrame(rows)


def eua(first="2025-12-01", last="2027-01-29") -> pd.DataFrame:
    days = pd.bdate_range(first, last)
    return pd.DataFrame({"auction_date": days, "price_eur_t": 70 + 0.1 * (days - BASE).days,
                         "known_from": days.tz_localize("UTC") + pd.Timedelta(hours=11)})


def eua_on_or_before(day) -> float:
    d = pd.bdate_range(end=pd.Timestamp(day), periods=1)[0]
    return 70 + 0.1 * (d - BASE).days


@pytest.fixture(scope="module")
def full():
    return GasInputs(settlements(), CAL, eua=eua())


# --- calendars --------------------------------------------------------------------------------------------------------

def test_expiry_is_the_second_uk_business_day_before_the_delivery_month():
    assert pd.Timestamp(CAL.expiry("2026-09")) == pd.Timestamp("2026-08-27")  # 31 August is a bank holiday (the scout's date)
    assert pd.Timestamp(CAL.expiry("2026-11")) == pd.Timestamp("2026-10-29")  # 1 November is a Sunday
    assert pd.Timestamp(CAL.expiry("2027-01")) == pd.Timestamp("2026-12-30")  # 25 and 28 December and 1 January are holidays
    assert pd.Timestamp(CAL.expiry("2026-05")) == pd.Timestamp("2026-04-29")
    assert CAL.front("2026-10-29") == "2026-11" and CAL.front("2026-10-30") == "2026-12"
    assert CAL.front("2026-12-30") == "2027-01" and CAL.front("2026-12-31") == "2027-02"


def test_ice_endex_is_open_on_uk_bank_holidays_and_closed_on_its_own_days():
    assert CAL.is_trading("2026-12-28") and CAL.is_trading("2026-08-31") and CAL.is_trading("2026-12-24")
    assert not CAL.is_trading("2026-12-25") and not CAL.is_trading("2026-04-03") and not CAL.is_trading("2026-04-06")
    assert not CAL.is_trading("2027-01-01") and not CAL.is_trading("2026-09-19")
    assert default_ice_closures([2026]) == [pd.Timestamp(d) for d in ("2026-01-01", "2026-04-03", "2026-04-06", "2026-12-25")]


def test_issue_time_is_1005_paris_on_the_day_before():
    assert issue_time("2026-09-18") == pd.Timestamp("2026-09-17T08:05Z")
    assert issue_time("2026-12-18") == pd.Timestamp("2026-12-17T09:05Z")
    assert issue_time("2026-10-26") == pd.Timestamp("2026-10-25T09:05Z")  # the clock went back that night


# --- check 1: construction ---------------------------------------------------------------------------------------------

CASES = [
    # delivery day, T1*, R3*, R9*, contract, age, roll in window
    ("2026-09-14", "2026-09-11", "2026-09-11", "2026-09-04", "2026-10", 3, False),  # Monday
    ("2026-09-15", "2026-09-11", "2026-09-11", "2026-09-04", "2026-10", 4, False),  # Tuesday
    ("2026-09-16", "2026-09-14", "2026-09-11", "2026-09-07", "2026-10", 2, False),  # Wednesday
    ("2026-09-17", "2026-09-15", "2026-09-14", "2026-09-08", "2026-10", 2, False),  # Thursday
    ("2026-09-18", "2026-09-16", "2026-09-15", "2026-09-09", "2026-10", 2, False),  # Friday
    ("2026-09-19", "2026-09-17", "2026-09-16", "2026-09-10", "2026-10", 2, False),  # Saturday
    ("2026-09-20", "2026-09-18", "2026-09-17", "2026-09-11", "2026-10", 2, False),  # Sunday
    ("2026-04-06", "2026-04-02", "2026-04-02", "2026-03-27", "2026-05", 4, False),  # Easter Monday (Good Friday closed)
    ("2026-04-07", "2026-04-02", "2026-04-02", "2026-03-27", "2026-05", 5, False),
    ("2026-04-08", "2026-04-02", "2026-04-02", "2026-03-30", "2026-05", 6, False),
    ("2026-04-09", "2026-04-07", "2026-04-02", "2026-03-31", "2026-05", 2, False),
    ("2026-12-26", "2026-12-24", "2026-12-23", "2026-12-17", "2027-01", 2, False),  # Christmas (25 Dec closed, 28 Dec open)
    ("2026-12-27", "2026-12-24", "2026-12-24", "2026-12-18", "2027-01", 3, False),
    ("2026-12-28", "2026-12-24", "2026-12-24", "2026-12-18", "2027-01", 4, False),
    ("2026-12-29", "2026-12-24", "2026-12-24", "2026-12-18", "2027-01", 5, False),
    ("2026-12-30", "2026-12-28", "2026-12-24", "2026-12-21", "2027-01", 2, False),
    ("2027-01-02", "2026-12-31", "2026-12-30", "2026-12-24", "2027-02", 2, True),   # roll at the year end
    ("2027-01-03", "2026-12-31", "2026-12-31", "2026-12-24", "2027-02", 3, False),  # 1 January closed
    ("2026-10-31", "2026-10-29", "2026-10-28", "2026-10-22", "2026-11", 2, False),  # Nov26 expires on 29 October
    ("2026-11-01", "2026-10-30", "2026-10-29", "2026-10-23", "2026-12", 2, True),   # the roll falls inside the change window
]


@pytest.mark.parametrize("day,t1,r3,r9,contract,age,roll", CASES)
def test_construction_by_hand(full, day, t1, r3, r9, contract, age, roll):
    f = full.day_features(day)
    assert not f["gas_missing"] and not f["gas_srmc_missing"]
    assert (f["gas_t1"], f["gas_r3"], f["gas_r9"]) == (pd.Timestamp(t1), pd.Timestamp(r3), pd.Timestamp(r9))
    assert f["gas_contract"] == contract and f["gas_age_days"] == age and f["gas_roll_in_window"] is roll
    assert f["gas_chg_lag1_eur"] == pytest.approx(value(contract, t1) - value(contract, r3))  # within one contract
    assert f["gas_chg_lag7_eur"] == pytest.approx(value(contract, t1) - value(contract, r9))
    tau = issue_time(day)
    assert f["gas_issue_time"] == tau and f["gas_known_from_t1"] == known(t1) <= tau
    d = pd.Timestamp(day)
    assert f["srmc_eur_mwh"] == pytest.approx((value(contract, t1) + 0.184 * eua_on_or_before(d - pd.Timedelta(days=2))) / 0.50)
    c3 = CAL.front(r3)  # the contract behind the D-1 auction, which differs from `contract` after a roll
    assert (c3 != contract) is roll
    assert f["srmc_ref_eur_mwh"] == pytest.approx((value(c3, r3) + 0.184 * eua_on_or_before(d - pd.Timedelta(days=3))) / 0.50)


def test_no_new_close_since_the_d1_auction_gives_zero_change(full):
    for day in ("2026-09-14", "2026-09-15", "2026-04-08", "2026-12-28"):
        assert full.day_features(day)["gas_chg_lag1_eur"] == 0.0


def test_missing_final_day_uses_the_day_before_in_the_same_contract():
    gas = GasInputs(settlements(drop={("2026-11", "2026-10-29")}), CAL, eua=eua())
    f = gas.day_features("2026-10-31")
    assert f["gas_t1"] == pd.Timestamp("2026-10-28") and f["gas_contract"] == "2026-11" and f["gas_age_days"] == 3
    assert f["gas_r3"] == pd.Timestamp("2026-10-28") and f["gas_chg_lag1_eur"] == 0.0 and not f["gas_missing"]
    # the next day rolls to Dec26 (all its values present); the cost behind the D-1 auction takes Rs one trading day before R3*,
    # where the front month is still Nov26 and has a value
    g = gas.day_features("2026-11-01")
    assert not g["gas_missing"] and g["gas_contract"] == "2026-12" and g["gas_r3"] == pd.Timestamp("2026-10-29")
    assert not g["gas_srmc_missing"] and g["gas_rs"] == pd.Timestamp("2026-10-28") and g["gas_c3"] == "2026-11"
    assert g["srmc_ref_eur_mwh"] == pytest.approx((value("2026-11", "2026-10-28") + 0.184 * eua_on_or_before("2026-10-29")) / 0.50)


def test_spark_reference_tolerance_and_its_limit():
    within = GasInputs(settlements(drop={("2026-11", "2026-10-29"), ("2026-11", "2026-10-28")}), CAL, eua=eua())
    f = within.day_features("2026-11-01")
    assert f["gas_rs"] == pd.Timestamp("2026-10-27") and f["gas_c3"] == "2026-11" and not f["gas_srmc_missing"]
    beyond = GasInputs(settlements(drop={("2026-11", d) for d in ("2026-10-29", "2026-10-28", "2026-10-27")}), CAL, eua=eua())
    g = beyond.day_features("2026-11-01")
    assert not g["gas_missing"] and g["gas_srmc_missing"] and g["gas_missing_reason"] == "no_rs" and np.isnan(g["srmc_eur_mwh"])
    bad = settlements(drop={("2026-11", "2026-10-29")}, extra=[
        {"contract": "2026-11", "trade_date": "2026-10-29", "value": 600.0, "known_from": known("2026-10-29")}])
    h = GasInputs(bad, CAL, eua=eua()).day_features("2026-11-01")
    assert not h["gas_missing"] and h["gas_srmc_missing"] and h["gas_missing_reason"] == "out_of_range_rs"


def test_missing_latest_close_within_and_beyond_the_tolerance():
    one = GasInputs(settlements(drop={("2026-10", "2026-09-16")}), CAL, eua=eua())
    f = one.day_features("2026-09-18")
    assert f["gas_t1"] == pd.Timestamp("2026-09-15") and f["gas_age_days"] == 3 and f["gas_r3"] == pd.Timestamp("2026-09-15")
    assert f["gas_chg_lag1_eur"] == 0.0 and f["gas_r9"] == pd.Timestamp("2026-09-09")
    two = GasInputs(settlements(drop={("2026-10", "2026-09-16"), ("2026-10", "2026-09-15")}), CAL, eua=eua())
    g = two.day_features("2026-09-18")
    assert g["gas_missing"] and g["gas_missing_reason"] == "no_t1" and np.isnan(g["gas_chg_lag1_eur"]) and np.isnan(g["gas_age_days"])
    assert two.day_features("2026-09-17")["gas_t1"] == pd.Timestamp("2026-09-14")  # T1 = 15 Sep missing, one day back is allowed


def test_missing_reference_within_and_beyond_the_tolerance():
    one = GasInputs(settlements(drop={("2026-10", "2026-09-16")}), CAL, eua=eua())
    f = one.day_features("2026-09-19")  # T1 = 17 Sep present, T3 = 16 Sep missing
    assert f["gas_r3"] == pd.Timestamp("2026-09-15") and f["gas_chg_lag1_eur"] == pytest.approx(0.02)
    two = GasInputs(settlements(drop={("2026-10", "2026-09-16"), ("2026-10", "2026-09-15")}), CAL, eua=eua())
    assert two.day_features("2026-09-19")["gas_r3"] == pd.Timestamp("2026-09-14")
    three = GasInputs(settlements(drop={("2026-10", d) for d in ("2026-09-16", "2026-09-15", "2026-09-14")}), CAL, eua=eua())
    g = three.day_features("2026-09-19")
    assert g["gas_missing"] and g["gas_missing_reason"] == "no_r3"
    nine = GasInputs(settlements(drop={("2026-10", d) for d in ("2026-09-10", "2026-09-09", "2026-09-08")}), CAL, eua=eua())
    assert nine.day_features("2026-09-19")["gas_missing_reason"] == "no_r9"


def test_value_first_known_after_the_issue_time_is_not_used():
    tau = issue_time("2026-09-18")  # 08:05 UTC on 17 September
    late = settlements(drop={("2026-10", "2026-09-16")}, extra=[
        {"contract": "2026-10", "trade_date": "2026-09-16", "value": value("2026-10", "2026-09-16"), "known_from": tau + pd.Timedelta(minutes=1)}])
    f = GasInputs(late, CAL, eua=eua()).day_features("2026-09-18")
    assert f["gas_t1"] == pd.Timestamp("2026-09-15") and f["gas_age_days"] == 3
    on_time = settlements(drop={("2026-10", "2026-09-16")}, extra=[
        {"contract": "2026-10", "trade_date": "2026-09-16", "value": value("2026-10", "2026-09-16"), "known_from": tau}])
    g = GasInputs(on_time, CAL, eua=eua()).day_features("2026-09-18")
    assert g["gas_t1"] == pd.Timestamp("2026-09-16") and g["gas_known_from_t1"] == tau


def test_a_revision_counts_only_from_its_own_known_from():
    revised = settlements(drop={("2026-10", "2026-09-16")}, extra=[
        {"contract": "2026-10", "trade_date": "2026-09-16", "value": 50.0, "known_from": pd.Timestamp("2026-09-16T18:00Z")},
        {"contract": "2026-10", "trade_date": "2026-09-16", "value": 51.0, "known_from": pd.Timestamp("2026-09-17T09:00Z")},
    ])
    gas = GasInputs(revised, CAL, eua=eua())
    assert gas.day_features("2026-09-18")["gas_chg_lag1_eur"] == pytest.approx(50.0 - value("2026-10", "2026-09-15"))
    assert gas.day_features("2026-09-19")["gas_chg_lag1_eur"] == pytest.approx(value("2026-10", "2026-09-17") - 51.0)


def test_value_outside_the_range_makes_the_day_missing():
    bad = settlements(drop={("2026-10", "2026-09-16")}, extra=[
        {"contract": "2026-10", "trade_date": "2026-09-16", "value": 0.5, "known_from": known("2026-09-16")}])
    f = GasInputs(bad, CAL, eua=eua()).day_features("2026-09-18")
    assert f["gas_missing"] and f["gas_missing_reason"] == "out_of_range"


def test_eua_rules_for_candidate_b():
    no_eua = GasInputs(settlements(), CAL)
    f = no_eua.day_features("2026-09-18")
    assert not f["gas_missing"] and f["gas_srmc_missing"] and np.isnan(f["srmc_eur_mwh"])
    # the auction of D-2 published after the issue time is replaced by the previous one
    table = eua()
    table.loc[table["auction_date"] == pd.Timestamp("2026-09-16"), "known_from"] = issue_time("2026-09-18") + pd.Timedelta(minutes=1)
    g = GasInputs(settlements(), CAL, eua=table).day_features("2026-09-18")
    assert g["srmc_eur_mwh"] == pytest.approx((value("2026-10", "2026-09-16") + 0.184 * eua_on_or_before("2026-09-15")) / 0.50)
    assert g["eua_known_from_d2"] <= issue_time("2026-09-18")


# --- hard data checks ---------------------------------------------------------------------------------------------------

def test_a_value_dated_after_expiry_stops_the_branch():
    after = settlements(extra=[{"contract": "2026-11", "trade_date": "2026-10-30", "value": 30.0, "known_from": known("2026-10-30")}])
    with pytest.raises(GasDataError, match="after their contract's expiry"):
        GasInputs(after, CAL)


def test_closure_day_repeat_is_dropped_and_a_different_value_stops_the_branch(full):
    repeat = settlements(extra=[{"contract": "2026-05", "trade_date": "2026-04-03", "value": value("2026-05", "2026-04-02"),
                                 "known_from": known("2026-04-03")}])
    gas = GasInputs(repeat, CAL, eua=eua())
    assert gas.day_features("2026-04-07") == full.day_features("2026-04-07")
    different = settlements(extra=[{"contract": "2026-05", "trade_date": "2026-04-03", "value": 99.0, "known_from": known("2026-04-03")}])
    with pytest.raises(GasDataError, match="non-trading day"):
        GasInputs(different, CAL)


def test_known_from_before_the_settlement_is_final_stops_the_branch():
    early = settlements(extra=[{"contract": "2026-10", "trade_date": "2026-09-16", "value": 1.0 + value("2026-10", "2026-09-16"),
                                "known_from": pd.Timestamp("2026-09-16T15:19Z")}])  # 17:19 Paris
    with pytest.raises(GasDataError, match="before the settlement was final"):
        GasInputs(early, CAL)
    with pytest.raises(GasDataError, match="lack columns"):
        GasInputs(settlements().drop(columns=["known_from"]), CAL)


# --- check 2: perturbation ------------------------------------------------------------------------------------------------

PERTURB_DAYS = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-18", "2026-09-20", "2026-04-07", "2026-04-09",
                "2026-10-31", "2026-11-01", "2026-12-29", "2026-12-30", "2027-01-02"]


def same(a: dict, b: dict) -> bool:
    """Equal key by key: floats to 1e-9 with NaN equal to NaN, everything else exactly."""
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            assert (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9, (k, x, y)
        else:
            assert (x is pd.NaT and y is pd.NaT) or x == y, (k, x, y)
    return True


def _features(f: dict) -> dict:
    return {k: f[k] for k in GAS + ["srmc_eur_mwh", "srmc_ref_eur_mwh", "gas_t1", "gas_r3", "gas_r9", "gas_contract"]}


@pytest.mark.parametrize("day", PERTURB_DAYS)
def test_perturbation_after_d1_or_after_the_issue_time_changes_nothing(full, day):
    rng = np.random.default_rng(1)
    base = settlements()
    d, tau = pd.Timestamp(day), issue_time(day)
    perturbed = base.copy()
    later = pd.to_datetime(perturbed["trade_date"]) >= d - pd.Timedelta(days=1)
    perturbed.loc[later, "value"] = perturbed.loc[later, "value"] + rng.uniform(5, 50, int(later.sum()))
    # revisions of every earlier value, all known one minute after the issue time
    revisions = base[~later].copy()
    revisions["value"] = revisions["value"] + rng.uniform(5, 50, len(revisions))
    revisions["known_from"] = tau + pd.Timedelta(minutes=1)
    perturbed = pd.concat([perturbed, revisions], ignore_index=True)
    table = eua()
    table.loc[table["auction_date"] >= d - pd.Timedelta(days=1), "price_eur_t"] += 30.0
    assert same(_features(GasInputs(perturbed, CAL, eua=table).day_features(day)), _features(full.day_features(day)))
    # power check: the same kind of change to the value at T1* does move the feature
    t1 = full.day_features(day)["gas_t1"].strftime("%Y-%m-%d")
    moved = base.copy()
    moved.loc[moved["trade_date"] == t1, "value"] += 7.0
    assert GasInputs(moved, CAL, eua=eua()).day_features(day)["gas_chg_lag7_eur"] == pytest.approx(full.day_features(day)["gas_chg_lag7_eur"] + 7.0)


# --- checks 3 and 4: hard gate and companion column --------------------------------------------------------------------

def gas_for_synthetic() -> GasInputs:
    """Hand-built settlements covering the synthetic inputs table (September 2023 to April 2024)."""
    contracts = [p.strftime("%Y-%m") for p in pd.period_range("2023-07", "2024-07", freq="M")]
    frame = settlements(first="2023-06-01", last="2024-05-31", contracts=contracts,
                        drop={("2024-01", "2023-12-22"), ("2024-01", "2023-12-27"), ("2024-02", "2024-01-23")})
    return GasInputs(frame, CAL, eua=eua("2023-06-01", "2024-05-31"))


@pytest.fixture(scope="module")
def synth():
    return synthetic_inputs(), gas_for_synthetic()


def test_build_features_needs_gas_for_gas_sets_and_ignores_it_otherwise(synth):
    inputs, gas = synth
    with pytest.raises(ValueError, match="needs the gas data"):
        build_features(inputs, "honest_v2_gas")
    plain, with_gas = build_features(inputs, "honest_v2"), build_features(inputs, "honest_v2", gas=gas)
    pd.testing.assert_frame_equal(plain.X, with_gas.X)
    pd.testing.assert_frame_equal(plain.info, with_gas.info)
    control = build_features(inputs, "honest_v2_t23")
    pd.testing.assert_frame_equal(plain.X, control.X)
    b = build_features(inputs, "honest_v2_gas_srmc", gas=gas)
    assert list(b.X.columns) == HONEST_V2 + GAS + GAS_SRMC
    row = b.X.index[2000]
    assert b.X.loc[row, "spark_lag1_eur"] == pytest.approx(b.X.loc[row, "price_lag1"] - gas.day_features(b.info.loc[row, "delivery_day"])["srmc_ref_eur_mwh"])
    assert "srmc_ref_eur_mwh" not in b.info and {"gas_known_from_t1", "gas_roll_in_window", "gas_contract", "gas_features_missing"} <= set(b.info)
    # days whose gas features are missing (29 December 2023: T1 = 27 Dec and the trading day before, 22 Dec (25 and 26 Dec closed), both absent for Jan24)
    missing = b.info.loc[b.info["delivery_day"] == pd.Timestamp("2023-12-29"), "gas_features_missing"]
    assert missing.all() and b.X.loc[missing.index, GAS].isna().all().all()
    assert build_features(inputs, "honest_v2_gas_rel", gas=gas).target_offset.equals(b.X["price_lag1_mean"])


def test_companion_assertion_rejects_a_value_known_after_the_issue_time(synth):
    inputs, gas = synth
    table = build_features(inputs, "honest_v2_gas", gas=gas)
    assert_known_before_issue(table.info)
    bad = table.info.copy()
    row = bad.index[3000]
    bad.loc[row, "gas_known_from_r3"] = bad.loc[row, "gas_issue_time"] + pd.Timedelta(seconds=1)
    with pytest.raises(LookaheadError, match="gas_known_from_r3"):
        assert_known_before_issue(bad)


@pytest.fixture
def fast_gbm(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)


def test_hard_gate_stops_the_backtest(synth, monkeypatch):
    inputs, gas = synth
    table = build_features(inputs, "honest_v2_gas", gas=gas)
    # a known_from after the issue time stops the run instead of being recorded
    tampered = build_features(inputs, "honest_v2_gas", gas=gas)
    rows = tampered.info["delivery_day"] == pd.Timestamp("2024-03-12")
    tampered.info.loc[rows, "gas_known_from_t1"] = tampered.info.loc[rows, "gas_issue_time"] + pd.Timedelta(hours=1)
    with pytest.raises(LookaheadError):
        walk_forward(tampered, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, train_start="2023-01-01")
    # a feature whose rule passes the 12:00 gate but not the 10:05 pre-market deadline stops it too
    monkeypatch.setitem(timing.TIMINGS, "at_1100", InputTiming("at_1100", lambda idx: timing._local_clock(timing.delivery_days(idx), -1, 11), "test"))
    monkeypatch.setitem(features_module.FEATURE_TIMINGS, "gas_age_days", "at_1100")
    with pytest.raises(LookaheadError, match="pre-market"):
        walk_forward(table, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, train_start="2023-01-01")
    monkeypatch.setitem(features_module.FEATURE_TIMINGS, "gas_age_days", "wind_solar_forecast")
    with pytest.raises(LookaheadError, match="12:00"):
        walk_forward(table, "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None, train_start="2023-01-01")
    # an existing set keeps the old behaviour: recorded, not stopped
    assert not walk_forward(build_features(inputs, "extended"), "2024-03-01", "2024-04-01", models=("linear",), log=lambda *_: None).point_in_time


def test_train_start_and_relative_target(synth):
    inputs, gas = synth
    table = build_features(inputs, "honest_v2", gas=gas)
    days = table.info["delivery_day"]
    bt = walk_forward(table, "2024-02-01", "2024-03-01", models=("linear",), log=lambda *_: None, train_start="2023-12-01")
    expected = int(table.y[(days >= pd.Timestamp("2023-12-01")) & (days < pd.Timestamp("2024-02-01"))].notna().sum())
    assert bt.predictions["train_rows"].iloc[0] == expected
    rel = build_features(inputs, "honest_v2_gas_rel", gas=gas)
    out = walk_forward(rel, "2024-02-01", "2024-03-01", models=("linear",), log=lambda *_: None, train_start="2023-01-01").predictions
    train, test = rel.info["delivery_day"] < pd.Timestamp("2024-02-01"), out.index
    manual = fit_predict("linear", rel.X[train], rel.y[train] - rel.X.loc[train, "price_lag1_mean"], rel.X.loc[test]) + rel.X.loc[test, "price_lag1_mean"].to_numpy()
    np.testing.assert_allclose(out["linear"].to_numpy(), manual)


def test_run_backtest_adds_the_control_and_scores_missing_days_with_it(synth, fast_gbm):
    inputs, gas = synth
    with pytest.raises(ValueError, match="must be trained from 2023-01-01"):
        run_backtest(inputs, "2024-01-01", "2024-02-01", feature_sets=("honest_v2_gas",), gas=gas, log=lambda *_: None)
    results = run_backtest(inputs, "2023-12-01", "2024-02-01", feature_sets=("honest_v2_gas", "honest_v2_gas_srmc"), gas=gas,
                           train_start="2023-01-01", log=lambda *_: None)
    assert list(results) == ["honest_v2_t23", "honest_v2_gas", "honest_v2_gas_srmc"]
    control = results["honest_v2_t23"]["backtest"].predictions
    for name in ("honest_v2_gas", "honest_v2_gas_srmc"):
        preds = results[name]["backtest"].predictions
        assert results[name]["backtest"].point_in_time
        fallback = preds["gas_fallback"]
        assert fallback.any() and not fallback.all()
        assert (preds.loc[fallback, "gbm"] == control.loc[fallback[fallback].index, "gbm"]).all()
        assert (preds.loc[~fallback, "gbm"] != control.loc[fallback[~fallback].index, "gbm"]).any()


# --- check 5: reissue equality ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backtest_tables(synth):
    inputs, gas = synth
    return {name: build_features(inputs, name, gas=gas) for name in ("honest_v2_gas", "honest_v2_gas_srmc", "honest_v2_gas_rel")}


@pytest.mark.parametrize("day", ["2024-03-12", "2024-03-18", "2024-04-02", "2024-01-02"])
def test_reissue_from_the_snapshot_equals_the_backtest_row(synth, backtest_tables, day):
    inputs, gas = synth
    tau = issue_time(day)
    snapshot = gas.settlements[pd.to_datetime(gas.settlements["known_from"], utc=True) <= tau]
    eua_snapshot = gas.eua[pd.to_datetime(gas.eua["known_from"], utc=True) <= tau]
    reissue_gas = GasInputs(snapshot, CAL, eua=eua_snapshot)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    reissue_inputs = inputs[inputs.index < end]
    for name in ("honest_v2_gas", "honest_v2_gas_srmc", "honest_v2_gas_rel"):
        backtest = backtest_tables[name]
        reissue = build_features(reissue_inputs, name, gas=reissue_gas)
        rows = backtest.info["delivery_day"] == pd.Timestamp(day)
        pd.testing.assert_frame_equal(backtest.X[rows], reissue.X.loc[backtest.X.index[rows]])
        pd.testing.assert_frame_equal(backtest.info[rows], reissue.info.loc[backtest.info.index[rows]])


# --- the command line -----------------------------------------------------------------------------------------------------

def test_backtest_command_with_gas_directory(tmp_path, monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    data = tmp_path / "forecast"
    data.mkdir()
    synthetic_inputs().to_csv(data / "inputs.csv", index_label="timestamp_utc")
    gas_dir = tmp_path / "gas"
    gas_dir.mkdir()
    gas = gas_for_synthetic()
    gas.settlements.to_csv(gas_dir / "settlements.csv", index=False)
    pd.DataFrame({"date": default_ice_closures(range(2022, 2028))}).to_csv(gas_dir / "ice_closures.csv", index=False)
    gas.eua.to_csv(gas_dir / "eua.csv", index=False)
    loaded = load_gas_dir(gas_dir)
    assert same(loaded.day_features("2024-03-12"), gas.day_features("2024-03-12"))
    out = tmp_path / "reports"
    cli.main(["forecast-backtest", "--data", str(data / "inputs.csv"), "--out", str(out), "--test-start", "2024-03-01",
              "--test-end", "2024-04-01", "--feature-sets", "honest_v2_gas_rel", "--gas", str(gas_dir), "--train-start", "2023-01-01"])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) >= {"honest_v2_t23", "honest_v2_gas_rel"} and summary["honest_v2_gas_rel"]["point_in_time"] is True
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "ttf-gas C" in text and "the ttf-gas control" in text
    predictions = pd.read_csv(out / "predictions_honest_v2_gas_rel.csv")
    assert "gas_known_from_t1" in predictions and not {"srmc_ref_eur_mwh", "gas_chg_lag1_eur"} & set(predictions)
