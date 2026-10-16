"""The ttf-gas calendar from sourced and default closures, the raw DBnomics loader, the EUA known_from rule, and the
pre-feature checks of `thermo-fr gas-checks`, on hand-built data only (no real gas or EEX values)."""

import json

import numpy as np
import pandas as pd
import pytest

from thermo_fr import cli
from thermo_fr.forecast import gas_checks
from thermo_fr.forecast.gas import (
    Calendars,
    GasDataError,
    GasInputs,
    contract_from_name,
    default_ice_closures,
    eua_known_from,
    ice_closure_table,
    known_from_secondary,
    load_dbnomics_json,
    load_gas_dir,
    read_gas_dir,
    settlement_issues,
    settlements_from_dbnomics,
)

# the scout's sourced closures (ICE circular E24027 and the 2026 trading schedule): public calendar dates, no prices
SOURCED = ["2024-12-25", "2024-12-26", "2025-01-01", "2026-01-01", "2026-04-03", "2026-04-06", "2026-12-25"]
YEARS = range(2021, 2028)
TABLE = ice_closure_table(SOURCED, YEARS)
CAL = Calendars.from_closure_table(TABLE, YEARS)
BASE = pd.Timestamp("2020-01-01")
LISTED_DAYS = 120  # each synthetic contract trades for the last 120 calendar days before its expiry


def value(contract, day) -> float:
    """Encodes contract and date, and moves by up to about 1.5 EUR/MWh a day so the sign test of the agreement check has days."""
    months = (pd.Period(contract, "M") - pd.Period("2020-01", "M")).n
    days = (pd.Timestamp(day) - BASE).days
    return 20 + months / 10 + 0.01 * days + 3 * np.sin(days / 2.0)


def synthetic_settlements(first="2022-08-01", last="2026-10-15", drop=(), cal=CAL) -> pd.DataFrame:
    rows = []
    for c in [p.strftime("%Y-%m") for p in pd.period_range("2022-11", "2027-03", freq="M")]:
        expiry = pd.Timestamp(cal.expiry(c))
        for d in pd.bdate_range(max(pd.Timestamp(first), expiry - pd.Timedelta(days=LISTED_DAYS)), min(pd.Timestamp(last), expiry)):
            if cal.is_trading(d) and (c, d.strftime("%Y-%m-%d")) not in drop:
                rows.append({"contract": c, "trade_date": d.normalize(), "value": round(value(c, d), 3)})
    return pd.DataFrame(rows)


def dbnomics_pages(frame: pd.DataFrame, na_on=()) -> list[dict]:
    """Two raw pages in the v22 series format (observations=1), contract named in the series name, some 'NA' values."""
    docs = []
    for i, (contract, part) in enumerate(frame.groupby("contract")):
        name = pd.Period(contract, "M").strftime("%b%y")
        periods = [d.strftime("%Y-%m-%d") for d in part["trade_date"]]
        values = ["NA" if p in na_on else v for p, v in zip(periods, part["value"])]
        docs.append({"series_code": f"D.{5460494 + i}", "series_name": f"Dutch TTF Gas Futures - {name}", "period": periods,
                     "value": values, "indexed_at": "2026-10-16T05:30:00Z"})
    half = len(docs) // 2
    return [{"series": {"docs": docs[:half], "num_found": len(docs)}}, {"series": {"docs": docs[half:], "num_found": len(docs)}}]


def eex_table(frame: pd.DataFrame, noise=0.02, shift_days=0, offset=0.0, seed=3) -> pd.DataFrame:
    """A stand-in for the owner's EEX snapshot: Nov26 from 2026-09-01, DBnomics values plus small noise."""
    rng = np.random.default_rng(seed)
    part = frame[(frame["contract"] == "2026-11") & (frame["trade_date"] >= "2026-09-01")].copy()
    part["value"] = part["value"].shift(shift_days) + offset + rng.uniform(-noise, noise, len(part))
    return part.dropna()


def write_dir(tmp_path, frame=None, eua=True, expiries=True, extra_raw=(), na_on=()):
    frame = synthetic_settlements() if frame is None else frame
    gas_dir = tmp_path / "gas"
    (gas_dir / "dbnomics").mkdir(parents=True)
    pages = dbnomics_pages(pd.concat([frame, pd.DataFrame(list(extra_raw))], ignore_index=True) if extra_raw else frame, na_on)
    for i, page in enumerate(pages):
        (gas_dir / "dbnomics" / f"page_{i}.json").write_text(json.dumps(page))
    pd.DataFrame({"date": SOURCED, "source_url": "test"}).to_csv(gas_dir / "ice_closures.csv", index=False)
    if expiries:
        rows = [{"contract_month": c, "last_trading_day": pd.Timestamp(CAL.rule_expiry(c)).strftime("%Y-%m-%d"),
                 "source": "ICE published expiry calendar" if c >= "2026-11" else "RULE-DERIVED"}
                for c in sorted(frame["contract"].unique())]
        pd.DataFrame(rows).to_csv(gas_dir / "ice_expiries.csv", index=False)
    if eua:
        days = pd.bdate_range("2022-08-01", "2026-10-15")
        pd.DataFrame({"auction_date": days.strftime("%Y-%m-%d"), "price_eur_t": 70 + 0.1 * (days - BASE).days}).to_csv(gas_dir / "eua.csv", index=False)
    eex = tmp_path / "eex.csv"
    eex_table(frame).to_csv(eex, index=False)
    return gas_dir, eex


# --- calendar --------------------------------------------------------------------------------------------------------------

def test_closure_table_flags_sourced_and_default_days():
    flags = dict(zip(TABLE["date"].dt.strftime("%Y-%m-%d"), TABLE["sourced"]))
    assert all(flags[d] for d in SOURCED)
    # 2023 is unsourced: the default pattern, weekdays only (26 Dec 2023 is a Tuesday, 25 Dec a Monday, 1 Jan a Sunday)
    assert flags["2023-04-07"] is False and flags["2023-04-10"] is False and flags["2023-12-25"] is False and flags["2023-12-26"] is False
    assert "2023-01-01" not in flags and "2022-12-26" in flags and "2022-12-27" not in flags  # no substitute days
    # inside a sourced span the default does not apply: 2026 has no 26 Dec (a Saturday) and nothing beyond the schedule
    assert not any(d.startswith("2026") and not flags[d] for d in flags)
    # 2025 after 1 January is unsourced: Good Friday 18 April 2025 is a default closure
    assert flags["2025-04-18"] is False and not CAL.is_trading("2025-04-18") and CAL.is_default_closure("2025-04-18")
    assert not CAL.is_default_closure("2026-04-03") and not CAL.is_trading("2026-04-03")
    assert default_ice_closures([2026]) == [pd.Timestamp(d) for d in ("2026-01-01", "2026-04-03", "2026-04-06", "2026-12-25")]


def test_ice_published_expiry_wins_over_the_rule():
    cal = Calendars.from_closure_table(TABLE, YEARS, expiry_overrides={"2026-11": "2026-10-28"})
    assert pd.Timestamp(cal.expiry("2026-11")) == pd.Timestamp("2026-10-28")
    assert pd.Timestamp(cal.rule_expiry("2026-11")) == pd.Timestamp("2026-10-29") and cal.front("2026-10-29") == "2026-12"


def test_values_on_default_closures_are_dropped_and_listed_not_a_stop():
    frame = synthetic_settlements(first="2025-03-01", last="2025-05-30")
    frame["known_from"] = known_from_secondary(frame["trade_date"]).to_numpy()
    extra = pd.DataFrame([{"contract": "2025-05", "trade_date": pd.Timestamp("2025-04-18"), "value": 99.0,
                           "known_from": pd.Timestamp("2025-04-19T07:00Z")}])
    clean, issues = settlement_issues(pd.concat([frame, extra]), CAL)
    assert len(issues["default_closure_values"]) == 1 and not len(issues["non_trading_not_repeat"])
    assert not issues["default_closure_values"]["repeat"].iloc[0]  # 99.0 differs from the previous settlement
    assert pd.Timestamp("2025-04-18") not in set(clean["trade_date"])
    GasInputs(pd.concat([frame, extra]), CAL)  # no GasDataError
    # the same differing value on a sourced closure stops the branch
    sourced = pd.DataFrame([{"contract": "2026-05", "trade_date": pd.Timestamp("2026-04-03"), "value": 99.0,
                             "known_from": pd.Timestamp("2026-04-04T07:00Z")}])
    later = synthetic_settlements(first="2026-03-01", last="2026-04-30")
    later["known_from"] = known_from_secondary(later["trade_date"]).to_numpy()
    with pytest.raises(GasDataError, match="non-trading day"):
        GasInputs(pd.concat([later, sourced]), CAL)


# --- EUA and the raw DBnomics loader -------------------------------------------------------------------------------------

def test_eua_known_from_is_mandatory_and_follows_the_rule():
    rule = eua_known_from(["2026-07-15", "2026-01-15"])
    assert list(rule) == [pd.Timestamp("2026-07-15T11:00Z"), pd.Timestamp("2026-01-15T12:00Z")]  # 13:00 Paris
    later = eua_known_from(["2026-07-15", "2026-07-16"], ["2026-07-15T18:00Z", "2026-07-16T08:00Z"])
    assert list(later) == [pd.Timestamp("2026-07-15T18:00Z"), pd.Timestamp("2026-07-16T11:00Z")]  # a fetch stamp counts only if later
    frame = synthetic_settlements(first="2026-08-01", last="2026-09-30")
    frame["known_from"] = known_from_secondary(frame["trade_date"]).to_numpy()
    with pytest.raises(GasDataError, match="known_from"):
        GasInputs(frame, CAL, eua=pd.DataFrame({"auction_date": ["2026-09-01"], "price_eur_t": [70.0]}))


def test_contract_month_from_series_names():
    assert contract_from_name("Dutch TTF Gas Futures - Dec22") == "2022-12"
    assert contract_from_name("TTF December 2022") == "2022-12" and contract_from_name("Mar 23") == "2023-03"
    assert contract_from_name("Market data") is None and contract_from_name("Dec22 vs Jan23") is None


def test_load_dbnomics_json_and_known_from(tmp_path):
    frame = synthetic_settlements(first="2026-07-01", last="2026-08-31")
    pages = dbnomics_pages(frame, na_on={"2026-07-15"})
    paths = []
    for i, page in enumerate(pages):
        paths.append(tmp_path / f"p{i}.json")
        paths[-1].write_text(json.dumps(page))
    raw = load_dbnomics_json(paths)
    assert set(raw.columns) == {"series_code", "series_name", "contract", "trade_date", "value"}
    assert len(raw) == len(frame) - int((frame["trade_date"] == "2026-07-15").sum())  # 'NA' skipped
    assert raw["contract"].min() >= "2026-07" and pd.Timestamp("2026-07-15") not in set(raw["trade_date"])
    s = settlements_from_dbnomics(raw)
    assert s.loc[0, "known_from"] == known_from_secondary([s.loc[0, "trade_date"]])[0]
    assert known_from_secondary(["2026-09-16"])[0] == pd.Timestamp("2026-09-17T07:00Z")  # 09:00 Paris the next day
    # git-history versions: accepted when the raw value is the latest version, refused otherwise
    history = s.copy()
    history["known_from"] = pd.to_datetime(history["known_from"], utc=True) - pd.Timedelta(hours=10)
    revised = history.iloc[[0]].assign(value=history.iloc[0]["value"] + 1, known_from=history.iloc[0]["known_from"] - pd.Timedelta(hours=1))
    both = pd.concat([revised, history], ignore_index=True)
    assert len(settlements_from_dbnomics(raw, both)) == len(raw) + 1
    with pytest.raises(GasDataError, match="differ from its latest version"):
        settlements_from_dbnomics(raw, history.assign(value=history["value"] + 0.001))
    # a series without a month in its name needs series_map
    pages[0]["series"]["docs"][0]["series_name"] = "Dutch TTF Gas Futures"
    paths[0].write_text(json.dumps(pages[0]))
    with pytest.raises(GasDataError, match="series_map"):
        load_dbnomics_json(paths)
    code = pages[0]["series"]["docs"][0]["series_code"]
    contract = sorted(frame["contract"].unique())[0]
    assert len(load_dbnomics_json(paths, {code: contract})) == len(raw)
    # two series giving different values for one contract and day
    clash = {"series": {"docs": [dict(pages[0]["series"]["docs"][0], series_code="D.1", series_name="X " + pd.Period(contract, "M").strftime("%b%y"),
                                      value=[v + 1 if v != "NA" else v for v in pages[0]["series"]["docs"][0]["value"]])]}}
    (tmp_path / "p9.json").write_text(json.dumps(clash))
    with pytest.raises(GasDataError, match="different values"):
        load_dbnomics_json(paths + [tmp_path / "p9.json"], {code: contract})


def test_read_gas_dir_assembles_calendar_expiries_and_eua(tmp_path):
    gas_dir, _ = write_dir(tmp_path, frame=synthetic_settlements(first="2026-06-01", last="2026-10-15"))
    parts = read_gas_dir(gas_dir)
    assert parts["calendars"].expiry_overrides["2026-11"] == np.datetime64("2026-10-29")
    assert parts["eua"]["known_from"].notna().all()
    assert pd.Timestamp(parts["eua"]["known_from"].iloc[0]) == eua_known_from([parts["eua"]["auction_date"].iloc[0]])[0]
    gas = load_gas_dir(gas_dir)
    assert not gas.day_features("2026-09-18")["gas_srmc_missing"]


def test_eex_snapshot_format(tmp_path):
    path = tmp_path / "20261016T080000Z_gas_front_month_table.json"
    header = ["shortCode", "maturityDate", "tradeDate", "totVolTrdd", "grossOpenInt", "grossOpenIntSz", "netOpenInt", "netOpenIntSz", "settlPx"]
    data = [["X", 202611, "2026-09-01", 1, 1, 1, None, None, 30.5], ["X", 20261201, "2026-09-01", 1, 1, 1, None, None, 31.0],
            ["X", 202611, "2026-09-02", 1, 1, 1, None, None, None]]
    path.write_text(json.dumps({"header": header, "data": data, "uOM": "MWh", "currency": "EUR"}))
    table = gas_checks.load_eex_tables([path])
    assert list(table["contract"]) == ["2026-11", "2026-12"] and list(table["value"]) == [30.5, 31.0]


# --- the agreement check ----------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dbn():
    frame = synthetic_settlements(first="2026-06-01", last="2026-10-15")
    rng = np.random.default_rng(5)
    frame["value"] = frame["value"] + np.cumsum(rng.normal(0, 1.0, len(frame)))  # moves large enough for the sign test
    return frame


def test_agreement_passes_on_close_series(dbn):
    ok, lines = gas_checks.agreement(dbn, eex_table(dbn), "2026-10-16")
    assert ok, lines
    assert any("five largest level differences" in line for line in lines)


@pytest.mark.parametrize("kwargs,failing", [
    ({"offset": 0.3}, "levels MAE"),
    ({"shift_days": 1, "noise": 0.0}, "lag 0 above"),
    ({"noise": 0.4}, "daily changes MAE"),
])
def test_agreement_fails_on_level_dating_or_noise(dbn, kwargs, failing):
    ok, lines = gas_checks.agreement(dbn, eex_table(dbn, **kwargs), "2026-10-16")
    assert not ok and any(line.startswith("FAIL") and failing in line for line in lines), lines


def test_agreement_needs_thirty_common_trade_dates(dbn):
    ok, lines = gas_checks.agreement(dbn, eex_table(dbn), "2026-10-05")
    assert not ok and "too few" in lines[-1]


# --- the command end to end -------------------------------------------------------------------------------------------------

def run(gas_dir, eex, capsys, extra=()):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["gas-checks", "--gas", str(gas_dir), "--eex", str(eex), "--check-date", "2026-10-16", *extra])
        raise SystemExit(0)
    return stopped.value.code, capsys.readouterr().out


@pytest.fixture(scope="module")
def base_frame():
    return synthetic_settlements()


def test_clean_directory_passes_every_check(tmp_path, capsys, base_frame):
    gas_dir, eex = write_dir(tmp_path, base_frame)
    code, out = run(gas_dir, eex, capsys)
    assert code == 0, out
    assert "Stop rules breached: none" in out and "Candidates withdrawn: none" in out
    assert "[ok] M1" in out and "[ok] M3" in out and "[ok] agreement with the EEX settlements" in out
    assert "2026-11   2026-10-29" in out  # the expiry listing


def test_value_after_expiry_and_missing_days_stop(tmp_path, capsys, base_frame):
    late = [{"contract": "2026-10", "trade_date": pd.Timestamp("2026-09-30"), "value": 40.0}]
    gas_dir, eex = write_dir(tmp_path, base_frame, extra_raw=late)
    code, out = run(gas_dir, eex, capsys)
    assert code == 1 and "[STOP] hard: after_expiry" in out and "2026-10 2026-09-30" in out
    assert "M2 and M3" in out and "not computed" in out


def test_missing_runs_and_m1_stop(tmp_path, capsys, base_frame):
    cov = gas_checks.front_coverage(base_frame, CAL, *gas_checks.M1_SPAN)
    drop_days = cov["day"].iloc[::14].tolist()  # about 7% of trading days, no runs
    runs = [cov["day"].iloc[i:i + 3].tolist() for i in range(100, 700, 100)]  # six runs of three
    drop = {(cov.loc[cov["day"] == d, "front"].iloc[0], d.strftime("%Y-%m-%d")) for d in drop_days + sum(runs, [])}
    frame = base_frame[[(c, d.strftime("%Y-%m-%d")) not in drop for c, d in zip(base_frame["contract"], base_frame["trade_date"])]]
    gas_dir, eex = write_dir(tmp_path, frame)
    code, out = run(gas_dir, eex, capsys)
    assert code == 1 and "[STOP] M1" in out and "[STOP] runs of missing front-month days" in out
    assert "6 runs of 3 or more" in out


def test_identical_runs_and_expiry_file_mismatch_stop(tmp_path, capsys, base_frame):
    frame = base_frame.copy()
    for contract in ("2023-03", "2023-06", "2023-09", "2023-12", "2024-03", "2024-06"):
        rows = frame.index[frame["contract"] == contract][10:13]
        frame.loc[rows, "value"] = frame.loc[rows[0], "value"]
    gas_dir, eex = write_dir(tmp_path, frame)
    expiries = pd.read_csv(gas_dir / "ice_expiries.csv")
    expiries.loc[expiries["contract_month"] == "2023-05", "last_trading_day"] = "2023-04-26"
    expiries.to_csv(gas_dir / "ice_expiries.csv", index=False)
    code, out = run(gas_dir, eex, capsys)
    assert code == 1 and "[STOP] identical consecutive settlements" in out and "6 runs" in out
    assert "[STOP] expiry file against the rule" in out and "2023-05: file 2023-04-26" in out


def test_without_eua_candidate_b_is_withdrawn_only(tmp_path, capsys, base_frame):
    gas_dir, eex = write_dir(tmp_path, base_frame, eua=False)
    code, out = run(gas_dir, eex, capsys)
    assert code == 2, out
    assert "[WITHDRAW B] M2 selection, B (honest_v2_gas_srmc)" in out and "[ok] M2 selection, A (honest_v2_gas)" in out
    assert "Candidates withdrawn: B" in out


def test_agreement_failure_and_load_error_stop(tmp_path, capsys, base_frame):
    gas_dir, _ = write_dir(tmp_path, base_frame)
    shifted = tmp_path / "eex_shifted.csv"
    rng = np.random.default_rng(2)
    noisy = base_frame.copy()
    noisy["value"] = noisy["value"] + rng.normal(0, 0.5, len(noisy))
    eex_table(noisy, shift_days=1, noise=0.0).to_csv(shifted, index=False)
    code, out = run(gas_dir, shifted, capsys)
    assert code == 1 and "[STOP] agreement with the EEX settlements" in out
    page = json.loads((gas_dir / "dbnomics" / "page_0.json").read_text())
    doc = page["series"]["docs"][-1]
    page["series"]["docs"].append(dict(doc, series_code="D.1", value=[v + 1 if v != "NA" else v for v in doc["value"]]))
    (gas_dir / "dbnomics" / "page_0.json").write_text(json.dumps(page))
    code, out = run(gas_dir, shifted, capsys)  # two series give different values for one contract: loading stops
    assert code == 1 and "[STOP] loading the gas directory" in out and "different values" in out


def test_m2_counts_strict_days_per_candidate(base_frame):
    frame = base_frame.copy()
    frame["known_from"] = known_from_secondary(frame["trade_date"]).to_numpy()
    days = pd.bdate_range("2022-08-01", "2026-10-15")
    eua = pd.DataFrame({"auction_date": days, "price_eur_t": 70.0, "known_from": eua_known_from(days).to_numpy()})
    gas = GasInputs(frame, CAL, eua=eua)
    strict = pd.date_range("2024-03-01", "2024-03-31")
    checks = gas_checks.check_m2(gas, strict)
    assert [c.name for c in checks][:2] == ["M2 selection, A (honest_v2_gas) and C (honest_v2_gas_rel)", "M2 selection, B (honest_v2_gas_srmc)"]
    assert "0 of 31 strict delivery days" in checks[0].lines[0] and all(c.ok for c in checks[:2])
    assert checks[2].lines[0].startswith("0 of 0") and not checks[2].ok  # no strict holdout day given: counted as a breach


def test_research_mode_substitutes_and_skips_m3_and_agreement(tmp_path, capsys, base_frame):
    cov = gas_checks.front_coverage(base_frame, CAL, "2024-03-01", "2024-03-29")
    gap = {(r.front, r.day.strftime("%Y-%m-%d")) for r in cov.itertuples()}  # every front-month value of March 2024 missing
    frame = base_frame[[(c, d.strftime("%Y-%m-%d")) not in gap for c, d in zip(base_frame["contract"], base_frame["trade_date"])]]
    gas_dir, eex = write_dir(tmp_path, frame)
    page = json.loads((gas_dir / "dbnomics" / "page_0.json").read_text())
    page["series"]["docs"].append({"series_code": "D.9", "series_name": "Daily – Summer23", "period": ["2023-01-03"], "value": [30.0]})
    (gas_dir / "dbnomics" / "page_0.json").write_text(json.dumps(page))
    code, out = run(gas_dir, eex, capsys, extra=("--research",))
    assert code == 0, out
    assert "1 other series (quarters, seasons, years, strips) skipped" in out
    assert "M3\n    does not apply" in out and "not applicable to the research measurement" in out
    assert "substituted days 20 of 989" in out and "next-contract substitution on" in out
    assert "M2 selection, R1 (research_ttf_ideal_chg) and R2 (research_ttf_ideal_rel)" in out and "M2 holdout" not in out
    code, out = run(gas_dir, eex, capsys)  # the v4 checks on the same gap: a run of 20 missing front-month days, no substitution
    assert code == 0 and "2024-03-01 to 2024-03-28: 20 days, front 2024-04" in out and "1 runs of 3 or more" in out


def test_default_closure_values_count_against_the_calendar_only_when_they_differ():
    frame = synthetic_settlements(first="2023-03-01", last="2025-05-30")
    frame["known_from"] = known_from_secondary(frame["trade_date"]).to_numpy()
    defaults = [d for d in TABLE.loc[~TABLE["sourced"], "date"] if pd.Timestamp("2023-03-01") <= d <= pd.Timestamp("2025-05-30")]
    previous = frame.sort_values("trade_date").groupby("contract").apply(
        lambda part: part.set_index("trade_date")["value"], include_groups=False)

    def closure_rows(differ: bool) -> pd.DataFrame:
        rows = []
        for day in defaults:
            live = [c for c in sorted(frame["contract"].unique()) if CAL.expiry(c) >= np.datetime64(day.date())
                    and (previous[c].index < day).any()][:2]
            for contract in live:
                before = previous[contract][previous[contract].index < day]
                rows.append({"contract": contract, "trade_date": day, "value": before.iloc[-1] + (1.0 if differ else 0.0),
                             "known_from": pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1, hours=6)})
        return pd.DataFrame(rows)

    for differ, evidence in ((False, False), (True, True)):
        extra = closure_rows(differ)
        assert len(extra) > gas_checks.DEFAULT_CLOSURE_EVIDENCE
        _, issues = settlement_issues(pd.concat([frame, extra], ignore_index=True), CAL)
        assert len(issues["default_closure_values"]) == len(extra)
        assert bool(issues["default_closure_values"]["repeat"].all()) is (not differ)
        text = "\n".join(next(c for c in gas_checks.check_settlement_issues(issues) if c.name.startswith("soft")).lines)
        assert ("evidence against the default calendar" in text and "no evidence" not in text) is evidence
        assert (f"{len(extra)} repeat the previous settlement" in text) is (not differ)
