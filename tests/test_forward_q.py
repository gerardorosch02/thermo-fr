"""The forward log of Q: what the entry point may read, the write-once record, the manifest chain and the end-date lock."""

import json

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import forward_q as fq
from thermo_fr.forecast.french_rl import HONEST_START, OPERATOR_COLUMNS
from thermo_fr.forecast.inputs import INPUT_COLUMNS


def _table(start="2024-04-25", end="2024-05-10"):
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1h", inclusive="left")
    return pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)


def test_expected_days_are_the_registered_hundred():
    days = fq.expected_days()
    assert len(days) == 100 and days[0] == pd.Timestamp("2026-12-01") and days[-1] == pd.Timestamp("2027-04-17")
    assert set(days.dayofweek) == {1, 2, 3, 4, 5}


def test_operator_forecast_is_kept_only_before_the_honest_start():
    table = _table()
    tau = pd.Timestamp("2024-05-09 08:05", tz="UTC")
    known = fq.as_known_q(table, tau)
    days = fq.delivery_days(known.index)
    for column in OPERATOR_COLUMNS:
        assert known.loc[np.asarray(days < HONEST_START), column].notna().all()
        assert known.loc[np.asarray(days >= HONEST_START), column].isna().all()


def test_operator_forecast_published_after_tau_stays_blank():
    table = _table("2024-04-20", "2024-04-30")
    tau = pd.Timestamp("2024-04-27 08:05", tz="UTC")  # 10:05 Paris on 27 April: the forecast of the 28th is due at 18:00 that day
    known = fq.as_known_q(table, tau)
    days = fq.delivery_days(known.index)
    assert known.loc[np.asarray(days >= pd.Timestamp("2024-04-28")), "solar_fc_mw"].isna().all()
    assert known.loc[np.asarray(days <= pd.Timestamp("2024-04-27")), "solar_fc_mw"].notna().all()


def test_day_hours_follow_the_clock_changes():
    assert len(fq.day_hours("2026-10-25")) == 25 and len(fq.day_hours("2027-03-28")) == 23 and len(fq.day_hours("2026-12-01")) == 24


def test_record_is_written_once(tmp_path):
    record = {"delivery_day": "2026-12-01", "morning_run_id": 7, "issued_at_utc": "2026-11-30T09:20:00Z", "forecasts": {}}
    path = fq.write_record(tmp_path, record)
    assert json.loads(path.read_text())["morning_run_id"] == 7
    with pytest.raises(FileExistsError):
        fq.write_record(tmp_path, record)


def test_manifest_chain_shows_an_edit_and_a_changed_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "2026-12-01_run000001_20261130T091500Z.csv").write_text("a,b\n1,2\n")
    (fresh / "2026-12-01_run000001_20261130T091500Z.models.json").write_text("{}")
    first = fq.capture(fresh, tmp_path / "out")
    assert first["added"] == 2 and fq.capture(fresh, tmp_path / "out")["added"] == 0
    check = fq.verify_manifest(tmp_path / "out" / "manifest.jsonl")
    assert check["chain_ok"] and not check["changed"] and not check["missing"] and check["entries"] == 2
    (fresh / "2026-12-01_run000001_20261130T091500Z.csv").write_text("a,b\n1,3\n")
    assert fq.verify_manifest(tmp_path / "out" / "manifest.jsonl")["changed"] == ["fresh/2026-12-01_run000001_20261130T091500Z.csv"]
    assert fq.capture(fresh, tmp_path / "out")["changed_since_listed"] == ["fresh/2026-12-01_run000001_20261130T091500Z.csv"]
    manifest = tmp_path / "out" / "manifest.jsonl"
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join([lines[0].replace('"bytes": 8', '"bytes": 9'), lines[1]]) + "\n")
    assert not fq.verify_manifest(manifest)["chain_ok"]


def test_outcomes():
    assert fq.outcome({"upper_one_sided_95": -0.01, "lower_one_sided_95": -0.4}, 95, 1.0) == "supported"
    assert fq.outcome({"upper_one_sided_95": 0.3, "lower_one_sided_95": 0.01}, 95, 1.0) == "contradicted"
    assert fq.outcome({"upper_one_sided_95": 0.1, "lower_one_sided_95": -0.4}, 95, 1.0) == "inconclusive"
    assert fq.outcome({"upper_one_sided_95": -0.01, "lower_one_sided_95": -0.4}, 89, 1.0) == "inconclusive"
    assert fq.outcome({"upper_one_sided_95": -0.01, "lower_one_sided_95": -0.4}, 95, 0.85) == "inconclusive"


def test_statistics_are_locked_before_the_end_date(tmp_path):
    with pytest.raises(PermissionError):
        fq.score(tmp_path, db=tmp_path / "none.db", today="2027-02-01")


def test_chosen_record_is_the_last_one_before_the_window():
    records = [{"delivery_day": "2026-12-01", "issued_at_utc": "2026-11-30T09:20:00Z", "before_window": True, "flags": []},
               {"delivery_day": "2026-12-01", "issued_at_utc": "2026-11-30T09:50:00Z", "before_window": True, "flags": []},
               {"delivery_day": "2026-12-01", "issued_at_utc": "2026-11-30T10:40:00Z", "before_window": False, "flags": ["late"]}]
    assert fq.chosen_records(records)["2026-12-01"]["issued_at_utc"] == "2026-11-30T09:50:00Z"


def test_switch_takes_the_tree_at_or_above_the_threshold():
    index = pd.date_range("2026-12-01", periods=24, freq="h", tz="UTC")
    g, l0 = pd.Series(100.0, index=index), pd.Series(60.0, index=index)
    assert fq.SWITCH_DISTANCE == 1.6
    below, branch = fq.switch_forecast(g, l0, 1.5999)
    assert branch == "ensemble" and (below == 80.0).all()
    at, branch = fq.switch_forecast(g, l0, 1.6)
    assert branch == "tree" and (at == g).all()
    above, branch = fq.switch_forecast(g, l0, 3.2)
    assert branch == "tree" and (above == g).all()


def test_switch_falls_back_to_the_tree_when_an_input_is_missing():
    index = pd.date_range("2026-12-01", periods=24, freq="h", tz="UTC")
    g, l0 = pd.Series(100.0, index=index), pd.Series(60.0, index=index)
    for lear_curve, distance in ((None, 0.4), (l0, None), (l0, float("nan"))):
        out, branch = fq.switch_forecast(g, lear_curve, distance)
        assert branch == "fallback" and (out == g).all()

