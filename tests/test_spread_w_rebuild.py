"""The January rebuild of the test of W. Unit tests need no data. The end-to-end tests replay the last days of the opened
window (design data, to 2026-10-10) from the static folder named by SPREAD_LIVE_ROOT and are skipped without it. No price of
the test window is read and no error against one is computed."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import forward_q as fq
from thermo_fr.forecast import spread_live as sl
from thermo_fr.forecast import spread_w_test as swt
from thermo_fr.forecast.inputs import INPUT_COLUMNS, load_inputs
from thermo_fr.rebuild import spread_w_rebuild as rb
from thermo_fr.rebuild import spread_w_score as score

ROOT = os.environ.get("SPREAD_LIVE_ROOT")
needs_data = pytest.mark.skipif(not ROOT or not Path(ROOT, "static", "reference.json").exists(), reason="set SPREAD_LIVE_ROOT to a folder with static/")


def _window_file(folder: Path, day: str, stamp: str, values: dict | None = None) -> Path:
    index = pd.date_range(pd.Timestamp(day).tz_localize("Europe/Paris") - pd.Timedelta(days=3), periods=96, freq="h").tz_convert("UTC")
    table = pd.DataFrame(np.nan, index=index, columns=INPUT_COLUMNS)
    for column, value in (values or {}).items():
        table[column] = value
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day}_spread_{stamp}.csv"
    table.to_csv(path, index_label="timestamp_utc")
    return path


def test_the_logger_code_hash_does_not_cover_the_rebuild_package():
    assert not any("rebuild" in str(path) for path in sl.code_files())


def test_the_eve_window_is_the_latest_one_saved_on_the_eve_by_10_05_paris(tmp_path):
    live = tmp_path / "live"
    _window_file(live / "french_windows", "2026-11-23", "20261121T090000Z")  # two days before: not the eve
    _window_file(live / "french_windows", "2026-11-23", "20261122T070000Z")  # 08:00 Paris on the eve
    wanted = _window_file(live / "french_windows", "2026-11-23", "20261122T090320Z")  # 10:03 Paris on the eve
    _window_file(live / "french_windows", "2026-11-23", "20261122T090600Z")  # 10:06 Paris: after the gate
    _window_file(live / "french_windows", "2026-11-24", "20261122T090000Z")  # another delivery day
    src = rb.Sources(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, live=live)
    assert rb.eve_window(src, pd.Timestamp("2026-11-23")) == wanted
    assert rb.eve_window(src, pd.Timestamp("2026-11-25")) is None


def test_only_the_load_forecast_and_the_nuclear_output_come_from_a_saved_window(tmp_path):
    path = _window_file(tmp_path, "2026-11-23", "20261122T090000Z", {"load_fc_mw": 50000.0, "price_eur_mwh": 1.0, "temp_fc_c": 99.0})
    french = load_inputs(path).astype(float)
    french[:] = 7.0
    out = rb.with_window(french, path, pd.Timestamp("2026-11-23"))
    day = rb._days(out)
    assert (out.loc[day <= pd.Timestamp("2026-11-23"), "load_fc_mw"] == 50000.0).all()  # the saved value wins
    assert (out.loc[day > pd.Timestamp("2026-11-23"), "load_fc_mw"] == 7.0).all()  # nothing after the delivery day
    assert (out["nuclear_mw"] == 7.0).all()  # the window holds none: the refetched value stays
    assert (out["price_eur_mwh"] == 7.0).all() and (out["temp_fc_c"] == 7.0).all()  # prices and weather are never taken from a window


def test_extra_rows_have_no_target_and_keep_the_refetched_rows_untouched():
    hours = pd.date_range("2026-11-22T23:00Z", periods=48, freq="h")
    frame = pd.DataFrame({"delivery_day": [pd.Timestamp("2026-11-23")] * 24 + [pd.Timestamp("2026-11-24")] * 24, "hour": list(range(24)) * 2, "fr": 1.0, "de": 2.0, "spread": -1.0}, index=hours)
    sets = {"H": pd.DataFrame({"a": np.arange(48.0)}, index=hours), "G": pd.DataFrame({"a": np.arange(48.0), "b": 0.0}, index=hours)}
    feat = {"frame": frame, "y": frame["spread"], "sets": sets}
    saved_inputs = {name: table.iloc[:24] + 100 for name, table in sets.items()}
    wide = rb.augmented(feat, {pd.Timestamp("2026-11-23"): saved_inputs})
    assert len(wide["frame"]) == 72 and wide["frame"].index.is_unique
    extra = wide["frame"].index[48:]
    assert wide["y"].loc[extra].isna().all() and (wide["frame"].loc[extra, "delivery_day"] == pd.Timestamp("2026-11-23")).all()
    assert wide["sets"]["G"].iloc[:48].equals(sets["G"]) and (wide["sets"]["G"].loc[extra, "a"].to_numpy() == np.arange(24.0) + 100).all()
    assert rb.augmented(feat, {}) is feat


def test_a_hole_stops_the_run_unless_an_amendment_names_it():
    hours = pd.date_range("2025-11-19T23:00Z", "2026-11-23T22:00Z", freq="h")
    days = pd.DatetimeIndex(hours.tz_convert("Europe/Paris").normalize().tz_localize(None))
    frame = pd.DataFrame({"delivery_day": days, "fr": 1.0, "de": 1.0}, index=hours)
    frame.loc[days == pd.Timestamp("2026-11-21"), "fr"] = np.nan
    feat = {"frame": frame, "sets": {"C": pd.DataFrame({"x": 1.0}, index=hours)}}
    found = rb.coverage_check(feat, "2026-11-20", "2026-11-23")
    assert not found["pass"] and found["new_holes_not_accepted"] == {"fr": ["2026-11-21"]}
    assert rb.coverage_check(feat, "2026-11-20", "2026-11-23", {"fr": ["2026-11-21"]})["pass"]
    assert not rb.coverage_check(feat, "2026-11-20", "2026-11-23", {"fr": ["2026-11-22"]})["pass"]  # another day does not cover this one


def test_the_window_cannot_be_assembled_before_the_read_date(tmp_path, monkeypatch):
    monkeypatch.setattr(swt, "_today", lambda: pd.Timestamp("2027-01-18"))
    with pytest.raises(SystemExit, match="sealed"):
        rb.rebuild(rb.Sources(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path), tmp_path / "out", swt.FIRST, swt.LAST, None)
    with pytest.raises(SystemExit, match="replay"):
        rb.main(["replay", "--out", str(tmp_path), "--static", "s", "--french", "f", "--german-raw", "g", "--wind", "w", "--solar", "s", "--first", "2026-10-11", "--last", "2026-10-12"])


# ---- end to end, on design days ----------------------------------------------------------------------------------------------

def _sources(live=None) -> rb.Sources:
    static = Path(ROOT) / "static"
    return rb.Sources(static, static / "inputs.csv", static / "raw_first_test", static / "german_wind_points.csv.gz", static / "german_solar_points.csv.gz", live=live)


def _frame(folder: Path) -> pd.DataFrame:
    frame = pd.read_csv(folder / "test_predictions.csv.gz", index_col=0, parse_dates=["delivery_day"])
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    out = tmp_path_factory.mktemp("plain")
    checks = rb.rebuild(_sources(), out, "2026-10-04", "2026-10-10", None, reproduce_from="2026-10-04", scramble_days=("2026-10-10",), log=lambda *_: None)
    return out, checks


@needs_data
def test_the_rebuild_reproduces_the_frozen_forecasts_of_the_last_stored_day(plain):
    out, checks = plain
    reference = json.loads((Path(ROOT) / "static" / "reference.json").read_text())
    frame = _frame(out)
    day = frame[frame["delivery_day"] == pd.Timestamp(sl.SELFTEST_DAY)]
    assert [sl.iso(t) for t in day.index] == reference["timestamps_utc"]
    for name in sl.MODELS:
        assert np.abs(day[name].to_numpy() - np.asarray(reference["forecasts"][name])).max() <= 1e-9, name
    assert all(checks[k]["pass"] for k in ("1_german_overlap", "2_french_and_points", "3_coverage", "5_scramble", "6_eligible_hours"))
    assert not checks["2_french_and_points"]["differences_found"]
    assert not checks["4_stored_forecasts_reproduce"]["pass"] and not checks["all_pass"]  # no stored forecasts were given: nothing may be scored
    assert (frame["provenance"] == "rebuilt").all() and len(frame) == 168


@needs_data
def test_the_files_written_are_what_the_scoring_code_reads(plain, monkeypatch):
    out, checks = plain
    frame, schedule = score.load(out / "test_predictions.csv.gz", out / "schedule.csv")
    for name, value in (("FIRST", "2026-10-04"), ("LAST", "2026-10-10"), ("EXPECTED_DAYS", 7), ("EXPECTED_HOURS", 168)):
        monkeypatch.setattr(swt, name, value)
    score.validate(frame)  # design days: scoring them is allowed, and the point here is only that the files are accepted
    hours = score.availability(frame, schedule)
    assert hours["all"]["scheduled_hours"] == hours["all"]["issued_hours"] == hours["all"]["eligible_hours"] == 168 and hours["rebuilt"]["days"] == 7
    six = checks["6_eligible_hours"]
    assert six["pass"] and six["every_hour_of_the_window_accounted_for"] and not six["floor"]["met"] and six["floor"]["least_days"] == 95  # seven days are under the floor, which does not stop the run


@needs_data
def test_a_saved_window_changes_the_forecast_of_its_own_day_only_and_a_saved_record_is_scored_as_saved(plain, tmp_path):
    base = _frame(plain[0])
    live = tmp_path / "live"
    stored = load_inputs(Path(ROOT) / "static" / "inputs.csv").astype(float)
    days = rb._days(stored)
    window = stored[(days >= pd.Timestamp("2026-09-29")) & (days <= pd.Timestamp("2026-10-09"))].copy()
    window.loc[rb._days(window) == pd.Timestamp("2026-10-09"), "load_fc_mw"] *= 1.10  # the saved load forecast of the day differs from the refetched one
    window["price_eur_mwh"] += 500.0  # a price in the window must not be read
    (live / "french_windows").mkdir(parents=True)
    window.to_csv(live / "french_windows" / "2026-10-09_spread_20261008T080200Z.csv", index_label="timestamp_utc")  # 10:02 Paris on the eve
    reference = json.loads((Path(ROOT) / "static" / "reference.json").read_text())
    paths = sl.Paths(root=tmp_path, live_dir=live)
    record = {"delivery_day": "2026-10-10", "dry_run": False, "saved_before_auction": True, "all_fetches_finished_by_cutoff": True, "code_sha256": "test", "flags": [],
              "timestamps_utc": reference["timestamps_utc"], "forecasts": {m: [v + 1.0 for v in reference["forecasts"][m]] for m in sl.MODELS},
              "evidence": {"hours": 24, "yesterdays_spread_present_hours": 24, "input_hours_missing_for_D": {"s_l2": 24}},
              "inputs": {sl.SECOND_SOURCE: {"hours_taken_from_it": 48}}, "fingerprint": reference["fingerprint"]}
    path = sl.write_once(sl.record_path(paths, "2026-10-10"), record)
    fq.manifest_append(paths.records / "manifest.jsonl", [{"kind": "spread_record", "path": path.name, "sha256": fq.sha256_file(path), "bytes": path.stat().st_size}])
    out = tmp_path / "out"
    rb.rebuild(_sources(live), out, "2026-10-04", "2026-10-10", None, reproduce_from="2026-10-04", scramble_days=(), log=lambda *_: None)
    frame, report = _frame(out), json.loads((out / "rebuild_report.json").read_text())
    assert report["days_by_class"] == {"saved": 1, "rebuilt_french_saved": 1, "rebuilt": 5}
    by_day = frame.groupby("delivery_day")["provenance"].first()
    assert by_day[pd.Timestamp("2026-10-09")] == "rebuilt_french_saved" and by_day[pd.Timestamp("2026-10-10")] == "saved"
    other = ~frame["delivery_day"].isin([pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-10")])
    models = list(rb.ALL_MODELS)
    assert np.abs(frame.loc[other, models].to_numpy() - base.loc[other, models].to_numpy()).max() <= 1e-9  # no other day moves: the saved value is never a training row
    ninth = frame["delivery_day"] == pd.Timestamp("2026-10-09")
    assert np.abs(frame.loc[ninth, "W"].to_numpy() - base.loc[ninth, "W"].to_numpy()).max() > 1e-6  # the day itself is forecast from the saved load forecast
    tenth = frame["delivery_day"] == pd.Timestamp("2026-10-10")
    for name in sl.MODELS:  # the saved forecast is the one scored
        assert np.abs(frame.loc[tenth, name].to_numpy() - (np.asarray(reference["forecasts"][name]) + 1.0)).max() <= 1e-12
    assert np.abs(frame.loc[tenth, "C"].to_numpy() - base.loc[tenth, "C"].to_numpy()).max() <= 1e-9  # C is rebuilt on a saved day
    assert frame.loc[:, ["spread", "yesterday"]].equals(base.loc[:, ["spread", "yesterday"]])  # realised values never come from a window
    saved = report["saved_days"]
    assert saved["with_hours_from_the_second_source"] == {"2026-10-10": 48} and saved["inputs_that_differ_from_the_rebuild"] == {}
    assert all(abs(v["largest"] - 1.0) < 1e-9 for v in saved["forecast_difference_from_the_rebuild"].values())
    load = report["saved_against_refetched_french_inputs"]["load_fc_mw"]
    assert load["values_compared"] == 24 and load["values_that_differ_by_more_than_1_percent"] == 24 and load["stated_beside_the_result"]
    schedule = pd.read_csv(out / "schedule.csv", index_col=0, keep_default_na=False)
    assert (schedule.loc[schedule["delivery_day"] == "2026-10-10", "inputs_missing"] == 1).all()


@needs_data
def test_a_day_without_a_realised_price_is_left_out_with_its_reason_and_the_day_after_stays_eligible(plain):
    from thermo_fr.forecast import spread_eval as se

    data, _ = rb.join(_sources(), "2026-10-10")
    feat = se.features(data)
    frame = feat["frame"]
    eighth, ninth = np.asarray(frame["delivery_day"] == pd.Timestamp("2026-10-08")), np.asarray(frame["delivery_day"] == pd.Timestamp("2026-10-09"))
    frame.loc[eighth, ["fr", "spread"]] = np.nan  # as if ENTSO-E never published the French price of the 8th
    feat["y"] = frame["spread"]
    feat["yesterday"] = feat["yesterday"].copy()
    feat["yesterday"][ninth] = np.nan  # so the 9th has no spread of the previous day
    pred = _frame(plain[0])[list(rb.ALL_MODELS)]
    classes = {day: "rebuilt" for day in pd.date_range("2026-10-04", "2026-10-10")}
    out, schedule, _ = rb.assemble(feat, pred, {}, classes, {}, "2026-10-04", "2026-10-10")
    assert out["delivery_day"].nunique() == 6 and len(out) == 144 and pd.Timestamp("2026-10-08") not in set(out["delivery_day"])
    assert (schedule.loc[schedule["delivery_day"] == pd.Timestamp("2026-10-08"), "unscored_reason"] == f"{score.NO_SPREAD}: no French price").all()
    on_ninth = out["delivery_day"] == pd.Timestamp("2026-10-09")
    assert on_ninth.sum() == 24 and out.loc[on_ninth, "yesterday"].isna().all() and out.loc[~on_ninth, "yesterday"].notna().all()  # eligible without yesterday's spread
    assert out[list(score.COMMON)].notna().all().all() and (schedule.loc[out.index, "unscored_reason"] == "").all()


def test_stop_condition_4_needs_the_reconstruction_by_hash_and_a_passing_self_test(tmp_path, monkeypatch):
    hours = pd.date_range("2026-10-09T22:00Z", periods=24, freq="h")
    pred = pd.DataFrame({m: np.arange(24.0) for m in rb.ALL_MODELS}, index=hours)
    kept = tmp_path / "kept"
    kept.mkdir()
    pred.to_csv(kept / "comparator_predictions.csv.gz")
    good, bad = tmp_path / "selftest.json", tmp_path / "failed.json"
    good.write_text('HW done\n{"pass": true, "verdict": "forecasts identical to the reference machine", "code_matches_reference": true}')
    bad.write_text('{"pass": false, "verdict": "forecasts differ"}')
    found = rb.reproduce_check(pred, kept, good)
    assert not found["pass"] and not found["is_the_reconstruction_of_2026_11_23"] and found["largest_difference"]["W"] == 0.0  # the numbers agree, but this is not the registered reconstruction
    monkeypatch.setitem(rb.RECONSTRUCTIONS, "comparator_predictions.csv.gz", rb.se.sha256(kept / "comparator_predictions.csv.gz"))
    found = rb.reproduce_check(pred, kept, good)
    assert found["pass"] and found["basis"] == "met under the exception of 2026-11-23" and found["reported_not_required"] == ["HW", "GK", "GM"]
    assert not rb.reproduce_check(pred, kept, bad)["pass"] and not rb.reproduce_check(pred, kept, None)["pass"]  # the workspace must pass the self-test on the day
    assert not rb.reproduce_check(pred + 1e-6, kept, good)["pass"]
