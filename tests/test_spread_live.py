"""The morning logger of the spread forecasts. Unit tests need no data. The end-to-end tests replay 2026-10-10, a day of the
opened window, from a folder of static files named by SPREAD_LIVE_ROOT (they are skipped without it). No network is used and
no price of the test window is read."""

import io
import json
import os
import shutil
import time
import urllib.error
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import spread_live as sl

ROOT = os.environ.get("SPREAD_LIVE_ROOT")
needs_data = pytest.mark.skipif(not ROOT or not Path(ROOT, "static", "reference.json").exists(), reason="set SPREAD_LIVE_ROOT to a folder with static/")


def test_gates_are_paris_times_on_the_eve_in_winter_and_summer():
    g = {k: sl.iso(v) for k, v in sl.gates(pd.Timestamp("2026-11-21")).items()}
    assert g == {"earliest": "2026-11-20T09:00:00Z", "cutoff": "2026-11-20T09:05:00Z", "last_chance": "2026-11-20T09:25:00Z", "auction": "2026-11-20T11:00:00Z"}
    g = sl.gates(pd.Timestamp("2026-07-02"))
    assert (sl.iso(g["cutoff"]), sl.iso(g["auction"])) == ("2026-07-01T08:05:00Z", "2026-07-01T10:00:00Z")


def test_a_day_has_one_record_created_whole_and_never_replaced(tmp_path):
    path = tmp_path / "issued" / "2026-11-21.json"
    sl.write_once(path, {"a": 1})
    with pytest.raises(FileExistsError):
        sl.write_once(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1} and [f.name for f in path.parent.iterdir()] == ["2026-11-21.json"]  # no temporary file left behind


def test_an_unreadable_record_is_reported_and_stops_nothing(tmp_path):
    paths = sl.Paths(tmp_path)
    sl.write_once(sl.record_path(paths, "2026-11-21"), {"delivery_day": "2026-11-21", "flags": [], "issued_by_cutoff": True})
    (paths.records / "issued" / "2026-11-22.json").write_text('{"delivery_day": "2026-11-2')  # cut off mid-write by something else
    status = sl.status(paths)
    assert status["records"] == 1 and status["unreadable_records"] == ["2026-11-22.json"]
    assert status["manifest"]["records_not_in_the_manifest"] == ["2026-11-21.json", "2026-11-22.json"] and not status["manifest"]["ok"]


def test_the_next_delivery_day_is_tomorrow_in_paris():
    assert sl.next_delivery_day(pd.Timestamp("2026-11-20T09:02:00Z")) == pd.Timestamp("2026-11-21")
    assert sl.next_delivery_day(pd.Timestamp("2026-11-20T23:30:00Z")) == pd.Timestamp("2026-11-22")  # already the 21st in Paris


def test_chunks_cover_the_range_in_pieces_of_at_most_31_days():
    pieces = sl.chunks(pd.Timestamp("2026-10-10"), pd.Timestamp("2026-11-21"))
    assert pieces[0][0] == "202610100000" and pieces[-1][1] == "202611210000"
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:])) and len(pieces) == 2
    assert sl.chunks(pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-21")) == [("202611100000", "202611210000")]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_the_german_fetch_saves_each_answer_untouched_never_writes_the_key_and_survives_one_failing_item(tmp_path):
    calls = []

    def opener(request, timeout=None):
        url = request.full_url
        calls.append(url)
        if "documentType=A65" in url:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, io.BytesIO(b"bad token SECRET"))
        return FakeResponse(b"<Publication_MarketDocument>" + url.split("documentType=")[1][:3].encode() + b"</Publication_MarketDocument>")

    result = sl.fetch_german_raw(pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-21"), tmp_path / "20261120T090200Z", key="SECRET", opener=opener)
    assert len(calls) == 3 and all("securityToken=SECRET" in c for c in calls)
    assert sorted(result["files"]) == ["prices_DE_LU_202611100000_202611210000.xml", "wind_solar_forecast_DE_LU_202611100000_202611210000.xml"]
    assert list(result["errors"]) == ["load_forecast_DE_LU_202611100000_202611210000.xml"] and "SECRET" not in json.dumps(result)
    for name in result["files"]:
        assert b"SECRET" not in (tmp_path / "20261120T090200Z" / name).read_bytes()


def test_a_folder_with_an_item_missing_still_parses_to_empty_series(tmp_path):
    parsed = sl.parse_german_folder(tmp_path)
    assert parsed["de"].empty and list(parsed["tso"].columns) == ["load", "onshore", "offshore", "solar"] and parsed["tso"].empty


def test_saved_windows_are_ordered_by_issue_time_and_cut_at_the_time_asked(tmp_path):
    production, root = tmp_path / "prod", tmp_path / "root"
    (production / "data" / "shadow" / "fresh").mkdir(parents=True)
    (root / "live" / "french_windows").mkdir(parents=True)
    for name in ("2026-11-20_run000422_20261119T103234Z.csv", "2026-11-21_run000430_20261120T091500Z.csv"):
        (production / "data" / "shadow" / "fresh" / name).write_text("x")
    (production / "data" / "shadow" / "fresh" / "2026-11-20_run000422_20261119T103234Z.models.json").write_text("{}")
    (root / "live" / "french_windows" / "2026-11-21_spread_20261120T090210Z.csv").write_text("x")
    paths = sl.Paths(root, production)
    names = [p.name for _, p in sl.window_files(paths, pd.Timestamp("2026-11-20T09:05:00Z"))]
    assert names == ["2026-11-20_run000422_20261119T103234Z.csv", "2026-11-21_spread_20261120T090210Z.csv"]  # the 09:15 production run is after the cutoff asked for
    assert len(sl.window_files(paths, pd.Timestamp("2026-11-20T10:00:00Z"))) == 3


def test_the_first_point_value_seen_for_an_hour_is_kept(tmp_path):
    from thermo_fr.forecast import spread_eval as se

    index = pd.date_range("2026-11-18T00:00Z", periods=24, freq="h")
    stored = se.Data(french=None, de=None, tso=None, wind_pts=pd.DataFrame({"de_wind_pt_A_ms": 5.0}, index=index), solar_pts=pd.DataFrame({"de_solar_pt_A_wm2": 1.0}, index=index), last="")
    folder = tmp_path / "live" / "german_points"
    folder.mkdir(parents=True)
    later = pd.date_range("2026-11-18T12:00Z", periods=36, freq="h")
    for stamp, value in (("20261119T090200Z", 7.0), ("20261120T090200Z", 9.0)):
        frame = pd.DataFrame({"de_wind_pt_A_ms": value}, index=later if value == 7.0 else later + pd.Timedelta(hours=24))
        frame.index.name = "timestamp_utc"
        frame.to_csv(folder / f"{stamp}_wind.csv.gz")
    wind, _, used = sl.points_tables(sl.Paths(tmp_path), stored, pd.Timestamp("2026-11-21T00:00Z"))
    assert wind.loc["2026-11-18T13:00Z", "de_wind_pt_A_ms"] == 5.0  # the static file wins where it has a value
    assert wind.loc["2026-11-19T06:00Z", "de_wind_pt_A_ms"] == 7.0  # then the first fetch that saw the hour
    assert wind.loc["2026-11-20T06:00Z", "de_wind_pt_A_ms"] == 9.0 and len(used) == 2
    early, _, used = sl.points_tables(sl.Paths(tmp_path), stored, pd.Timestamp("2026-11-19T12:00Z"))
    assert len(used) == 1 and early.index.max() == pd.Timestamp("2026-11-19T23:00Z")


@pytest.fixture
def replay_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(Path(ROOT, "static"), root / "static", target_is_directory=True)
    return sl.Paths(root, tmp_path / "no-production")


@needs_data
def test_the_selftest_rebuilds_the_frozen_batch_forecasts_through_the_loggers_own_path(replay_root):
    result = sl.selftest(replay_root, log=lambda *_: None)
    assert result["pass"] and result["hours"] == 24 and max(result["forecasts"]["largest_difference"].values()) < 1e-9
    assert result["tables"]["identical_to_the_reference_machine"] and result["verdict"] == "forecasts identical to the reference machine"
    assert max(result["lightgbm_on_identical_synthetic_data"]["largest_difference_from_the_reference_machine"].values()) == 0.0


def steps_with_a_failed_german_fetch(paths, day, started, stamp):
    def fails():
        raise RuntimeError("source down")

    return {"french_window": lambda: {"files": {}, "items": {}, "errors": {}}, "german_raw": fails, "german_points": lambda: {"errors": {"de_wind_pt_A_ms": "timeout"}}}


def ticking(start, step=20):
    now = {"t": pd.Timestamp(start)}

    def clock():
        now["t"] += pd.Timedelta(seconds=step)
        return now["t"]

    return now, clock


@needs_data
def test_one_morning_end_to_end_without_network(replay_root, monkeypatch):
    reference = json.loads((replay_root.static / "reference.json").read_text())
    quiet = lambda *_: None  # noqa: E731
    monkeypatch.setattr(sl, "fetch_steps", steps_with_a_failed_german_fetch)
    # 10:02 Paris on the eve of 2026-10-10: a source is down, so the first run writes nothing and leaves it to the next trigger
    now, clock = ticking("2026-10-09T08:02:00Z")
    first = sl.issue(replay_root, clock=clock, log=quiet)
    assert first["status"] == "postponed" and not sl.record_path(replay_root, "2026-10-10").exists()
    # 10:31 Paris: past the last chance, so it issues with what there is and says so
    now["t"] = pd.Timestamp("2026-10-09T08:31:00Z")
    out = sl.issue(replay_root, clock=clock, log=quiet)
    assert out["status"] == "issued" and "german_raw failed" in out["flags"] and "german_points: 1 item(s) failed" in out["flags"]
    assert "a fetch finished after 10:05 Paris" in out["flags"] and "production folder not found: no production window is read" in out["flags"]
    record = json.loads(Path(out["record"]).read_text())
    assert Path(out["record"]).name == "2026-10-10.json"
    assert record["saved_before_auction"] and not record["issued_by_cutoff"] and not record["all_fetches_finished_by_cutoff"] and not record["scored"]
    assert record["cutoff_utc"] == "2026-10-09T08:05:00Z" and record["issued_at_utc"] < record["auction_close_utc"]
    assert record["timestamps_utc"] == reference["timestamps_utc"]
    for model in sl.MODELS:  # the static files hold the same inputs, so the forecasts are the frozen ones
        assert np.max(np.abs(np.asarray(record["forecasts"][model]) - np.asarray(reference["forecasts"][model]))) < 1e-9
    # the stored raw files hold German prices and ENTSO-E forecasts of the delivery day itself: dropped, and counted
    assert record["inputs"]["dropped"]["german_price_hours_of_D_or_later"] >= 24
    assert record["inputs"]["latest_german_price_hour_utc"] < "2026-10-09T22:00:00Z"
    assert all(record["evidence"]["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values())
    assert record["training"]["W"]["last_training_day"] < record["training"]["W"]["block_start"] <= "2026-10-10"
    assert len(record["static_sha256"]["raw_first_test"]) == 64 and len(record["code_sha256"]) == 64
    status = sl.status(replay_root)
    assert status["records"] == 1 and status["manifest"]["ok"] and status["manifest"]["lines"] == 1
    assert [a["outcome"] for a in status["attempts_without_a_record"]] == ["postponed"]
    # a second trigger the same morning writes nothing
    assert sl.issue(replay_root, clock=clock, log=quiet)["status"] == "already issued"
    assert not (replay_root.records / "running" / "2026-10-10.json").exists()


@needs_data
def test_runs_at_the_wrong_time_write_nothing(replay_root, monkeypatch):
    quiet = lambda *_: None  # noqa: E731
    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {})
    # ten past midnight Paris on the eve (a task catching up overnight): before 10:00, so nothing is fetched or written
    _, clock = ticking("2026-10-08T22:10:00Z")
    early = sl.issue(replay_root, clock=clock, log=quiet)
    assert early["status"] == "too early" and early["delivery_day"] == "2026-10-10"
    # after the auction closed
    _, clock = ticking("2026-10-09T10:00:01Z")
    assert sl.issue(replay_root, clock=clock, log=quiet)["status"] == "missed"
    # started in time but finished after the auction
    times = iter([pd.Timestamp("2026-10-09T09:58:00Z")] + [pd.Timestamp("2026-10-09T10:00:30Z")] * 50)
    late = sl.issue(replay_root, clock=lambda: next(times), log=quiet)
    assert late["status"] == "missed" and "finished after" in late["reason"]
    assert not (replay_root.records / "issued").exists()
    assert [a["outcome"] for a in sl.status(replay_root)["attempts_without_a_record"]] == ["too early", "missed", "missed"]


@needs_data
def test_a_task_that_fires_a_little_early_waits_for_ten_and_a_second_process_stands_back(replay_root, monkeypatch):
    quiet = lambda *_: None  # noqa: E731
    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {})
    now, clock = ticking("2026-10-09T07:58:30Z", step=5)  # 09:58:30 Paris
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now["t"] += pd.Timedelta(seconds=seconds)

    running = replay_root.records / "running" / "2026-10-10.json"
    running.parent.mkdir(parents=True)
    running.write_text(json.dumps({"started_utc": "2026-10-09T07:59:00Z", "pid": 1}))  # another process started a minute ago
    assert sl.issue(replay_root, clock=clock, log=quiet, sleep=sleep)["status"] == "another run is in progress"
    running.unlink()
    now["t"] = pd.Timestamp("2026-10-09T07:58:30Z")
    out = sl.issue(replay_root, clock=clock, log=quiet, sleep=sleep)
    assert out["status"] == "issued" and 60 < slept[-1] <= 90
    record = json.loads(Path(out["record"]).read_text())
    assert record["started_at_utc"] >= "2026-10-09T08:00:00Z" and record["issued_by_cutoff"] and record["all_fetches_finished_by_cutoff"]


@needs_data
def test_a_failure_while_building_leaves_a_trace_and_no_record(replay_root, monkeypatch):
    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {})
    monkeypatch.setattr(sl, "build_data", lambda *a, **k: (_ for _ in ()).throw(ValueError("broken table")))
    _, clock = ticking("2026-10-09T08:02:00Z")
    out = sl.issue(replay_root, clock=clock, log=lambda *_: None)
    assert out["status"] == "failed" and "broken table" in out["error"] and not sl.record_path(replay_root, "2026-10-10").exists()
    assert sl.status(replay_root)["attempts_without_a_record"][-1]["outcome"] == "failed"


@needs_data
def test_a_dry_run_writes_only_under_dry_run_at_any_time_of_day(replay_root, monkeypatch):
    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {})
    _, clock = ticking("2026-10-09T15:00:00Z")  # the afternoon: the auction is long closed
    out = sl.issue(replay_root, dry_run=True, clock=clock, log=lambda *_: None)
    assert out["status"] == "dry run written" and "/live/dry_run/records/issued/" in out["record"].replace("\\", "/")
    assert not (replay_root.root / "live" / "records").exists()
    assert json.loads(Path(out["record"]).read_text())["dry_run"] is True


def test_the_points_fetch_writes_one_table_per_kind_in_the_static_columns_and_keeps_a_failed_point_missing(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("p", Path(sl.__file__).resolve().parents[3] / "scripts" / "fetch_german_points.py")
    pts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pts)
    columns = {kind: [cfg["column"].format(name) for name, _, _ in cfg["points"]] for kind, cfg in pts.KINDS.items()}
    times = [f"2026-11-21T{h:02d}:00" for h in range(24)]

    def opener(request, timeout=None):
        url = request.full_url
        assert "start_date=2026-11-16" in url and "end_date=2026-11-21" in url and "models=best_match" in url
        if "latitude=54.5&" in url and "wind_speed" in url:
            raise urllib.error.URLError("down")
        variable = url.split("hourly=")[1].split("&")[0]
        return FakeResponse(json.dumps({"hourly": {"time": times, variable: [36.0] * 24}}).encode())

    monkey_sleep = sl.time.sleep
    sl.time.sleep = lambda *_: None
    try:
        out = sl.fetch_german_points(pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-21"), tmp_path, "20261120T090200Z", columns, opener=opener, pause=0)
    finally:
        sl.time.sleep = monkey_sleep
    assert list(out["errors"]) == ["de_wind_pt_NordseeKueste_ms"] and out["wind"]["hours"] == 24 and out["wind"]["hours_all_points_present"] == 0
    wind = pd.read_csv(tmp_path / out["wind"]["file"], index_col=0)
    assert list(wind.columns) == columns["wind"] and wind["de_wind_pt_NordseeKueste_ms"].isna().all()
    assert wind["de_wind_pt_Ostholstein_ms"].iloc[0] == pytest.approx(10.0)  # 36 km/h is 10 m/s
    solar = pd.read_csv(tmp_path / out["solar"]["file"], index_col=0)
    assert out["solar"]["hours_all_points_present"] == 24 and solar.iloc[0, 0] == 36.0


def test_differing_columns_compares_values_not_storage_types():
    a = pd.DataFrame({"x": [1.0, np.nan, 3.0], "flag": [True, False, True], "y": [np.nan, np.nan, np.nan]})
    b = pd.DataFrame({"x": [1.0, np.nan, 3.0], "flag": [1.0, 0.0, 1.0], "y": pd.Series([None, None, None], dtype=object)})
    assert sl.differing_columns(a, b) == {}
    c = b.assign(x=[1.0, 2.0, 3.5])
    assert sl.differing_columns(a, c) == {"x": {"hours": 2, "missing_in_one_only": 1, "largest_difference": 0.5}}


def test_fingerprints_name_the_column_that_differs():
    table = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [0.1, 0.2, 0.3]})
    same = sl.compare_fingerprints(sl.fingerprint(table), sl.fingerprint(table.copy()))
    other = sl.compare_fingerprints(sl.fingerprint(table.assign(b=[0.1, 0.2, 0.3000001])), sl.fingerprint(table))
    assert same == {} and list(other) == ["b"]
    assert list(sl.compare_fingerprints(sl.fingerprint(table.assign(a=[1.0, 2.0, 0.0])), sl.fingerprint(table))) == ["a"]  # a zero is not a missing value


def test_an_item_absent_for_the_delivery_day_is_an_error_only_if_it_is_due_by_the_gate():
    items = {"prices": {"status": "absent", "message": ""}, "nuclear_actual": {"status": "absent", "message": ""}, "wind_solar_forecast": {"status": "absent", "message": ""},
             "load_forecast": {"status": "absent", "message": ""}, "weather_issued": {"status": "present", "message": ""}, "wind_points": {"status": "error", "message": "timeout"}}
    errors = {item: v["message"] or "failed" for item, v in items.items() if v["status"] == "error"}
    absent = sorted(item for item, v in items.items() if v["status"] == "absent" and item in sl.DUE_BY_THE_GATE)
    assert errors == {"wind_points": "timeout"} and absent == ["load_forecast"]


@needs_data
def test_the_selftest_verdict_on_another_machine(replay_root, monkeypatch):
    curves, evidence, probe = sl.selftest_material(replay_root, log=lambda *_: None)
    rng = np.random.default_rng(0)

    def machine(noise=0.0, last_bits=False, wrong_table=False, wrong_library=False):
        import copy

        ev = copy.deepcopy(evidence)
        if last_bits:
            ev["fingerprint"]["all_rows_to_D"]["bits"]["s_m7"] = "0" * 16
        if wrong_table:
            ev["fingerprint"]["all_rows_to_D"]["columns"]["s_m7"][1] += 5.0
        pr = copy.deepcopy(probe)
        if wrong_library:
            pr["frozen_settings"][0] += 0.3
        monkeypatch.setattr(sl, "selftest_material", lambda paths, log=print: (curves + noise * rng.standard_normal(curves.shape), ev, pr))
        return sl.selftest(replay_root, log=lambda *_: None)

    same = machine()
    assert same["pass"] and same["verdict"] == "forecasts identical to the reference machine"
    laptop = machine(noise=2.5, last_bits=True)  # what the owner's laptop showed: equal tables to 1e-9, not bit for bit, forecasts a few EUR/MWh apart
    assert laptop["pass"] and "machine-dependent numbers" in laptop["verdict"] and not laptop["tables"]["bit_for_bit_identical"]
    assert not machine(noise=2.5)["pass"]  # bit-identical tables cannot explain different forecasts
    assert not machine(noise=15.0, last_bits=True)["pass"]  # outside the band
    assert not machine(noise=2.5, last_bits=True, wrong_table=True)["pass"]
    assert not machine(noise=2.5, last_bits=True, wrong_library=True)["pass"]


def test_the_fetches_run_side_by_side_and_a_hung_source_does_not_hold_the_others():
    _, clock = ticking("2026-10-09T08:00:00Z", step=1)

    def slow(value):
        def step():
            time.sleep(0.4)
            return {"value": value}
        return step

    def fails():
        raise RuntimeError("source down")

    began = time.monotonic()
    out = sl.run_fetches({"a": slow(1), "b": slow(2), "c": fails, "d": lambda: time.sleep(30)}, clock, {"a": 5, "b": 5, "c": 5, "d": 1})
    assert time.monotonic() - began < 2.5  # not 0.4 + 0.4 + 30: side by side, and the hung one is given up on after its second
    assert list(out) == ["a", "b", "c", "d"] and out["a"]["result"] == {"value": 1} and out["b"]["result"] == {"value": 2}
    assert isinstance(out["c"]["error"], RuntimeError) and isinstance(out["d"]["error"], TimeoutError) and "result" not in out["d"]
    assert all(box["began"] <= box["done"] for box in out.values())

    def exits():
        raise SystemExit(3)

    ended = sl.run_fetches({"e": exits}, clock, {"e": 5})["e"]  # whatever ends a fetch is written down, and is not called a timeout
    assert isinstance(ended["error"], SystemExit) and ended["done"] is not None


def test_a_fetched_file_is_whole_or_not_there(tmp_path):
    path = tmp_path / "20261120T090012Z_wind.csv.gz"
    sl.write_whole(path, b"abc")
    assert path.read_bytes() == b"abc" and [f.name for f in tmp_path.iterdir()] == [path.name]
    frame = pd.DataFrame({"a": [1.5, np.nan]}, index=pd.to_datetime(["2026-11-20T00:00Z", "2026-11-20T01:00Z"]))
    frame.to_csv(tmp_path / "direct.csv")
    sl.write_whole(tmp_path / "whole.csv", frame.to_csv().encode("utf-8"))  # how the French window is written
    assert (tmp_path / "whole.csv").read_bytes() == (tmp_path / "direct.csv").read_bytes()
    (tmp_path / "whole.csv").unlink(), (tmp_path / "direct.csv").unlink()
    (tmp_path / "20261121T090012Z_wind.csv.gz.part").write_bytes(b"half")  # what a fetch cut short leaves behind
    assert [f.name for f in sorted(tmp_path.glob("*.csv.gz"))] == [path.name]  # the name the logger reads does not match it


def test_a_rounding_difference_is_told_apart_from_a_value_that_moved():
    a = pd.DataFrame({"x": [100.0, 2.0, np.nan], "y": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"x": [100.0 + 1.5e-14, 2.0, np.nan], "y": [1.0, 2.0, 3.0]})
    assert list(sl.differing_columns(a, b)) == ["x"] and sl.differing_columns(a, b, sl.SCRAMBLE_TOLERANCE) == {}
    moved = pd.DataFrame({"x": [100.0, 2.0, np.nan], "y": [1.0, 2.0 + 1e-6, 3.0]})
    assert list(sl.differing_columns(a, moved, sl.SCRAMBLE_TOLERANCE)) == ["y"]
    gone = pd.DataFrame({"x": [100.0, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})  # missing in one only always counts
    assert sl.differing_columns(a, gone, sl.SCRAMBLE_TOLERANCE)["x"]["missing_in_one_only"] == 1
    wild = pd.DataFrame({"x": [np.inf, 2.0, np.nan], "y": [1.0, 2.0, -np.inf]})  # an infinite value is never the same as a finite one
    for tolerance in (0.0, sl.SCRAMBLE_TOLERANCE):
        assert set(sl.differing_columns(a, wild, tolerance)) == {"x", "y"} and sl.differing_columns(wild, wild, tolerance) == {}
    assert sl.differing_columns(a, wild)["x"] == {"hours": 1, "missing_in_one_only": 0, "largest_difference": None}  # no finite gap to report


@needs_data
def test_a_missing_price_of_yesterday_is_issued_at_once_and_a_missing_load_forecast_waits(replay_root, monkeypatch):
    quiet = lambda *_: None  # noqa: E731
    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {})
    real = sl.build_data
    day = pd.Timestamp("2026-10-10")

    def without(what):
        def build(paths, d, upto, live=True):
            data, info = real(paths, d, upto=upto, live=live)
            if what == "german price of yesterday":
                de = data.de.copy()
                de[np.asarray(sl.delivery_days(de.index) == day - pd.Timedelta(days=1))] = np.nan
                return replace(data, de=de), info
            french = data.french.copy()
            french.loc[np.asarray(sl.delivery_days(french.index) == day), "load_fc_mw"] = np.nan
            return replace(data, french=french), info
        return build

    # 10:02 Paris, the load forecast of D is not there yet: nothing is written, the next trigger tries again
    monkeypatch.setattr(sl, "build_data", without("load forecast"))
    _, clock = ticking("2026-10-09T08:02:00Z", step=1)
    waited = sl.issue(replay_root, clock=clock, log=quiet)
    assert waited["status"] == "postponed" and "load forecast" in waited["reason"] and not sl.record_path(replay_root, "2026-10-10").exists()
    # 10:02 Paris, yesterday's German price is not on the source: issued at once, inputs left missing, and the record says so
    monkeypatch.setattr(sl, "build_data", without("german price of yesterday"))
    _, clock = ticking("2026-10-09T08:02:00Z", step=1)
    out = sl.issue(replay_root, clock=clock, log=quiet)
    assert out["status"] == "issued" and any(flag.startswith("yesterday's spread is missing") for flag in out["flags"])
    record = json.loads(Path(out["record"]).read_text())
    assert record["issued_by_cutoff"] and record["evidence"]["yesterdays_spread_present_hours"] == 0
    assert record["evidence"]["input_hours_missing_for_D"]["s_l1"] == 24 and record["evidence"]["input_hours_missing_for_D"]["de_l1"] == 24
    assert all(len(record["forecasts"][m]) == 24 and np.isfinite(record["forecasts"][m]).all() for m in sl.MODELS)
    assert all(record["evidence"]["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values())


@needs_data
def test_each_fetch_is_timed_by_its_own_finish_and_a_failed_price_request_waits(replay_root, monkeypatch):
    quiet = lambda *_: None  # noqa: E731
    now = {"t": pd.Timestamp("2026-10-09T08:02:00Z")}  # 10:02 Paris; this clock moves only when a step moves it
    real = sl.build_data
    day = pd.Timestamp("2026-10-10")

    def without_yesterdays_german_price(paths, d, upto, live=True):
        data, info = real(paths, d, upto=upto, live=live)
        de = data.de.copy()
        de[np.asarray(sl.delivery_days(de.index) == day - pd.Timedelta(days=1))] = np.nan
        return replace(data, de=de), info

    def steps(request_errors):
        def german():
            time.sleep(0.5)  # still running when the French fetch is done
            now["t"] = pd.Timestamp("2026-10-09T08:06:00Z")  # and finished a minute after the gate
            return {"folder": "x", "files": {}, "errors": {n: "URLError" for n in request_errors}, "request_errors": request_errors}
        return lambda *a: {"french_window": lambda: {"files": {}, "items": {}, "errors": {}}, "german_raw": german, "german_points": lambda: {"errors": {}}}

    monkeypatch.setattr(sl, "build_data", without_yesterdays_german_price)
    # the German price request failed and yesterday's spread is missing: trying again can bring it, so nothing is written
    monkeypatch.setattr(sl, "fetch_steps", steps(["prices_DE_LU_202610010000_202610110000.xml"]))
    waited = sl.issue(replay_root, clock=lambda: now["t"], log=quiet)
    assert waited["status"] == "postponed" and "the German price request failed" in waited["reason"]
    assert waited["fetch"]["french_window"]["finished_by_cutoff"] and not waited["fetch"]["german_raw"]["finished_by_cutoff"]
    assert waited["fetch"]["french_window"]["finished_utc"] == "2026-10-09T08:02:00Z" and waited["fetch"]["german_raw"]["finished_utc"] == "2026-10-09T08:06:00Z"
    # the same morning with every request answered: the price is simply not on the source, and the record is written at once
    now["t"] = pd.Timestamp("2026-10-09T08:02:00Z")
    monkeypatch.setattr(sl, "fetch_steps", steps([]))
    out = sl.issue(replay_root, clock=lambda: now["t"], log=quiet)
    assert out["status"] == "issued" and "a fetch finished after 10:05 Paris" in out["flags"]
    record = json.loads(Path(out["record"]).read_text())
    assert record["fetches_ran"] == "side by side" and not record["all_fetches_finished_by_cutoff"]
    assert [record["fetch"][n]["finished_by_cutoff"] for n in ("french_window", "german_raw", "german_points")] == [True, False, True]


@needs_data
def test_a_rounding_difference_under_the_scramble_is_reported_and_not_called_a_leak_and_a_late_load_forecast_is_named(replay_root, monkeypatch):
    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {})
    real_scrambled, real_build = sl.se.scrambled, sl.build_data
    day = pd.Timestamp("2026-10-10")

    def one_unit_in_the_last_place(data, first):
        out = real_scrambled(data, first)
        de = out.de.copy()
        rows = np.asarray(sl.delivery_days(de.index) == day - pd.Timedelta(days=1))
        de[rows] = np.nextafter(de[rows].to_numpy(float), np.inf)
        return replace(out, de=de)

    def without_the_load_forecast(paths, d, upto, live=True):
        data, info = real_build(paths, d, upto=upto, live=live)
        french = data.french.copy()
        french.loc[np.asarray(sl.delivery_days(french.index) == day), "load_fc_mw"] = np.nan
        return replace(data, french=french), info

    monkeypatch.setattr(sl.se, "scrambled", one_unit_in_the_last_place)
    monkeypatch.setattr(sl, "build_data", without_the_load_forecast)
    _, clock = ticking("2026-10-09T08:31:00Z", step=1)  # 10:31 Paris: past the last chance to wait for the load forecast
    out = sl.issue(replay_root, clock=clock, log=lambda *_: None)
    assert out["status"] == "issued"
    assert any(flag.startswith("the French load forecast of D is missing") for flag in out["flags"])
    assert any(flag.startswith("an input of D differs by rounding only") for flag in out["flags"])
    assert "an input of D depends on a value not known at the gate" not in out["flags"]
    evidence = json.loads(Path(out["record"]).read_text())["evidence"]
    assert all(evidence["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values()) and evidence["scramble_tolerance"] == 1e-9
    assert not any(evidence["inputs_bit_identical_under_the_scramble"].values()) and evidence["inputs_that_moved_under_the_scramble"] == {}
    rounding = evidence["inputs_that_differ_only_by_rounding_under_the_scramble"]
    assert "de_l1" in rounding and all(detail["largest_difference"] < 1e-9 for detail in rounding.values())


# ---- the second source of the coupled German price (amendment 5) --------------------------------------------------------------

def answer_of(hourly: pd.Series, unit: str = "EUR / MWh", leave_out=()) -> bytes:
    """An energy-charts answer holding each hourly value as four equal quarter hours, except the quarter hours named."""
    stamps, prices = [], []
    for hour, value in hourly.items():
        for quarter in range(4):
            t = hour + pd.Timedelta(minutes=15 * quarter)
            if t not in leave_out:
                stamps.append(int(t.timestamp()))
                prices.append(None if pd.isna(value) else float(value))
    return json.dumps({"license_info": "CC BY 4.0", "unix_seconds": stamps, "price": prices, "unit": unit, "deprecated": False}).encode()


def test_the_second_source_is_read_as_complete_hours_in_eur_per_mwh_and_nothing_else():
    hours = pd.date_range("2026-11-19T23:00Z", periods=4, freq="h")
    answer = json.loads(answer_of(pd.Series([10.0, 20.0, np.nan, 40.0], index=hours), leave_out={hours[3] + pd.Timedelta(minutes=45)}))
    answer["unix_seconds"] += [int(hours[1].timestamp()) + 60]  # a stamp off the quarter-hour grid is not a price of the auction
    answer["price"] += [999.0]
    answer["price"][1] = 14.0  # the quarter hours of the first hour are 10, 14, 10, 10
    held = sl.parse_second_source(json.dumps(answer).encode())
    assert list(held.index) == [hours[0], hours[1]] and held.iloc[0] == 11.0 and held.iloc[1] == 20.0  # an hour with a missing or absent quarter hour is not held
    with pytest.raises(ValueError):
        sl.parse_second_source(answer_of(pd.Series([10.0], index=hours[:1]), unit="ct / kWh"))
    with pytest.raises(ValueError):
        sl.parse_second_source(b"Too Many Requests")
    # a truth value, a text and a number no auction could give are not prices, and the hour they belong to is not held
    odd = json.loads(answer_of(pd.Series([10.0, 20.0, 30.0, 40.0], index=hours)))
    odd["price"][0], odd["price"][4], odd["price"][8] = True, "20.0", 1e6
    assert list(sl.parse_second_source(json.dumps(odd).encode()).index) == [hours[3]]


def test_the_second_source_fetch_asks_once_for_days_before_D_waits_when_told_to_and_saves_the_answer_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(sl.time, "sleep", lambda seconds: None)
    hours = pd.date_range("2026-11-10T23:00Z", "2026-11-21T22:00Z", freq="h")
    body, calls = answer_of(pd.Series(50.0, index=hours)), []

    def opener(request, timeout=None):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", None, io.BytesIO(b"Too Many Requests"))
        return FakeResponse(body)

    result = sl.fetch_second_source(pd.Timestamp("2026-11-11"), pd.Timestamp("2026-11-21"), tmp_path / "second", "20261121T090000Z", opener=opener)
    assert len(calls) == 2 and calls[1].endswith("bzn=DE-LU&start=2026-11-11&end=2026-11-21")
    assert (tmp_path / "second" / "20261121T090000Z.json").read_bytes() == body and result["bytes"] == len(body) and result["complete_hours_held"] == len(hours)
    # an answer that cannot be read is kept on disk and reported as a failure of the fetch
    with pytest.raises(RuntimeError, match="saved as 20261121T093000Z.json, sha256 [0-9a-f]{64}, 6 bytes"):
        sl.fetch_second_source(pd.Timestamp("2026-11-11"), pd.Timestamp("2026-11-21"), tmp_path / "second", "20261121T093000Z", opener=lambda request, timeout=None: FakeResponse(b"<html>"))
    assert (tmp_path / "second" / "20261121T093000Z.json").read_bytes() == b"<html>"
    # saved answers are read up to the time asked, the later fetch winning, and a broken one holds nothing
    (tmp_path / "live" / sl.SECOND_SOURCE).mkdir(parents=True)
    for stamp, content in (("20261120T090000Z", answer_of(pd.Series(1.0, index=hours[:48]))), ("20261121T090000Z", answer_of(pd.Series(2.0, index=hours[24:72]))),
                           ("20261121T093000Z", b"<html>"), ("20261122T090000Z", answer_of(pd.Series(3.0, index=hours)))):
        (tmp_path / "live" / sl.SECOND_SOURCE / f"{stamp}.json").write_bytes(content)
    held, used = sl.second_source_held(sl.Paths(tmp_path), pd.Timestamp("2026-11-21T12:00Z"))
    assert [u["file"] for u in used] == ["20261120T090000Z.json", "20261121T090000Z.json"] and list(held.iloc[[0, 23, 24, 71]]) == [1.0, 1.0, 2.0, 2.0] and len(held) == 72
    assert all(len(u["sha256"]) == 64 for u in used)  # every file read is named with its hash, also one a given-up fetch finished later
    # an hour a later answer no longer holds is still read from the earlier one
    (tmp_path / "live" / sl.SECOND_SOURCE / "20261121T100000Z.json").write_bytes(answer_of(pd.Series(np.nan, index=hours[:24])))
    again, used = sl.second_source_held(sl.Paths(tmp_path), pd.Timestamp("2026-11-21T12:00Z"))
    assert len(used) == 3 and again.equals(held)


def test_an_hour_entsoe_lacks_is_taken_from_the_second_source_only_when_the_two_agree_and_never_for_D():
    day = pd.Timestamp("2026-11-22")
    hours = pd.date_range("2026-11-09T23:00Z", "2026-11-22T22:00Z", freq="h")  # delivery days 10 to 22 November
    truth = pd.Series(np.round(np.random.default_rng(0).uniform(20, 120, len(hours)), 2), index=hours)
    days = sl.delivery_days(hours)
    hole = np.asarray((days == pd.Timestamp("2026-11-20")) | (days == pd.Timestamp("2026-11-21")))
    entsoe = truth[np.asarray(days < day)].copy()
    entsoe[hole[: len(entsoe)]] = np.nan  # ENTSO-E answers with nothing for the 20th and the 21st
    # the two agree on every hour both hold: the 48 hours are taken, counted by day, and no hour ENTSO-E holds is touched
    filled, info = sl.with_second_source(entsoe, truth, day)
    assert info["trusted_this_morning"] and info["hours_taken_from_it"] == 48 and info["hours_taken_from_it_by_delivery_day"] == {"2026-11-20": 24, "2026-11-21": 24}
    assert info["hours_of_D_or_later_in_its_answers_never_read"] == 24 and filled.index.max() == pd.Timestamp("2026-11-21T22:00Z")
    assert filled.notna().all() and (filled[hole[: len(filled)]] == truth[hole]).all() and (filled[~hole[: len(filled)]] == entsoe.dropna()).all()
    assert info["hours_both_hold_over_the_ten_days_before_D"] == 8 * 24 and info["largest_difference_eur_mwh"] == 0.0
    # one quarter hour of the last ten days differs by a single cent, a quarter of a cent on the hour: the second source may be
    # another auction, so nothing is taken
    other = truth.copy()
    other.iloc[100] += 0.0025
    same, info = sl.with_second_source(entsoe, other, day)
    assert same is entsoe and not info["trusted_this_morning"] and info["hours_taken_from_it"] == 0 and info["hours_it_holds_that_entsoe_lacks"] == 48
    assert info["of_which_agree"] == info["hours_both_hold_over_the_ten_days_before_D"] - 1 and info["agreement_tolerance_eur_mwh"] == 0.001
    rounding = truth + 1e-12  # a difference in the last places of the mean is no disagreement
    assert sl.with_second_source(entsoe, rounding, day)[1]["trusted_this_morning"]
    # fewer than 24 hours held by both over the ten days: there is nothing to check it against, so nothing is taken
    thin = entsoe.copy()
    thin[np.asarray(sl.delivery_days(thin.index) >= pd.Timestamp("2026-11-12"))] = np.nan
    thin.iloc[-3:] = truth[thin.index[-3:]]
    same, info = sl.with_second_source(thin, truth, day)
    assert same is thin and not info["trusted_this_morning"] and info["hours_both_hold_over_the_ten_days_before_D"] == 3
    thin.iloc[-23:] = truth[thin.index[-23:]]  # 23 hours in common are not enough, 24 are
    assert not sl.with_second_source(thin, truth, day)[1]["trusted_this_morning"]
    thin.iloc[-24:] = truth[thin.index[-24:]]
    filled, info = sl.with_second_source(thin, truth, day)
    assert info["trusted_this_morning"] and info["hours_both_hold_over_the_ten_days_before_D"] == 24 and info["hours_taken_from_it"] == 9 * 24
    # a difference older than ten days does not count against it, and the second source holding nothing changes nothing
    old = truth.copy()
    old.iloc[5] += 5.0
    assert sl.with_second_source(entsoe, old, day)[1]["trusted_this_morning"]
    same, info = sl.with_second_source(entsoe, truth.iloc[:0], day)
    assert same is entsoe and info["hours_taken_from_it"] == 0 and not info["trusted_this_morning"]
    # delivery days the stored files hold are never filled, whatever the second source holds
    early = pd.date_range("2026-09-30T22:00Z", "2026-10-12T21:00Z", freq="h")
    stored = pd.Series(30.0, index=early)
    stored.iloc[:48] = np.nan
    filled, info = sl.with_second_source(stored, pd.Series(30.0, index=early), pd.Timestamp("2026-10-13"))
    assert info["hours_taken_from_it"] == 0 and filled.iloc[:48].isna().all()


@needs_data
def test_a_morning_on_which_entsoe_lacks_the_german_price_of_the_eve(replay_root, monkeypatch):
    quiet = lambda *_: None  # noqa: E731
    monkeypatch.setattr(sl, "LIVE_DATA_FROM", "2026-09-25")  # the replay day lies before the live stretch; this lets its eve be filled
    reference = json.loads((replay_root.static / "reference.json").read_text())
    day = pd.Timestamp("2026-10-10")
    real = sl.german_refetched
    whole = real(replay_root, pd.Timestamp("2026-10-09T08:00Z"))[0]["de"]

    def entsoe_without_the_eve(paths, upto, live=True):
        new, used = real(paths, upto, live=live)
        return {**new, "de": new["de"][np.asarray(sl.delivery_days(new["de"].index) != day - pd.Timedelta(days=1))]}, used

    def steps(second, request_errors=()):
        german = {"folder": "x", "files": {}, "errors": {n: "URLError" for n in request_errors}, "request_errors": list(request_errors)}
        return lambda paths, d, started, stamp: {"french_window": lambda: {"files": {}, "items": {}, "errors": {}}, "german_raw": lambda: german,
                                                 sl.SECOND_SOURCE: lambda: second(paths, stamp), "german_points": lambda: {"errors": {}}}

    def answers(series):
        return lambda paths, stamp: sl.fetch_second_source(day - pd.Timedelta(days=10), day - pd.Timedelta(days=1), paths.live / sl.SECOND_SOURCE, stamp,
                                                           opener=lambda request, timeout=None: FakeResponse(answer_of(series)))

    def down(paths, stamp):
        raise RuntimeError("HTTP 429 from energy-charts.info")

    recent = whole[np.asarray(sl.delivery_days(whole.index) >= day - pd.Timedelta(days=10))]
    monkeypatch.setattr(sl, "german_refetched", entsoe_without_the_eve)
    # 10:02 Paris: the second source cannot be reached and the German price of the eve is missing, so the run leaves it to the next
    monkeypatch.setattr(sl, "fetch_steps", steps(down))
    _, clock = ticking("2026-10-09T08:02:00Z", step=1)
    waited = sl.issue(replay_root, clock=clock, log=quiet)
    assert waited["status"] == "postponed" and "second source failed" in waited["reason"] and not sl.record_path(replay_root, "2026-10-10").exists()
    # 10:02 Paris, the second source answers with another auction's prices: nothing is taken, the record is written at once and says why
    wrong = recent + 3.0
    monkeypatch.setattr(sl, "fetch_steps", steps(answers(wrong)))
    dry = sl.issue(replay_root, dry_run=True, clock=clock, log=quiet)
    assert dry["status"] == "dry run written" and any(flag.startswith("the second source holds German prices that the files fetched from ENTSO-E lack but was not used: it disagrees with ENTSO-E on 216 of 216") for flag in dry["flags"])
    assert any(flag.startswith("yesterday's spread is missing") for flag in dry["flags"])
    # 10:02 Paris, the second source answers with the coupled prices: the 24 hours of the eve are taken from it and the inputs are whole.
    # The price request to ENTSO-E failed as well that morning; with the German side whole there is nothing to wait for
    monkeypatch.setattr(sl, "fetch_steps", steps(answers(recent), request_errors=["prices_DE_LU_202609290000_202610110000.xml"]))
    _, clock = ticking("2026-10-09T08:02:00Z", step=1)
    out = sl.issue(replay_root, clock=clock, log=quiet)
    assert out["status"] == "issued" and not any(flag.startswith("yesterday's spread is missing") for flag in out["flags"])
    assert "the coupled German price of 24 hour(s) before D came from the second source (energy-charts.info) because the files fetched from ENTSO-E held none" in out["flags"]
    assert "german_raw: 1 item(s) failed" in out["flags"]
    record = json.loads(Path(out["record"]).read_text())
    second = record["inputs"][sl.SECOND_SOURCE]
    assert second["trusted_this_morning"] and second["hours_taken_from_it_by_delivery_day"] == {"2026-10-09": 24} and len(second["files_used"]) == 1
    # the answer of this test also holds the prices of D and of the day after, as the stored raw files do: they are counted and never read
    assert second["hours_both_hold_over_the_ten_days_before_D"] == 9 * 24 and second["hours_of_D_or_later_in_its_answers_never_read"] == 48
    assert record["inputs"]["latest_german_price_hour_utc"] < "2026-10-09T22:00:00Z"
    assert record["fetch"][sl.SECOND_SOURCE]["sha256"] and record["fetch"][sl.SECOND_SOURCE]["finished_by_cutoff"] and record["all_fetches_finished_by_cutoff"]
    assert [u["sha256"] for u in second["files_used"]] == [record["fetch"][sl.SECOND_SOURCE]["sha256"]] and second["of_which_agree"] == 9 * 24
    assert record["evidence"]["yesterdays_spread_present_hours"] == 24 and record["evidence"]["german_price_hours_missing_on_the_eve"] == 0
    assert not sl.compare_fingerprints(record["fingerprint"]["inputs_of_D"], reference["fingerprint"]["inputs_of_D"])  # the inputs of D are those of the whole ENTSO-E series
    assert all(record["evidence"]["inputs_identical_when_everything_unknown_at_the_gate_is_scrambled"].values())
    assert all(len(record["forecasts"][m]) == 24 and np.isfinite(record["forecasts"][m]).all() for m in sl.MODELS)


@needs_data
def test_a_second_source_that_is_down_does_not_hold_up_a_morning_on_which_entsoe_has_the_price(replay_root, monkeypatch):
    def down():
        raise RuntimeError("HTTP 429 from energy-charts.info")

    monkeypatch.setattr(sl, "fetch_steps", lambda *a: {"french_window": lambda: {"files": {}, "items": {}, "errors": {}}, "german_raw": lambda: {"folder": "x", "files": {}, "errors": {}, "request_errors": []},
                                                       sl.SECOND_SOURCE: down, "german_points": lambda: {"errors": {}}})
    reference = json.loads((replay_root.static / "reference.json").read_text())
    _, clock = ticking("2026-10-09T08:02:00Z", step=1)
    out = sl.issue(replay_root, clock=clock, log=lambda *_: None)
    assert out["status"] == "issued" and f"{sl.SECOND_SOURCE} failed" in out["flags"]
    record = json.loads(Path(out["record"]).read_text())
    assert record["inputs"][sl.SECOND_SOURCE]["hours_taken_from_it"] == 0 and record["inputs"][sl.SECOND_SOURCE]["files_used"] == []
    for model in sl.MODELS:  # with nothing taken from the second source the forecasts are the frozen ones
        assert np.max(np.abs(np.asarray(record["forecasts"][model]) - np.asarray(reference["forecasts"][model]))) < 1e-9


@needs_data
def test_the_last_run_of_a_morning_issues_without_the_german_price_when_the_second_source_answers_nonsense(replay_root, monkeypatch):
    monkeypatch.setattr(sl, "LIVE_DATA_FROM", "2026-09-25")
    day = pd.Timestamp("2026-10-10")
    real = sl.german_refetched

    def entsoe_without_the_eve(paths, upto, live=True):
        new, used = real(paths, upto, live=live)
        return {**new, "de": new["de"][np.asarray(sl.delivery_days(new["de"].index) != day - pd.Timedelta(days=1))]}, used

    def nonsense(paths, stamp):
        return sl.fetch_second_source(day - pd.Timedelta(days=10), day - pd.Timedelta(days=1), paths.live / sl.SECOND_SOURCE, stamp, opener=lambda request, timeout=None: FakeResponse(b"<html>busy</html>"))

    monkeypatch.setattr(sl, "german_refetched", entsoe_without_the_eve)
    monkeypatch.setattr(sl, "fetch_steps", lambda paths, d, started, stamp: {"french_window": lambda: {"files": {}, "items": {}, "errors": {}}, "german_raw": lambda: {"folder": "x", "files": {}, "errors": {}, "request_errors": []},
                                                                           sl.SECOND_SOURCE: lambda: nonsense(paths, stamp), "german_points": lambda: {"errors": {}}})
    _, clock = ticking("2026-10-09T08:31:00Z", step=1)  # 10:31 Paris: past the last chance
    out = sl.issue(replay_root, clock=clock, log=lambda *_: None)
    assert out["status"] == "issued" and f"{sl.SECOND_SOURCE} failed" in out["flags"] and any(flag.startswith("yesterday's spread is missing") for flag in out["flags"])
    assert not any("came from the second source" in flag for flag in out["flags"])
    record = json.loads(Path(out["record"]).read_text())
    error = record["fetch"][sl.SECOND_SOURCE]["error"]
    saved = list((replay_root.live / sl.SECOND_SOURCE).glob("*.json"))
    assert len(saved) == 1 and saved[0].read_bytes() == b"<html>busy</html>"  # the answer is kept, and the record names it with its hash
    assert saved[0].name in error and sl.hashlib.sha256(b"<html>busy</html>").hexdigest() in error
    assert record["inputs"][sl.SECOND_SOURCE]["files_used"] == [] and record["inputs"][sl.SECOND_SOURCE]["hours_taken_from_it"] == 0
    assert record["evidence"]["german_price_hours_missing_on_the_eve"] == 24 and record["evidence"]["yesterdays_spread_present_hours"] == 0


def test_the_morning_asks_the_second_source_for_the_ten_days_before_D_and_never_for_D(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(sl, "fetch_second_source", lambda first, last, folder, stamp: asked.append((first, last, folder, stamp)) or {"errors": {}})
    day = pd.Timestamp("2026-11-23")
    steps = sl.fetch_steps(sl.Paths(tmp_path), day, pd.Timestamp("2026-11-22T09:00:10Z"), "20261122T090010Z")
    assert list(steps) == ["french_window", "german_raw", sl.SECOND_SOURCE, "german_points"] and set(sl.FETCH_BUDGET_SECONDS) == set(steps)
    steps[sl.SECOND_SOURCE]()
    assert asked == [(pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-22"), tmp_path / "live" / sl.SECOND_SOURCE, "20261122T090010Z")]
    asked.clear()  # in the first days of the live stretch the request starts where the stored files end
    sl.fetch_steps(sl.Paths(tmp_path), pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-14T08:00:10Z"), "x")[sl.SECOND_SOURCE]()
    assert asked[0][:2] == (pd.Timestamp("2026-10-11"), pd.Timestamp("2026-10-14"))


def test_saved_answers_that_cannot_be_combined_stop_no_record(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise NotImplementedError("a stamp beyond what the calendar holds")

    one = pd.DatetimeIndex(["2026-11-21T10:00Z"])
    de, tso = pd.Series([50.0], index=one), pd.DataFrame(1.0, index=one, columns=list(sl.se.TSO_COLUMNS))
    monkeypatch.setattr(sl, "with_second_source", broken)
    monkeypatch.setattr(sl.se, "load_stored", lambda *a: sl.se.Data(french=None, de=de.iloc[:0], tso=tso.iloc[:0], wind_pts=None, solar_pts=None, last=""))
    monkeypatch.setattr(sl, "german_refetched", lambda *a, **k: ({"de": de, "tso": tso}, []))
    monkeypatch.setattr(sl, "french_history", lambda *a, **k: (pd.DataFrame({"price_eur_mwh": de}), {"last_window_issued_utc": None}))
    monkeypatch.setattr(sl, "points_tables", lambda *a, **k: (None, None, []))
    data, info = sl.build_data(sl.Paths(tmp_path), pd.Timestamp("2026-11-23"), upto=pd.Timestamp("2026-11-22T09:00Z"))
    assert info[sl.SECOND_SOURCE]["could_not_be_read"].startswith("NotImplementedError") and info[sl.SECOND_SOURCE]["hours_taken_from_it"] == 0
    assert info[sl.SECOND_SOURCE]["files_used"] == [] and data.de.equals(de)  # the ENTSO-E series goes on untouched
