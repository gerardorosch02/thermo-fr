"""The morning logger of the spread forecasts. Unit tests need no data. The end-to-end tests replay 2026-10-10, a day of the
opened window, from a folder of static files named by SPREAD_LIVE_ROOT (they are skipped without it). No network is used and
no price of the test window is read."""

import io
import json
import os
import shutil
import urllib.error
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
    assert result["pass"] and result["hours"] == 24 and max(result["largest_difference"].values()) < 1e-9


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
    assert out["status"] == "issued" and "german_raw failed" in out["flags"] and "german_points: 1 item(s) missing or failed" in out["flags"]
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
