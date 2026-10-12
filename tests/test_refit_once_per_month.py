"""The refit runs once per month: metadata of the current month stops a second trigger, the laptop dispatches it on the first weekday
only, the metadata records the trigger and the local dashboard shows the last refit (red without one by the 5th)."""

import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from thermo_fr import cli
from thermo_fr.dashboard import data as q
from thermo_fr.forecast import dispatch as d
from thermo_fr.forecast import refit as r
from thermo_fr.forecast.gitsync import GitSyncError
from thermo_fr.forecast.schedule import decide
from thermo_fr.forecast.store import Store

MONDAY = pd.Timestamp("2026-11-02T07:10:00Z")  # Monday 2 November 2026, the first weekday of the month


def write_meta(model_dir, feature_set, fitted_at, trigger=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {"feature_set": feature_set, "model_file": f"{feature_set}.txt", "fitted_at_utc": fitted_at}
    if trigger:
        meta["trigger"] = trigger
    (model_dir / f"{feature_set}.json").write_text(json.dumps(meta))


def test_a_refit_of_the_current_month_in_every_set_is_reported(tmp_path):
    write_meta(tmp_path, "honest_v2", "2026-11-02T02:14:00Z", "cron")
    write_meta(tmp_path, "honest", "2026-11-02T02:31:00Z", "cron")
    done = r.refit_this_month(tmp_path, ("honest_v2", "honest"), now=MONDAY + pd.Timedelta(days=20))
    assert done == {"month": "2026-11", "fitted_at_utc": "2026-11-02T02:31:00Z", "trigger": "cron", "feature_sets": ["honest_v2", "honest"]}


def test_no_metadata_a_previous_month_or_one_set_behind_means_the_refit_runs(tmp_path):
    assert r.refit_this_month(tmp_path, ("honest_v2",), now=MONDAY) is None
    write_meta(tmp_path, "honest_v2", "2026-10-10T09:27:37Z")  # last month, written before the trigger field existed
    assert r.refit_this_month(tmp_path, ("honest_v2",), now=MONDAY) is None
    write_meta(tmp_path, "honest_v2", "2026-11-02T07:20:00Z", "dispatch")
    assert r.refit_this_month(tmp_path, ("honest_v2",), now=MONDAY)["trigger"] == "dispatch"
    assert r.refit_this_month(tmp_path, ("honest_v2", "honest"), now=MONDAY) is None  # the fallback set has no metadata yet
    write_meta(tmp_path, "honest", "2026-10-10T09:30:00Z")
    assert r.refit_this_month(tmp_path, ("honest_v2", "honest"), now=MONDAY) is None  # and here it is a month behind


def test_months_are_counted_on_the_paris_calendar(tmp_path):
    assert r.refit_month("2026-10-31T23:30:00Z") == "2026-11"  # 00:30 Paris on 1 November
    assert r.refit_month("2026-06-30T22:30:00Z") == "2026-07"  # 00:30 Paris on 1 July (summer time)
    write_meta(tmp_path, "honest_v2", "2026-10-31T23:30:00Z", "cron")
    assert r.refit_this_month(tmp_path, ("honest_v2",), now=MONDAY) is not None
    assert r.refit_this_month(tmp_path, ("honest_v2",), now="2026-11-30T23:30:00Z") is None  # already 1 December in Paris


def _drop_new_handlers(before):
    root = logging.getLogger("thermo_fr")
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


def test_cli_once_per_month_exits_without_fetching_and_otherwise_passes_the_trigger(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_fetch_and_refit(start, end, **kw):
        calls.append(kw)
        return {}

    monkeypatch.setattr(r, "fetch_and_refit", fake_fetch_and_refit)
    model_dir = tmp_path / "model"
    this_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    write_meta(model_dir, "honest_v2", this_month, "dispatch")
    write_meta(model_dir, "honest", this_month, "dispatch")
    before_meta = (model_dir / "honest_v2.json").read_text()
    common = ["refit-model", "--out", str(model_dir), "--feature-sets", "honest_v2", "honest", "--logs-dir", str(tmp_path / "logs"),
              "--cache-dir", str(tmp_path / "cache"), "--csv-dir", str(tmp_path / "csv"), "--data", str(tmp_path / "inputs.csv")]
    handlers = list(logging.getLogger("thermo_fr").handlers)
    try:
        cli.main(common + ["--once-per-month", "--trigger", "cron"])
        out = capsys.readouterr().out
        assert "already refitted this month: honest_v2, honest fitted" in out and "(dispatch trigger)" in out and "nothing written" in out
        assert calls == [] and (model_dir / "honest_v2.json").read_text() == before_meta
        # Last month's metadata, or no --once-per-month, lets the refit run, with the trigger passed through
        write_meta(model_dir, "honest", "2026-09-01T02:00:00Z", "cron")
        cli.main(common + ["--once-per-month", "--trigger", "cron"])
        cli.main(common + ["--trigger", "dispatch"])
    finally:
        _drop_new_handlers(handlers)
    assert [c["trigger"] for c in calls] == ["cron", "dispatch"] and all(c["feature_sets"] == ("honest_v2", "honest") for c in calls)


def test_refit_model_records_the_trigger_in_the_metadata(tmp_path, monkeypatch):
    from test_forecast_features import synthetic_inputs
    from thermo_fr.forecast import models

    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 10)
    history = synthetic_inputs("2023-09-01", "2024-03-01", issued_from="2024-01-10")
    meta = r.refit_model(history, out_dir=tmp_path, holdout_days=0, log=lambda *_: None, params={}, save_proxies=False, trigger="cron",
                         now=pd.Timestamp("2024-03-01T02:00Z"))
    assert meta["trigger"] == "cron" and json.loads((tmp_path / "honest.json").read_text())["trigger"] == "cron"
    assert r.refit_this_month(tmp_path, ("honest",), now="2024-03-20T10:00Z")["trigger"] == "cron"
    with pytest.raises(ValueError, match="trigger must be one of"):
        r.refit_model(history, out_dir=tmp_path, holdout_days=0, log=lambda *_: None, params={}, save_proxies=False, trigger="scheduled")


def _gh_for(now):
    sent = []

    def runner(cmd, **kw):
        if cmd[1:3] == ["workflow", "run"]:
            sent.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1:3] == ["run", "list"]:
            created = pd.Timestamp(now).strftime("%Y-%m-%dT%H:%M:%SZ")
            return subprocess.CompletedProcess(cmd, 0, json.dumps([{"databaseId": 5, "number": 12, "status": "queued", "conclusion": None,
                                                                    "createdAt": created, "url": "https://github.com/example/runs/5"}]), "")
        raise AssertionError(cmd)

    return runner, sent


@pytest.mark.parametrize("day, dispatched", [
    ("2026-11-02T07:10:00Z", True),   # Monday 2 November: the first weekday
    ("2026-11-01T07:10:00Z", False),  # Sunday 1 November
    ("2026-11-03T07:10:00Z", False),  # the day after: the task fires again, nothing is sent
    ("2026-08-01T05:10:00Z", False),  # Saturday 1 August
    ("2026-08-03T05:10:00Z", True),   # Monday 3 August
    ("2026-12-01T07:10:00Z", True),   # Tuesday 1 December
])
def test_the_scheduled_refit_dispatch_fires_on_the_first_weekday_only(tmp_path, monkeypatch, day, dispatched):
    monkeypatch.setattr(d, "find_gh", lambda gh="gh": "gh")
    runner, sent = _gh_for(day)
    store = Store(tmp_path / "db.sqlite")
    result = d.dispatch_and_record(store, tmp_path, step="refit", kind="scheduled", runner=runner, sleep=lambda s: None, wait_s=1, poll_s=0,
                                   now=pd.Timestamp(day), first_weekday_only=True, log=_Quiet())
    run = q.last_publish_run(store, "dispatch-refit")
    store.close()
    if dispatched:
        assert result["status"] == "ok" and sent[0][1:] == ["workflow", "run", "forecast.yml", "--ref", "main", "-f", "step=refit"]
        assert run["status"] == "ok" and "dispatched refit on main: run #12" in run["message"] and "once per month" in run["message"]
    else:
        assert result["status"] == "skipped" and sent == [] and run["status"] == "skipped"
        assert "not the first weekday of the month" in run["message"]


def test_without_the_flag_a_refit_dispatch_goes_out_any_day_and_the_forecast_dispatch_keeps_its_command(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "find_gh", lambda gh="gh": "gh")
    runner, sent = _gh_for("2026-11-03T07:10:00Z")
    store = Store(tmp_path / "db.sqlite")
    assert d.dispatch_and_record(store, tmp_path, step="refit", kind="manual", runner=runner, sleep=lambda s: None, wait_s=1, poll_s=0,
                                 now=pd.Timestamp("2026-11-03T07:10:00Z"), log=_Quiet())["status"] == "ok"
    assert len(sent) == 1 and q.last_publish_run(store, "dispatch-refit")["kind"] == "manual" and q.last_publish_run(store, "dispatch") is None
    store.close()


def test_cli_refit_dispatch_skipped_day_exits_zero(tmp_path, monkeypatch, capsys):
    def never(repo, **kw):
        raise AssertionError("nothing should be dispatched")

    monkeypatch.setattr(d, "dispatch_workflow", never)
    monkeypatch.setattr(d, "_moment", lambda now=None: pd.Timestamp("2026-11-03T07:10:00Z"))
    monkeypatch.chdir(tmp_path)
    handlers = list(logging.getLogger("thermo_fr").handlers)
    try:
        cli.main(["dispatch", "--step", "refit", "--first-weekday-only", "--db", str(tmp_path / "db.sqlite"), "--repo", str(tmp_path),
                  "--kind", "test", "--wait", "0"])
    finally:
        _drop_new_handlers(handlers)
    assert capsys.readouterr().out.startswith("skipped: refit not dispatched: 2026-11-03 is not the first weekday")


def test_schedule_step_decides_a_dispatched_refit():
    step, reason = decide(MONDAY.to_pydatetime(), "workflow_dispatch", "", "refit")
    assert step == "refit" and "once per month" in reason
    assert decide(pd.Timestamp("2026-11-03T02:01:00Z").to_pydatetime(), "schedule", "0 2 1-3 * *")[0] == "skip"  # the backup cron, day after


def test_refit_panel_levels_and_source(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    assert q.refit_panel(model_dir, now=MONDAY)["level"] == "missing"
    write_meta(model_dir, "honest_v2", "2026-10-10T09:27:37Z")  # last month, before the trigger field
    due = q.refit_panel(model_dir, now=MONDAY)
    assert due["level"] == "due" and due["trigger"] == "unknown" and "red from the 5th" in due["message"] and "2026-10-10 09:27 UTC" in due["message"]
    assert q.refit_panel(model_dir, now="2026-11-04T23:30:00Z")["level"] == "overdue"  # 5 November 00:30 in Paris
    assert q.refit_panel(model_dir, now="2026-11-04T22:30:00Z")["level"] == "due"
    assert q.refit_panel(model_dir, now="2026-11-20T10:00:00Z")["level"] == "overdue"
    write_meta(model_dir, "honest_v2", "2026-11-02T07:20:00Z", "dispatch")
    ok = q.refit_panel(model_dir, now="2026-11-20T10:00:00Z")
    assert ok["level"] == "ok" and ok["this_month"] and ok["trigger"] == "dispatch" and "trigger dispatch" in ok["message"]
    assert ok["source"].startswith("the local checkout")
    # With a repository the metadata comes from origin/main, where the workflow commits the refit; the local file is the fallback
    import thermo_fr.forecast.gitsync as gitsync

    remote = json.dumps({"fitted_at_utc": "2026-11-02T02:20:00Z", "trigger": "cron"})
    monkeypatch.setattr(gitsync, "fetch_file", lambda repo, path, **kw: remote if path == "published/model/honest_v2.json" else None)
    panel = q.refit_panel(Path("published/model"), now="2026-11-20T10:00:00Z", repo=tmp_path)
    assert panel["source"] == "origin/main" and panel["trigger"] == "cron"

    def down(repo, path, **kw):
        raise GitSyncError("git fetch origin main: could not resolve host")

    monkeypatch.setattr(gitsync, "fetch_file", down)
    fallback = q.refit_panel(model_dir, now="2026-11-20T10:00:00Z", repo=tmp_path)
    assert fallback["trigger"] == "dispatch" and "origin/main unreachable" in fallback["source"]


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
