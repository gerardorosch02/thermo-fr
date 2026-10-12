"""The laptop's dispatch of the Actions run: the gh command it sends, no secret on it, and the run record (ok, failed)."""

import json
import logging
import subprocess

import pandas as pd
import pytest

from thermo_fr.forecast import dispatch as d
from thermo_fr.forecast.store import Store
from thermo_fr.dashboard.data import last_publish_run


def fake_gh(created_runs, fail_dispatch=False):
    calls = []

    def runner(cmd, **kw):
        calls.append((cmd, kw))
        if cmd[1:3] == ["workflow", "run"]:
            if fail_dispatch:
                return subprocess.CompletedProcess(cmd, 1, "", "HTTP 401: Bad credentials")
            created_runs.append({"databaseId": 99, "number": 7, "status": "queued", "conclusion": None,
                                 "createdAt": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "url": "https://github.com/example/thermo-fr/actions/runs/99"})
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1:3] == ["run", "list"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(created_runs), "")
        raise AssertionError(cmd)

    return runner, calls


def test_dispatch_sends_workflow_run_with_the_step_and_finds_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "find_gh", lambda gh="gh": "gh")
    monkeypatch.setenv("ENTSOE_API_KEY", "not-for-the-command-line")
    runner, calls = fake_gh(created_runs=[{"databaseId": 1, "number": 1, "status": "completed", "conclusion": "success",
                                           "createdAt": "2026-10-09T16:19:37Z", "url": "https://github.com/example/old"}])
    run = d.dispatch_workflow(tmp_path, step="morning-run", runner=runner, sleep=lambda s: None, wait_s=1, poll_s=0)
    assert run["number"] == 7 and run["status"] == "queued"
    sent = calls[0][0]
    assert sent[1:] == ["workflow", "run", "forecast.yml", "--ref", "main", "-f", "step=morning-run"]
    for cmd, kw in calls:
        assert "not-for-the-command-line" not in " ".join(cmd) and "--token" not in cmd
        assert kw["env"]["GH_PROMPT_DISABLED"] == "1" and kw["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_dispatch_times_out_when_no_run_appears(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "find_gh", lambda gh="gh": "gh")

    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "[]" if cmd[1] == "run" else "", "")

    with pytest.raises(d.DispatchError, match="no workflow_dispatch run"):
        d.dispatch_workflow(tmp_path, runner=runner, sleep=lambda s: None, wait_s=0, poll_s=0)


def test_dispatch_records_ok_and_failed_runs_in_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "find_gh", lambda gh="gh": "gh")
    store = Store(tmp_path / "db.sqlite")
    runner, _ = fake_gh(created_runs=[])
    noon_summer = pd.Timestamp("2026-07-15T10:10Z")  # 12:10 Paris: a late dispatch
    ok = d.dispatch_and_record(store, tmp_path, kind="scheduled", runner=runner, now=noon_summer, sleep=lambda s: None, wait_s=1, poll_s=0,
                               log=_Quiet())
    assert ok["status"] == "ok" and "run #7" in ok["message"] and "after the market window" in ok["message"]
    last = last_publish_run(store, "dispatch")
    assert last["status"] == "ok" and last["kind"] == "scheduled"
    runner, _ = fake_gh(created_runs=[], fail_dispatch=True)
    failed = d.dispatch_and_record(store, tmp_path, kind="scheduled", runner=runner, now=pd.Timestamp("2026-07-15T08:06Z"),
                                   sleep=lambda s: None, wait_s=1, poll_s=0, log=_Quiet())
    assert failed["status"] == "failed" and "Bad credentials" in failed["message"] and "pre-market run at 10:06 Paris" in failed["message"]
    last = last_publish_run(store, "dispatch")
    assert last["status"] == "failed" and "Bad credentials" in last["message"]
    assert store.runs()["command"].eq("dispatch").sum() == 2
    store.close()


def test_a_network_error_is_retried_once_after_sixty_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "find_gh", lambda gh="gh": "gh")
    store = Store(tmp_path / "db.sqlite")
    created, state, slept = [], {"calls": 0}, []

    def flaky(cmd, **kw):
        if cmd[1:3] == ["workflow", "run"]:
            state["calls"] += 1
            if state["calls"] == 1:
                return subprocess.CompletedProcess(cmd, 1, "", "error connecting to api.github.com: dial tcp: lookup api.github.com: no such host")
            created.append({"databaseId": 5, "number": 8, "status": "queued", "conclusion": None,
                            "createdAt": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "url": "https://github.com/example/8"})
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps(created), "")

    ok = d.dispatch_and_record(store, tmp_path, kind="scheduled", runner=flaky, sleep=slept.append, wait_s=1, poll_s=0, log=_Quiet())
    assert ok["status"] == "ok" and ok["attempts"] == 2 and "no such host" in ok["message"] and "run #8" in ok["message"]
    assert slept[0] == d.RETRY_DELAY_S == 60.0 and state["calls"] == 2
    assert last_publish_run(store, "dispatch")["status"] == "ok"
    # two failures in a row: one retry only, then the failure is recorded
    always = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "network is unreachable")  # noqa: E731
    slept.clear()
    failed = d.dispatch_and_record(store, tmp_path, kind="scheduled", runner=always, sleep=slept.append, wait_s=1, poll_s=0, log=_Quiet())
    assert failed["status"] == "failed" and failed["attempts"] == 2 and slept == [60.0] and "after 2 attempt(s)" in failed["message"]
    assert last_publish_run(store, "dispatch")["status"] == "failed"
    store.close()


def test_cli_dispatch_exits_nonzero_on_failure(tmp_path, monkeypatch, capsys):
    from thermo_fr import cli

    def refuse(repo, **kw):
        raise d.DispatchError("HTTP 401: Bad credentials")

    monkeypatch.setattr(d, "dispatch_workflow", refuse)
    monkeypatch.setattr(d, "RETRY_DELAY_S", 0.0)  # the one retry, without the real minute
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger("thermo_fr")
    handlers_before = list(root.handlers)
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["dispatch", "--db", str(tmp_path / "db.sqlite"), "--repo", str(tmp_path), "--kind", "test", "--wait", "0"])
    finally:
        for handler in list(root.handlers):  # the command's log handlers point at this test's captured stderr and tmp_path
            if handler not in handlers_before:
                root.removeHandler(handler)
                handler.close()
    assert exc.value.code == 1 and "failed: dispatch of morning-run failed after 2 attempt(s): HTTP 401" in capsys.readouterr().out
    s = Store(tmp_path / "db.sqlite")
    assert s.runs().iloc[0]["status"] == "failed"
    s.close()


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
