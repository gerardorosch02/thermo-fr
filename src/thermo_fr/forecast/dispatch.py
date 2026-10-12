"""Trigger the GitHub Actions forecast workflow from the laptop (workflow_dispatch) and record the outcome in the local store.

GitHub's schedule trigger is delivered late by hours, or not at all, at busy times: on 10 and 11 October 2026 the 13:15 UTC settle
cron ran at 17:51 and 18:04 UTC and the four weekday morning crons never ran. A forecast that has to be issued between 10:05 and
11:00 Paris cannot ride on it. The laptop therefore dispatches the pre-market run itself, through `gh workflow run`, at 09:05 and
09:30 London (10:05 and 10:30 Paris all year, since London and Paris change clocks together); the crons stay in the workflow as a
backup and the workflow's `schedule-step` treats a dispatched run exactly like a cron run (see forecast/schedule.py), so a dispatch that
only starts after the window is labelled "issued after the market window, not tradeable" and never replaces a pre-market version.

No secret touches the command line: gh authenticates with the login stored in its own keyring (`gh auth login` once), and the
environment disables every prompt so a scheduled run that cannot authenticate fails with a message instead of hanging. Each dispatch
is recorded as a run (command "dispatch") in the local store, failures included, which the local dashboard shows in red.
"""

import os
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from .schedule import decide
from .store import Store

WORKFLOW = "forecast.yml"
GH_FALLBACKS = (r"C:\Program Files\GitHub CLI\gh.exe",)
NO_PROMPT_ENV = {"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


class DispatchError(RuntimeError):
    """gh failed, or the run did not appear in time."""


def find_gh(gh: str = "gh") -> str:
    found = shutil.which(gh)
    if found:
        return found
    for candidate in GH_FALLBACKS:
        if Path(candidate).exists():
            return candidate
    raise DispatchError("GitHub CLI (gh) not found on PATH; install it and run `gh auth login` once as this user")


def _run(args, cwd, runner=subprocess.run):
    result = runner(list(args), cwd=str(cwd), capture_output=True, text=True, env={**os.environ, **NO_PROMPT_ENV})
    if result.returncode != 0:
        raise DispatchError(f"{' '.join(Path(a).name if i == 0 else a for i, a in enumerate(args))}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def dispatch_workflow(repo, step: str = "morning-run", ref: str = "main", workflow: str = WORKFLOW, gh: str = "gh", runner=subprocess.run,
                      wait_s: float = 45.0, poll_s: float = 5.0, now=None, sleep=time.sleep) -> dict:
    """Dispatch `step` on `ref` and wait for the run to appear; returns its number, url and status."""
    exe = find_gh(gh)
    started = pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")
    _run([exe, "workflow", "run", workflow, "--ref", ref, "-f", f"step={step}"], repo, runner)
    deadline = time.monotonic() + wait_s
    while True:
        sleep(poll_s)
        listing = _run([exe, "run", "list", "--workflow", workflow, "--event", "workflow_dispatch", "--branch", ref, "--limit", "5",
                        "--json", "databaseId,number,status,conclusion,createdAt,url"], repo, runner)
        runs = json.loads(listing or "[]")
        fresh = [r for r in runs if pd.Timestamp(r["createdAt"]) >= started - pd.Timedelta(seconds=90)]
        if fresh:
            run = sorted(fresh, key=lambda r: r["createdAt"])[-1]
            return {"number": run["number"], "url": run["url"], "status": run["status"], "created_at": run["createdAt"]}
        if time.monotonic() >= deadline:
            raise DispatchError(f"gh workflow run returned but no workflow_dispatch run of {workflow} appeared within {wait_s:.0f} s")


def expected_label(now=None) -> str:
    """What schedule-step will make of a morning-run dispatched now (the same rule the workflow applies)."""
    moment = (pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")).to_pydatetime()
    step, reason = decide(moment, "workflow_dispatch", "", "morning-run")
    return f"{step}: {reason}"


def dispatch_and_record(store: Store, repo, step: str = "morning-run", kind: str = "manual", ref: str = "main", gh: str = "gh",
                        runner=subprocess.run, log=None, now=None, **kw) -> dict:
    """Dispatch and record the outcome as a run in the store (ok or failed); never raises for a failed dispatch."""
    import logging

    log = log or logging.getLogger("thermo_fr.dispatch")
    run_id = store.start_run("dispatch", kind, None, now=now)
    expected = expected_label(now) if step == "morning-run" else f"{step}: manual dispatch"
    try:
        run = dispatch_workflow(repo, step=step, ref=ref, gh=gh, runner=runner, now=now, **kw)
        message = f"dispatched {step} on {ref}: run #{run['number']} {run['status']} ({run['url']}); expected {expected}"
        log.info(message)
        store.finish_run(run_id, "ok", message, now=now)
        return {"status": "ok", "message": message, "run": run}
    except (DispatchError, OSError, ValueError) as exc:
        message = f"dispatch of {step} failed: {exc}; expected {expected}"
        log.error(message)
        store.finish_run(run_id, "failed", message, now=now)
        return {"status": "failed", "message": message}
