"""Trigger the GitHub Actions forecast workflow from the laptop (workflow_dispatch) and record the outcome in the local store.

GitHub's schedule trigger is delivered late by hours, or not at all, at busy times: on 10 and 11 October 2026 the 13:15 UTC settle
cron ran at 17:51 and 18:04 UTC and the four weekday morning crons never ran. A forecast that has to be issued between 10:05 and
11:00 Paris cannot ride on it. The laptop therefore dispatches the pre-market run itself, through `gh workflow run`, at 09:05 and
09:30 London (10:05 and 10:30 Paris all year, since London and Paris change clocks together), and the workflow's `schedule-step` decides
a dispatched run by the Paris clock (see forecast/schedule.py), so a dispatch that only starts after the window is labelled "issued
after the market window, not tradeable" and never replaces a pre-market version. The weekday morning crons were removed from the
workflow on 2026-10-13: on 12 October GitHub delivered them 9 to 12 hours late, which cannot back up a pre-market forecast and only
adds after-window versions. The settle is dispatched the same way at 14:20 London (`--step settle`), so the public record updates the
same afternoon; its 13:15 UTC cron stays as a backup, and a second settle of the same day changes nothing (the same prices are stored
again, nothing new is scored, and the workflow commits nothing when only the stamps in status.json changed).

The monthly refit is dispatched the same way (`thermo-fr dispatch --step refit --first-weekday-only`, a scheduled task on the 1st to
the 3rd at 07:00 London that dispatches on the first weekday of the month only), with the refit cron as a backup; the refit itself is
idempotent (`thermo-fr refit-model --once-per-month`), so whichever trigger reaches the workflow second exits as "already refitted
this month" without committing. Refit dispatches are recorded under the command "dispatch-refit".

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

from .schedule import PARIS, decide, first_weekday_of_month
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


def _moment(now=None) -> pd.Timestamp:
    return pd.Timestamp(now).tz_convert("UTC") if now is not None else pd.Timestamp.now(tz="UTC")


def expected_label(now=None, step: str = "morning-run") -> str:
    """What schedule-step will make of `step` dispatched now (the same rule the workflow applies)."""
    decided, reason = decide(_moment(now).to_pydatetime(), "workflow_dispatch", "", step)
    return f"{decided}: {reason}"


def run_command(step: str) -> str:
    """The command a dispatch is recorded under in the store: "dispatch" for the forecast, "dispatch-<step>" otherwise."""
    return "dispatch" if step == "morning-run" else f"dispatch-{step}"


def not_first_weekday(now=None) -> str | None:
    """A reason to skip when today in Paris is not the first weekday of the month, else None."""
    today = _moment(now).tz_convert(PARIS).date()
    first = first_weekday_of_month(today)
    if today == first:
        return None
    return f"{today} is not the first weekday of the month ({first}); the task fires on the 1st to the 3rd and dispatches once"


RETRY_DELAY_S = 60.0  # after waking from sleep the network can take a few seconds to return


def dispatch_and_record(store: Store, repo, step: str = "morning-run", kind: str = "manual", ref: str = "main", gh: str = "gh",
                        runner=subprocess.run, log=None, now=None, retries: int = 1, retry_delay_s: float | None = None,
                        sleep=time.sleep, first_weekday_only: bool = False, **kw) -> dict:
    """Dispatch and record the outcome as a run in the store (ok, skipped or failed); never raises for a failed dispatch.

    A network or gh error is retried once after `retry_delay_s` seconds before a failure is recorded; the second scheduled dispatch
    of the morning is the next chance after that. With `first_weekday_only` nothing is sent unless today (Paris) is the first weekday
    of the month: the refit task fires on the 1st to the 3rd and the other days are recorded as skipped.
    """
    import logging

    log = log or logging.getLogger("thermo_fr.dispatch")
    retry_delay_s = RETRY_DELAY_S if retry_delay_s is None else retry_delay_s
    run_id = store.start_run(run_command(step), kind, None, now=now)
    expected = expected_label(now, step)
    if first_weekday_only:
        reason = not_first_weekday(now)
        if reason is not None:
            message = f"{step} not dispatched: {reason}"
            log.info(message)
            store.finish_run(run_id, "skipped", message, now=now)
            return {"status": "skipped", "message": message, "attempts": 0}
    attempt, errors = 0, []
    while True:
        attempt += 1
        try:
            run = dispatch_workflow(repo, step=step, ref=ref, gh=gh, runner=runner, now=now, sleep=sleep, **kw)
            retried = f" (succeeded on attempt {attempt} after: {'; '.join(errors)})" if errors else ""
            message = f"dispatched {step} on {ref}: run #{run['number']} {run['status']} ({run['url']}); expected {expected}{retried}"
            log.info(message)
            store.finish_run(run_id, "ok", message, now=now)
            return {"status": "ok", "message": message, "run": run, "attempts": attempt}
        except (DispatchError, OSError, ValueError) as exc:
            errors.append(str(exc))
            if attempt <= retries:
                log.warning("dispatch attempt %d failed (%s); retrying in %.0f s", attempt, exc, retry_delay_s)
                sleep(retry_delay_s)
                continue
            message = f"dispatch of {step} failed after {attempt} attempt(s): {' | '.join(errors)}; expected {expected}"
            log.error(message)
            store.finish_run(run_id, "failed", message, now=now)
            return {"status": "failed", "message": message, "attempts": attempt}
