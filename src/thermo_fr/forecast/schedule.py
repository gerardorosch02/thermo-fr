"""Which step a GitHub Actions run should perform, from the clock in Paris.

GitHub cron is UTC and Paris moves between UTC+1 and UTC+2, while the two
deadlines that matter are Paris times: the ENTSO-E load forecast is due at
10:00 on the day before delivery, and the EEX day-ahead future is traded from
about 11:15, so a forecast that is to be scored against the market must be
issued before then. The pre-market window is therefore 10:05 to 11:00 Paris:
a run that starts before 10:05 would be ahead of the load forecast deadline,
one that starts at or after 11:00 could finish after the window opened.

The pre-market run is triggered by the laptop (`thermo-fr dispatch`, a
workflow_dispatch at 10:05 and 10:30 Paris) and decided here by that clock
rule: skipped before the load forecast deadline, pre-market inside the window,
and from 11:00 Paris a late run for information only, whose version the
dashboards label "issued after the market window, not tradeable"; a later
version never replaces a pre-market one, since the headline is the last version
issued before the window (market.headline_version). The workflow had weekday
morning crons as a backup until 2026-10-13; they were removed because GitHub
delivered them hours late (on 12 October 2026 between 9 and 12 hours late), so
they could not back up a pre-market forecast and only added after-window
versions and failure risk. Two crons remain, both backups for a laptop
dispatch: the settle at 13:15 UTC (the laptop dispatches it at 14:20 London)
and the refit at 02:00 UTC on the 1st to the 3rd, which proceeds on the first
weekday of the month only (the laptop dispatches it at 07:00 London that day).
The settle is idempotent (a second settle stores the same prices again and
scores nothing new) and the refit runs once per month
(`thermo-fr refit-model --once-per-month`), so a backup that arrives after the
dispatch changes nothing. Any other schedule string is skipped.
"""

import argparse
import datetime as dt
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
SETTLE_CRON = "15 13 * * *"
REFIT_CRON = "0 2 1-3 * *"
PREMARKET_WINDOW = (dt.time(10, 5), dt.time(11, 0))  # Paris, start inclusive, end exclusive


def first_weekday_of_month(day: dt.date) -> dt.date:
    first = day.replace(day=1)
    if first.weekday() > 4:
        first += dt.timedelta(days=(7 - first.weekday()) % 7)
    return first


def decide(now_utc: dt.datetime, event: str, schedule: str = "", dispatch_step: str = "") -> tuple[str, str]:
    """(step, reason). step is morning-run, settle, refit or skip."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=dt.timezone.utc)
    paris = now_utc.astimezone(PARIS)
    if event == "workflow_dispatch":
        step = dispatch_step or "morning-run"
        if step == "refit":
            return step, "dispatched refit; once per month, a second trigger in the same month exits as already refitted"
        if step == "settle":
            return step, "dispatched settle; a second settle of the same day stores the same prices again and scores nothing new"
        if step != "morning-run":
            return step, "manual dispatch"
        clock = paris.time()
        if clock < PREMARKET_WINDOW[0]:
            return "skip", f"dispatched at {paris:%H:%M} Paris, before the 10:00 load forecast deadline"
        if clock >= PREMARKET_WINDOW[1]:
            return "morning-run", f"dispatched late run for information at {paris:%H:%M} Paris, after the market window"
        return "morning-run", f"dispatched pre-market run at {paris:%H:%M} Paris"
    if schedule == SETTLE_CRON:
        return "settle", "daily settlement (backup cron)"
    if schedule == REFIT_CRON:
        if paris.date() == first_weekday_of_month(paris.date()):
            return "refit", "first weekday of the month (backup cron)"
        return "skip", "refit runs on the first weekday of the month only"
    return "skip", f"unknown schedule {schedule!r}; the morning crons were removed on 2026-10-13 (the laptop dispatches the pre-market run)"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Print the step a scheduled run should perform (for the GitHub Actions workflow).")
    parser.add_argument("--event", required=True, help="github.event_name")
    parser.add_argument("--schedule", default="", help="github.event.schedule")
    parser.add_argument("--input", dest="dispatch_step", default="", help="the step chosen on a manual dispatch")
    parser.add_argument("--now", default=None, help="UTC time to decide for, ISO format; default the current time")
    args = parser.parse_args(argv)
    now = dt.datetime.fromisoformat(args.now) if args.now else dt.datetime.now(dt.timezone.utc)
    step, reason = decide(now, args.event, args.schedule, args.dispatch_step)
    print(f"step={step}")
    print(f"reason={reason}")


if __name__ == "__main__":
    main()
