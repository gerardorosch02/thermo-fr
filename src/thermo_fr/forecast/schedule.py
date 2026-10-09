"""Which step a scheduled GitHub Actions run should perform, from the clock in Paris.

GitHub cron is UTC and Paris moves between UTC+1 and UTC+2, while the two
deadlines that matter are Paris times: the ENTSO-E load forecast is due at
10:00 on the day before delivery, and the EEX day-ahead future is traded from
about 11:15, so a forecast that is to be scored against the market must be
issued before then. The workflow therefore fires several pre-market crons
(08:10, 08:35, 09:10 and 09:35 UTC on weekdays) and this module keeps only
those that start inside the pre-market window, 10:05 to 11:00 Paris: in
summer the 08:10 and 08:35 UTC runs (10:10 and 10:35 Paris), in winter the
09:10 and 09:35 UTC runs. A run that starts before 10:05 Paris would be ahead
of the load forecast deadline; one that starts at or after 11:00 could finish
after the window opened. Both are skipped. One later cron (10:30 UTC, 12:30
Paris in summer and 11:30 in winter) always runs, for information only: its
version is issued after the market window and the dashboards label it so.

GitHub starts scheduled workflows late at busy times, often by five to fifteen
minutes, which is why two pre-market crons sit about 25 minutes apart in each
season instead of one.
"""

import argparse
import datetime as dt
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
PREMARKET_CRONS = ("10 8 * * 1-5", "35 8 * * 1-5", "10 9 * * 1-5", "35 9 * * 1-5")
LATE_CRON = "30 10 * * 1-5"
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
        return dispatch_step or "morning-run", "manual dispatch"
    if schedule == SETTLE_CRON:
        return "settle", "daily settlement"
    if schedule == REFIT_CRON:
        if paris.date() == first_weekday_of_month(paris.date()):
            return "refit", "first weekday of the month"
        return "skip", "refit runs on the first weekday of the month only"
    if schedule == LATE_CRON:
        return "morning-run", f"late run for information at {paris:%H:%M} Paris, after the market window"
    if schedule in PREMARKET_CRONS:
        clock = paris.time()
        if clock < PREMARKET_WINDOW[0]:
            return "skip", f"{paris:%H:%M} Paris is before the 10:00 load forecast deadline"
        if clock >= PREMARKET_WINDOW[1]:
            return "skip", f"{paris:%H:%M} Paris could finish after the market window opens at 11:15"
        return "morning-run", f"pre-market run at {paris:%H:%M} Paris"
    return "skip", f"unknown schedule {schedule!r}"


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
