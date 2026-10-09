"""The workflow's step decision from the Paris clock: pre-market window all year, late run for information, refit day."""

import datetime as dt

from thermo_fr.forecast.schedule import LATE_CRON, PREMARKET_CRONS, REFIT_CRON, SETTLE_CRON, decide, first_weekday_of_month


def utc(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc)


def test_summer_keeps_the_08_utc_runs_and_skips_the_09_utc_ones():
    # Paris is UTC+2: 08:10 UTC is 10:10 Paris, 09:10 UTC is 11:10 Paris
    assert decide(utc("2026-07-15T08:12:00"), "schedule", PREMARKET_CRONS[0])[0] == "morning-run"
    assert decide(utc("2026-07-15T08:40:00"), "schedule", PREMARKET_CRONS[1])[0] == "morning-run"
    step, reason = decide(utc("2026-07-15T09:12:00"), "schedule", PREMARKET_CRONS[2])
    assert step == "skip" and "market window" in reason
    assert decide(utc("2026-07-15T09:38:00"), "schedule", PREMARKET_CRONS[3])[0] == "skip"


def test_winter_keeps_the_09_utc_runs_and_skips_the_08_utc_ones():
    # Paris is UTC+1: 08:10 UTC is 09:10 Paris (before the load forecast deadline), 09:10 UTC is 10:10 Paris
    step, reason = decide(utc("2026-01-15T08:12:00"), "schedule", PREMARKET_CRONS[0])
    assert step == "skip" and "load forecast deadline" in reason
    assert decide(utc("2026-01-15T08:40:00"), "schedule", PREMARKET_CRONS[1])[0] == "skip"
    assert decide(utc("2026-01-15T09:12:00"), "schedule", PREMARKET_CRONS[2])[0] == "morning-run"
    assert decide(utc("2026-01-15T09:38:00"), "schedule", PREMARKET_CRONS[3])[0] == "morning-run"


def test_every_day_of_the_year_has_a_pre_market_run_even_with_a_late_start():
    for day in (dt.date(2026, 1, 15), dt.date(2026, 3, 30), dt.date(2026, 7, 15), dt.date(2026, 10, 26), dt.date(2026, 12, 15)):
        for delay in (0, 5, 15):  # minutes GitHub may start a scheduled workflow late
            kept = [cron for cron in PREMARKET_CRONS
                    if decide(dt.datetime.combine(day, dt.time(int(cron.split()[1]), int(cron.split()[0]) + delay), dt.timezone.utc),
                              "schedule", cron)[0] == "morning-run"]
            assert kept, (day, delay)
    # a run that only starts at 11:00 Paris or later is never kept
    assert decide(utc("2026-07-15T09:00:00"), "schedule", PREMARKET_CRONS[2])[0] == "skip"  # 11:00 Paris exactly
    assert decide(utc("2026-01-15T09:59:00"), "schedule", PREMARKET_CRONS[3])[0] == "morning-run"  # 10:59 Paris
    assert decide(utc("2026-01-15T10:00:00"), "schedule", PREMARKET_CRONS[3])[0] == "skip"  # 11:00 Paris


def test_late_run_settle_refit_and_dispatch():
    step, reason = decide(utc("2026-07-15T10:32:00"), "schedule", LATE_CRON)
    assert step == "morning-run" and "information" in reason and "12:32 Paris" in reason
    assert decide(utc("2026-07-15T13:16:00"), "schedule", SETTLE_CRON)[0] == "settle"
    assert decide(utc("2026-06-01T02:01:00"), "schedule", REFIT_CRON)[0] == "refit"  # Monday 1 June 2026
    assert decide(utc("2026-08-01T02:01:00"), "schedule", REFIT_CRON)[0] == "skip"  # Saturday; the refit waits for Monday 3 August
    assert decide(utc("2026-08-03T02:01:00"), "schedule", REFIT_CRON)[0] == "refit"
    assert first_weekday_of_month(dt.date(2026, 8, 20)) == dt.date(2026, 8, 3)
    assert decide(utc("2026-07-15T15:00:00"), "workflow_dispatch", "", "settle") == ("settle", "manual dispatch")
    assert decide(utc("2026-07-15T15:00:00"), "workflow_dispatch", "")[0] == "morning-run"
    assert decide(utc("2026-07-15T15:00:00"), "schedule", "0 0 * * *")[0] == "skip"


def test_cli_prints_github_output_lines(capsys):
    from thermo_fr import cli

    cli.main(["schedule-step", "--event", "schedule", "--schedule", PREMARKET_CRONS[0], "--now", "2026-07-15T08:12:00+00:00"])
    out = capsys.readouterr().out
    assert "step=morning-run" in out and "reason=pre-market run at 10:12 Paris" in out
