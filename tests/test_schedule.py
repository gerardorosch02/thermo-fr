"""The workflow's step decision: a dispatched morning-run follows the Paris clock rule all year; only the settle and refit crons remain."""

import datetime as dt

from thermo_fr.forecast.schedule import REFIT_CRON, SETTLE_CRON, decide, first_weekday_of_month


def utc(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc)


def test_a_dispatched_morning_run_follows_the_clock_rule_all_year():
    # summer: 10:05 Paris is 08:05 UTC; winter: 09:05 UTC. The laptop dispatches at 09:05 and 09:30 London, 10:05 and 10:30 Paris
    for day, offset in (("2026-07-15", 2), ("2026-01-15", 1), ("2026-10-26", 1), ("2026-03-30", 2)):
        def at(hh, mm):
            return utc(f"{day}T{hh - offset:02d}:{mm:02d}:00")
        step, reason = decide(at(10, 0), "workflow_dispatch", "", "morning-run")
        assert step == "skip" and "load forecast deadline" in reason
        step, reason = decide(at(10, 6), "workflow_dispatch", "", "morning-run")
        assert step == "morning-run" and reason == "dispatched pre-market run at 10:06 Paris"
        assert decide(at(10, 31), "workflow_dispatch", "", "morning-run")[0] == "morning-run"
        assert decide(at(10, 59), "workflow_dispatch", "", "morning-run")[1].startswith("dispatched pre-market")
        step, reason = decide(at(11, 0), "workflow_dispatch", "", "morning-run")
        assert step == "morning-run" and "after the market window" in reason  # information only, labelled by the dashboards
        step, reason = decide(at(12, 10), "workflow_dispatch", "", "morning-run")
        assert step == "morning-run" and "12:10 Paris, after the market window" in reason
    step, reason = decide(utc("2026-07-15T15:00:00"), "workflow_dispatch", "")
    assert step == "morning-run" and "after the market window" in reason  # the default input is the morning-run


def test_settle_and_refit_crons_remain_as_backups_and_every_other_schedule_is_skipped():
    assert decide(utc("2026-07-15T13:16:00"), "schedule", SETTLE_CRON) == ("settle", "daily settlement (backup cron)")
    assert decide(utc("2026-10-12T21:43:59"), "schedule", SETTLE_CRON)[0] == "settle"  # delivered eight hours late: harmless
    assert decide(utc("2026-06-01T02:01:00"), "schedule", REFIT_CRON)[0] == "refit"  # Monday 1 June 2026
    assert decide(utc("2026-08-01T02:01:00"), "schedule", REFIT_CRON)[0] == "skip"  # Saturday; the refit waits for Monday 3 August
    assert decide(utc("2026-08-03T02:01:00"), "schedule", REFIT_CRON)[0] == "refit"
    assert first_weekday_of_month(dt.date(2026, 8, 20)) == dt.date(2026, 8, 3)
    # the removed morning crons: a workflow file that still carried one would be skipped, not run late
    for old in ("10 8 * * 1-5", "35 8 * * 1-5", "10 9 * * 1-5", "35 9 * * 1-5", "30 10 * * 1-5", "0 0 * * *"):
        step, reason = decide(utc("2026-07-15T08:12:00"), "schedule", old)
        assert step == "skip" and "unknown schedule" in reason


def test_dispatched_settle_and_refit_say_why_a_second_trigger_is_harmless():
    step, reason = decide(utc("2026-07-15T13:20:00"), "workflow_dispatch", "", "settle")
    assert step == "settle" and "second settle" in reason
    step, reason = decide(utc("2026-11-02T07:00:00"), "workflow_dispatch", "", "refit")
    assert step == "refit" and "once per month" in reason


def test_cli_prints_github_output_lines(capsys):
    from thermo_fr import cli

    cli.main(["schedule-step", "--event", "workflow_dispatch", "--input", "morning-run", "--now", "2026-07-15T08:12:00+00:00"])
    out = capsys.readouterr().out
    assert "step=morning-run" in out and "reason=dispatched pre-market run at 10:12 Paris" in out
    cli.main(["schedule-step", "--event", "schedule", "--schedule", SETTLE_CRON, "--now", "2026-07-15T13:16:00+00:00"])
    assert "step=settle" in capsys.readouterr().out
