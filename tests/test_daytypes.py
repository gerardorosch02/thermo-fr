"""Day types, bridge days, holiday neighbours and the comparable (same-type) day."""

import pandas as pd

from thermo_fr.forecast.daytypes import SATURDAY, SUNDAY_OR_HOLIDAY, WORKING_DAY, day_table, same_type_day


def test_day_types_and_bridge_days_on_known_dates():
    table = day_table(pd.DatetimeIndex([
        "2024-05-08",  # Wednesday, Victory Day
        "2024-05-09",  # Ascension Thursday
        "2024-05-10",  # the Friday pont
        "2024-05-11",  # Saturday
        "2024-05-13",  # a plain Monday
        "2025-05-01",  # Thursday, Labour Day
        "2025-05-02",  # pont
        "2024-07-14",  # a Sunday holiday
        "2024-12-24", "2024-12-25", "2024-12-31", "2025-01-02", "2025-01-03",
    ]))
    assert table.loc["2024-05-08", "day_type"] == SUNDAY_OR_HOLIDAY and table.loc["2024-05-08", "holiday"] == 1
    assert table.loc["2024-05-09", ["day_type", "holiday", "post_holiday"]].tolist() == [SUNDAY_OR_HOLIDAY, 1, 1]
    assert table.loc["2024-05-08", "pre_holiday"] == 1  # the eve of Ascension
    assert table.loc["2024-05-10", ["day_type", "bridge_day", "post_holiday"]].tolist() == [WORKING_DAY, 1, 1]
    assert table.loc["2024-05-11", ["day_type", "bridge_day"]].tolist() == [SATURDAY, 0]
    assert table.loc["2024-05-13", ["day_type", "bridge_day", "pre_holiday", "post_holiday"]].tolist() == [WORKING_DAY, 0, 0, 0]
    assert table.loc["2025-05-02", "bridge_day"] == 1 and table.loc["2025-05-01", "bridge_day"] == 0
    assert table.loc["2024-07-14", ["dow", "holiday", "day_type"]].tolist() == [6, 1, SUNDAY_OR_HOLIDAY]
    assert table.loc[["2024-12-24", "2024-12-25", "2024-12-31", "2025-01-02", "2025-01-03"], "year_end_break"].tolist() == [1, 1, 1, 1, 0]
    assert table.loc["2024-12-24", "pre_holiday"] == 1 and table.loc["2024-12-25", "day_type"] == SUNDAY_OR_HOLIDAY


def test_same_type_day_is_the_most_recent_comparable_day():
    days = pd.DatetimeIndex(["2024-05-13", "2024-05-11", "2024-05-09", "2024-05-10", "2024-12-25", "2026-10-10", "2026-10-12", "2025-05-01"])
    comparable = same_type_day(days)
    assert [str(d.date()) for d in comparable] == [
        "2024-05-10",  # Monday -> the Friday before (a working day, even though it was a pont)
        "2024-05-04",  # Saturday -> the Saturday before
        "2024-05-08",  # Ascension -> the holiday the day before
        "2024-05-07",  # pont Friday -> Tuesday, the last working day (Wednesday and Thursday were holidays)
        "2024-12-22",  # Christmas Wednesday -> the Sunday before
        "2026-10-03",  # Saturday -> Saturday
        "2026-10-09",  # Monday -> Friday
        "2025-04-27",  # 1 May (Thursday) -> the Sunday before
    ]
    # Always strictly earlier, never more than two weeks back.
    gap = (days - comparable).days
    assert (gap >= 1).all() and (gap <= 14).all()


def test_tables_are_in_input_order_and_empty_input_is_fine():
    assert day_table(pd.DatetimeIndex([])).empty and len(same_type_day(pd.DatetimeIndex([]))) == 0
    year = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    table = day_table(year)
    assert len(table) == 365 and table["holiday"].sum() == 11 and table["bridge_day"].sum() >= 2  # 2 May and 15 August 2025
    assert set(table["day_type"].unique()) == {WORKING_DAY, SATURDAY, SUNDAY_OR_HOLIDAY}
