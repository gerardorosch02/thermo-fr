"""Calendar structure of French delivery days: day types, holidays and their neighbours.

Prices follow demand, and demand follows the working calendar more than the
weekday number alone says. A public holiday behaves like a Sunday, the Friday
after Ascension Thursday or the Monday before a Tuesday holiday is a "pont"
(bridge day) with much of industry closed, the days around Christmas and New
Year are a holiday week, and the day after a holiday ramps back up. All of
this is known years ahead, so every feature here passes the 12:00 gate.

Day types:
    0  working day (Monday to Friday, not a public holiday)
    1  Saturday (not a public holiday)
    2  Sunday or public holiday

The most recent earlier day of the same type is the natural "comparable"
day for a price lag: the previous Friday for a Monday, the previous Saturday
for a Saturday, the previous Sunday (or holiday) for a holiday that falls on
a Thursday. Its price is public before the gate as long as the comparable
day is D-1 or earlier, which it always is.

Public holidays come from the `holidays` package (national French calendar:
New Year, Easter Monday, 1 and 8 May, Ascension, Whit Monday, 14 July,
Assumption, All Saints, Armistice, Christmas). Regional extras (Good Friday
and 26 December in Alsace-Moselle) are not counted.
"""

import holidays
import numpy as np
import pandas as pd

WORKING_DAY, SATURDAY, SUNDAY_OR_HOLIDAY = 0, 1, 2
MAX_LOOKBACK_DAYS = 14
YEAR_END_BREAK = ((12, 24), (1, 2))  # 24 December to 2 January inclusive

COLUMNS = ["dow", "holiday", "day_type", "bridge_day", "pre_holiday", "post_holiday", "year_end_break"]


def french_holidays(days: pd.DatetimeIndex) -> np.ndarray:
    """1 where the day is a French public holiday."""
    days = pd.DatetimeIndex(days)
    if len(days) == 0:
        return np.zeros(0, dtype=int)
    calendar = holidays.France(years=range(days.year.min() - 1, days.year.max() + 2))
    return np.array([d.date() in calendar for d in days], dtype=int)


def day_table(days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per distinct day (naive midnight timestamps) with the COLUMNS above."""
    unique = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days).normalize())))
    if len(unique) == 0:
        return pd.DataFrame(columns=COLUMNS, dtype=int)
    # one day of margin on each side so that the neighbours of the first and last day are known
    full = pd.date_range(unique.min() - pd.Timedelta(days=1), unique.max() + pd.Timedelta(days=1), freq="D")
    holiday = french_holidays(full)
    dow = full.dayofweek.to_numpy()
    non_working = (dow >= 5) | (holiday == 1)
    day_type = np.where(holiday == 1, SUNDAY_OR_HOLIDAY, np.where(dow == 5, SATURDAY, np.where(dow == 6, SUNDAY_OR_HOLIDAY, WORKING_DAY)))
    prev_non_working = np.roll(non_working, 1)
    next_non_working = np.roll(non_working, -1)
    bridge = (~non_working) & prev_non_working & next_non_working
    pre_holiday = np.roll(holiday, -1)
    post_holiday = np.roll(holiday, 1)
    month, day = full.month.to_numpy(), full.day.to_numpy()
    (m1, d1), (m2, d2) = YEAR_END_BREAK
    year_end = ((month == m1) & (day >= d1)) | ((month == m2) & (day <= d2))
    table = pd.DataFrame(
        {
            "dow": dow,
            "holiday": holiday,
            "day_type": day_type,
            "bridge_day": bridge.astype(int),
            "pre_holiday": pre_holiday,
            "post_holiday": post_holiday,
            "year_end_break": year_end.astype(int),
        },
        index=full,
    )
    return table.loc[unique]


def same_type_day(days: pd.DatetimeIndex, max_back: int = MAX_LOOKBACK_DAYS) -> pd.DatetimeIndex:
    """For each day, the most recent earlier day of the same type (within max_back days; else the day before)."""
    days = pd.DatetimeIndex(pd.DatetimeIndex(days).normalize())
    if len(days) == 0:
        return days
    full = pd.date_range(days.min() - pd.Timedelta(days=max_back), days.max(), freq="D")
    types = day_table(full)["day_type"].reindex(full).to_numpy()
    result = np.empty(len(days), dtype="datetime64[ns]")
    position = {d: i for i, d in enumerate(full)}
    for k, day in enumerate(days):
        i = position[day]
        found = day - pd.Timedelta(days=1)
        for back in range(1, max_back + 1):
            if types[i - back] == types[i]:
                found = full[i - back]
                break
        result[k] = found.to_datetime64()
    return pd.DatetimeIndex(result)
