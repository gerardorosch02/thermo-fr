"""When each input becomes known, relative to the day-ahead auction.

The French day-ahead auction (SDAC) closes at 12:00 Paris time on the day
before delivery. A forecast for delivery day D may only use information that
existed at that moment, which this module calls the gate for D.

Every feature column has a timing rule: a function from the valid hour of a
row (UTC) to the moment that feature's value became public (UTC). The rules
encode what is documented, not what the API returns, because the ENTSO-E API
returns no publication time for historical data (its createdDateTime is the
query time). Sources:

- Day-ahead prices for delivery day X are published by the market coupling
  shortly after the auction, taken here as 13:00 Paris on X-1. So the price
  of D-1 is known at the gate for D, and the price of D is not.
- Day-ahead total load forecast: Regulation (EU) 543/2013, Article 6(1)(b),
  "published no later than two hours before the gate closure of the day-ahead
  market", so 10:00 Paris on D-1, and "updated when significant changes
  occur". The API holds the latest version, so a later update cannot be ruled
  out; the deadline is used.
- Day-ahead wind and solar generation forecast: Article 14(1)(d), published
  "no later than 18:00 Brussels time, one day before actual delivery", so
  it may arrive after the auction. Taken as 18:00 Paris on D-1, which fails
  the gate. That is why the honest feature set excludes it.
- Weather forecasts from the Open-Meteo previous-runs archive, lead day 2: the
  run initialised 48 to 53 hours before the valid hour, counted as available
  six hours after initialisation (global models are disseminated within four
  to six hours). The latest run used for any hour of D is 18 UTC on D-2.
- Weather from the historical-forecast archive: stitched from the latest run
  before each hour, so it is taken as known one hour before valid time. It
  is a training proxy only and always fails the gate.
- Calendar features are known in advance.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..config import LOCAL_TZ

GATE_HOUR = 12  # Paris local time, on the day before delivery
PRICE_PUBLICATION_HOUR = 13
LOAD_FORECAST_HOUR = 10
WIND_SOLAR_FORECAST_HOUR = 18
WEATHER_RUN_CYCLE_HOURS = 6
WEATHER_AVAILABILITY_LAG = pd.Timedelta(hours=6)
WEATHER_LEAD_DAYS = 2
FAR_PAST = pd.Timestamp("1900-01-01", tz="UTC")


class LookaheadError(AssertionError):
    """A feature for delivery day D uses information published after the gate for D."""


def delivery_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local (Paris) calendar day of each valid hour, as naive midnight timestamps."""
    local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
    return pd.DatetimeIndex(local.normalize().tz_localize(None))


def gate_closure(days) -> pd.DatetimeIndex:
    """12:00 Paris on the day before each delivery day, in UTC."""
    days = pd.DatetimeIndex(pd.to_datetime(days))
    local = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_HOUR)).tz_localize(LOCAL_TZ)
    return local.tz_convert("UTC")


def gate_for(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Gate closure for the delivery day of each valid hour."""
    return gate_closure(delivery_days(index))


def _local_clock(days, offset_days: int, hour: int) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(pd.to_datetime(days))
    local = (days + pd.Timedelta(days=offset_days) + pd.Timedelta(hours=hour)).tz_localize(LOCAL_TZ)
    return local.tz_convert("UTC")


def price_lag_issue(lag_days: int) -> Callable:
    """Price of delivery day D - lag was published at 13:00 Paris on D - lag - 1."""

    def rule(index):
        return _local_clock(delivery_days(index), -lag_days - 1, PRICE_PUBLICATION_HOUR)

    return rule


def load_forecast_issue(index) -> pd.DatetimeIndex:
    return _local_clock(delivery_days(index), -1, LOAD_FORECAST_HOUR)


def wind_solar_forecast_issue(index) -> pd.DatetimeIndex:
    return _local_clock(delivery_days(index), -1, WIND_SOLAR_FORECAST_HOUR)


def weather_issued_issue(index, lead_days: int = WEATHER_LEAD_DAYS) -> pd.DatetimeIndex:
    """Run initialisation (valid - lead, floored to the 6-hour cycle) plus the availability lag."""
    valid = pd.DatetimeIndex(index).tz_convert("UTC")
    run = (valid - pd.Timedelta(days=lead_days)).floor(f"{WEATHER_RUN_CYCLE_HOURS}h")
    return run + WEATHER_AVAILABILITY_LAG


def weather_proxy_issue(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).tz_convert("UTC") - pd.Timedelta(hours=1)


def calendar_issue(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([FAR_PAST] * len(index))


@dataclass(frozen=True)
class InputTiming:
    name: str
    issue: Callable[[pd.DatetimeIndex], pd.DatetimeIndex]
    description: str


TIMINGS = {
    "calendar": InputTiming("calendar", calendar_issue, "Hour, weekday, month and public holidays are known in advance."),
    "price_lag1": InputTiming("price_lag1", price_lag_issue(1), "Day-ahead price of D-1, published about 13:00 Paris on D-2."),
    "price_lag2": InputTiming("price_lag2", price_lag_issue(2), "Day-ahead price of D-2, published about 13:00 Paris on D-3."),
    "price_lag7": InputTiming("price_lag7", price_lag_issue(7), "Day-ahead price of D-7, published about 13:00 Paris on D-8."),
    "load_forecast": InputTiming(
        "load_forecast", load_forecast_issue,
        "ENTSO-E day-ahead total load forecast, due two hours before gate closure (10:00 Paris on D-1).",
    ),
    "wind_solar_forecast": InputTiming(
        "wind_solar_forecast", wind_solar_forecast_issue,
        "ENTSO-E day-ahead wind and solar forecast, due by 18:00 Paris on D-1: after the auction.",
    ),
    "weather_issued": InputTiming(
        "weather_issued", weather_issued_issue,
        "Open-Meteo previous-runs archive, lead day 2: runs of D-2, available by 00:00 UTC on D-1.",
    ),
    "weather_proxy": InputTiming(
        "weather_proxy", weather_proxy_issue,
        "Open-Meteo historical-forecast archive: latest run before each hour, not point in time.",
    ),
}


def issue_times(index: pd.DatetimeIndex, timing_name: str) -> pd.DatetimeIndex:
    return TIMINGS[timing_name].issue(pd.DatetimeIndex(index))


def lateness(index: pd.DatetimeIndex, timing_name: str) -> pd.TimedeltaIndex:
    """How long after the gate each value became known (negative means before)."""
    return issue_times(index, timing_name) - gate_for(index)


def check_point_in_time(index: pd.DatetimeIndex, feature_timings: dict[str, str]) -> None:
    """Raise LookaheadError if any feature's issue time is after the gate for any row.

    `feature_timings` maps feature column names to the names in TIMINGS.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return
    gate = gate_for(index)
    offenders = []
    for feature, timing_name in feature_timings.items():
        late = issue_times(index, timing_name) - gate
        if (late > pd.Timedelta(0)).any():
            worst = late.max()
            offenders.append(f"{feature} ({timing_name}): up to {worst} after the gate")
    if offenders:
        raise LookaheadError("Features use information published after 12:00 Paris on D-1: " + "; ".join(offenders))
