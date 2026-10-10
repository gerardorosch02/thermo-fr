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
- Wind and solar generation proxies (forecast/wind_proxy.py,
  forecast/solar_proxy.py): hub-height wind or radiation forecasts with the
  same lead-day-2 timing as the weather above, turned into MW with weights
  refitted at the start of each month on actual generation up to two days
  before the month. Actual generation per type is published within an hour
  of the operating period (Article 16(1)(a)), so the calibration data is
  taken as known at 01:00 Paris on the day before the month starts. The rule
  is the later of the two.
- The same-type price lag (the same hour of the most recent earlier day of
  the same type, see daytypes.py) is the price of a day no later than D-1,
  published at 13:00 Paris on the day before that day: at the latest 13:00
  on D-2.
- Calendar features, including day types, bridge days and the holiday
  neighbours, are known in advance.
- Actual nuclear generation (ENTSO-E A75, psrType B14) is published within
  an hour of the operating period (Article 16(1)(a)); on 2026-10-10 the
  latest quarter-hour was about 50 minutes old. The rules allow two hours.
  The full day D-2 is therefore known by 02:00 Paris on D-1, and the hours
  of D-1 that end by NUCLEAR_D1_CUTOFF_HOUR (08:00 Paris) are known by 10:00
  Paris on D-1, before the earliest pre-market run.
- Neighbour day-ahead prices (DE-LU, BE, NL, ES, IT-North, CH) for delivery
  day D-1 clear with the same coupling as France (Switzerland runs its own
  auction, also on D-2 for D-1), taken as 13:00 Paris on D-2 like the
  French price lag. The Swiss result for D itself, although it clears before
  the French gate, is not used: it is not reliably out by the pre-market
  issue time.
- The v2 residual load (load forecast minus the wind and solar proxies minus
  the latest known nuclear generation) is known when its latest component is,
  the load forecast at 10:00 Paris on D-1.

Two deadlines are checked. The gate (12:00 Paris on D-1) is the hard
constraint every honest feature must meet. The pre-market issue time,
PREMARKET_ISSUE (10:05 Paris on D-1, the start of the earliest scheduled
run), is the moment the forecast is actually produced; check_point_in_time
with deadline="premarket" verifies that every feature is known by then.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ

GATE_HOUR = 12  # Paris local time, on the day before delivery
PREMARKET_ISSUE = (10, 5)  # Paris, D-1: the earliest scheduled pre-market run starts here (forecast/schedule.py)
PRICE_PUBLICATION_HOUR = 13
NUCLEAR_PUBLICATION_LAG_HOURS = 2  # conservative; the regulation says one hour and about 50 minutes was observed
NUCLEAR_D1_CUTOFF_HOUR = 8  # Paris: D-1 hours ending by then are used, known by 10:00 Paris
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


def premarket_issue_for(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """10:05 Paris on the day before the delivery day of each valid hour, in UTC: when the earliest pre-market run starts."""
    days = delivery_days(index)
    local = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=PREMARKET_ISSUE[0], minutes=PREMARKET_ISSUE[1])).tz_localize(LOCAL_TZ)
    return local.tz_convert("UTC")


def _local_clock(days, offset_days: int, hour: int) -> pd.DatetimeIndex:
    """`hour` o'clock Paris on each day plus offset, in UTC. On the clock-change days a skipped hour moves forward and a repeated hour takes
    its first (summer-time) occurrence, the earlier of the two, so a rule never claims a later publication than the clock allows."""
    days = pd.DatetimeIndex(pd.to_datetime(days))
    naive = days + pd.Timedelta(days=offset_days) + pd.Timedelta(hours=hour)
    local = naive.tz_localize(LOCAL_TZ, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward")
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


PROXY_LAG_DAYS = 2  # the generation proxies are calibrated on actual generation to two days before the month
GENERATION_PUBLICATION_LAG_HOURS = 1


def proxy_calibration_issue(index) -> pd.DatetimeIndex:
    """Actual generation of (first day of the delivery month - 2) is public one hour after that day ends."""
    days = delivery_days(index)
    month_start = pd.DatetimeIndex(days.to_period("M").to_timestamp())
    return _local_clock(month_start, -(PROXY_LAG_DAYS - 1), GENERATION_PUBLICATION_LAG_HOURS)


def generation_proxy_issue(index) -> pd.DatetimeIndex:
    """The later of the weather run (lead day 2) and the calibration data; shared by the wind and solar proxies."""
    weather = weather_issued_issue(index)
    calibration = proxy_calibration_issue(index)
    return pd.DatetimeIndex(weather.where(weather >= calibration, calibration))


# kept under their old names for the wind proxy
WIND_PROXY_LAG_DAYS = PROXY_LAG_DAYS
wind_proxy_calibration_issue = proxy_calibration_issue
wind_proxy_issue = generation_proxy_issue


def price_lag_same_type_issue(index) -> pd.DatetimeIndex:
    """Price of the most recent earlier day of the same type, published at 13:00 Paris on the day before that day."""
    from .daytypes import same_type_day

    comparable = same_type_day(delivery_days(index))
    return _local_clock(comparable, -1, PRICE_PUBLICATION_HOUR)


def calendar_issue(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([FAR_PAST] * len(index))


def nuclear_d2_issue(index) -> pd.DatetimeIndex:
    """The last hour of D-2 ends at 00:00 Paris on D-1; its generation is public NUCLEAR_PUBLICATION_LAG_HOURS later."""
    return _local_clock(delivery_days(index), -1, NUCLEAR_PUBLICATION_LAG_HOURS)


def nuclear_d1_issue(index) -> pd.DatetimeIndex:
    """D-1 hours ending by NUCLEAR_D1_CUTOFF_HOUR Paris are public NUCLEAR_PUBLICATION_LAG_HOURS after that hour."""
    return _local_clock(delivery_days(index), -1, NUCLEAR_D1_CUTOFF_HOUR + NUCLEAR_PUBLICATION_LAG_HOURS)


def neighbour_price_lag1_issue(index) -> pd.DatetimeIndex:
    """Neighbour prices of D-1, published with the coupling results about 13:00 Paris on D-2."""
    return price_lag_issue(1)(index)


def residual_v2_issue(index) -> pd.DatetimeIndex:
    """The latest of the load forecast, the generation proxies and the D-1 nuclear hours."""
    candidates = [load_forecast_issue(index), generation_proxy_issue(index), nuclear_d1_issue(index)]
    latest = candidates[0]
    for other in candidates[1:]:
        latest = pd.DatetimeIndex(latest.where(latest >= other, other))
    return latest


@dataclass(frozen=True)
class InputTiming:
    name: str
    issue: Callable[[pd.DatetimeIndex], pd.DatetimeIndex]
    description: str


TIMINGS = {
    "calendar": InputTiming("calendar", calendar_issue, "Hour, weekday, month, public holidays, day types and bridge days are known in advance."),
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
    "wind_proxy": InputTiming(
        "wind_proxy", generation_proxy_issue,
        "Wind generation proxy: lead-day-2 hub-height wind forecasts, weights fitted on actual generation to two days before the month.",
    ),
    "solar_proxy": InputTiming(
        "solar_proxy", generation_proxy_issue,
        "Solar generation proxy: lead-day-2 radiation forecasts, weights fitted on actual generation to two days before the month.",
    ),
    "price_lag_same_type": InputTiming(
        "price_lag_same_type", price_lag_same_type_issue,
        "Day-ahead price of the most recent earlier day of the same type (D-1 or earlier), published about 13:00 Paris the day before it.",
    ),
    "nuclear_d2": InputTiming(
        "nuclear_d2", nuclear_d2_issue,
        "ENTSO-E actual nuclear generation of D-2 (A75, B14), public within two hours of each hour: by 02:00 Paris on D-1.",
    ),
    "nuclear_d1": InputTiming(
        "nuclear_d1", nuclear_d1_issue,
        "ENTSO-E actual nuclear generation of the D-1 hours ending by 08:00 Paris, public by 10:00 Paris on D-1.",
    ),
    "neighbour_price_lag1": InputTiming(
        "neighbour_price_lag1", neighbour_price_lag1_issue,
        "Day-ahead prices of DE-LU, BE, NL, ES, IT-North and CH for D-1, published about 13:00 Paris on D-2.",
    ),
    "residual_v2": InputTiming(
        "residual_v2", residual_v2_issue,
        "Load forecast minus wind and solar proxies minus the latest known nuclear generation: known with the load forecast, 10:00 Paris on D-1.",
    ),
}


def issue_times(index: pd.DatetimeIndex, timing_name: str) -> pd.DatetimeIndex:
    return TIMINGS[timing_name].issue(pd.DatetimeIndex(index))


def lateness(index: pd.DatetimeIndex, timing_name: str) -> pd.TimedeltaIndex:
    """How long after the gate each value became known (negative means before)."""
    return issue_times(index, timing_name) - gate_for(index)


def check_point_in_time(index: pd.DatetimeIndex, feature_timings: dict[str, str], deadline: str = "gate") -> None:
    """Raise LookaheadError if any feature's issue time is after the deadline for any row.

    `feature_timings` maps feature column names to the names in TIMINGS.
    `deadline` is "gate" (12:00 Paris on D-1, the auction) or "premarket"
    (10:05 Paris on D-1, when the earliest scheduled run starts).
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return
    if deadline == "gate":
        limit, label = gate_for(index), "12:00 Paris on D-1"
    elif deadline == "premarket":
        limit, label = premarket_issue_for(index), "the pre-market issue time, 10:05 Paris on D-1"
    else:
        raise ValueError(f"deadline must be 'gate' or 'premarket', not {deadline!r}")
    offenders = []
    for feature, timing_name in feature_timings.items():
        late = issue_times(index, timing_name) - limit
        if (late > pd.Timedelta(0)).any():
            worst = late.max()
            offenders.append(f"{feature} ({timing_name}): up to {worst} after the deadline")
    if offenders:
        raise LookaheadError(f"Features use information published after {label}: " + "; ".join(offenders))
