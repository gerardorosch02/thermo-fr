"""Synthetic French-like data with known answers, for tests and the offline demo."""

import holidays
import numpy as np
import pandas as pd

from .config import LOCAL_TZ


def make_synthetic(
    start: str = "2021-01-01",
    end: str = "2026-01-01",
    threshold: float = 15.0,
    gradient: float = 2400.0,
    price_gradient: float = 4.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Hourly UTC frame with temperature, load_mw and price_eur_mwh.

    Load and price respond to the daily mean temperature through a heating
    threshold, so a correct model should recover `threshold`, `gradient` and
    `price_gradient`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    local = idx.tz_convert(LOCAL_TZ)
    dates = pd.DatetimeIndex(pd.to_datetime(local.date))
    days = dates.unique()

    # Daily temperature: seasonal cycle plus persistent weather anomalies.
    doy = days.dayofyear.to_numpy()
    seasonal = 12.5 - 8.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    anomaly = np.zeros(len(days))
    for i in range(1, len(days)):
        anomaly[i] = 0.8 * anomaly[i - 1] + rng.normal(0, 1.6)
    t_day = pd.Series(seasonal + anomaly, index=days)

    fr = holidays.France(years=range(days.year.min(), days.year.max() + 1))
    dow = days.dayofweek
    is_holiday = np.array([d.date() in fr for d in days])
    weekday_effect = np.select([dow == 5, dow == 6], [-6000.0, -9000.0], 0.0) - 7000.0 * is_holiday

    month_keys = days.to_period("M")
    months = month_keys.unique()
    load_level = pd.Series(rng.normal(0, 1500, len(months)), index=months)
    gas_level = pd.Series(rng.uniform(60, 200, len(months)), index=months)

    heating = np.maximum(threshold - t_day.to_numpy(), 0.0)
    load_day = 48000 + gradient * heating + weekday_effect + load_level.reindex(month_keys).to_numpy()
    price_day = (
        gas_level.reindex(month_keys).to_numpy()
        + price_gradient * heating
        - 10.0 * (dow >= 5)
        + rng.normal(0, 8, len(days))
    )

    hour = local.hour.to_numpy()
    profile = 3000.0 * np.sin(2 * np.pi * (hour - 6) / 24)  # zero-mean over a normal day
    temp_cycle = 3.0 * np.sin(2 * np.pi * (hour - 9) / 24)

    by_hour = lambda s: pd.Series(s, index=days).reindex(dates).to_numpy()
    frame = pd.DataFrame(
        {
            "temperature": by_hour(t_day.to_numpy()) + temp_cycle,
            "load_mw": by_hour(load_day) + profile + rng.normal(0, 800, len(idx)),
            "price_eur_mwh": by_hour(price_day) + rng.normal(0, 5, len(idx)),
        },
        index=idx,
    )
    return frame
