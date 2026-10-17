"""Hour-label audit of the ENTSO-E actual generation series (A75) against references whose labels are independent.

For an actual series a and a reference r, the monthly lag is the time by which a lags r, estimated from the correlation of
their hourly changes, corr(a(T) - a(T-1), r(T+k) - r(T+k-1)) for k = -2..2, at the peak refined by a parabola through the
three highest points (so it is fractional). A series labelled one hour late shows a lag one hour larger than its usual one.
Only the change of the lag between periods matters: series with different physics have different usual lags.

The references: for solar, Open-Meteo historical-forecast radiation (radiation_proxy_wm2, on start-of-hour labels) and the
ENTSO-E day-ahead solar forecast; for wind, the ENTSO-E day-ahead wind forecast and the cube of the Open-Meteo 100 m wind
speed; for nuclear, the day-ahead price and the day-ahead load forecast (French nuclear modulates with residual demand).
The references themselves are audited against each other, so that a shift in a reference is not read as one in the
actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import hour_labels

LAGS = (-2, -1, 0, 1, 2)


def monthly_lag(actual: pd.Series, reference: pd.Series, lags=LAGS) -> pd.Series:
    """Hours by which `actual` lags `reference`, per UTC month (positive: actual is later)."""
    a, r = actual.astype(float).diff(), reference.astype(float).diff()
    month = pd.PeriodIndex(a.index.tz_convert("UTC").tz_localize(None), freq="M")
    out = {}
    for m in sorted(set(month)):
        rows = month == m
        aa = a[rows]
        if aa.notna().sum() < 24 * 10:
            continue
        cs = np.array([aa.corr(r.shift(-k).reindex(aa.index)) for k in lags])
        if np.isnan(cs).all():
            continue
        i = int(np.nanargmax(cs))
        frac = 0.0
        if 0 < i < len(cs) - 1 and not np.isnan(cs[[i - 1, i + 1]]).any():
            y0, y1, y2 = cs[i - 1], cs[i], cs[i + 1]
            denom = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        out[str(m)] = -(lags[i] + frac)
    return pd.Series(out, dtype=float)


def pairs(table: pd.DataFrame) -> dict:
    wind = table["wind_onshore_mw"].add(table["wind_offshore_mw"].fillna(0.0))
    wind_fc = table["wind_onshore_fc_mw"].add(table["wind_offshore_fc_mw"].fillna(0.0))
    radiation = table["radiation_proxy_wm2"] if hour_labels.aligned() else hour_labels.to_start_of_hour(table["radiation_proxy_wm2"])
    cube = table["wind100_proxy_ms"] ** 3
    return {
        "solar actual vs Open-Meteo radiation": (table["solar_mw"], radiation),
        "solar actual vs ENTSO-E solar forecast": (table["solar_mw"], table["solar_fc_mw"]),
        "wind actual vs ENTSO-E wind forecast": (wind, wind_fc),
        "wind actual vs Open-Meteo wind speed cubed": (wind, cube),
        "nuclear actual vs day-ahead price": (table["nuclear_mw"], table["price_eur_mwh"]),
        "nuclear actual vs load forecast": (table["nuclear_mw"], table["load_fc_mw"]),
        "reference: solar forecast vs Open-Meteo radiation": (table["solar_fc_mw"], radiation),
        "reference: wind forecast vs Open-Meteo wind speed cubed": (wind_fc, cube),
        "reference: price vs load forecast": (table["price_eur_mwh"], table["load_fc_mw"]),
    }


def audit(table: pd.DataFrame, period: tuple[str, str], end: str | None = None) -> dict:
    """Mean monthly lag inside `period` (first and last month, inclusive) against the other months, per pair, and the shift
    (inside minus outside) with the outside months' standard deviation."""
    if end is not None:
        table = table[table.index < pd.Timestamp(end, tz="UTC")]
    out, monthly = {}, {}
    for name, (a, r) in pairs(table).items():
        lag = monthly_lag(a, r)
        monthly[name] = lag
        inside = (lag.index >= period[0]) & (lag.index <= period[1])
        out[name] = {"inside_h": round(float(lag[inside].mean()), 2), "outside_h": round(float(lag[~inside].mean()), 2),
                     "shift_h": round(float(lag[inside].mean() - lag[~inside].mean()), 2),
                     "outside_sd_h": round(float(lag[~inside].std()), 2), "months_inside": int(inside.sum()),
                     "months_outside": int((~inside).sum())}
    return {"period": list(period), "pairs": out, "monthly": pd.DataFrame(monthly)}
