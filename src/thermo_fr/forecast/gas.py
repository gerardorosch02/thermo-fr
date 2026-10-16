"""Gas inputs for the ttf-gas experiment: ICE Endex TTF futures settlements as known at the issue time.

The pre-registration (branch ttf-gas, docs/experiments.md once logged) fixes the rules implemented here:

- Two calendars, both passed in as data and never inferred from gaps in the series. UK business days (weekdays less England
  and Wales bank holidays) serve only the contract expiry. ICE Endex trading days (weekdays less ICE Endex closure days)
  serve everything else; ICE Endex is open on UK bank holidays.
- The last trading day E(c) of contract month c is the second UK business day before the first calendar day of c.
  front(T) is the earliest-delivery contract with E(c) on or after T.
- Every settlement value carries `known_from` (UTC), the moment it was available through the source. A value counts for
  delivery day D only if known_from <= tau, tau = 10:05 Paris on D-1 (the pre-market issue time of timing.py). When a
  (contract, trade date) pair has several rows (a revision), the row with the latest known_from at or before tau is used.
- As-of rule, per delivery day D, with T1, T3, T9 the last trading days on or before D-2, D-3, D-9:
  1. T1* is the latest trading day on or before T1, at most T1_TOLERANCE trading days earlier, for which front(T1*) has a
     value known by tau; c = front(T1*). Otherwise the gas features of D are missing.
  2. R3* is the latest trading day on or before min(T3, T1*), at most REF_TOLERANCE trading days earlier, for which c has
     a value known by tau; R9* the same from T9. Otherwise the gas features of D are missing.
  3. gas_chg_lag1_eur = S_c(T1*) - S_c(R3*), gas_chg_lag7_eur = S_c(T1*) - S_c(R9*), gas_age_days = D - T1* in days.
  4. No value is carried forward beyond these limits and no other feed is substituted.
- Candidate B: srmc_eur_mwh = (S_c(T1*) + emission_factor x EUA on or before D-2) / efficiency, and the reference cost behind
  the D-1 auction, (S_c3(R3*) + emission_factor x EUA on or before D-3) / efficiency with c3 = front(R3*), from which
  features.py builds spark_lag1_eur = price_lag1 - that reference. EUA is the latest primary auction on or before the
  cutoff (several on one day are averaged), carried forward; if the EUA table has known_from, it must also be <= tau.
- A value outside VALUE_RANGE makes the features of every day that uses it missing (the service rule of the
  pre-registration, applied identically in the backtest).

Hard data checks (validate_settlements) raise GasDataError, which stops the branch: a value dated after its contract's
expiry; a value dated on a non-trading day that differs from the contract's previous settlement (an equal one is a repeat
and is dropped); a known_from earlier than the settlement itself (17:20 Paris on the trade date).

Nothing here downloads anything or reads a file; cli.py loads a git-ignored directory into GasInputs.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from dateutil.easter import easter

from ..config import LOCAL_TZ
from .timing import PREMARKET_ISSUE, LookaheadError

GAS_TRAIN_START = "2023-01-01"  # every model of the experiment, candidates and control, is trained from this delivery day
T1_TOLERANCE = 1  # trading days the latest close may be older than T1
REF_TOLERANCE = 2  # trading days a reference close may be older than min(T3 or T9, T1*)
VALUE_RANGE = (1.0, 500.0)  # EUR/MWh; outside it the features of the day are missing
EFFICIENCY = 0.50  # gas plant efficiency, gross calorific basis [OPEN-2, the owner confirms]
EMISSION_FACTOR = 0.184  # tCO2 per MWh of gas, gross calorific basis [OPEN-2]
SETTLEMENT_FINAL = (17, 20)  # Paris (= Amsterdam) local time on the trade date: ICE Endex settlement prices final
KNOWN_FROM_COLUMNS = ("gas_known_from_t1", "gas_known_from_r3", "gas_known_from_r9", "gas_known_from_c3",
                      "eua_known_from_d2", "eua_known_from_d3")


class GasDataError(ValueError):
    """A hard data check of the pre-registration failed: the branch stops."""


def default_ice_closures(years: Iterable[int]) -> list[pd.Timestamp]:
    """The 2026 pattern (1 January, Good Friday, Easter Monday, 25 December), the pre-registered default for a year whose ICE
    Endex schedule cannot be sourced. Weekend dates are kept; they change nothing."""
    out = []
    for year in years:
        e = pd.Timestamp(easter(year))
        out += [pd.Timestamp(year, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1), pd.Timestamp(year, 12, 25)]
    return sorted(out)


def uk_bank_holidays(years: Iterable[int]) -> list[pd.Timestamp]:
    """England and Wales bank holidays from the `holidays` package (subdivision ENG), observed days included."""
    import holidays

    return sorted(pd.Timestamp(d) for d in holidays.country_holidays("GB", subdiv="ENG", years=list(years)))


def _days(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype="datetime64[D]")
    return np.array(sorted({pd.Timestamp(v).strftime("%Y-%m-%d") for v in values}), dtype="datetime64[D]")


def _d64(day) -> np.datetime64:
    return np.datetime64(pd.Timestamp(day).strftime("%Y-%m-%d"), "D")


def _ts(day: np.datetime64) -> pd.Timestamp:
    return pd.Timestamp(day)


def contract_key(value) -> str:
    """A contract month as 'YYYY-MM' (accepts 'YYYY-MM', a Period, a date in the delivery month)."""
    return pd.Period(value, freq="M").strftime("%Y-%m")


@dataclass(frozen=True)
class Calendars:
    """UK business days (expiry only) and ICE Endex trading days (everything else)."""

    ice_closures: np.ndarray
    uk_holidays: np.ndarray

    @classmethod
    def build(cls, ice_closures: Sequence, uk_holidays: Sequence | None = None, years: Iterable[int] | None = None) -> "Calendars":
        if uk_holidays is None:
            if years is None:
                raise ValueError("pass uk_holidays or the years to take them from the holidays package")
            uk_holidays = uk_bank_holidays(years)
        return cls(_days(ice_closures), _days(uk_holidays))

    def is_trading(self, day) -> bool:
        return bool(np.is_busday(_d64(day), holidays=self.ice_closures))

    def last_trading_on_or_before(self, day) -> np.datetime64:
        return np.busday_offset(_d64(day), 0, roll="backward", holidays=self.ice_closures)

    def trading_days_back(self, day: np.datetime64, n: int) -> np.datetime64:
        """The trading day n trading days before `day` (itself a trading day)."""
        return np.busday_offset(day, -n, roll="backward", holidays=self.ice_closures)

    def expiry(self, contract) -> np.datetime64:
        """Second UK business day before the first calendar day of the delivery month."""
        first = _d64(pd.Period(contract, freq="M").start_time)
        return np.busday_offset(first, -2, roll="forward", holidays=self.uk_holidays)

    def front(self, day) -> str:
        """The earliest-delivery contract not yet expired on `day` (expiry on or after it)."""
        d = _d64(day)
        month = pd.Period(_ts(d), freq="M")
        while self.expiry(month) < d:
            month += 1
        return month.strftime("%Y-%m")


def issue_time(day) -> pd.Timestamp:
    """tau: 10:05 Paris on D-1, in UTC (the pre-market issue time of timing.py)."""
    local = pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=PREMARKET_ISSUE[0], minutes=PREMARKET_ISSUE[1])
    return local.tz_localize(LOCAL_TZ).tz_convert("UTC")


def _utc(values) -> pd.Series:
    out = pd.to_datetime(values, utc=True)
    return out


def validate_settlements(settlements: pd.DataFrame, calendars: Calendars) -> pd.DataFrame:
    """Normalise the columns and run the hard checks; returns the rows that enter the features.

    Columns: contract (month), trade_date (day), value (EUR/MWh), known_from (UTC). Raises GasDataError on a value dated
    after its contract's expiry, a differing value on a non-trading day, or a known_from before the settlement was final.
    """
    required = {"contract", "trade_date", "value", "known_from"}
    missing = required - set(settlements.columns)
    if missing:
        raise GasDataError(f"settlements lack columns {sorted(missing)}")
    frame = pd.DataFrame({
        "contract": [contract_key(c) for c in settlements["contract"]],
        "trade_date": pd.to_datetime(settlements["trade_date"]).dt.tz_localize(None).dt.normalize().to_numpy(),
        "value": pd.to_numeric(settlements["value"]).astype(float).to_numpy(),
        "known_from": _utc(settlements["known_from"]).to_numpy(),
    })
    frame["known_from"] = pd.to_datetime(frame["known_from"], utc=True)
    if frame["value"].isna().any() or frame["known_from"].isna().any():
        raise GasDataError("settlements contain empty values or empty known_from times")

    expiries = {c: _ts(calendars.expiry(c)) for c in frame["contract"].unique()}
    late = frame[frame["trade_date"] > frame["contract"].map(expiries)]
    if len(late):
        first = late.iloc[0]
        raise GasDataError(f"{len(late)} value(s) dated after their contract's expiry, e.g. {first['contract']} on "
                           f"{first['trade_date']:%Y-%m-%d} (expiry {expiries[first['contract']]:%Y-%m-%d})")

    final = (frame["trade_date"] + pd.Timedelta(hours=SETTLEMENT_FINAL[0], minutes=SETTLEMENT_FINAL[1])).dt.tz_localize(LOCAL_TZ)
    early = frame[frame["known_from"] < final.dt.tz_convert("UTC")]
    if len(early):
        first = early.iloc[0]
        raise GasDataError(f"{len(early)} value(s) known before the settlement was final, e.g. {first['contract']} on "
                           f"{first['trade_date']:%Y-%m-%d} known from {first['known_from']}")

    trading = np.is_busday(frame["trade_date"].to_numpy().astype("datetime64[D]"), holidays=calendars.ice_closures)
    drop = []
    first_known = frame.sort_values("known_from").drop_duplicates(["contract", "trade_date"], keep="first")
    for i in np.flatnonzero(~trading):
        row = frame.iloc[i]
        earlier = first_known[(first_known["contract"] == row["contract"]) & (first_known["trade_date"] < row["trade_date"])]
        previous = earlier.sort_values("trade_date").iloc[-1]["value"] if len(earlier) else None
        if previous is None or row["value"] != previous:
            raise GasDataError(f"{row['contract']} has a value on the non-trading day {row['trade_date']:%Y-%m-%d} that is not a "
                               "repeat of its previous settlement")
        drop.append(frame.index[i])
    return frame.drop(index=drop).reset_index(drop=True)


@dataclass
class GasInputs:
    """Contract settlements with their known-from times, the two calendars, and the EUA auctions (candidate B)."""

    settlements: pd.DataFrame
    calendars: Calendars
    eua: pd.DataFrame | None = None  # auction_date, price_eur_t, optional known_from (UTC)
    efficiency: float = EFFICIENCY
    emission_factor: float = EMISSION_FACTOR
    _store: dict = field(default_factory=dict, init=False, repr=False)
    _eua: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        clean = validate_settlements(self.settlements, self.calendars)
        store: dict = {}
        for (contract, day), part in clean.sort_values("known_from").groupby(["contract", "trade_date"], sort=False):
            store[(contract, _d64(day))] = (list(part["known_from"]), list(part["value"]))
        self._store = store
        if self.eua is not None:
            eua = pd.DataFrame({
                "auction_date": pd.to_datetime(self.eua["auction_date"]).dt.tz_localize(None).dt.normalize(),
                "price": pd.to_numeric(self.eua["price_eur_t"]).astype(float),
            })
            if "known_from" in self.eua:
                eua["known_from"] = pd.to_datetime(self.eua["known_from"], utc=True).to_numpy()
            self._eua = eua.sort_values("auction_date").reset_index(drop=True)

    def value(self, contract: str, day: np.datetime64, tau: pd.Timestamp):
        """(value, known_from) of the contract's settlement on `day` as known at tau, or None."""
        entry = self._store.get((contract, day))
        if entry is None:
            return None
        known, values = entry
        i = bisect.bisect_right(known, tau)
        if i == 0:
            return None
        return values[i - 1], known[i - 1]

    def eua_on_or_before(self, cutoff: pd.Timestamp, tau: pd.Timestamp):
        """(mean price of the latest auction day on or before cutoff known by tau, its known_from or NaT), or None."""
        if self._eua is None:
            return None
        eua = self._eua[self._eua["auction_date"] <= cutoff]
        if "known_from" in eua:
            eua = eua[eua["known_from"] <= tau]
        if eua.empty:
            return None
        last = eua[eua["auction_date"] == eua["auction_date"].max()]
        known = last["known_from"].max() if "known_from" in last else pd.NaT
        return float(last["price"].mean()), known

    def day_features(self, day) -> dict:
        """Features and info for one delivery day, exactly as the as-of rule defines them."""
        return day_features(day, self)

    def feature_table(self, days) -> pd.DataFrame:
        """One row per delivery day (index), feature and info columns; asserts every value used was known by tau."""
        days = pd.DatetimeIndex(pd.to_datetime(sorted(set(pd.DatetimeIndex(days))))).normalize()
        table = pd.DataFrame([day_features(d, self) for d in days], index=days)
        assert_known_before_issue(table)
        return table


def _empty(day, tau, reason) -> dict:
    return {
        "gas_chg_lag1_eur": np.nan, "gas_chg_lag7_eur": np.nan, "gas_age_days": np.nan, "srmc_eur_mwh": np.nan,
        "srmc_ref_eur_mwh": np.nan, "gas_contract": None, "gas_t1": pd.NaT, "gas_r3": pd.NaT, "gas_r9": pd.NaT,
        **{c: pd.NaT for c in KNOWN_FROM_COLUMNS}, "gas_roll_in_window": False, "gas_missing": True, "gas_srmc_missing": True,
        "gas_missing_reason": reason, "gas_issue_time": tau,
    }


def day_features(day, gas: GasInputs) -> dict:
    """The as-of rule for one delivery day D (see the module docstring)."""
    cal = gas.calendars
    D = _d64(day)
    tau = issue_time(_ts(D))
    t1 = cal.last_trading_on_or_before(D - np.timedelta64(2, "D"))
    t3 = cal.last_trading_on_or_before(D - np.timedelta64(3, "D"))
    t9 = cal.last_trading_on_or_before(D - np.timedelta64(9, "D"))

    t1_star = contract = latest = None
    for k in range(T1_TOLERANCE + 1):
        candidate = cal.trading_days_back(t1, k)
        front = cal.front(candidate)
        found = gas.value(front, candidate, tau)
        if found is not None:
            t1_star, contract, latest = candidate, front, found
            break
    if t1_star is None:
        return _empty(day, tau, "no_t1")

    def reference(t_ref):
        start = min(t_ref, t1_star)
        for k in range(REF_TOLERANCE + 1):
            candidate = cal.trading_days_back(start, k)
            found = gas.value(contract, candidate, tau)
            if found is not None:
                return candidate, found
        return None

    r3, r9 = reference(t3), reference(t9)
    if r3 is None:
        return _empty(day, tau, "no_r3")
    if r9 is None:
        return _empty(day, tau, "no_r9")
    used = [latest[0], r3[1][0], r9[1][0]]
    if any(not (VALUE_RANGE[0] <= v <= VALUE_RANGE[1]) for v in used):
        return _empty(day, tau, "out_of_range")

    out = _empty(day, tau, "")
    out.update({
        "gas_chg_lag1_eur": latest[0] - r3[1][0],
        "gas_chg_lag7_eur": latest[0] - r9[1][0],
        "gas_age_days": float((D - t1_star).astype(int)),
        "gas_contract": contract,
        "gas_t1": _ts(t1_star), "gas_r3": _ts(r3[0]), "gas_r9": _ts(r9[0]),
        "gas_known_from_t1": latest[1], "gas_known_from_r3": r3[1][1], "gas_known_from_r9": r9[1][1],
        "gas_roll_in_window": cal.front(r3[0]) != contract,
        "gas_missing": False,
    })

    # candidate B: the gas-plant cost now and as priced behind the D-1 auction
    c3 = cal.front(r3[0])
    ref_gas = gas.value(c3, r3[0], tau)
    eua_now = gas.eua_on_or_before(_ts(D - np.timedelta64(2, "D")), tau)
    eua_ref = gas.eua_on_or_before(_ts(D - np.timedelta64(3, "D")), tau)
    if ref_gas is None or eua_now is None or eua_ref is None or not (VALUE_RANGE[0] <= ref_gas[0] <= VALUE_RANGE[1]):
        out["gas_missing_reason"] = "no_srmc_inputs"
        return out
    out.update({
        "srmc_eur_mwh": (latest[0] + gas.emission_factor * eua_now[0]) / gas.efficiency,
        "srmc_ref_eur_mwh": (ref_gas[0] + gas.emission_factor * eua_ref[0]) / gas.efficiency,
        "gas_known_from_c3": ref_gas[1], "eua_known_from_d2": eua_now[1], "eua_known_from_d3": eua_ref[1],
        "gas_srmc_missing": False,
    })
    return out


def assert_known_before_issue(info: pd.DataFrame) -> None:
    """The companion check: every known_from used for a delivery day is at or before that day's issue time."""
    if "gas_issue_time" not in info:
        return
    tau = pd.to_datetime(info["gas_issue_time"], utc=True)
    for column in KNOWN_FROM_COLUMNS:
        if column not in info:
            continue
        known = pd.to_datetime(info[column], utc=True)
        late = known.notna() & (known > tau)
        if late.any():
            raise LookaheadError(f"{column}: {int(late.sum())} value(s) known after the 10:05 Paris issue time, first at row {late.idxmax()}")


def load_gas_dir(path, efficiency: float = EFFICIENCY, emission_factor: float = EMISSION_FACTOR) -> GasInputs:
    """GasInputs from a git-ignored directory:

    - settlements.csv: contract, trade_date, value, known_from (UTC, ISO 8601)
    - ice_closures.csv: date (ICE Endex closure days, from ICE's schedules)
    - uk_holidays.csv (optional): date; otherwise the holidays package, subdivision ENG
    - eua.csv (optional, candidate B): auction_date, price_eur_t, optional known_from
    """
    path = Path(path)
    settlements = pd.read_csv(path / "settlements.csv")
    closures = pd.read_csv(path / "ice_closures.csv")["date"]
    uk = pd.read_csv(path / "uk_holidays.csv")["date"] if (path / "uk_holidays.csv").exists() else None
    years = range(pd.to_datetime(settlements["trade_date"]).min().year - 1, pd.to_datetime(settlements["trade_date"]).max().year + 2)
    eua = pd.read_csv(path / "eua.csv") if (path / "eua.csv").exists() else None
    return GasInputs(settlements, Calendars.build(closures, uk, years=years), eua=eua, efficiency=efficiency,
                     emission_factor=emission_factor)
