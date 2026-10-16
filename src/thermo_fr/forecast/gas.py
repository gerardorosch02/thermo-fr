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
  the D-1 auction, (S_c3(Rs) + emission_factor x EUA on or before D-3) / efficiency, where Rs is the latest trading day on
  or before R3*, at most REF_TOLERANCE trading days earlier, on which front(Rs) has a value known by tau, and c3 =
  front(Rs). features.py builds spark_lag1_eur = price_lag1 - that reference. EUA is the latest primary auction on or
  before the cutoff (several on one day are averaged), carried forward, and must be known by tau (its known_from is
  mandatory, see eua_known_from). Without Rs or EUA, B's features are missing; A is unaffected.
- A value outside VALUE_RANGE among S_c(T1*), S_c(R3*), S_c(R9*) makes all gas features of the day missing; S_c3(Rs)
  outside it makes B's missing. The rule never skips to an earlier close and applies in the backtest and in service.

Hard data checks (settlement_issues, validate_settlements) raise GasDataError, which stops the branch: a value dated after
its contract's expiry; a value on a weekend or a sourced ICE closure that differs from the contract's previous settlement
or has none (an equal one is a repeat and is dropped); a known_from earlier than 17:20 Amsterdam local time on the trade
date. A value on an unsourced default closure is dropped and listed, not a stop.

Nothing here downloads anything or reads a file; cli.py loads a git-ignored directory into GasInputs.
"""

from __future__ import annotations

import bisect
import json
import re
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


# Spans covered by a sourced ICE Endex document (ice_closures.csv): circular E24027 (Christmas and New Year 2024/25) and the
# 2026 trading schedule. Outside them the default pattern applies (pre-registration v4, section 2).
SOURCED_SPANS = (("2024-12-24", "2025-01-01"), ("2026-01-01", "2026-12-31"))


def default_ice_closures(years: Iterable[int]) -> list[pd.Timestamp]:
    """The pre-registered default for unsourced dates: 1 January, Good Friday, Easter Monday, 25 and 26 December, each only when
    it falls on a weekday; no substitute days."""
    out = []
    for year in years:
        e = pd.Timestamp(easter(year))
        days = [pd.Timestamp(year, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1), pd.Timestamp(year, 12, 25),
                pd.Timestamp(year, 12, 26)]
        out += [d for d in days if d.dayofweek < 5]
    return sorted(out)


def ice_closure_table(sourced: Sequence, years: Iterable[int], sourced_spans=SOURCED_SPANS) -> pd.DataFrame:
    """Every ICE Endex closure day with a flag: sourced (from ICE documents) or default (the pattern, on dates outside every
    sourced span). The table is built from documents only, never from gaps in the data."""
    spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in sourced_spans]
    sourced_days = sorted({pd.Timestamp(d).normalize() for d in sourced})
    outside = [d for d in default_ice_closures(years) if not any(a <= d <= b for a, b in spans) and d not in sourced_days]
    table = pd.DataFrame({"date": sourced_days + outside, "sourced": [True] * len(sourced_days) + [False] * len(outside)})
    return table.sort_values("date").reset_index(drop=True)


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
    """UK business days (expiry only) and ICE Endex trading days (everything else).

    `default_closures` is the subset of `ice_closures` that comes from the default pattern rather than an ICE document;
    `expiry_overrides` maps a contract month to ICE's published last trading day, which wins over the rule.
    """

    ice_closures: np.ndarray
    uk_holidays: np.ndarray
    default_closures: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[D]"))
    expiry_overrides: dict = field(default_factory=dict)

    @classmethod
    def build(cls, ice_closures: Sequence, uk_holidays: Sequence | None = None, years: Iterable[int] | None = None,
              default_closures: Sequence = (), expiry_overrides: dict | None = None) -> "Calendars":
        if uk_holidays is None:
            if years is None:
                raise ValueError("pass uk_holidays or the years to take them from the holidays package")
            uk_holidays = uk_bank_holidays(years)
        overrides = {contract_key(c): _d64(d) for c, d in (expiry_overrides or {}).items()}
        return cls(_days(ice_closures), _days(uk_holidays), _days(default_closures), overrides)

    @classmethod
    def from_closure_table(cls, table: pd.DataFrame, years: Iterable[int], uk_holidays: Sequence | None = None,
                           expiry_overrides: dict | None = None) -> "Calendars":
        """From ice_closure_table(): all closures, with the default ones flagged."""
        return cls.build(table["date"], uk_holidays, years=years, default_closures=table.loc[~table["sourced"].astype(bool), "date"],
                         expiry_overrides=expiry_overrides)

    def is_default_closure(self, day) -> bool:
        return bool(np.isin(_d64(day), self.default_closures))

    def rule_expiry(self, contract) -> np.datetime64:
        """Second UK business day before the first calendar day of the delivery month."""
        first = _d64(pd.Period(contract, freq="M").start_time)
        return np.busday_offset(first, -2, roll="forward", holidays=self.uk_holidays)

    def is_trading(self, day) -> bool:
        return bool(np.is_busday(_d64(day), holidays=self.ice_closures))

    def last_trading_on_or_before(self, day) -> np.datetime64:
        return np.busday_offset(_d64(day), 0, roll="backward", holidays=self.ice_closures)

    def trading_days_back(self, day: np.datetime64, n: int) -> np.datetime64:
        """The trading day n trading days before `day` (itself a trading day)."""
        return np.busday_offset(day, -n, roll="backward", holidays=self.ice_closures)

    def expiry(self, contract) -> np.datetime64:
        """ICE's published last trading day where given, else the rule."""
        key = contract_key(contract)
        return self.expiry_overrides[key] if key in self.expiry_overrides else self.rule_expiry(key)

    def front(self, day) -> str:
        """The earliest-delivery contract not yet expired on `day` (expiry on or after it)."""
        d = _d64(day)
        month = pd.Period(_ts(d), freq="M")
        while self.expiry(month) < d:
            month += 1
        return month.strftime("%Y-%m")


def next_contract(contract: str) -> str:
    return (pd.Period(contract, freq="M") + 1).strftime("%Y-%m")


def known_from_ideal(trade_dates) -> pd.Series:
    """Research timing rule gas_settle_ideal: a settlement of trading day T is known from 17:20 Amsterdam local time on T (the
    exchange's own final-settlement time), whatever the feed actually delivered."""
    dates = pd.to_datetime(pd.Series(trade_dates)).dt.tz_localize(None).dt.normalize()
    local = dates + pd.Timedelta(hours=SETTLEMENT_FINAL[0], minutes=SETTLEMENT_FINAL[1])
    return local.dt.tz_localize("Europe/Amsterdam").dt.tz_convert("UTC")


def issue_time(day) -> pd.Timestamp:
    """tau: 10:05 Paris on D-1, in UTC (the pre-market issue time of timing.py)."""
    local = pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=PREMARKET_ISSUE[0], minutes=PREMARKET_ISSUE[1])
    return local.tz_localize(LOCAL_TZ).tz_convert("UTC")


def _utc(values) -> pd.Series:
    out = pd.to_datetime(values, utc=True)
    return out


def normalise_settlements(settlements: pd.DataFrame) -> pd.DataFrame:
    """contract 'YYYY-MM', trade_date naive day, value float, known_from UTC; raises GasDataError on missing columns or blanks."""
    required = {"contract", "trade_date", "value", "known_from"}
    missing = required - set(settlements.columns)
    if missing:
        raise GasDataError(f"settlements lack columns {sorted(missing)}")
    frame = pd.DataFrame({
        "contract": [contract_key(c) for c in settlements["contract"]],
        "trade_date": pd.to_datetime(settlements["trade_date"]).dt.tz_localize(None).dt.normalize().to_numpy(),
        "value": pd.to_numeric(settlements["value"]).astype(float).to_numpy(),
        "known_from": pd.to_datetime(settlements["known_from"], utc=True).to_numpy(),
    })
    frame["known_from"] = pd.to_datetime(frame["known_from"], utc=True)
    if frame["value"].isna().any() or frame["known_from"].isna().any():
        raise GasDataError("settlements contain empty values or empty known_from times")
    return frame


HARD_ISSUES = ("after_expiry", "known_too_early", "non_trading_not_repeat")


def settlement_issues(settlements: pd.DataFrame, calendars: Calendars) -> tuple[pd.DataFrame, dict]:
    """The expiry and closure-day checks of the pre-registration, every offending row listed.

    Returns the rows that enter the features and a dict of DataFrames:
    - after_expiry (hard): a value dated after its contract's last trading day;
    - known_too_early (hard): a known_from before 17:20 Amsterdam local time on the trade date;
    - non_trading_not_repeat (hard): a value on a weekend or a sourced closure that differs from the contract's previous
      settlement (its latest earlier trade date, as first published), or has none;
    - non_trading_repeats (dropped): such values equal to the previous settlement;
    - default_closure_values (soft, dropped and listed): any value on an unsourced default closure day.
    """
    frame = normalise_settlements(settlements)
    expiries = {c: _ts(calendars.expiry(c)) for c in frame["contract"].unique()}
    after = frame["trade_date"] > frame["contract"].map(expiries)
    final = (frame["trade_date"] + pd.Timedelta(hours=SETTLEMENT_FINAL[0], minutes=SETTLEMENT_FINAL[1])).dt.tz_localize(LOCAL_TZ)
    early = frame["known_from"] < final.dt.tz_convert("UTC")
    days = frame["trade_date"].to_numpy().astype("datetime64[D]")
    trading = np.is_busday(days, holidays=calendars.ice_closures)
    default = np.isin(days, calendars.default_closures)
    hard_closed = ~trading & ~default
    first_known = frame.sort_values("known_from").drop_duplicates(["contract", "trade_date"], keep="first")
    repeat = np.zeros(len(frame), dtype=bool)
    for i in np.flatnonzero(hard_closed):
        row = frame.iloc[i]
        earlier = first_known[(first_known["contract"] == row["contract"]) & (first_known["trade_date"] < row["trade_date"])]
        previous = earlier.sort_values("trade_date").iloc[-1]["value"] if len(earlier) else None
        repeat[i] = previous is not None and row["value"] == previous
    issues = {
        "after_expiry": frame[after].assign(expiry=frame.loc[after, "contract"].map(expiries)),
        "known_too_early": frame[early],
        "non_trading_not_repeat": frame[hard_closed & ~repeat],
        "non_trading_repeats": frame[hard_closed & repeat],
        "default_closure_values": frame[default],
    }
    clean = frame[trading & ~after.to_numpy() & ~early.to_numpy()].reset_index(drop=True)
    return clean, issues


def validate_settlements(settlements: pd.DataFrame, calendars: Calendars) -> pd.DataFrame:
    """settlement_issues(), raising GasDataError on any hard issue; returns the rows that enter the features."""
    return _raise_hard(*settlement_issues(settlements, calendars))


def _raise_hard(clean: pd.DataFrame, issues: dict) -> pd.DataFrame:
    messages = {
        "after_expiry": "dated after their contract's expiry",
        "known_too_early": "known before the settlement was final (17:20 Amsterdam time on the trade date)",
        "non_trading_not_repeat": "on a non-trading day (weekend or sourced closure) that are not a repeat of the previous settlement",
    }
    for name in HARD_ISSUES:
        bad = issues[name]
        if len(bad):
            first = bad.iloc[0]
            raise GasDataError(f"{len(bad)} value(s) {messages[name]}, e.g. {first['contract']} on {first['trade_date']:%Y-%m-%d}")
    return clean


def eua_known_from(auction_dates, fetched_at=None) -> pd.Series:
    """13:00 Paris on the auction day (results follow the 11:00 CET close of bidding), or the first fetch stamp of the report
    copy that contains the result if that is later."""
    dates = pd.to_datetime(pd.Series(auction_dates)).dt.tz_localize(None).dt.normalize()
    rule = (dates + pd.Timedelta(hours=13)).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")
    if fetched_at is None:
        return rule
    fetched = pd.to_datetime(pd.Series(fetched_at), utc=True).reset_index(drop=True)
    rule = rule.reset_index(drop=True)
    return rule.where(fetched.isna() | (fetched <= rule), fetched)


@dataclass
class GasInputs:
    """Contract settlements with their known-from times, the two calendars, and the EUA auctions (candidate B)."""

    settlements: pd.DataFrame
    calendars: Calendars
    eua: pd.DataFrame | None = None  # auction_date, price_eur_t, known_from (UTC, mandatory: eua_known_from())
    efficiency: float = EFFICIENCY
    emission_factor: float = EMISSION_FACTOR
    mode: str = "dbnomics"  # "ideal": the research measurement's idealised timing (known_from_ideal); research sets require it
    substitute_next: bool = False  # research only: when front(T) has no value on T, the next contract stands in for that day
    _store: dict = field(default_factory=dict, init=False, repr=False)
    _eua: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        clean, self.issues = settlement_issues(self.settlements, self.calendars)
        _raise_hard(clean, self.issues)
        store: dict = {}
        for (contract, day), part in clean.sort_values("known_from").groupby(["contract", "trade_date"], sort=False):
            store[(contract, _d64(day))] = (list(part["known_from"]), list(part["value"]))
        self._store = store
        if self.eua is not None:
            eua = pd.DataFrame({
                "auction_date": pd.to_datetime(self.eua["auction_date"]).dt.tz_localize(None).dt.normalize(),
                "price": pd.to_numeric(self.eua["price_eur_t"]).astype(float),
            })
            if "known_from" not in self.eua or self.eua["known_from"].isna().any():
                raise GasDataError("every EUA value needs a known_from time (eua_known_from())")
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
        """(mean price of the latest auction day on or before cutoff known by tau, its latest known_from), or None."""
        if self._eua is None:
            return None
        eua = self._eua[(self._eua["auction_date"] <= cutoff) & (self._eua["known_from"] <= tau)]
        if eua.empty:
            return None
        last = eua[eua["auction_date"] == eua["auction_date"].max()]
        known = last["known_from"].max()
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
        "srmc_ref_eur_mwh": np.nan, "gas_contract": None, "gas_t1": pd.NaT, "gas_r3": pd.NaT, "gas_r9": pd.NaT, "gas_rs": pd.NaT,
        "gas_c3": None,
        **{c: pd.NaT for c in KNOWN_FROM_COLUMNS}, "gas_roll_in_window": False, "gas_next_contract": False, "gas_missing": True, "gas_srmc_missing": True,
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
    substituted = False
    for k in range(T1_TOLERANCE + 1):
        candidate = cal.trading_days_back(t1, k)
        front = cal.front(candidate)
        found = gas.value(front, candidate, tau)
        if found is None and gas.substitute_next:  # research only: the next contract stands in for the whole day
            nxt = next_contract(front)
            found = gas.value(nxt, candidate, tau)
            if found is not None:
                front, substituted = nxt, True
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
        "gas_next_contract": substituted,
        "gas_missing": False,
    })

    # candidate B: the gas-plant cost now and as priced behind the D-1 auction. Rs is the latest trading day on or before R3*,
    # at most REF_TOLERANCE trading days earlier, on which front(Rs) has a value known by tau; c3 = front(Rs).
    rs = None
    for k in range(REF_TOLERANCE + 1):
        candidate = cal.trading_days_back(r3[0], k)
        c3 = cal.front(candidate)
        found = gas.value(c3, candidate, tau)
        if found is not None:
            rs = (candidate, c3, found)
            break
    eua_now = gas.eua_on_or_before(_ts(D - np.timedelta64(2, "D")), tau)
    eua_ref = gas.eua_on_or_before(_ts(D - np.timedelta64(3, "D")), tau)
    if rs is None or eua_now is None or eua_ref is None:
        out["gas_missing_reason"] = "no_rs" if rs is None else "no_eua"
        return out
    if not (VALUE_RANGE[0] <= rs[2][0] <= VALUE_RANGE[1]):
        out["gas_missing_reason"] = "out_of_range_rs"
        return out
    ref_gas = rs[2]
    out.update({
        "gas_rs": _ts(rs[0]), "gas_c3": rs[1],
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


# --- loaders (the laptop's git-ignored directory; nothing derived is written anywhere) ------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}
_NAME_RE = re.compile(r"(?<![A-Za-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s\-/']*(\d{4}|\d{2})(?!\d)", re.I)
SECONDARY_KNOWN_FROM = (9, 0)  # Paris, the calendar day after the trade date: the fixed rule when no git history exists


def contract_from_name(name: str) -> str | None:
    """'YYYY-MM' from a series name such as 'Dutch TTF Gas Futures - Dec22' or '... December 2022'; None unless exactly one
    month-year appears."""
    found = set()
    for month, year in _NAME_RE.findall(name or ""):
        y = int(year) + (2000 if len(year) == 2 else 0)
        found.add(f"{y:04d}-{MONTHS[month[:3].lower()]:02d}")
    return found.pop() if len(found) == 1 else None


def _docs(obj) -> list:
    if isinstance(obj, list):
        return obj
    if "series" in obj and isinstance(obj["series"], dict):
        return obj["series"].get("docs", [])
    return obj.get("docs", [])


def load_dbnomics_json(paths, series_map: dict | None = None, monthly_only: bool = False) -> pd.DataFrame:
    """The raw DBnomics pages saved by the laptop script (v22 series endpoint, observations=1): one row per observation with
    series_code, series_name, contract ('YYYY-MM'), trade_date and value. 'NA' observations are skipped. The contract comes
    from `series_map` (series_code -> month) where given, else from the series name; an unresolved series or two differing
    values for one (contract, trade date) raise GasDataError. With `monthly_only`, series whose name names no single month
    (quarters, seasons, calendar and gas years, strips) are skipped instead, and their count is in frame.attrs."""
    rows, unresolved, skipped = [], [], []
    for path in sorted(Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)):
        for doc in _docs(json.loads(Path(path).read_text(encoding="utf-8"))):
            code, name = doc.get("series_code"), doc.get("series_name", "")
            contract = (series_map or {}).get(code) or contract_from_name(name)
            if contract is None:
                (skipped if monthly_only else unresolved).append(f"{code} ({name!r})")
                continue
            for period, value in zip(doc.get("period", []), doc.get("value", [])):
                if value is None or value == "NA" or (isinstance(value, float) and np.isnan(value)):
                    continue
                rows.append({"series_code": code, "series_name": name, "contract": contract_key(contract),
                             "trade_date": pd.Timestamp(period), "value": float(value)})
    if unresolved:
        raise GasDataError(f"no contract month for series {', '.join(unresolved)}; add them to series_map.csv")
    frame = pd.DataFrame(rows, columns=["series_code", "series_name", "contract", "trade_date", "value"])
    frame = frame.drop_duplicates(["contract", "trade_date", "value"])
    clash = frame[frame.duplicated(["contract", "trade_date"], keep=False)]
    if len(clash):
        first = clash.iloc[0]
        raise GasDataError(f"{len(clash)} rows give different values for one contract and trade date, e.g. {first['contract']} "
                           f"on {first['trade_date']:%Y-%m-%d} (series {sorted(clash['series_code'].unique())})")
    frame = frame.sort_values(["contract", "trade_date"]).reset_index(drop=True)
    frame.attrs["skipped_series"] = len(skipped)
    frame.attrs["monthly_series"] = int(frame["series_code"].nunique())
    return frame


def known_from_secondary(trade_dates) -> pd.Series:
    """The secondary rule: 09:00 Paris on the calendar day after the trade date."""
    dates = pd.to_datetime(pd.Series(trade_dates)).dt.tz_localize(None).dt.normalize()
    local = dates + pd.Timedelta(days=1, hours=SECONDARY_KNOWN_FROM[0], minutes=SECONDARY_KNOWN_FROM[1])
    return local.dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")


def settlements_from_dbnomics(raw: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Settlements with known_from. Without `history`, every value gets the secondary rule. With `history` (the git-history
    reconstruction: contract, trade_date, value, known_from, one row per published version), the history is the settlement
    table, after checking that each raw value equals the latest version for its (contract, trade date)."""
    if history is None:
        out = raw[["contract", "trade_date", "value"]].copy()
        out["known_from"] = known_from_secondary(out["trade_date"]).to_numpy()
        return out.reset_index(drop=True)
    hist = normalise_settlements(history)
    latest = hist.sort_values("known_from").drop_duplicates(["contract", "trade_date"], keep="last")
    merged = raw.merge(latest, on=["contract", "trade_date"], how="left", suffixes=("", "_history"))
    bad = merged[merged["value_history"].isna() | (merged["value"] != merged["value_history"])]
    if len(bad):
        first = bad.iloc[0]
        raise GasDataError(f"{len(bad)} raw values are absent from the history or differ from its latest version, e.g. "
                           f"{first['contract']} on {first['trade_date']:%Y-%m-%d}")
    return hist


def read_gas_dir(path, ideal: bool = False) -> dict:
    """The pieces of a gas directory (git-ignored, on the laptop):

    - dbnomics/*.json: raw DBnomics pages (or settlements.csv: contract, trade_date, value, known_from)
    - known_from.csv (optional): the git-history versions, contract, trade_date, value, known_from
    - series_map.csv (optional): series_code, contract
    - ice_closures.csv: date (sourced ICE Endex closures); the default pattern is added for unsourced dates
    - ice_expiries.csv (optional): contract_month, last_trading_day, source (an ICE-published date wins over the rule)
    - uk_holidays.csv (optional): date; otherwise the holidays package, subdivision ENG
    - eua.csv (optional, candidate B): auction_date, price_eur_t, and known_from or fetched_at (else 13:00 Paris on the day)

    The raw pages are read with monthly_only (the dataset also holds quarters, seasons, years and strips). With `ideal` (the
    research measurement only), every value gets known_from_ideal and known_from.csv is ignored.
    """
    path = Path(path)
    raw = None
    if (path / "settlements.csv").exists():
        settlements = pd.read_csv(path / "settlements.csv")
    else:
        series_map = None
        if (path / "series_map.csv").exists():
            m = pd.read_csv(path / "series_map.csv", dtype=str)
            series_map = dict(zip(m["series_code"], m["contract"]))
        raw = load_dbnomics_json(sorted((path / "dbnomics").glob("*.json")), series_map, monthly_only=True)
        if ideal:
            settlements = raw[["contract", "trade_date", "value"]].copy()
            settlements["known_from"] = known_from_ideal(settlements["trade_date"]).to_numpy()
        else:
            history = pd.read_csv(path / "known_from.csv") if (path / "known_from.csv").exists() else None
            settlements = settlements_from_dbnomics(raw, history)
    trade = pd.to_datetime(settlements["trade_date"])
    last_contract = max(int(contract_key(c)[:4]) for c in settlements["contract"].unique())
    years = range(trade.min().year - 1, max(trade.max().year, last_contract) + 2)  # UK holidays through the last expiry
    closures = ice_closure_table(pd.read_csv(path / "ice_closures.csv")["date"], years)
    expiries = pd.read_csv(path / "ice_expiries.csv", dtype=str) if (path / "ice_expiries.csv").exists() else None
    overrides = {}
    if expiries is not None:
        published = expiries[expiries["source"].str.startswith("ICE published")]
        overrides = dict(zip(published["contract_month"], published["last_trading_day"]))
    uk = pd.read_csv(path / "uk_holidays.csv")["date"] if (path / "uk_holidays.csv").exists() else None
    eua = None
    if (path / "eua.csv").exists():
        eua = pd.read_csv(path / "eua.csv")
        if "known_from" not in eua:
            fetched = eua["fetched_at"] if "fetched_at" in eua else None
            eua["known_from"] = eua_known_from(eua["auction_date"], fetched).to_numpy()
    calendars = Calendars.from_closure_table(closures, years, uk_holidays=uk, expiry_overrides=overrides)
    return {"raw": raw, "settlements": settlements, "closures": closures, "expiries": expiries, "calendars": calendars,
            "eua": eua, "years": years}


def load_gas_dir(path, efficiency: float = EFFICIENCY, emission_factor: float = EMISSION_FACTOR, ideal: bool = False) -> GasInputs:
    """GasInputs from a gas directory (read_gas_dir); raises GasDataError on any hard data check. `ideal` builds the research
    measurement's inputs: idealised known_from and the next-contract substitution."""
    parts = read_gas_dir(path, ideal=ideal)
    return GasInputs(parts["settlements"], parts["calendars"], eua=parts["eua"], efficiency=efficiency,
                     emission_factor=emission_factor, mode="ideal" if ideal else "dbnomics", substitute_next=ideal)
