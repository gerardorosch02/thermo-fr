"""Pre-feature checks of the ttf-gas pre-registration (v4, section 4), run on the laptop's gas data before any feature.

`thermo-fr gas-checks` prints one report and exits non-zero when a stop rule holds (1), or when only a candidate is withdrawn
(2). It writes nothing. The checks, with the limits fixed in the pre-registration:

- hard expiry and closure-day checks (gas.settlement_issues): a value after expiry, a known_from before the settlement was
  final, a non-repeat on a weekend or sourced closure stop the branch; repeats are dropped; values on unsourced default
  closures are dropped and listed (more than five reported as evidence against the default, never a stop);
- the expiry file: a rule-derived row that disagrees with the rule stops (the calendars disagree); an ICE-published date
  that differs is listed and wins;
- every contract's expiry, last value and the trading days between them, listed;
- M1: trading days on which front(T) has no value at all, 2022-12-01 to 2026-10-10 and separately 2026-06-19 to
  2026-10-10, stop above 5%;
- runs of three or more consecutive trading days without a front-month value over the M1 span, stop above five runs;
- M2: strict delivery days whose gas features are missing under the as-of rule, per candidate, on the selection window and
  the holdout; above 5% the candidate is withdrawn, and if it is A (whose features C shares) the branch stops;
- M3: new-close delivery days of the selection window (D-2 a trading day) whose D-2 close was not known by 10:05 on D-1,
  stop above 20%;
- data quality: every front contract from Feb23 to the end of the M1 span present, the holdout's contracts (Aug26 to Nov26)
  present, the current contract (Nov26) present, runs of three or more identical consecutive settlements of one contract
  listed and more than five stop;
- the agreement check against the owner's EEX settlements: at least 30 common trade dates, levels MAE <= 0.25 and mean
  difference within +/-0.15, daily changes MAE <= 0.15 and the same sign on >= 95% of days with an EEX change of at least
  0.5, change correlation at lag 0 >= 0.98 and above the correlations at a lag of one trade day either way. Only
  differences are printed, never levels.
All measures are reported either way, by year where the pre-registration asks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .gas import HARD_ISSUES, Calendars, GasDataError, GasInputs, _d64, _ts, contract_key, issue_time, read_gas_dir, settlement_issues

M1_SPAN = ("2022-12-01", "2026-10-10")
M1_HOLDOUT_SPAN = ("2026-06-19", "2026-10-10")
SELECTION = ("2024-02-17", "2026-06-30")
HOLDOUT = ("2026-07-01", "2026-10-10")
MISSING_LIMIT = 0.05  # M1 and M2
STALE_LIMIT = 0.20  # M3
RUN_LENGTH = 3
MAX_RUNS = 5  # runs of missing front-month days, and runs of identical settlements
DEFAULT_CLOSURE_EVIDENCE = 5
FIRST_CHECKED_FRONT = "2023-02"
HOLDOUT_CONTRACTS = ("2026-08", "2026-09", "2026-10", "2026-11")
CURRENT_CONTRACT = "2026-11"
AGREEMENT = {"min_dates": 30, "level_mae": 0.25, "level_mean": 0.15, "change_mae": 0.15, "sign_share": 0.95, "sign_move": 0.5,
             "corr0": 0.98}
CANDIDATES = {"A (honest_v2_gas) and C (honest_v2_gas_rel)": ("gas_missing",), "B (honest_v2_gas_srmc)": ("gas_missing", "gas_srmc_missing")}


@dataclass
class Check:
    name: str
    ok: bool  # False: the rule is breached
    stop: bool = False  # a breach stops the branch
    withdraws: tuple = ()  # candidates withdrawn by the breach
    lines: list = field(default_factory=list)


def trading_days(cal: Calendars, start, end) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return pd.DatetimeIndex([d for d in days if cal.is_trading(d)])


def front_coverage(clean: pd.DataFrame, cal: Calendars, start, end) -> pd.DataFrame:
    """Per trading day T: front(T) and whether the fetch has any value for it on T (known_from not considered)."""
    have = set(zip(clean["contract"], clean["trade_date"]))
    rows = [{"day": d, "front": cal.front(d), "has_value": (cal.front(d), d) in have} for d in trading_days(cal, start, end)]
    return pd.DataFrame(rows)


def _by_year(frame: pd.DataFrame, flag: str, day: str = "day") -> str:
    parts = []
    for year, part in frame.groupby(frame[day].dt.year):
        parts.append(f"{year}: {int(part[flag].sum())}/{len(part)} ({100 * part[flag].mean():.1f}%)")
    return "; ".join(parts)


def check_settlement_issues(issues: dict) -> list[Check]:
    out = []
    labels = {"after_expiry": "values dated after their contract's expiry",
              "known_too_early": "values with known_from before 17:20 Amsterdam time on the trade date",
              "non_trading_not_repeat": "values on a weekend or sourced closure that are not a repeat"}
    for name in HARD_ISSUES:
        bad = issues[name]
        lines = [f"{len(bad)} {labels[name]}"] + [f"  {r.contract} {r.trade_date:%Y-%m-%d}" for r in bad.itertuples()]
        out.append(Check(f"hard: {name}", ok=len(bad) == 0, stop=True, lines=lines))
    rep = issues["non_trading_repeats"]
    out.append(Check("repeats on non-trading days, dropped", ok=True,
                     lines=[f"{len(rep)} dropped"] + [f"  {r.contract} {r.trade_date:%Y-%m-%d}" for r in rep.itertuples()]))
    dflt = issues["default_closure_values"]
    lines = [f"{len(dflt)} values on unsourced default closures, dropped and listed"]
    lines += [f"  {r.contract} {r.trade_date:%Y-%m-%d}" for r in dflt.itertuples()]
    if len(dflt) > DEFAULT_CLOSURE_EVIDENCE:
        lines.append(f"  more than {DEFAULT_CLOSURE_EVIDENCE}: evidence against the default calendar (changes nothing)")
    out.append(Check("soft: values on default closures", ok=True, lines=lines))
    return out


def check_expiry_file(expiries: pd.DataFrame | None, cal: Calendars) -> Check:
    if expiries is None:
        return Check("expiry file", ok=True, lines=["no ice_expiries.csv: the rule alone"])
    rule_mismatch, ice_differs = [], []
    for r in expiries.itertuples():
        rule = _ts(cal.rule_expiry(r.contract_month))
        given = pd.Timestamp(r.last_trading_day)
        if rule != given:
            (ice_differs if str(r.source).startswith("ICE published") else rule_mismatch).append(
                f"  {r.contract_month}: file {given:%Y-%m-%d}, rule {rule:%Y-%m-%d} ({str(r.source)[:40]})")
    lines = [f"{len(expiries)} rows; {len(rule_mismatch)} rule-derived rows disagree with the rule; "
             f"{len(ice_differs)} ICE-published dates differ from the rule (ICE wins)"] + rule_mismatch + ice_differs
    return Check("expiry file against the rule", ok=not rule_mismatch, stop=True, lines=lines)


def expiry_listing(clean: pd.DataFrame, cal: Calendars) -> Check:
    lines = ["contract  expiry      last value  trading days between"]
    for contract, part in clean.groupby("contract"):
        expiry = _ts(cal.expiry(contract))
        last = part["trade_date"].max()
        gap = len(trading_days(cal, last + pd.Timedelta(days=1), expiry)) if last < expiry else 0
        lines.append(f"{contract}   {expiry:%Y-%m-%d}  {last:%Y-%m-%d}  {gap}")
    return Check("expiry, last value and gap per contract", ok=True, lines=lines)


def check_m1(clean: pd.DataFrame, cal: Calendars) -> list[Check]:
    out = []
    for label, span in (("M1", M1_SPAN), ("M1 holdout part", M1_HOLDOUT_SPAN)):
        cov = front_coverage(clean, cal, *span)
        cov["missing"] = ~cov["has_value"]
        share = float(cov["missing"].mean()) if len(cov) else 1.0
        lines = [f"{int(cov['missing'].sum())} of {len(cov)} trading days {span[0]} to {span[1]} without a front-month value "
                 f"({100 * share:.2f}%, limit {100 * MISSING_LIMIT:.0f}%)", "by year: " + _by_year(cov, "missing")]
        out.append(Check(label, ok=share <= MISSING_LIMIT, stop=True, lines=lines))
    return out


def missing_runs(clean: pd.DataFrame, cal: Calendars, span=M1_SPAN, length=RUN_LENGTH) -> list[tuple]:
    cov = front_coverage(clean, cal, *span)
    runs, current = [], []
    for row in cov.itertuples():
        if not row.has_value:
            current.append(row)
            continue
        if len(current) >= length:
            runs.append((current[0].day, current[-1].day, len(current), current[0].front))
        current = []
    if len(current) >= length:
        runs.append((current[0].day, current[-1].day, len(current), current[0].front))
    return runs


def check_runs(clean: pd.DataFrame, cal: Calendars) -> Check:
    runs = missing_runs(clean, cal)
    lines = [f"{len(runs)} runs of {RUN_LENGTH} or more consecutive trading days without a front-month value (limit {MAX_RUNS})"]
    lines += [f"  {a:%Y-%m-%d} to {b:%Y-%m-%d}: {n} days, front {c}" for a, b, n, c in runs]
    return Check("runs of missing front-month days", ok=len(runs) <= MAX_RUNS, stop=True, lines=lines)


def identical_runs(clean: pd.DataFrame, length=RUN_LENGTH) -> list[tuple]:
    first = clean.sort_values("known_from").drop_duplicates(["contract", "trade_date"], keep="first")
    runs = []
    for contract, part in first.sort_values("trade_date").groupby("contract"):
        values, dates = part["value"].to_numpy(), part["trade_date"].to_numpy()
        start = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                if i - start >= length:
                    runs.append((contract, pd.Timestamp(dates[start]), pd.Timestamp(dates[i - 1]), i - start))
                start = i
    return runs


def check_data_quality(clean: pd.DataFrame, cal: Calendars) -> list[Check]:
    present = set(clean["contract"])
    last_front = cal.front(M1_SPAN[1])
    needed = [p.strftime("%Y-%m") for p in pd.period_range(FIRST_CHECKED_FRONT, last_front, freq="M")]
    absent = [c for c in needed if c not in present]
    out = [Check("every front contract from Feb23 present", ok=not absent, stop=True,
                 lines=[f"{len(needed)} contracts {needed[0]} to {needed[-1]}; absent: {', '.join(absent) or 'none'}"])]
    absent_h = [c for c in HOLDOUT_CONTRACTS if c not in present]
    out.append(Check("holdout contracts present (Aug26 to Nov26)", ok=not absent_h, stop=True, lines=[f"absent: {', '.join(absent_h) or 'none'}"]))
    later = sorted(c for c in present if c >= CURRENT_CONTRACT)
    out.append(Check("current contracts in the fetch (Nov26 onward)", ok=CURRENT_CONTRACT in present, stop=True,
                     lines=[f"contracts from {CURRENT_CONTRACT}: {', '.join(later) or 'none'}"]))
    runs = identical_runs(clean)
    lines = [f"{len(runs)} runs of {RUN_LENGTH} or more identical consecutive settlements of one contract (limit {MAX_RUNS})"]
    lines += [f"  {c} {a:%Y-%m-%d} to {b:%Y-%m-%d}: {n} values" for c, a, b, n in runs]
    out.append(Check("identical consecutive settlements", ok=len(runs) <= MAX_RUNS, stop=True, lines=lines))
    return out


def window_days(span, strict_days=None) -> pd.DatetimeIndex:
    days = pd.date_range(*span, freq="D")
    return days if strict_days is None else days[days.isin(pd.DatetimeIndex(strict_days))]


def check_m2(gas: GasInputs, strict_days=None) -> list[Check]:
    out = []
    note = "strict delivery days" if strict_days is not None else "all delivery days (no inputs table given, strictness not applied)"
    for label, span in (("selection", SELECTION), ("holdout", HOLDOUT)):
        days = window_days(span, strict_days)
        table = gas.feature_table(days)
        for candidate, flags in CANDIDATES.items():
            if table.empty:  # nothing to measure counts as a breach
                missing, reasons = pd.Series(dtype=bool), {}
            else:
                missing = table[list(flags)].any(axis=1)
                reasons = table.loc[missing, "gas_missing_reason"].value_counts().to_dict()
            share = float(missing.mean()) if len(missing) else 1.0
            frame = pd.DataFrame({"day": pd.DatetimeIndex(table.index), "missing": missing.to_numpy()})
            is_a = candidate.startswith("A")
            lines = [f"{int(missing.sum())} of {len(missing)} {note} {span[0]} to {span[1]} without gas features "
                     f"({100 * share:.2f}%, limit {100 * MISSING_LIMIT:.0f}%)", f"reasons: {reasons}", "by year: " + _by_year(frame, "missing")]
            breach = share > MISSING_LIMIT
            withdraws = (("A", "C") if is_a else ("B",)) if breach else ()
            out.append(Check(f"M2 {label}, {candidate}", ok=not breach, stop=is_a, withdraws=withdraws, lines=lines))
    return out


def check_m3(gas: GasInputs) -> Check:
    cal = gas.calendars
    rows = []
    for day in pd.date_range(*SELECTION, freq="D"):
        d2 = day - pd.Timedelta(days=2)
        if not cal.is_trading(d2):
            continue
        known = gas.value(cal.front(d2), _d64(d2), issue_time(day)) is not None
        rows.append({"day": day, "stale": not known})
    frame = pd.DataFrame(rows)
    share = float(frame["stale"].mean()) if len(frame) else 1.0
    lines = [f"{int(frame['stale'].sum())} of {len(frame)} new-close delivery days {SELECTION[0]} to {SELECTION[1]} whose D-2 close "
             f"was not known by 10:05 on D-1 ({100 * share:.2f}%, limit {100 * STALE_LIMIT:.0f}%)", "by year: " + _by_year(frame, "stale")]
    return Check("M3", ok=share <= STALE_LIMIT, stop=True, lines=lines)


def load_eex_tables(paths) -> pd.DataFrame:
    """The owner's EEX TTF settlement snapshots (JSON with header and data, columns maturityDate, tradeDate, settlPx), or a
    CSV with contract, trade_date, value. Later files win for the same contract and trade date."""
    frames = []
    for path in sorted(Path(p) for p in paths):
        if path.suffix == ".csv":
            f = pd.read_csv(path)[["contract", "trade_date", "value"]]
        else:
            o = json.loads(path.read_text(encoding="utf-8"))
            raw = pd.DataFrame(o["data"], columns=o["header"])
            f = pd.DataFrame({"contract": [_maturity_month(m) for m in raw["maturityDate"]], "trade_date": raw["tradeDate"],
                              "value": raw["settlPx"]})
        f["contract"] = [contract_key(c) for c in f["contract"]]
        f["trade_date"] = pd.to_datetime(f["trade_date"]).dt.tz_localize(None).dt.normalize()
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        frames.append(f.dropna(subset=["value"]))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["contract", "trade_date", "value"])
    return table.drop_duplicates(["contract", "trade_date"], keep="last").reset_index(drop=True)


def _maturity_month(value) -> str:
    text = str(value).strip()
    if text.isdigit() and len(text) in (6, 8):
        return f"{text[:4]}-{text[4:6]}"
    return contract_key(text)


def agreement(dbnomics: pd.DataFrame, eex: pd.DataFrame, check_date) -> tuple[bool, list[str]]:
    """The agreement check; returns (pass, report lines). Only differences are printed."""
    a = AGREEMENT
    latest = dbnomics.sort_values("known_from").drop_duplicates(["contract", "trade_date"], keep="last") if "known_from" in dbnomics else dbnomics
    pairs = latest[["contract", "trade_date", "value"]].merge(eex, on=["contract", "trade_date"], suffixes=("_dbn", "_eex"))
    pairs = pairs[pairs["trade_date"] <= pd.Timestamp(check_date) - pd.Timedelta(days=1)].sort_values(["contract", "trade_date"])
    n_dates = pairs["trade_date"].nunique()
    lines = [f"{len(pairs)} common (contract, trade date) pairs, {n_dates} common trade dates (need {a['min_dates']}); contracts "
             f"{', '.join(sorted(pairs['contract'].unique())) or 'none'}"]
    if n_dates < a["min_dates"]:
        return False, lines + ["FAIL: too few common trade dates"]
    pairs["diff"] = pairs["value_dbn"] - pairs["value_eex"]
    g = pairs.groupby("contract")
    pairs["chg_dbn"], pairs["chg_eex"] = g["value_dbn"].diff(), g["value_eex"].diff()
    pairs["dbn_next"], pairs["dbn_prev"] = g["chg_dbn"].shift(-1), g["chg_dbn"].shift(1)
    ch = pairs.dropna(subset=["chg_dbn", "chg_eex"])
    level_mae, level_mean = pairs["diff"].abs().mean(), pairs["diff"].mean()
    change_mae = (ch["chg_dbn"] - ch["chg_eex"]).abs().mean()
    big = ch[ch["chg_eex"].abs() >= a["sign_move"]]
    sign_share = float((np.sign(big["chg_dbn"]) == np.sign(big["chg_eex"])).mean()) if len(big) else float("nan")
    corr0 = ch["chg_dbn"].corr(ch["chg_eex"])
    corr_lead = ch["dbn_next"].corr(ch["chg_eex"])  # DBnomics one trade day later against EEX
    corr_lag = ch["dbn_prev"].corr(ch["chg_eex"])
    conditions = {
        f"levels MAE {level_mae:.3f} <= {a['level_mae']}": level_mae <= a["level_mae"],
        f"levels mean difference {level_mean:+.3f} within +/-{a['level_mean']}": abs(level_mean) <= a["level_mean"],
        f"daily changes MAE {change_mae:.3f} <= {a['change_mae']}": change_mae <= a["change_mae"],
        f"same sign on {sign_share:.1%} of {len(big)} days with an EEX change >= {a['sign_move']} (need {a['sign_share']:.0%})":
            len(big) > 0 and sign_share >= a["sign_share"],
        f"change correlation lag 0 {corr0:.4f} >= {a['corr0']}": corr0 >= a["corr0"],
        f"lag 0 above lag +1 ({corr_lead:.4f}) and lag -1 ({corr_lag:.4f})": corr0 > corr_lead and corr0 > corr_lag,
    }
    lines += [("pass  " if ok else "FAIL  ") + text for text, ok in conditions.items()]
    lines.append("five largest level differences (DBnomics minus EEX):")
    for r in pairs.reindex(pairs["diff"].abs().sort_values(ascending=False).index).head(5).itertuples():
        lines.append(f"  {r.contract} {r.trade_date:%Y-%m-%d} {r.diff:+.3f}")
    lines.append("five largest daily change differences:")
    ch = ch.assign(cdiff=ch["chg_dbn"] - ch["chg_eex"])
    for r in ch.reindex(ch["cdiff"].abs().sort_values(ascending=False).index).head(5).itertuples():
        lines.append(f"  {r.contract} {r.trade_date:%Y-%m-%d} {r.cdiff:+.3f}")
    return all(conditions.values()), lines


def strict_days_from_inputs(path) -> pd.DatetimeIndex:
    """Paris delivery days with at least one strict hour (all three as-issued weather columns present) in the inputs table."""
    inputs = pd.read_csv(path, index_col=0, usecols=["timestamp_utc", "temp_fc_c", "wind100_fc_ms", "radiation_fc_wm2"])
    inputs.index = pd.to_datetime(inputs.index, utc=True)
    strict = inputs.notna().all(axis=1)
    days = inputs.index[strict].tz_convert(LOCAL_TZ).normalize().tz_localize(None)
    return pd.DatetimeIndex(days.unique())


def run_checks(gas_dir, eex_paths=(), inputs_path=None, check_date=None) -> list[Check]:
    """Every pre-feature check on a gas directory; nothing is written."""
    try:
        parts = read_gas_dir(gas_dir)
    except GasDataError as error:
        return [Check("loading the gas directory", ok=False, stop=True, lines=[str(error)])]
    cal = parts["calendars"]
    sourced = int(parts["closures"]["sourced"].sum())
    checks = [Check("calendars", ok=True, lines=[f"ICE Endex closures: {sourced} sourced, {len(parts['closures']) - sourced} default "
                                                 f"(unsourced dates); years {parts['years'].start} to {parts['years'].stop - 1}"])]
    clean, issues = settlement_issues(parts["settlements"], cal)
    checks += check_settlement_issues(issues)
    checks.append(check_expiry_file(parts["expiries"], cal))
    checks.append(expiry_listing(clean, cal))
    checks += check_m1(clean, cal)
    checks.append(check_runs(clean, cal))
    checks += check_data_quality(clean, cal)
    check_date = pd.Timestamp(check_date) if check_date else pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()
    if eex_paths:
        ok, lines = agreement(clean, load_eex_tables(eex_paths), check_date)
        checks.append(Check("agreement with the EEX settlements", ok=ok, stop=True, lines=lines))
    else:
        checks.append(Check("agreement with the EEX settlements", ok=False, stop=True, lines=["not run: no EEX snapshot given"]))
    if any(issues[name].shape[0] for name in HARD_ISSUES):
        checks.append(Check("M2 and M3", ok=False, stop=True, lines=["not computed: a hard data check failed"]))
        return checks
    gas = GasInputs(parts["settlements"], cal, eua=parts["eua"])
    strict = strict_days_from_inputs(inputs_path) if inputs_path else None
    checks += check_m2(gas, strict)
    checks.append(check_m3(gas))
    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["ttf-gas pre-feature checks (pre-registration v4, section 4)", ""]
    for c in checks:
        status = "ok" if c.ok else ("STOP" if c.stop else ("WITHDRAW " + "/".join(c.withdraws) if c.withdraws else "BREACH"))
        lines.append(f"[{status}] {c.name}")
        lines += [f"    {line}" for line in c.lines]
    stops = [c.name for c in checks if not c.ok and c.stop]
    withdrawn = sorted({w for c in checks if not c.ok for w in c.withdraws})
    lines += ["", f"Stop rules breached: {', '.join(stops) if stops else 'none'}",
              f"Candidates withdrawn: {', '.join(withdrawn) if withdrawn else 'none'}"]
    return "\n".join(lines)


def exit_code(checks: list[Check]) -> int:
    if any(not c.ok and c.stop for c in checks):
        return 1
    if any(not c.ok and c.withdraws for c in checks):
        return 2
    return 0
