"""Nuclear availability as known at the issue time, and hydro reservoir levels (branch nuclear-hydro).

Pre-registration: nuclear_hydro_prereg.md (scratchpad), sections 2 and 4. Nothing here reads prices.

Nuclear (section 2). tau is 10:05 Paris on D-1, the pre-market issue time.
- Perimeter: EDF REMIT rows with filiere "Nucléaire", less Fessenheim 1 and 2 (closed 2020). Flamanville 3 counts from its
  first grid connection, 21 December 2024 at 11:48 Paris (EDF press release of that day): quarter-hours that start before it
  leave the unit out.
- Version in force at tau: each version's effective publication time is the running maximum of its publication time over
  the notice's versions up to it; the version in force is the highest version whose effective time is at or before tau. A
  notice with no such version does not exist at tau. The export's `status` is never used to select versions.
- A version whose status is Annulée or Supprimée is the notice's cancellation (it is always the notice's highest version, which
  `register_checks` verifies): when it is the version in force the notice contributes nothing.
- Types: Planifiée and Fortuite are used; every chronicle type (Chronique PMIN, Chronique aFRR, and the older labels Chronique
  and "Chronique.") is excluded.
- Quarter-hour grid in UTC over the Paris delivery day (92, 96 or 100 quarter-hours). A version applies to a quarter-hour if
  [date_de_debut, date_de_fin) overlaps it. Unit availability is the smaller of the unit's Pmax and the lowest
  puissance_disponible_mw among the applicable versions in force; fleet availability is the sum over units; hours are the
  mean of their four quarter-hours.
- Pmax: puissance_maximale_mw of the unit's most recently published non-chronicle version known at tau (effective time); if
  none is known, its earliest-published version ever (counted).
- Publication stamps: the full-history export truncates them to the minute, the rolling export keeps seconds (the two agree to
  the truncation on every common version). A version known only from the history counts as published at the end of its
  minute (stamp + 59 s), so a version stamped in the minute of tau is never treated as known earlier than it may have been.

Hydro (section 4).
- ENTSO-E A72 FR: the week ending on Sunday W is known from 23:59 Paris on the third working day after W (Monday to Friday,
  less French public holidays; Regulation 543/2013 Article 16(2)(d)). The value used is the latest week known at tau, if it
  is at most three weeks older than the latest due week; otherwise the features are missing.
- BFE ogd17: the row dated Monday t is known from t + 25 days (00:00 Paris). The CH anomaly is TotalCH content as a share of
  TotalCH maximum (percent) minus the mean of the same ISO week over every earlier ISO year from 2000.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd

from ..data.edf_remit import CANCELLED, DELETED, FORCED, NUCLEAR, PLANNED

LOCAL_TZ = "Europe/Paris"
USED_TYPES = (PLANNED, FORCED)
CANCELLING = (CANCELLED, DELETED)
EXCLUDED_UNITS = ("FESSENHEIM 1", "FESSENHEIM 2")
FLAMANVILLE_3 = "FLAMANVILLE 3"
FLAMANVILLE_3_GRID = pd.Timestamp("2024-12-21 11:48").tz_localize(LOCAL_TZ).tz_convert("UTC")
MINUTE_STAMP_MARGIN = pd.Timedelta(seconds=59)
QH = pd.Timedelta(minutes=15)
QH_NS = QH.value
NEVER = np.iinfo(np.int64).max

NUCLEAR_FEATURES = ["nuc_avail_mw", "nuc_avail_chg_mw", "residual_avail_mw"]
HYDRO_FEATURES = ["fr_hydro_fill_mwh", "fr_hydro_chg4w_mwh", "ch_hydro_anom_pct"]
A72_TOLERANCE = pd.Timedelta(weeks=3)
A72_WORKING_DAYS = 3
BFE_LAG = pd.Timedelta(days=25)


# ----------------------------------------------------------------------------------------------------------------------
# timing helpers


def issue_tau(day) -> pd.Timestamp:
    """10:05 Paris on D-1, in UTC."""
    local = pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=5)
    return local.tz_localize(LOCAL_TZ).tz_convert("UTC")


def reference_tau(day) -> pd.Timestamp:
    """12:00 Paris on D-2: the D-1 auction, when the D-1 reference availability is taken."""
    local = pd.Timestamp(day).normalize() - pd.Timedelta(days=2) + pd.Timedelta(hours=12)
    return local.tz_localize(LOCAL_TZ).tz_convert("UTC")


def day_quarter_hours(day) -> pd.DatetimeIndex:
    """UTC starts of the quarter-hours of Paris delivery day D (92, 96 or 100)."""
    start = pd.Timestamp(day).normalize().tz_localize(LOCAL_TZ)
    end = (pd.Timestamp(day).normalize() + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")


# ----------------------------------------------------------------------------------------------------------------------
# register: merge of the two exports, checks


def merge_exports(history: pd.DataFrame, rolling: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """One register from the full-history export (minute stamps) and the rolling export (second stamps).

    - Duplicate (notice, version) rows within an export keep the row with the latest publication stamp.
    - A version in both exports keeps the history's fields and the rolling export's exact publication time.
    - A version only in the history counts as published at stamp + 59 s.
    - A version only in the rolling export is added as it is.
    `published_raw` keeps the stamp as exported; `published` is the time used by the rules.
    """
    key = ["notice", "version"]
    facts = {"history_rows": int(len(history)), "history_duplicate_keys": int(history.duplicated(key).sum())}
    h = history.sort_values(key + ["published"]).drop_duplicates(key, keep="last").copy()
    h["published_raw"] = h["published"]
    h["source"] = "history"
    h["published"] = h["published_raw"] + MINUTE_STAMP_MARGIN
    if rolling is None:
        return h.reset_index(drop=True), facts
    facts["rolling_rows"] = int(len(rolling))
    facts["rolling_duplicate_keys"] = int(rolling.duplicated(key).sum())
    r = rolling.sort_values(key + ["published"]).drop_duplicates(key, keep="last").copy()
    exact = r.set_index(key)["published"]
    hk = pd.MultiIndex.from_frame(h[key])
    common = hk.isin(exact.index)
    h.loc[common, "published"] = exact.reindex(hk[common]).to_numpy()
    h.loc[common, "source"] = "both"
    rk = pd.MultiIndex.from_frame(r[key])
    extra = r[~rk.isin(hk)].copy()
    extra["published_raw"] = extra["published"]
    extra["source"] = "rolling"
    facts["common_versions"] = int(common.sum())
    facts["rolling_only_versions"] = int(len(extra))
    facts["history_last_stamp"] = str(history["published"].max())
    facts["rolling_only_published_before_history_end"] = int((extra["published"] <= history["published"].max() + MINUTE_STAMP_MARGIN).sum())
    merged = pd.concat([h, extra], ignore_index=True).sort_values(key).reset_index(drop=True)
    return merged, facts


def export_agreement(history: pd.DataFrame, rolling: pd.DataFrame, fuel: str | None = NUCLEAR) -> dict:
    """Field-by-field agreement of the versions present in both exports (status excluded: it is the state at export time)."""
    key = ["notice", "version"]
    h = history.drop_duplicates(key, keep="last")
    r = rolling.drop_duplicates(key, keep="last")
    if fuel is not None:
        h, r = h[h["fuel"] == fuel], r[r["fuel"] == fuel]
    m = h.merge(r, on=key, suffixes=("_h", "_r"))
    out = {"common_versions": int(len(m))}
    for c in ("unit", "fuel", "type", "start", "end", "max_mw", "available_mw", "cause", "information"):
        a, b = m[f"{c}_h"], m[f"{c}_r"]
        out[f"mismatch_{c}"] = int((~((a == b) | (a.isna() & b.isna()))).sum())
    out["published_history_equals_rolling_floored_to_minute"] = bool((m["published_h"] == m["published_r"].dt.floor("min")).all())
    bad = m[m["max_mw_h"] != m["max_mw_r"]]
    out["max_mw_mismatches"] = bad[key + ["unit_h", "max_mw_h", "max_mw_r", "published_h"]].astype(str).to_dict("records")
    rk = r.set_index(key).index
    hk = h.set_index(key).index
    # every version of a notice the rolling export holds, published before the history's last stamp, must be in the history
    early = r[(r["published"] < history["published"].max()) & ~rk.isin(hk)]
    out["rolling_versions_before_history_end_missing_from_history"] = int(len(early))
    # the rolling export drops versions whose event ended before its window; history-only versions of shared notices must be those
    shared = h[h["notice"].isin(r["notice"]) & ~hk.isin(rk)]
    out["history_only_versions_of_shared_notices"] = int(len(shared))
    out["of_which_event_end_before_rolling_window"] = int((shared["end"] < r["end"].min()).sum())
    return out


def register_checks(register: pd.DataFrame) -> dict:
    """Assumptions the rule relies on, verified on the nuclear rows: cancellation versions are the highest of their notice,
    a notice never changes type or unit, and chronicles never reduce availability."""
    n = register[register["fuel"] == NUCLEAR]
    top = n.groupby("notice")["version"].transform("max")
    cancel = n["status"].isin(CANCELLING)
    chron = ~n["type"].isin(USED_TYPES)
    return {
        "cancellation_versions": int(cancel.sum()),
        "cancellation_versions_not_highest": int((cancel & (n["version"] != top)).sum()),
        "notices_with_several_types": int((n.groupby("notice")["type"].nunique() > 1).sum()),
        "notices_with_several_units": int((n.groupby("notice")["unit"].nunique() > 1).sum()),
        "chronicle_rows": int(chron.sum()),
        "chronicle_types": sorted(n.loc[chron, "type"].unique().tolist()),
        "chronicle_rows_with_reduction": int((n.loc[chron, "unavailable_mw"] > 0).sum()),
        "used_rows_missing_available": int(n.loc[~chron, "available_mw"].isna().sum()),
        "rows_with_end_not_after_start": int((n["end"] <= n["start"]).sum()),
    }


# ----------------------------------------------------------------------------------------------------------------------
# availability as known at tau


@dataclass
class Counts:
    """Data counts accumulated over every call (section 5, reported diagnostics)."""

    calls: int = 0
    notices_used: set = field(default_factory=set)
    cancelled_applied: set = field(default_factory=set)
    deleted_applied: set = field(default_factory=set)
    missing_available_skipped: set = field(default_factory=set)
    units_without_known_version: int = 0  # unit-calls whose Pmax came from the earliest version ever

    def summary(self) -> dict:
        return {"calls": self.calls, "notices_used": len(self.notices_used), "cancelled_notices_applied": len(self.cancelled_applied),
                "deleted_notices_applied": len(self.deleted_applied),
                "notices_skipped_missing_available": len(self.missing_available_skipped),
                "unit_calls_without_known_version": self.units_without_known_version}


class NuclearRegister:
    """The nuclear part of the merged register, prepared for repeated as-of queries."""

    def __init__(self, register: pd.DataFrame):
        n = register[(register["fuel"] == NUCLEAR) & ~register["unit"].isin(EXCLUDED_UNITS)].copy()
        n = n.sort_values(["notice", "version"]).reset_index(drop=True)
        n["effective"] = n.groupby("notice")["published"].cummax()
        self.chronicles_excluded = int((~n["type"].isin(USED_TYPES)).sum())
        self.chronicle_notices_excluded = int(n.loc[~n["type"].isin(USED_TYPES), "notice"].nunique())
        used = n[n["type"].isin(USED_TYPES)].reset_index(drop=True)
        nxt = used.groupby("notice")["effective"].shift(-1)
        self.units = sorted(n["unit"].unique())
        unit_code = {u: i for i, u in enumerate(self.units)}
        self.frame = used
        self.notice = used["notice"].to_numpy()
        self.unit = used["unit"].map(unit_code).to_numpy()
        self.eff = used["effective"].astype("int64").to_numpy()
        filled = nxt.fillna(pd.Timestamp(0, tz="UTC")).astype("int64").to_numpy()
        self.next_eff = np.where(nxt.isna().to_numpy(), NEVER, filled).astype(np.int64)
        self.start = used["start"].astype("int64").to_numpy()
        self.end = used["end"].astype("int64").to_numpy()
        self.available = used["available_mw"].to_numpy(dtype=float)
        self.cancelled = used["status"].eq(CANCELLED).to_numpy()
        self.deleted = used["status"].eq(DELETED).to_numpy()
        # Pmax history per unit: non-chronicle versions ordered by effective time (ties by notice and version order)
        self.pmax_eff, self.pmax_val, self.pmax_first = [], [], []
        for u in self.units:
            rows = used[used["unit"] == u]
            order = np.lexsort((rows["version"].to_numpy(), rows["notice"].to_numpy(), rows["effective"].astype("int64").to_numpy()))
            self.pmax_eff.append(rows["effective"].astype("int64").to_numpy()[order])
            self.pmax_val.append(rows["max_mw"].to_numpy(dtype=float)[order])
            self.pmax_first.append(float(rows["max_mw"].to_numpy(dtype=float)[order][0]) if len(rows) else np.nan)
        self.fl3 = unit_code.get(FLAMANVILLE_3)
        self.counts = Counts()

    def pmax(self, u: int, t: int) -> tuple[float, bool]:
        """(Pmax of unit u as known at t, whether a version was known)."""
        k = np.searchsorted(self.pmax_eff[u], t, side="right")
        if k == 0:
            return self.pmax_first[u], False
        return float(self.pmax_val[u][k - 1]), True

    def in_force(self, tau) -> np.ndarray:
        t = pd.Timestamp(tau).value
        return (self.eff <= t) & (t < self.next_eff)

    def unit_availability(self, qh: pd.DatetimeIndex, tau) -> np.ndarray:
        """Matrix (units x quarter-hours) of availability as known at tau; Flamanville 3 is 0 before its grid connection."""
        t = pd.Timestamp(tau).value
        q = pd.DatetimeIndex(qh).asi8
        origin, n_q = q[0], len(q)
        if not np.all(np.diff(q) == QH_NS):
            raise ValueError("quarter-hours must be a regular 15-minute grid")
        win_start, win_end = origin, origin + n_q * QH_NS
        self.counts.calls += 1
        mat = np.empty((len(self.units), n_q))
        for u in range(len(self.units)):
            p, known = self.pmax(u, t)
            if not known:
                self.counts.units_without_known_version += 1
            mat[u, :] = p
        if self.fl3 is not None:
            mat[self.fl3, q < FLAMANVILLE_3_GRID.value] = 0.0
        force = (self.eff <= t) & (t < self.next_eff) & (self.start < win_end) & (self.end > win_start)
        rows = np.flatnonzero(force)
        for i in rows:
            if self.cancelled[i]:
                self.counts.cancelled_applied.add(self.notice[i])
                continue
            if self.deleted[i]:
                self.counts.deleted_applied.add(self.notice[i])
                continue
            if not np.isfinite(self.available[i]):
                self.counts.missing_available_skipped.add(self.notice[i])
                continue
            a = max(0, (self.start[i] - origin) // QH_NS)
            b = min(n_q, -((origin - self.end[i]) // QH_NS))  # ceil((end - origin) / QH)
            if b <= a:
                continue
            self.counts.notices_used.add(self.notice[i])
            u = self.unit[i]
            seg = mat[u, a:b]
            if u == self.fl3:
                live = q[a:b] >= FLAMANVILLE_3_GRID.value
                seg[live] = np.minimum(seg[live], self.available[i])
            else:
                np.minimum(seg, self.available[i], out=seg)
        return mat

    def fleet(self, qh: pd.DatetimeIndex, tau) -> np.ndarray:
        return self.unit_availability(qh, tau).sum(axis=0)

    def fleet_day(self, day, tau=None) -> pd.Series:
        """Hourly fleet availability (MW, mean of the quarter-hours) for Paris delivery day D as known at tau (default 10:05
        Paris on D-1), indexed by the UTC hour start."""
        tau = issue_tau(day) if tau is None else tau
        qh = day_quarter_hours(day)
        values = self.fleet(qh, tau)
        return pd.Series(values, index=qh).groupby(qh.floor("h")).mean().rename("nuc_avail_mw")


def nuclear_daily_table(reg: NuclearRegister, days, log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(hourly, daily): hourly `nuc_avail_mw` for every hour of `days` as known at 10:05 Paris on D-1 (UTC hour index); daily
    the mean for D as known at that tau, the mean for D-1 as known at 12:00 Paris on D-2, their difference `nuc_avail_chg_mw`,
    and both taus."""
    hourly, daily = [], []
    for k, D in enumerate(pd.DatetimeIndex(days)):
        tau, ref = issue_tau(D), reference_tau(D)
        qh = day_quarter_hours(D)
        cur = reg.fleet(qh, tau)
        prev = reg.fleet(day_quarter_hours(D - pd.Timedelta(days=1)), ref)
        hourly.append(pd.Series(cur, index=qh).groupby(qh.floor("h")).mean())
        daily.append({"delivery_day": D, "tau": tau, "reference_tau": ref, "nuc_avail_mean_mw": float(cur.mean()),
                      "nuc_avail_prev_mean_mw": float(prev.mean()), "nuc_avail_chg_mw": float(cur.mean() - prev.mean())})
        if log and k % 200 == 0:
            log(f"  nuclear availability: {D:%Y-%m-%d}")
    h = pd.concat(hourly).rename("nuc_avail_mw")
    h.index.name = "timestamp_utc"
    return h.to_frame(), pd.DataFrame(daily).set_index("delivery_day")


def ex_post_hourly(reg: NuclearRegister, start_day, end_day, tau) -> pd.Series:
    """Hourly fleet availability over Paris days [start_day, end_day] from every version known at one time tau (the latest
    state when tau is the export time); used by the coverage gate's physical check."""
    start = pd.Timestamp(start_day).tz_localize(LOCAL_TZ)
    end = (pd.Timestamp(end_day) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    qh = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    values = reg.fleet(qh, tau)
    return pd.Series(values, index=qh).groupby(qh.floor("h")).mean().rename("nuc_avail_ex_post_mw")


# ----------------------------------------------------------------------------------------------------------------------
# hydro


def french_working_days_after(day: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th French working day (Monday to Friday, not a French public holiday) after `day`."""
    fr = holidays.France(years=range(day.year, day.year + 2))
    d, k = pd.Timestamp(day), 0
    while k < n:
        d += pd.Timedelta(days=1)
        if d.dayofweek < 5 and d.date() not in fr:
            k += 1
    return d


def a72_known_from(week_end) -> pd.Timestamp:
    """23:59 Paris on the third French working day after the Sunday that ends the week, in UTC."""
    d = french_working_days_after(pd.Timestamp(week_end).normalize(), A72_WORKING_DAYS)
    return (d + pd.Timedelta(hours=23, minutes=59)).tz_localize(LOCAL_TZ).tz_convert("UTC")


def a72_table(a72: pd.DataFrame, zone: str = "10YFR-RTE------C") -> pd.DataFrame:
    """Weekly FR series with its known-from time, indexed by week_end (the Sunday)."""
    z = a72[a72["zone"] == zone].copy()
    z = z.set_index(pd.DatetimeIndex(z["week_end"]).normalize()).sort_index()
    z["known_from"] = [a72_known_from(w) for w in z.index]
    return z[["stored_mwh", "known_from"]]


def a72_asof(table: pd.DataFrame, tau, first_week=None) -> dict:
    """The A72 features as known at tau: the latest week known at tau, if at most three weeks older than the latest due week.

    The latest due week is the latest Sunday whose publication deadline is at or before tau, whether or not the file holds it."""
    tau = pd.Timestamp(tau)
    local = tau.tz_convert(LOCAL_TZ).tz_localize(None)
    # latest due Sunday: walk back from the last Sunday before tau's local date until its deadline has passed
    w = (local.normalize() - pd.Timedelta(days=(local.dayofweek + 1) % 7 or 7))
    while a72_known_from(w) > tau:
        w -= pd.Timedelta(days=7)
    due = w
    known = table[(table["known_from"] <= tau) & table["stored_mwh"].notna()]
    if known.empty:
        return {"fr_hydro_fill_mwh": np.nan, "fr_hydro_chg4w_mwh": np.nan, "a72_week": pd.NaT, "a72_due_week": due,
                "a72_known_from": pd.NaT, "a72_age_weeks": np.nan}
    week = known.index.max()
    age = (due - week) / pd.Timedelta(weeks=1)
    if due - week > A72_TOLERANCE:
        return {"fr_hydro_fill_mwh": np.nan, "fr_hydro_chg4w_mwh": np.nan, "a72_week": week, "a72_due_week": due,
                "a72_known_from": known.at[week, "known_from"], "a72_age_weeks": age}
    value = float(known.at[week, "stored_mwh"])
    earlier = week - pd.Timedelta(weeks=4)
    prev = float(known.at[earlier, "stored_mwh"]) if earlier in known.index else np.nan
    return {"fr_hydro_fill_mwh": value, "fr_hydro_chg4w_mwh": value - prev, "a72_week": week, "a72_due_week": due,
            "a72_known_from": known.at[week, "known_from"], "a72_age_weeks": age}


def bfe_table(bfe: pd.DataFrame) -> pd.DataFrame:
    """BFE rows with share (percent of TotalCH maximum), ISO-week climatology over earlier ISO years from 2000, anomaly and
    known-from time, indexed by the row date."""
    b = bfe.copy()
    b = b.set_index(pd.DatetimeIndex(b["date"]).normalize()).sort_index()
    b["share_pct"] = 100.0 * b["totalch_speicherinhalt_gwh"] / b["totalch_max_speicherinhalt_gwh"]
    iso = b.index.isocalendar()
    b["iso_year"], b["iso_week"] = iso["year"].astype(int).to_numpy(), iso["week"].astype(int).to_numpy()
    clim = []
    for (y, w) in zip(b["iso_year"], b["iso_week"]):
        past = b[(b["iso_week"] == w) & (b["iso_year"] < y) & (b["iso_year"] >= 2000)]["share_pct"]
        clim.append(float(past.mean()) if len(past) else np.nan)
    b["climatology_pct"] = clim
    b["ch_hydro_anom_pct"] = b["share_pct"] - b["climatology_pct"]
    b["known_from"] = (b.index + BFE_LAG).tz_localize(LOCAL_TZ).tz_convert("UTC")
    return b


def bfe_asof(table: pd.DataFrame, tau) -> dict:
    known = table[table["known_from"] <= pd.Timestamp(tau)]
    if known.empty:
        return {"ch_hydro_anom_pct": np.nan, "bfe_date": pd.NaT, "bfe_known_from": pd.NaT}
    row = known.index.max()
    return {"ch_hydro_anom_pct": float(known.at[row, "ch_hydro_anom_pct"]), "bfe_date": row, "bfe_known_from": known.at[row, "known_from"]}


def hydro_daily_table(a72: pd.DataFrame, bfe: pd.DataFrame, days) -> pd.DataFrame:
    """Hydro features per delivery day as known at 10:05 Paris on D-1, with the weeks and rows used and their known-from times."""
    at, bt = a72_table(a72), bfe_table(bfe)
    out = []
    for D in pd.DatetimeIndex(days):
        tau = issue_tau(D)
        out.append({"delivery_day": D, "tau": tau, **a72_asof(at, tau), **bfe_asof(bt, tau)})
    frame = pd.DataFrame(out).set_index("delivery_day")
    for c in ("a72_known_from", "bfe_known_from"):
        late = frame[c].notna() & (pd.to_datetime(frame[c], utc=True) > frame["tau"])
        if late.any():
            raise AssertionError(f"hydro value known after tau on {int(late.sum())} days ({c})")
    return frame


# ----------------------------------------------------------------------------------------------------------------------
# hourly inputs for the feature builder


def hourly_inputs(index: pd.DatetimeIndex, nuclear_hourly: pd.DataFrame | None, nuclear_daily: pd.DataFrame | None,
                  hydro_daily: pd.DataFrame | None) -> pd.DataFrame:
    """The new inputs on the hourly UTC index of the inputs table: nuc_avail_mw (per hour), nuc_avail_chg_mw and the three
    hydro values (per Paris delivery day)."""
    index = pd.DatetimeIndex(index)
    days = pd.DatetimeIndex(index.tz_convert(LOCAL_TZ).normalize().tz_localize(None))
    out = pd.DataFrame(index=index)
    if nuclear_hourly is not None:
        out["nuc_avail_mw"] = nuclear_hourly["nuc_avail_mw"].reindex(index).to_numpy()
        out["nuc_avail_chg_mw"] = nuclear_daily["nuc_avail_chg_mw"].reindex(days).to_numpy()
    if hydro_daily is not None:
        for c in HYDRO_FEATURES:
            out[c] = hydro_daily[c].reindex(days).to_numpy()
    return out
