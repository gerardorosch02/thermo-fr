"""Nuclear availability as known at issue and hydro levels (branch nuclear-hydro, nuclear_hydro_prereg.md section 8): construction
tests on hand-built registers, perturbation tests (anything published after tau changes nothing for D), and the hard gates."""

import json

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr import cli
from thermo_fr.data.edf_remit import (ACTIVE, AFRR_CHRONICLE, CANCELLED, DELETED, FORCED, INACTIVE, NUCLEAR, PLANNED, PMIN_CHRONICLE,
                                      load_edf_remit)
from thermo_fr.forecast import features, lear, models, nh_sprint, timing
from thermo_fr.forecast import nuclear_hydro as nh
from thermo_fr.forecast.backtest import walk_forward
from thermo_fr.forecast.features import build_features, feature_timings
from thermo_fr.forecast.timing import LookaheadError, issue_times

HEADER = ("perimetre_juridique;perimetre_spatial;status;identifiant;numero_de_version;nom;filiere;date_de_publication;"
          "date_de_debut;date_de_fin;type;cause;information_complementaire;puissance_maximale_mw;puissance_disponible_mw")
D = pd.Timestamp("2025-01-15")
TAU = nh.issue_tau(D)  # 2025-01-14 09:05 UTC
DAY_START, DAY_END = "2025-01-14T23:00:00+00:00", "2025-01-15T23:00:00+00:00"
EARLY = "2024-06-01T08:00:00+00:00"


def row(notice, version, status, unit, published, start, end, max_mw, available_mw, type_=PLANNED, fuel=NUCLEAR):
    return {"perimetre_juridique": "EDF SA + Luminus", "perimetre_spatial": "France", "status": status, "identifiant": notice,
            "numero_de_version": version, "nom": unit, "filiere": fuel, "date_de_publication": published,
            "date_de_debut": start, "date_de_fin": end, "type": type_, "cause": "x", "information_complementaire": "",
            "puissance_maximale_mw": max_mw, "puissance_disponible_mw": available_mw}


def pmax_rows(units=("U1", "U2", "U3", "U4"), pmax=1000):
    """An old, finished notice per unit so every unit's Pmax is known long before tau."""
    return [row(f"P_{u}", 1, ACTIVE, u, EARLY, "2024-06-02T00:00:00+00:00", "2024-06-03T00:00:00+00:00", pmax, 0) for u in units]


def load(tmp_path, rows, name="r.csv"):
    path = tmp_path / name
    frame = pd.DataFrame(rows)
    path.write_text(HEADER + "\n" + frame.to_csv(sep=";", header=False, index=False), encoding="utf-8-sig")
    return load_edf_remit(path)


def register_of(tmp_path, rows):
    merged, _ = nh.merge_exports(load(tmp_path, rows), None)
    return nh.NuclearRegister(merged)


def rolling_register_of(tmp_path, rows):
    """Second-precision stamps (the rolling export), so a version is known exactly from its stamp."""
    frame = load(tmp_path, rows)
    merged, _ = nh.merge_exports(frame.iloc[0:0], frame)
    return nh.NuclearRegister(merged)


def hourly_of(reg, day=D, tau=None):
    return reg.fleet_day(day, tau)


# --- construction ------------------------------------------------------------------------------------------------------------

def test_three_versions_one_published_after_tau(tmp_path):
    rows = pmax_rows(("U1",)) + [
        row("A", 1, INACTIVE, "U1", "2025-01-10T08:00:00+00:00", DAY_START, DAY_END, 1000, 0),
        row("A", 2, INACTIVE, "U1", "2025-01-13T08:00:00+00:00", DAY_START, DAY_END, 1000, 400),
        row("A", 3, ACTIVE, "U1", "2025-01-14T12:00:00+00:00", DAY_START, DAY_END, 1000, 900),  # after tau
    ]
    reg = rolling_register_of(tmp_path, rows)
    assert (hourly_of(reg) == 400).all() and len(hourly_of(reg)) == 24
    assert (hourly_of(reg, tau=pd.Timestamp("2025-01-14T13:00Z")) == 900).all()
    assert (hourly_of(reg, tau=pd.Timestamp("2025-01-11T00:00Z")) == 0).all()
    assert (hourly_of(reg, tau=pd.Timestamp("2025-01-01T00:00Z")) == 1000).all()  # the notice does not exist yet


@pytest.mark.parametrize("status", [CANCELLED, DELETED])
def test_cancellation_or_deletion_before_and_after_tau(tmp_path, status):
    rows = pmax_rows(("U1", "U2")) + [
        row("E", 1, INACTIVE, "U1", "2025-01-05T08:00:00+00:00", DAY_START, DAY_END, 1000, 0),
        row("E", 2, status, "U1", "2025-01-09T08:00:00+00:00", DAY_START, DAY_END, 1000, 0),  # before tau: nothing
        row("F", 1, INACTIVE, "U2", "2025-01-05T08:00:00+00:00", DAY_START, DAY_END, 1000, 200, type_=FORCED),
        row("F", 2, status, "U2", "2025-01-14T10:00:00+00:00", DAY_START, DAY_END, 1000, 0, type_=FORCED),  # after tau: v1 applies
    ]
    reg = rolling_register_of(tmp_path, rows)
    assert (hourly_of(reg) == 1000 + 200).all()
    counts = reg.counts.summary()
    assert counts[{CANCELLED: "cancelled_notices_applied", DELETED: "deleted_notices_applied"}[status]] == 1
    assert (hourly_of(reg, tau=pd.Timestamp("2025-01-14T11:00Z")) == 2000).all()


def test_overlapping_notices_on_one_unit_take_the_minimum_not_a_sum(tmp_path):
    rows = pmax_rows(("U1",)) + [
        row("A", 1, ACTIVE, "U1", "2025-01-10T08:00:00+00:00", DAY_START, DAY_END, 1000, 600),
        row("B", 1, ACTIVE, "U1", "2025-01-10T09:00:00+00:00", "2025-01-15T09:00:00+00:00", "2025-01-15T12:00:00+00:00", 1000, 300),
        row("C", 1, ACTIVE, "U1", "2025-01-10T09:00:00+00:00", "2025-01-15T09:00:00+00:00", "2025-01-15T10:00:00+00:00", 1000, 700),
    ]
    h = hourly_of(rolling_register_of(tmp_path, rows))
    assert h[pd.Timestamp("2025-01-15T08:00Z")] == 600
    assert h[pd.Timestamp("2025-01-15T09:00Z")] == 300 and h[pd.Timestamp("2025-01-15T11:00Z")] == 300
    assert h[pd.Timestamp("2025-01-15T12:00Z")] == 600


def test_chronicles_never_reduce_availability(tmp_path):
    rows = pmax_rows(("U1",)) + [
        row(f"K{i}", 1, ACTIVE, "U1", "2025-01-10T08:00:00+00:00", DAY_START, DAY_END, 0, 0, type_=t)
        for i, t in enumerate([PMIN_CHRONICLE, AFRR_CHRONICLE, "Chronique", "Chronique."])
    ]
    reg = rolling_register_of(tmp_path, rows)
    assert (hourly_of(reg) == 1000).all()
    assert reg.chronicles_excluded == 4
    # a chronicle's Pmax of 0 is not a Pmax either
    assert reg.pmax(0, TAU.value) == (1000.0, True)


def test_quarter_hour_overlap_at_the_edges_of_the_hour(tmp_path):
    rows = pmax_rows(("U1", "U2", "U3")) + [
        row("A", 1, ACTIVE, "U1", "2025-01-10T08:00:00+00:00", "2025-01-15T10:10:00+00:00", "2025-01-15T10:50:00+00:00", 1000, 0),
        row("B", 1, ACTIVE, "U2", "2025-01-10T08:00:00+00:00", "2025-01-15T12:15:00+00:00", "2025-01-15T12:30:00+00:00", 1000, 0),
        row("C", 1, ACTIVE, "U3", "2025-01-10T08:00:00+00:00", "2025-01-15T13:00:00+00:00", "2025-01-15T14:00:00+00:00", 1000, 0),
    ]
    h = hourly_of(rolling_register_of(tmp_path, rows))
    assert h[pd.Timestamp("2025-01-15T10:00Z")] == 2000  # A touches all four quarter-hours of 10:00 UTC
    assert h[pd.Timestamp("2025-01-15T09:00Z")] == 3000 and h[pd.Timestamp("2025-01-15T11:00Z")] == 3000
    assert h[pd.Timestamp("2025-01-15T12:00Z")] == 3000 - 1000 / 4  # B covers one quarter-hour
    assert h[pd.Timestamp("2025-01-15T13:00Z")] == 2000 and h[pd.Timestamp("2025-01-15T14:00Z")] == 3000  # end is exclusive


@pytest.mark.parametrize("day,hours,start,end", [("2025-03-30", 23, "2025-03-29T23:00:00+00:00", "2025-03-30T22:00:00+00:00"),
                                                  ("2025-10-26", 25, "2025-10-25T22:00:00+00:00", "2025-10-26T23:00:00+00:00")])
def test_clock_change_days(tmp_path, day, hours, start, end):
    rows = pmax_rows(("U1", "U2")) + [
        row("A", 1, ACTIVE, "U1", "2025-01-10T08:00:00+00:00", start, end, 1000, 250),
        # U2 out for the first two UTC hours of the day only
        row("B", 1, ACTIVE, "U2", "2025-01-10T08:00:00+00:00", start, (pd.Timestamp(start) + pd.Timedelta(hours=2)).isoformat(), 1000, 0),
    ]
    reg = rolling_register_of(tmp_path, rows)
    assert len(nh.day_quarter_hours(day)) == hours * 4
    h = reg.fleet_day(pd.Timestamp(day))
    assert len(h) == hours and h.index[0] == pd.Timestamp(start) and h.index[-1] == pd.Timestamp(end) - pd.Timedelta(hours=1)
    assert (h.iloc[:2] == 250).all() and (h.iloc[2:] == 1250).all()
    _, daily = nh.nuclear_daily_table(reg, [pd.Timestamp(day)])
    assert daily["nuc_avail_mean_mw"].iloc[0] == pytest.approx((2 * 250 + (hours - 2) * 1250) / hours)


def test_fessenheim_excluded_and_flamanville_3_from_grid_connection(tmp_path):
    rows = [row("P1", 1, ACTIVE, "FESSENHEIM 1", EARLY, "2024-06-02T00:00:00+00:00", "2024-06-03T00:00:00+00:00", 880, 0),
            row("P3", 1, ACTIVE, "FLAMANVILLE 3", EARLY, "2024-06-02T00:00:00+00:00", "2024-06-03T00:00:00+00:00", 1620, 0)]
    reg = rolling_register_of(tmp_path, rows)
    assert "FESSENHEIM 1" not in reg.units and "FLAMANVILLE 3" in reg.units
    assert nh.FLAMANVILLE_3_GRID == pd.Timestamp("2024-12-21T10:48Z")
    h = reg.fleet_day(pd.Timestamp("2024-12-21"))
    assert h[pd.Timestamp("2024-12-21T09:00Z")] == 0 and h[pd.Timestamp("2024-12-21T10:00Z")] == 0  # 10:45 starts before 10:48
    assert h[pd.Timestamp("2024-12-21T11:00Z")] == 1620
    assert (reg.fleet_day(pd.Timestamp("2024-12-20")) == 0).all() and (reg.fleet_day(pd.Timestamp("2024-12-22")) == 1620).all()


def test_running_maximum_publication_time(tmp_path):
    rows = pmax_rows(("U1",)) + [
        row("A", 1, ACTIVE, "U1", "2025-01-14T08:00:00+00:00", DAY_START, DAY_END, 1000, 100),
        row("A", 2, ACTIVE, "U1", "2025-01-12T08:00:00+00:00", DAY_START, DAY_END, 1000, 500),  # stamped before v1
        row("A", 3, ACTIVE, "U1", "2025-01-14T12:00:00+00:00", DAY_START, DAY_END, 1000, 800),
    ]
    reg = rolling_register_of(tmp_path, rows)
    assert (hourly_of(reg, tau=pd.Timestamp("2025-01-13T00:00Z")) == 1000).all()  # v2 not known before v1 is
    assert (hourly_of(reg) == 500).all()  # at tau both are known: the higher version
    assert reg.frame.loc[reg.frame["notice"] == "A", "effective"].tolist() == [pd.Timestamp(x) for x in
                                                                                ("2025-01-14T08:00Z", "2025-01-14T08:00Z", "2025-01-14T12:00Z")]


def test_history_minute_stamps_count_from_the_end_of_their_minute(tmp_path):
    stamp = TAU.isoformat()  # 09:05:00 UTC, the minute of tau
    rows = pmax_rows(("U1",)) + [row("A", 1, ACTIVE, "U1", stamp, DAY_START, DAY_END, 1000, 0)]
    history = load(tmp_path, rows, "h.csv")
    merged, _ = nh.merge_exports(history, None)
    assert (nh.NuclearRegister(merged).fleet_day(D) == 1000).all()  # may have been published up to 59 s after tau
    rolling = load(tmp_path, rows, "r.csv")
    merged, facts = nh.merge_exports(history, rolling)
    assert facts["common_versions"] == 2
    assert (nh.NuclearRegister(merged).fleet_day(D) == 0).all()  # the exact stamp is at tau: known


def test_pmax_fallback_and_missing_available_are_counted(tmp_path):
    rows = [row("A", 1, ACTIVE, "U1", "2025-02-01T08:00:00+00:00", DAY_START, DAY_END, 1300, 0),  # published after tau
            row("B", 1, ACTIVE, "U2", EARLY, "2024-06-02T00:00:00+00:00", "2024-06-03T00:00:00+00:00", 900, 0),
            row("C", 1, ACTIVE, "U2", "2025-01-10T08:00:00+00:00", DAY_START, DAY_END, 900, np.nan)]
    reg = rolling_register_of(tmp_path, rows)
    assert (hourly_of(reg) == 1300 + 900).all()
    c = reg.counts.summary()
    assert c["unit_calls_without_known_version"] == 1 and c["notices_skipped_missing_available"] == 1


def test_change_feature_uses_d_minus_1_as_known_at_its_auction(tmp_path):
    d1_start, d1_end = "2025-01-13T23:00:00+00:00", "2025-01-14T23:00:00+00:00"
    rows = pmax_rows(("U1",)) + [
        row("A", 1, ACTIVE, "U1", "2025-01-12T10:00:00+00:00", d1_start, d1_end, 1000, 400),  # known at 12:00 on D-2
        row("A", 2, ACTIVE, "U1", "2025-01-13T12:00:00+00:00", d1_start, d1_end, 1000, 0),  # after the D-1 auction
        row("B", 1, ACTIVE, "U1", "2025-01-14T08:00:00+00:00", DAY_START, DAY_END, 1000, 900),
    ]
    reg = rolling_register_of(tmp_path, rows)
    assert nh.reference_tau(D) == pd.Timestamp("2025-01-13T11:00Z")
    _, daily = nh.nuclear_daily_table(reg, [D])
    assert daily.at[D, "nuc_avail_prev_mean_mw"] == 400 and daily.at[D, "nuc_avail_chg_mw"] == 500


def test_export_agreement_and_merge(tmp_path):
    rows = pmax_rows(("U1",)) + [row("A", 1, ACTIVE, "U1", "2025-01-10T08:00:37+00:00", DAY_START, DAY_END, 1000, 0)]
    rolling = load(tmp_path, rows + [row("A", 2, ACTIVE, "U1", "2025-01-20T08:00:00+00:00", DAY_START, DAY_END, 1000, 10)], "r.csv")
    hist_rows = [dict(r, date_de_publication=pd.Timestamp(r["date_de_publication"]).floor("min").isoformat()) for r in rows]
    history = load(tmp_path, hist_rows + hist_rows[-1:], "h.csv")  # a duplicated key
    agree = nh.export_agreement(history, rolling)
    assert agree["published_history_equals_rolling_floored_to_minute"] and agree["mismatch_available_mw"] == 0
    merged, facts = nh.merge_exports(history, rolling)
    assert facts["history_duplicate_keys"] == 1 and facts["rolling_only_versions"] == 1
    a1 = merged[(merged["notice"] == "A") & (merged["version"] == 1)]
    assert a1["published"].iloc[0] == pd.Timestamp("2025-01-10T08:00:37Z") and a1["source"].iloc[0] == "both"


# --- hydro ---------------------------------------------------------------------------------------------------------------------

def test_a72_third_working_day_over_a_french_holiday():
    # week ending Sunday 2025-04-20; Monday 21 April 2025 is Easter Monday
    assert nh.a72_known_from(pd.Timestamp("2025-04-20")) == pd.Timestamp("2025-04-24 23:59").tz_localize("Europe/Paris").tz_convert("UTC")
    # an ordinary week: Wednesday
    assert nh.a72_known_from(pd.Timestamp("2026-10-11")) == pd.Timestamp("2026-10-14 23:59").tz_localize("Europe/Paris").tz_convert("UTC")
    weeks = pd.date_range("2025-02-02", "2025-05-04", freq="W-SUN")
    a72 = pd.DataFrame({"zone": "10YFR-RTE------C", "week_start": weeks - pd.Timedelta(days=6), "week_end": weeks,
                        "stored_mwh": np.arange(len(weeks), dtype=float) * 100, "repeated": False, "created": pd.Timestamp("2026-10-16", tz="UTC")})
    table = nh.a72_table(a72)
    on_thursday = nh.a72_asof(table, nh.issue_tau(pd.Timestamp("2025-04-25")))  # 10:05 on Thursday 24 April: not yet due
    assert on_thursday["a72_week"] == pd.Timestamp("2025-04-13")
    on_friday = nh.a72_asof(table, nh.issue_tau(pd.Timestamp("2025-04-26")))
    assert on_friday["a72_week"] == pd.Timestamp("2025-04-20")
    w = on_friday["a72_week"]
    assert on_friday["fr_hydro_chg4w_mwh"] == table.at[w, "stored_mwh"] - table.at[w - pd.Timedelta(weeks=4), "stored_mwh"]


def test_a72_tolerance_of_three_weeks():
    weeks = pd.date_range("2025-01-05", "2025-04-27", freq="W-SUN")
    values = np.arange(len(weeks), dtype=float)
    a72 = pd.DataFrame({"zone": "10YFR-RTE------C", "week_start": weeks - pd.Timedelta(days=6), "week_end": weeks, "stored_mwh": values,
                        "repeated": False, "created": pd.Timestamp("2026-10-16", tz="UTC")})
    table = nh.a72_table(a72[a72["week_end"] <= pd.Timestamp("2025-03-30")])  # nothing after 30 March
    tau = nh.issue_tau(pd.Timestamp("2025-04-18"))  # latest due week ends 2025-04-13: two weeks after the last
    assert nh.a72_asof(table, tau)["a72_week"] == pd.Timestamp("2025-03-30")
    late = nh.a72_asof(table, nh.issue_tau(pd.Timestamp("2025-05-02")))  # due week 2025-04-27: four weeks after
    assert np.isnan(late["fr_hydro_fill_mwh"]) and late["a72_age_weeks"] == 4


def test_bfe_25_day_rule_and_climatology():
    dates = pd.date_range("2000-01-03", "2026-09-28", freq="W-MON")
    rng = np.random.default_rng(0)
    bfe = pd.DataFrame({"date": dates, "totalch_speicherinhalt_gwh": rng.uniform(1000, 8000, len(dates)), "totalch_max_speicherinhalt_gwh": 8800.0})
    table = nh.bfe_table(bfe)
    row = pd.Timestamp("2026-09-28")
    assert table.at[row, "known_from"] == pd.Timestamp("2026-10-23").tz_localize("Europe/Paris").tz_convert("UTC")
    assert nh.bfe_asof(table, nh.issue_tau(pd.Timestamp("2026-10-23")))["bfe_date"] == pd.Timestamp("2026-09-21")
    assert nh.bfe_asof(table, nh.issue_tau(pd.Timestamp("2026-10-24")))["bfe_date"] == row
    t = table.loc[row]
    same = table[(table["iso_week"] == t["iso_week"]) & (table["iso_year"] < t["iso_year"])]
    assert t["ch_hydro_anom_pct"] == pytest.approx(t["share_pct"] - same["share_pct"].mean()) and len(same) == 26


# --- perturbation: anything published after tau leaves D unchanged ------------------------------------------------------------

def perturbable_rows():
    rows = pmax_rows(("U1", "U2", "U3", "U4"))
    rng = np.random.default_rng(3)
    for i, start in enumerate(pd.date_range("2025-01-01", "2025-03-01", freq="3D", tz="UTC")):
        pub = start - pd.Timedelta(days=int(rng.integers(1, 9)), hours=int(rng.integers(0, 24)))
        unit = f"U{1 + i % 4}"
        end = start + pd.Timedelta(hours=int(rng.integers(5, 200)))
        rows.append(row(f"N{i}", 1, INACTIVE, unit, pub.isoformat(), start.isoformat(), end.isoformat(), 1000, float(rng.integers(0, 900))))
        rows.append(row(f"N{i}", 2, ACTIVE, unit, (pub + pd.Timedelta(hours=30)).isoformat(), start.isoformat(),
                        (end + pd.Timedelta(hours=12)).isoformat(), 1000, float(rng.integers(0, 900)), type_=FORCED))
    return rows


def test_versions_published_after_tau_change_nothing_for_d(tmp_path):
    days = pd.date_range("2025-01-20", "2025-02-20", freq="D")
    rows = perturbable_rows()
    base_h, base_d = nh.nuclear_daily_table(rolling_register_of(tmp_path, rows), days)
    rng = np.random.default_rng(9)
    for D_ in days[::5]:
        tau = nh.issue_tau(D_)
        extra = []
        for k in range(30):  # new notices, new versions of existing notices and cancellations, all published after tau
            pub = tau + pd.Timedelta(minutes=int(rng.integers(1, 5000)))
            notice = f"N{int(rng.integers(0, 20))}" if k % 2 else f"X{k}"
            version = 3 + k if notice.startswith("N") else 1
            status = CANCELLED if k % 7 == 0 else ACTIVE
            extra.append(row(notice, version, status, f"U{1 + k % 4}", pub.isoformat(), (D_ - pd.Timedelta(days=1)).isoformat() + "+00:00",
                             (D_ + pd.Timedelta(days=2)).isoformat() + "+00:00", 1000 + k, 0))
        h, d = nh.nuclear_daily_table(rolling_register_of(tmp_path, rows + extra), [D_])
        pd.testing.assert_frame_equal(h, base_h.loc[h.index])
        pd.testing.assert_frame_equal(d, base_d.loc[[D_]])
        # an alteration of a version published after tau
        altered = [dict(r, puissance_disponible_mw=5.0) if pd.Timestamp(r["date_de_publication"]) > tau else r for r in rows]
        h2, d2 = nh.nuclear_daily_table(rolling_register_of(tmp_path, altered), [D_])
        pd.testing.assert_frame_equal(h2, base_h.loc[h2.index])
        pd.testing.assert_frame_equal(d2, base_d.loc[[D_]])


def test_hydro_rows_not_yet_known_change_nothing_for_d():
    weeks = pd.date_range("2020-12-27", "2025-06-29", freq="W-SUN")
    a72 = pd.DataFrame({"zone": "10YFR-RTE------C", "week_start": weeks - pd.Timedelta(days=6), "week_end": weeks,
                        "stored_mwh": np.linspace(1e6, 3e6, len(weeks)), "repeated": False, "created": pd.Timestamp("2026-10-16", tz="UTC")})
    dates = pd.date_range("2000-01-03", "2025-06-30", freq="W-MON")
    bfe = pd.DataFrame({"date": dates, "totalch_speicherinhalt_gwh": np.linspace(1000, 8000, len(dates)) % 7000 + 500,
                        "totalch_max_speicherinhalt_gwh": 8800.0})
    days = pd.date_range("2025-03-01", "2025-04-30", freq="D")
    base = nh.hydro_daily_table(a72, bfe, days)
    for D_ in days[::7]:
        tau = nh.issue_tau(D_)
        a2 = a72.copy()
        not_due = [nh.a72_known_from(w) > tau for w in a2["week_end"]]
        a2.loc[not_due, "stored_mwh"] *= 7
        b2 = bfe.copy()
        young = (b2["date"] + nh.BFE_LAG).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC") > tau
        b2.loc[young, "totalch_speicherinhalt_gwh"] = 1.0
        again = nh.hydro_daily_table(a2, b2, [D_])
        pd.testing.assert_frame_equal(again, base.loc[[D_]])
        assert base.loc[D_, "a72_known_from"] <= tau and base.loc[D_, "bfe_known_from"] <= tau


@pytest.fixture(scope="module")
def nh_inputs():
    frame = synthetic_inputs("2022-01-01", "2024-05-01", issued_from="2023-06-01")
    local = frame.index.tz_convert("Europe/Paris")
    rng = np.random.default_rng(5)
    nh_frame = pd.DataFrame(index=frame.index)
    day = local.normalize().tz_localize(None)
    daily = pd.Series(rng.normal(0, 1, len(pd.unique(day))), index=pd.unique(day))
    nh_frame["nuc_avail_mw"] = 45000 + 3000 * np.sin(np.arange(len(frame)) / 500) + rng.normal(0, 300, len(frame))
    nh_frame["nuc_avail_chg_mw"] = daily.reindex(day).to_numpy() * 800
    nh_frame["fr_hydro_fill_mwh"] = 2e6 + 1e5 * daily.rolling(7, min_periods=1).mean().reindex(day).to_numpy()
    nh_frame["fr_hydro_chg4w_mwh"] = daily.reindex(day).to_numpy() * 1e4
    nh_frame["ch_hydro_anom_pct"] = daily.reindex(day).to_numpy() * 3
    frame["price_eur_mwh"] += -0.002 * (nh_frame["nuc_avail_mw"] - 45000)
    return frame, nh_frame


def test_every_new_column_has_a_rule_that_passes_both_deadlines(nh_inputs):
    hourly, nh_frame = nh_inputs
    for s in features.NH_SETS:
        timings = feature_timings(s)
        assert set(timings.values()) <= set(timing.TIMINGS)
        table = build_features(hourly, s, nh=nh_frame)
        timing.check_point_in_time(table.X.index, timings, deadline="gate")
        timing.check_point_in_time(table.X.index, timings, deadline="premarket")
    idx = pd.date_range("2025-01-15", periods=24, freq="1h", tz="Europe/Paris").tz_convert("UTC")
    for rule in ("nuclear_avail_asof", "hydro_a72_asof", "hydro_bfe_asof", "residual_avail"):
        assert (issue_times(idx, rule) == TAU).all()
    for v in ("n1", "n2", "n3"):
        cols = features.lear_columns(v)
        assert all(c in features.FEATURE_TIMINGS for c in cols)
        assert len(cols) == {"n1": 327, "n2": 305, "n3": 330}[v] < min(lear.LEAR_WINDOWS)


def test_feature_construction_and_residual(nh_inputs):
    hourly, nh_frame = nh_inputs
    table = build_features(hourly, "honest_v2_n3", nh=nh_frame)
    for c in features.NH_INPUTS:
        np.testing.assert_array_equal(table.X[c].to_numpy(), nh_frame[c].to_numpy())
    expect = table.X["load_fc_mw"] - table.X["wind_proxy_mw"] - table.X["solar_proxy_mw"] - nh_frame["nuc_avail_mw"]
    np.testing.assert_allclose(table.X["residual_avail_mw"].to_numpy(), expect.to_numpy())
    base = build_features(hourly, "honest_v2")
    pd.testing.assert_frame_equal(table.X[base.X.columns], base.X)
    with pytest.raises(ValueError, match="nuclear-hydro"):
        build_features(hourly, "honest_v2_n1")


def test_lear_design_blocks(nh_inputs):
    hourly, nh_frame = nh_inputs
    X0, Y0, b0 = lear.lear_design(hourly)
    X3, Y3, b3 = lear.lear_design(hourly, nh_frame, "n3")
    pd.testing.assert_frame_equal(X3[X0.columns], X0)
    assert list(X0.columns) == features.LEAR_COLUMNS and X3.shape[1] == 330
    Dd = pd.Timestamp("2023-11-15")
    local = hourly.index.tz_convert("Europe/Paris")
    on = (local.normalize().tz_localize(None) == Dd)
    for h in (0, 9, 23):
        assert X3.at[Dd, f"nuc_avail_d0_h{h:02d}"] == pytest.approx(nh_frame.loc[on & (local.hour == h), "nuc_avail_mw"].mean())
    assert X3.at[Dd, "nuc_avail_chg"] == nh_frame.loc[on, "nuc_avail_chg_mw"].iloc[0]
    assert X3.at[Dd, "ch_hydro_anom"] == nh_frame.loc[on, "ch_hydro_anom_pct"].iloc[0]
    autumn = pd.Timestamp("2023-10-29")
    two = (local.normalize().tz_localize(None) == autumn) & (local.hour == 2)
    assert two.sum() == 2 and X3.at[autumn, "nuc_avail_d0_h02"] == pytest.approx(nh_frame.loc[two, "nuc_avail_mw"].mean())


def test_forecast_for_d_unchanged_when_later_nh_values_change(nh_inputs):
    """The nh values of later days (built from versions published after tau_D) change neither LEAR nor the GBM member for D."""
    hourly, nh_frame = nh_inputs
    Dd = pd.Timestamp("2024-03-12")
    day = hourly.index.tz_convert("Europe/Paris").normalize().tz_localize(None)
    changed = nh_frame.copy()
    changed.loc[day > Dd] *= 1.7
    span = (Dd.strftime("%Y-%m-%d"), (Dd + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    sink_a, sink_b = [], []
    a = lear.lear_walk_forward(hourly, *span, windows=(364,), nh=nh_frame, variant="n3", coef_sink=sink_a)
    b = lear.lear_walk_forward(hourly, *span, windows=(364,), nh=changed, variant="n3", coef_sink=sink_b)
    np.testing.assert_allclose(a["lear"].to_numpy(), b["lear"].to_numpy(), rtol=0, atol=1e-9)
    assert sink_a == sink_b and len(sink_a) == 24 and {"nuc_own_hour", "nuc_block_sum", "ch_hydro_anom"} <= set(sink_a[0])
    moved = nh_frame.copy()
    moved.loc[day == Dd, "nuc_avail_mw"] += 20000  # D's own availability does matter
    c = lear.lear_walk_forward(hourly, *span, windows=(364,), nh=moved, variant="n1")
    d = lear.lear_walk_forward(hourly, *span, windows=(364,), nh=nh_frame, variant="n1")
    assert a["lear"].notna().all() and not np.allclose(c["lear"].to_numpy(), d["lear"].to_numpy())
    models_n = ("gbm",)
    t1 = build_features(hourly, "honest_v2_n3", nh=nh_frame)
    t2 = build_features(hourly, "honest_v2_n3", nh=changed)
    g1 = walk_forward(t1, "2024-03-01", "2024-04-01", models=models_n, log=lambda *_: None).predictions
    g2 = walk_forward(t2, "2024-03-01", "2024-04-01", models=models_n, log=lambda *_: None).predictions
    on = g1["delivery_day"] <= Dd
    np.testing.assert_array_equal(g1.loc[on, "gbm"].to_numpy(), g2.loc[on, "gbm"].to_numpy())


# --- hard gates --------------------------------------------------------------------------------------------------------------

def test_a_late_rule_stops_the_gbm_stage_and_lear(nh_inputs, monkeypatch):
    hourly, nh_frame = nh_inputs
    monkeypatch.setitem(features.FEATURE_TIMINGS, "nuc_avail_mw", "wind_solar_forecast")  # 18:00 on D-1
    with pytest.raises(LookaheadError):
        nh_sprint.gbm_stage(hourly, nh_frame, "2024-03-01", "2024-04-01", sets=["N1"])
    monkeypatch.setitem(lear.BLOCK_TIMINGS, "nuc_avail_chg", "wind_solar_forecast")
    monkeypatch.setitem(features.FEATURE_TIMINGS, "nuc_avail_chg", "wind_solar_forecast")
    with pytest.raises(LookaheadError):
        lear.lear_walk_forward(hourly, "2024-03-12", "2024-03-13", windows=(364,), nh=nh_frame, variant="n1")


def test_hydro_value_known_after_tau_stops_the_build(monkeypatch):
    weeks = pd.date_range("2024-12-29", "2025-06-29", freq="W-SUN")
    a72 = pd.DataFrame({"zone": "10YFR-RTE------C", "week_start": weeks - pd.Timedelta(days=6), "week_end": weeks, "stored_mwh": 1.0,
                        "repeated": False, "created": pd.Timestamp("2026-10-16", tz="UTC")})
    bfe = pd.DataFrame({"date": pd.date_range("2023-01-02", "2025-06-30", freq="W-MON"), "totalch_speicherinhalt_gwh": 10.0,
                        "totalch_max_speicherinhalt_gwh": 20.0})
    real = nh.a72_table
    monkeypatch.setattr(nh, "a72_table", lambda frame: real(frame).assign(known_from=lambda t: t["known_from"] - pd.Timedelta(days=10)))
    monkeypatch.setattr(nh, "a72_asof", lambda table, tau: {"fr_hydro_fill_mwh": 1.0, "fr_hydro_chg4w_mwh": 0.0, "a72_week": weeks[-1],
                                                            "a72_due_week": weeks[-1], "a72_known_from": tau + pd.Timedelta(hours=1), "a72_age_weeks": 0})
    with pytest.raises(AssertionError, match="after tau"):
        nh.hydro_daily_table(a72, bfe, [pd.Timestamp("2025-03-03")])


def test_holdout_refused_and_rule_on_a_synthetic_frame(nh_inputs, tmp_path, monkeypatch):
    hourly, nh_frame = nh_inputs
    with pytest.raises(ValueError, match="holdout"):
        nh_sprint.gbm_stage(hourly, nh_frame, "2026-06-01", "2026-08-01")
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=24 * 90, freq="1h", tz="UTC")
    day = pd.DatetimeIndex(idx.tz_convert("Europe/Paris").normalize().tz_localize(None))
    actual = 60 + rng.normal(0, 20, len(idx))
    base = pd.DataFrame({"delivery_day": day, "hour": idx.hour, "dow": day.dayofweek, "holiday": 0, "is_peak": 0, "actual": actual,
                         "strict": True, "wind_mw": rng.uniform(0, 1, len(idx)), "solar_mw": rng.uniform(0, 1, len(idx)),
                         "naive_day": actual, "naive_week": actual}, index=idx)
    gbm = {v: base.assign(gbm=actual + rng.normal(0, s, len(idx))) for v, s in (("S2", 12), ("N1", 6), ("N2", 12), ("N3", 12.5))}
    lear_ = {v: base.assign(lear=actual + rng.normal(0, s, len(idx))) for v, s in (("S2", 12), ("N1", 6), ("N2", 12), ("N3", 12.5))}
    frame = nh_sprint.assemble(gbm, lear_)
    hourly_p = pd.DataFrame({"price_eur_mwh": actual}, index=idx)
    nuc_daily = pd.DataFrame({"nuc_avail_chg_mw": rng.normal(0, 500, day.nunique())}, index=pd.unique(day))
    monkeypatch.setattr(nh_sprint, "REFERENCE_MAE", 0.0)
    report = nh_sprint.selection_report(frame, hourly_p, nuc_daily)
    assert report["candidates"]["N1"]["passes"] and not report["candidates"]["N3"]["passes"] and report["to_shadow"] == "N1"
    assert report["diagnostics"]["N1"]["paired_vs_S2"]["upper_one_sided_98_33"] < 0
    json.dumps(report, default=str)


def test_nh_cli_stages_resume_and_report(nh_inputs, tmp_path, monkeypatch):
    hourly, nh_frame = nh_inputs
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 20)
    data = tmp_path / "inputs.csv"
    hourly.to_csv(data, index_label="timestamp_utc")
    nhdir = tmp_path / "nh"
    nhdir.mkdir()
    nh_frame.to_csv(nhdir / "nh_hourly.csv", index_label="timestamp_utc")
    days = pd.unique(hourly.index.tz_convert("Europe/Paris").normalize().tz_localize(None))
    nuc = nh_frame.groupby(hourly.index.tz_convert("Europe/Paris").normalize().tz_localize(None))["nuc_avail_chg_mw"].first()
    nuc.rename_axis("delivery_day").to_frame().to_csv(nhdir / "nuclear_daily.csv")
    pd.DataFrame(index=pd.DatetimeIndex(days, name="delivery_day")).assign(x=1).to_csv(nhdir / "hydro_daily.csv")
    out = tmp_path / "run"
    common = ["--data", str(data), "--nh", str(nhdir), "--out", str(out), "--test-start", "2024-03-01", "--test-end", "2024-03-08"]
    real = lear.lear_walk_forward
    monkeypatch.setattr(lear, "lear_walk_forward", lambda *a, **k: real(*a, **{**k, "windows": (364,)}))
    cli.main(["nh-stage", "gbm", *common])
    for v in ("S2", "N1", "N2", "N3"):
        cli.main(["nh-stage", "lear", "--variant", v, *common])
    assert (out / "coef_N3_2024-03.csv").exists() and not (out / "coef_S2_2024-03.csv").exists()
    stamp = (out / "lear_N1_2024-03.csv").stat().st_mtime
    cli.main(["nh-stage", "lear", "--variant", "N1", *common])
    assert (out / "lear_N1_2024-03.csv").stat().st_mtime == stamp
    cli.main(["nh-report", *common])
    report = json.loads((out / "nh_report.json").read_text())
    assert set(report["candidates"]) == {"N1", "N2", "N3"} and "split_gain_shares" in report and "lear_coefficients" in report
