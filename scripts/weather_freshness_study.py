"""Fresher weather for the German wind proxy: the registered comparison on development days
(docs/prereg/weather_freshness_prereg.md). Nothing frozen is edited: the German wind table and the wind curve are replaced in
memory, and the frozen spread code does the rest.

    python weather_freshness_study.py tables                 build the two GFS tables, coverage, and the check that neither reads ahead
    python weather_freshness_study.py fit <arm> <curve>      arm: B (the stored table), S (stale GFS), F (fresh GFS); curve: built, fleet
    python weather_freshness_study.py summary                stop conditions, then the figures

Arms. S: each delivery hour from the latest GFS run started at least 48 hours before it (leads 48 to 53), which is the
project's present convention applied to GFS. F: every hour of delivery day D from the 00 UTC run of D-1 (leads 22 to 47).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thermo_fr.forecast import spread_eval as se, spread_w as sw  # noqa: E402
from thermo_fr.forecast.timing import delivery_days  # noqa: E402
from thermo_fr.forecast.wind_proxy import power_curve  # noqa: E402

WORK = Path("/home/claude/work/weather")
STATIC = Path("/home/claude/work/spread-live-root/static")
PACKED = WORK / "gfs_de_wind100m.csv.gz"
STORED = Path("/home/claude/work/spread-w/w_predictions.csv.gz")
FIRST_HOUR, FLEET_SD = "2024-01-01 00:00", 3.5
ARMS, CURVES = ("B", "S", "F"), ("built", "fleet")
VARIANTS = [("B", "built"), ("S", "built"), ("F", "built"), ("S", "fleet"), ("F", "fleet")]
MIN_GAIN = 0.25


def fleet_curve(v):
    """The wind curve averaged over a normal spread of speeds of FLEET_SD m/s around the point's value."""
    z = np.linspace(-2.5, 2.5, 11)
    w = np.exp(-z ** 2 / 2)
    w /= w.sum()
    return sum(wi * power_curve(np.clip(np.asarray(v, float) + FLEET_SD * zi, 0, None)) for wi, zi in zip(w, z))


def cube(packed=PACKED):
    """The fetched GFS history as an array [run, lead, point], with the runs' start times and the point names."""
    t = pd.read_csv(packed)
    t = t[t["init_utc"] >= "2023-01-01"]
    runs = pd.DatetimeIndex(sorted(pd.to_datetime(t["init_utc"].unique(), utc=True)))
    stored = pd.read_csv(STATIC / "german_wind_points.csv.gz", nrows=1).columns[1:]
    points = ["de_" + c[len("de_wind_pt_"):-len("_ms")] for c in stored]  # the stored table's own order
    assert set(points) == set(t["point"].unique()), "the fetched points must be the stored table's points"
    a = np.full((len(runs), 55, len(points)), np.nan)
    a[runs.get_indexer(pd.to_datetime(t["init_utc"], utc=True)), t["lead_h"].to_numpy(), pd.Index(points).get_indexer(t["point"])] = t["speed_ms"].to_numpy()
    return a, runs, points


def tables(a, runs, points) -> dict:
    """The two hourly tables of 100 m wind, labelled as the stored German table is."""
    last = (pd.Timestamp(se.DEV_LAST) + pd.Timedelta(hours=23)).tz_localize("Europe/Paris").tz_convert("UTC")  # the last hour of the last delivery day
    hours = pd.date_range(pd.Timestamp(FIRST_HOUR, tz="UTC"), last, freq="h")
    columns = [f"de_wind_pt_{p.split('_', 1)[1]}_ms" for p in points]
    stale_run = (hours - pd.Timedelta(hours=48)).floor("6h")
    fresh_run = (pd.DatetimeIndex(delivery_days(hours)) - pd.Timedelta(days=1)).tz_localize("UTC")
    out = {}
    for name, run in (("S", stale_run), ("F", fresh_run)):
        lead = ((hours - run) / pd.Timedelta(hours=1)).astype(int)
        idx = runs.get_indexer(run)
        ok = (idx >= 0) & (lead >= 0) & (lead <= 54)
        values = np.full((len(hours), len(points)), np.nan)
        values[ok] = a[idx[ok], lead[ok]]
        out[name] = pd.DataFrame(values, index=hours, columns=columns)
        out[name + "_lead"] = pd.Series(lead, index=hours)
        out[name + "_run"] = pd.Series(run, index=hours)
    return out


def stage_tables() -> dict:
    a, runs, points = cube()
    t = tables(a, runs, points)
    report = {"runs": len(runs), "first_run": str(runs[0]), "last_run": str(runs[-1]), "points": points, "packed_sha256": se.sha256(PACKED)}
    for arm in ("S", "F"):
        table, lead, run = t[arm], t[arm + "_lead"], t[arm + "_run"]
        report[arm] = {"hours": len(table), "hours_with_a_missing_point": int(table.isna().any(axis=1).sum()), "lead_min": int(lead.min()), "lead_max": int(lead.max()),
                       "mean_lead": round(float(lead.mean()), 2)}
        # no value of a delivery day comes from a run started after 10:05 Paris minus nothing: the latest run used for day D
        days = np.asarray(delivery_days(table.index))
        latest = run.groupby(days).max()
        gate = (pd.DatetimeIndex(latest.index) - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=5)).tz_localize("Europe/Paris").tz_convert("UTC")
        report[arm]["hours_from_the_newest_run_used_to_the_gate_min"] = round(float(((gate - pd.DatetimeIndex(latest.to_numpy())) / pd.Timedelta(hours=1)).min()), 2)
        table.reset_index(names="timestamp_utc").to_csv(WORK / f"table_{arm}.csv.gz", index=False)
    # scramble: every run started after a cut is replaced by noise; the tables up to the delivery day issued at that cut must not move
    rng = np.random.default_rng(0)
    checks = []
    for day in ("2024-03-31", "2025-07-15", "2025-10-26", "2026-06-30"):
        cut_run = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1)  # 00 UTC of D-1, the newest run arm F may use for D
        b = a.copy()
        later = runs > cut_run
        b[later] = rng.normal(50, 20, size=b[later].shape)
        u = tables(b, runs, points)
        keep = np.asarray(delivery_days(t["F"].index) <= pd.Timestamp(day))
        checks.append({"day": day, "runs_scrambled": int(later.sum()), **{arm: bool(np.array_equal(t[arm].to_numpy()[keep], u[arm].to_numpy()[keep], equal_nan=True)) for arm in ("S", "F")},
                       "rows_after_moved": bool(not np.array_equal(t["F"].to_numpy()[~keep], u["F"].to_numpy()[~keep], equal_nan=True)) if (~keep).any() else None})
    report["scramble"] = checks
    report["scramble_pass"] = all(c["S"] and c["F"] for c in checks) and all(c["rows_after_moved"] in (True, None) for c in checks)
    stored = se._hourly(STATIC / "german_wind_points.csv.gz").asfreq("h").reindex(t["S"].index)
    report["correlation_with_the_stored_table_by_point"] = {arm: round(float(np.mean([np.corrcoef(stored[c].dropna(), t[arm][c].reindex(stored[c].dropna().index))[0, 1] for c in t[arm].columns])), 3) for arm in ("S", "F")}
    (WORK / "tables_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


def stage_fit(arm: str, curve: str) -> None:
    started = time.time()
    data = se.load_stored(STATIC / "inputs.csv", STATIC / "neighbour_fundamentals.csv", STATIC / "neighbour_sequences.csv.gz", STATIC / "german_wind_points.csv.gz", STATIC / "german_solar_points.csv.gz")
    if arm != "B":
        table = pd.read_csv(WORK / f"table_{arm}.csv.gz")
        table.index = pd.to_datetime(table.pop("timestamp_utc"), utc=True)
        assert list(table.columns) == list(data.wind_pts.columns), "the arm's table must be labelled as the stored one"
        data = replace(data, wind_pts=table.asfreq("h"))
    if curve == "fleet":
        se.power_curve = fleet_curve
    feat = se.features(data)
    out = pd.DataFrame({"delivery_day": feat["frame"]["delivery_day"], "spread": feat["y"]})
    out["wind_px"] = feat["px"]["wind"].reindex(out.index).to_numpy()
    out["G"] = se.fit_months(feat, se.months_of(se.DEV_FIRST, se.DEV_LAST), models=("G",))["G"]
    print(arm, curve, "G fitted", round(time.time() - started), "s", flush=True)
    out["W"] = sw.fit_w(feat, se.DEV_FIRST, se.DEV_LAST)
    out = out[(out["delivery_day"] >= pd.Timestamp(se.DEV_FIRST)) & (out["delivery_day"] <= pd.Timestamp(se.DEV_LAST))]
    out.reset_index(names="timestamp_utc").to_csv(WORK / f"pred_{arm}_{curve}.csv.gz", index=False)
    print(arm, curve, "done", round(time.time() - started), "s, first proxy month", feat["first_month"].date(), flush=True)


def stage_summary() -> dict:
    stored = pd.read_csv(STORED, index_col=0)
    stored.index = pd.to_datetime(stored.index, utc=True)
    stored["delivery_day"] = pd.to_datetime(stored["delivery_day"])
    stored = stored[(stored["delivery_day"] >= pd.Timestamp(se.DEV_FIRST)) & (stored["delivery_day"] <= pd.Timestamp(se.DEV_LAST))]
    preds = {}
    for arm, curve in VARIANTS:
        p = pd.read_csv(WORK / f"pred_{arm}_{curve}.csv.gz")
        p.index = pd.to_datetime(p.pop("timestamp_utc"), utc=True)
        preds[f"{arm}_{curve}"] = p.reindex(stored.index)
    tables_report = json.loads((WORK / "tables_report.json").read_text())
    ref = preds["B_built"]
    stops = {"reference_G_reproduces": float((ref["G"] - stored["G"]).abs().max()), "reference_W_reproduces": float((ref["W"] - stored["W"]).abs().max()),
             "reference_proxy_reproduces": float((ref["wind_px"] - stored["wind_px"]).abs().max()), "tables_do_not_read_ahead": tables_report["scramble_pass"],
             "every_scored_hour_has_every_forecast": bool(all(p[["G", "W", "wind_px"]].notna().all().all() for p in preds.values())), "scored_hours": int(len(stored))}
    stops["all_pass"] = bool(stops["reference_G_reproduces"] < 1e-9 and stops["reference_W_reproduces"] < 1e-9 and stops["reference_proxy_reproduces"] < 1e-6
                             and stops["tables_do_not_read_ahead"] and stops["every_scored_hour_has_every_forecast"])
    result = {"stop_conditions": stops}
    if not stops["all_pass"]:
        print(json.dumps(result, indent=1))
        return result
    nf = se._hourly(STATIC / "neighbour_fundamentals.csv", usecols=["timestamp_utc", *se.TSO_COLUMNS.values()]).rename(columns={v: k for k, v in se.TSO_COLUMNS.items()})
    target = (nf["onshore"] + nf["offshore"]).reindex(stored.index)
    mw, eur = {}, {}
    for name, p in preds.items():
        e = (p["wind_px"] - target).dropna()
        d = pd.DataFrame({"px": p["wind_px"], "fc": target}).dropna()
        mw[name] = {"mae": round(float(e.abs().mean())), "bias": round(float(e.mean())), "corr": round(float(np.corrcoef(d.px, d.fc)[0, 1]), 4), "rank_corr": round(float(d.px.rank().corr(d.fc.rank())), 4),
                    "ramp_error": round(float((d.px.diff() - d.fc.diff()).abs().mean())),
                    "mae_by_delivery_hours": {f"{6 * k:02d}-{6 * k + 5:02d}": round(float(v)) for k, v in e.abs().groupby(e.index.tz_convert("Europe/Paris").hour // 6).mean().items()},
                    "hours": int(len(e))}
        eur[name] = {m: round(float((p[m] - stored["spread"]).abs().mean()), 3) for m in ("G", "W")}
    contrasts = {}
    for label, cand, base in (("fresh against stale, curve as built (primary)", "F_built", "S_built"), ("fresh against stale, fleet curve", "F_fleet", "S_fleet"),
                              ("stale GFS against the stored table", "S_built", "B_built"), ("fresh GFS against the stored table", "F_built", "B_built"),
                              ("fresh GFS with the fleet curve against the stored table", "F_fleet", "B_built"), ("fleet curve against built, fresh", "F_fleet", "F_built")):
        contrasts[label] = {}
        for m in ("W", "G"):
            frame = pd.DataFrame({"delivery_day": stored["delivery_day"], "spread": stored["spread"], "cand": preds[cand][m], "base": preds[base][m]})
            contrasts[label][m] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in se.paired(frame, "cand", "base").items()}
    primary = contrasts["fresh against stale, curve as built (primary)"]["W"]
    result.update({"wind_proxy_against_operator_forecast_mw": mw, "spread_mae_eur_mwh": eur, "contrasts_candidate_minus_base": contrasts,
                   "stored_for_reference": {m: round(float((stored[m] - stored["spread"]).abs().mean()), 3) for m in ("H", "G", "W", "C")}, "tables": {k: tables_report[k] for k in ("S", "F", "correlation_with_the_stored_table_by_point")}})
    result["reading"] = {"primary": primary, "rule": f"a gain of at least {MIN_GAIN} with the two-sided 95% interval below zero", "numerical_noise": "a difference between two models carries up to about 0.25 (amendment 3 of the test of W)"}
    (WORK / "freshness_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tables":
        stage_tables()
    elif stage == "fit":
        assert sys.argv[2] in ARMS and sys.argv[3] in CURVES
        stage_fit(sys.argv[2], sys.argv[3])
    elif stage == "summary":
        stage_summary()
