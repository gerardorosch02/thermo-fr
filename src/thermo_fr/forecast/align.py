"""Turn an inputs table stored as logged into the corrected basis of the solar alignment fix (step 1 of final_plan.md).

The history path (forecast-fetch, refit) produces the corrected basis by itself when data/hour_labels.HOUR_LABELS is
"start_of_hour". Tables fetched before the fix, data/forecast/inputs.csv and the laptop's saved morning windows, are on
the old labels; this module converts one without refetching:

1. every Open-Meteo preceding-hour column (radiation_fc_wm2, radiation_proxy_wm2, the 21 solar points) moves to the start
   of its hour (data/hour_labels.to_start_of_hour);
2. ENTSO-E actual solar, wind onshore, wind offshore and nuclear (the A75 documents) are relabelled by one hour on the
   days that the physical rule of amendment 2 flags on solar (fix_prereg.md amendment 1; forecast/solar_labels.py): the
   last hour of each flagged block becomes missing (0 for solar);
3. the solar and wind proxies are recalibrated on that basis exactly as the history path does it: rolling_proxy, weights
   refitted at the start of each month on the trailing 365 days ending two days before the month; and the weights the
   monthly refit would write, gen_proxy.latest_weights on the trailing 365 days. The nuclear lag features and
   residual_v2_mw follow from the table by their unchanged rules in forecast/features.py.

Notes for the later steps (review_step1.md):
- Step 2 takes honest_v2's and S2's baseline rows from the original data/forecast/inputs.csv, never from the corrected
  table (which also rewrites wind_proxy_mw, so a CSV round trip alone would differ by about 1e-13).
- Step 3's replay converts every saved morning window (data/shadow/fresh/) with align_window: radiation on start-of-hour
  labels, then both proxies recomputed from the converted points with the corrected weights. It never reuses the stored
  proxy columns of a window, which were made on the old labels with the old weights.

Nothing here reads a value published later than the uncorrected table does for the same delivery day: a start-of-hour
radiation value for [T, T+1) is the Open-Meteo value valid at T+1, from the same lead-day-2 run cycle as the last hour the
old labels used (tests/test_solar_align.py, and the timing rule radiation_issued in forecast/timing.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import hour_labels
from ..data.solar_points import POINT_COLUMNS
from ..data.wind_points import POINT_COLUMNS as WIND_POINT_COLUMNS
from . import solar_labels, solar_proxy, wind_proxy
from .gen_proxy import latest_weights

SELECTION = ("2024-02-17", "2026-07-01")  # the selection window, end exclusive


def align_inputs(table: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict, dict]:
    """(table on the corrected basis, report, latest proxy weights {"solar": ..., "wind": ...}) from a table stored as
    logged."""
    out = hour_labels.align_table(table)
    clear = solar_labels.national_clear_sky(out.index)
    offsets = solar_labels.entsoe_solar_offsets(out["solar_mw"], clear)
    days = solar_labels.flagged_days(offsets)
    out, missing = solar_labels.relabel_actuals(out, days)
    log(f"ENTSO-E actuals: median solar offset {offsets.attrs['median_offset_h']:.3f} h, {len(days)} days relabelled; "
        f"missing hours on them {missing}")
    proxy, fits = solar_proxy.rolling_proxy(out[POINT_COLUMNS], solar_proxy.actual_solar(out), log=log)
    out["solar_proxy_mw"] = proxy
    wind, wind_fits = wind_proxy.rolling_proxy(out[[c for c in WIND_POINT_COLUMNS if c in out]], wind_proxy.actual_wind(out), log=log)
    out["wind_proxy_mw"] = wind
    weights = {"solar": latest_weights(out, solar_proxy.actual_solar(out), solar_proxy.SOLAR),
               "wind": latest_weights(out, wind_proxy.actual_wind(out), wind_proxy.WIND)}
    report = {
        "hour_labels": hour_labels.START_OF_HOUR,
        "relabelled_solar_days": solar_labels.day_ranges(days),
        "relabelled_solar_day_count": int(len(days)),
        "median_offset_h": round(offsets.attrs["median_offset_h"], 3),
        "relabelled_series": list(missing),
        "missing_hours_on_flagged_days": missing,
        "solar_proxy_monthly_fits": fits,
        "wind_proxy_monthly_fits": wind_fits,
    }
    return out, report, weights


def proxy_mae(table: pd.DataFrame, window=SELECTION, kind: str = "solar") -> dict:
    """A generation proxy against actual generation of the same table, on the window's hours where both exist (MW)."""
    rows = (table.index >= pd.Timestamp(window[0], tz="UTC")) & (table.index < pd.Timestamp(window[1], tz="UTC"))
    actual = solar_proxy.actual_solar(table) if kind == "solar" else wind_proxy.actual_wind(table)
    both = pd.DataFrame({"proxy": table[f"{kind}_proxy_mw"], "actual": actual})[rows].dropna()
    err = both["actual"] - both["proxy"]
    late = (both.index.hour >= 15) & (both.index.hour <= 19)
    return {"hours": int(len(both)), "first": str(both.index.min()), "last": str(both.index.max()),
            "mae_mw": round(float(err.abs().mean()), 1), "bias_mw": round(float(err.mean()), 1),
            "mae_15_19_utc_mw": round(float(err[late].abs().mean()), 1),
            "rmse_mw": round(float(np.sqrt((err ** 2).mean())), 1)}


def run_alignment(inputs_path, out_path, report_dir, log=print) -> dict:
    """Convert inputs_path (as logged) to the corrected basis at out_path and write the step 1 report under report_dir:
    relabelled days, the cross-check against ENTSO-E's solar forecast, the wind and nuclear audit, the clear-sky labelling
    check before and after, and the solar proxy MAE before and after on the selection window only."""
    import json
    from pathlib import Path

    from . import label_audit
    from .inputs import load_inputs

    table = load_inputs(inputs_path)
    before_proxy, _ = solar_proxy.rolling_proxy(table[POINT_COLUMNS], solar_proxy.actual_solar(table))  # as logged, recomputed
    before_wind, _ = wind_proxy.rolling_proxy(table[WIND_POINT_COLUMNS], wind_proxy.actual_wind(table))
    aligned, report, weights = align_inputs(table, log=lambda *_: None)
    clear = solar_labels.national_clear_sky(table.index)
    sel = (table.index >= pd.Timestamp(SELECTION[0], tz="UTC")) & (table.index < pd.Timestamp(SELECTION[1], tz="UTC"))
    radiation_before = table[POINT_COLUMNS].mean(axis=1)[sel]
    radiation_after = aligned[POINT_COLUMNS].mean(axis=1)[sel]
    report["clear_sky_check_before"] = solar_labels.clear_sky_label_check(radiation_before, table["solar_mw"][sel], clear[sel])
    report["clear_sky_check_after"] = solar_labels.clear_sky_label_check(radiation_after, aligned["solar_mw"][sel], clear[sel])
    feature_before = table["radiation_fc_wm2"][sel]
    feature_after = aligned["radiation_fc_wm2"][sel]
    report["radiation_feature_check_before"] = solar_labels.clear_sky_label_check(feature_before, table["solar_mw"][sel], clear[sel])
    report["radiation_feature_check_after"] = solar_labels.clear_sky_label_check(feature_after, aligned["solar_mw"][sel], clear[sel])
    report["radiation_proxy_check_before"] = solar_labels.clear_sky_label_check(table["radiation_proxy_wm2"][sel], table["solar_mw"][sel], clear[sel])
    report["radiation_proxy_check_after"] = solar_labels.clear_sky_label_check(aligned["radiation_proxy_wm2"][sel], aligned["solar_mw"][sel], clear[sel])
    # scored against the corrected actuals in both cases (the relabelled days end before the proxy's first month)
    scored_before = aligned.assign(solar_proxy_mw=before_proxy, wind_proxy_mw=before_wind)
    report["stored_proxy_reproduced_max_abs_diff_mw"] = float(np.nanmax(np.abs((table["solar_proxy_mw"] - before_proxy).to_numpy())))
    report["stored_wind_proxy_reproduced_max_abs_diff_mw"] = float(np.nanmax(np.abs((table["wind_proxy_mw"] - before_wind).to_numpy())))
    report["proxy_mae_selection_before"] = proxy_mae(scored_before)
    report["proxy_mae_selection_after"] = proxy_mae(aligned)
    report["wind_proxy_mae_selection_before"] = proxy_mae(scored_before, kind="wind")
    report["wind_proxy_mae_selection_after"] = proxy_mae(aligned, kind="wind")
    report["wind_proxy_mae_selection_before_against_logged_actuals"] = proxy_mae(table.assign(wind_proxy_mw=before_wind), kind="wind")
    report["nuclear_lag_changes"] = nuclear_lag_changes(table, aligned)
    longest = max(report["relabelled_solar_days"], key=lambda r: r[2]) if report["relabelled_solar_days"] else ("2023-09", "2024-03", 0)
    first, last = longest[0][:7], longest[1][:7]  # the months of the longest relabelled run
    audit = label_audit.audit(table, (first, last), end=SELECTION[1])
    report["label_audit"] = audit["pairs"]
    out_path, report_dir = Path(out_path), Path(report_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out_path, index_label="timestamp_utc")
    (out_path.with_suffix(".json")).write_text(json.dumps({"hour_labels": hour_labels.START_OF_HOUR, "from": str(inputs_path),
                                                            "relabelled_solar_days": report["relabelled_solar_days"]}, indent=2))
    for name, w in weights.items():
        if w is not None:
            (report_dir / f"{name}_proxy.json").write_text(json.dumps(w, indent=2))
    audit["monthly"].round(3).to_csv(report_dir / "label_audit_monthly.csv", index_label="month")
    (report_dir / "align_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"corrected basis written to {out_path}; report in {report_dir}")
    return report


NUCLEAR_LAGS = ("nuclear_d2_mw", "nuclear_d2_mean_mw", "nuclear_d1_early_mw")


def nuclear_lag_changes(before: pd.DataFrame, after: pd.DataFrame, window=SELECTION, feature_set: str = "honest_v2") -> dict:
    """Strict delivery days of the window (the logged strict-row flag) on which any nuclear lag feature differs."""
    from .features import build_features

    a, b = build_features(before, feature_set), build_features(after, feature_set)
    days = a.info["delivery_day"]
    strict = a.info["weather_point_in_time"].to_numpy() & (days >= pd.Timestamp(window[0])).to_numpy() & (days < pd.Timestamp(window[1])).to_numpy()
    out = {}
    changed_any = pd.Series(False, index=a.X.index)
    for column in NUCLEAR_LAGS:
        x, y = a.X[column], b.X[column]
        diff = ~((x == y) | (x.isna() & y.isna()))
        changed_any |= diff
        out[column] = int(days[strict & diff.to_numpy()].nunique())
    changed_days = sorted(days[strict & changed_any.to_numpy()].unique())
    out["strict_days_changed"] = len(changed_days)
    out["first"] = str(changed_days[0].date()) if changed_days else None
    out["last"] = str(changed_days[-1].date()) if changed_days else None
    out["strict_days_in_window"] = int(days[strict].nunique())
    return out


def align_window(window: pd.DataFrame, solar_weights: dict, wind_weights: dict) -> pd.DataFrame:
    """A saved morning window (as logged) on the corrected basis, for step 3's replay: radiation moved to start-of-hour
    labels and both proxies recomputed from the window's own points with the corrected weights (never the stored ones)."""
    out = hour_labels.align_table(window)
    out["solar_proxy_mw"] = solar_proxy.apply_weights(out[[c for c in POINT_COLUMNS if c in out]], solar_weights)
    out["wind_proxy_mw"] = wind_proxy.apply_weights(out[[c for c in WIND_POINT_COLUMNS if c in out]], wind_weights)
    return out
