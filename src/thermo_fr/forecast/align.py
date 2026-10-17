"""Turn an inputs table stored as logged into the corrected basis of the solar alignment fix (step 1 of final_plan.md).

The history path (forecast-fetch, refit) produces the corrected basis by itself when data/hour_labels.HOUR_LABELS is
"start_of_hour". Tables fetched before the fix, data/forecast/inputs.csv and the laptop's saved morning windows, are on
the old labels; this module converts one without refetching:

1. every Open-Meteo preceding-hour column (radiation_fc_wm2, radiation_proxy_wm2, the 21 solar points) moves to the start
   of its hour (data/hour_labels.to_start_of_hour);
2. ENTSO-E actual solar is relabelled by one hour on the days that the physical rule of amendment 2 flags
   (forecast/solar_labels.relabel_entsoe_solar); the other actual series are only audited (forecast/label_audit.py);
3. the solar proxy is recalibrated on that basis exactly as the history path does it: solar_proxy.rolling_proxy, weights
   refitted at the start of each month on the trailing 365 days ending two days before the month; and the weights the
   monthly refit would write, gen_proxy.latest_weights on the trailing 365 days.

Nothing here reads a value published later than the uncorrected table does for the same delivery day: a start-of-hour
radiation value for [T, T+1) is the Open-Meteo value valid at T+1, from the same lead-day-2 run cycle as the last hour the
old labels used (tests/test_solar_align.py, and the timing rule radiation_issued in forecast/timing.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import hour_labels
from ..data.solar_points import POINT_COLUMNS
from . import solar_labels, solar_proxy
from .gen_proxy import latest_weights

SELECTION = ("2024-02-17", "2026-07-01")  # the selection window, end exclusive


def align_inputs(table: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict, dict | None]:
    """(table on the corrected basis, report, latest solar proxy weights) from a table stored as logged."""
    out = hour_labels.align_table(table)
    clear = solar_labels.national_clear_sky(out.index)
    relabelled, days, offsets = solar_labels.relabel_entsoe_solar(out["solar_mw"], clear)
    out["solar_mw"] = relabelled
    log(f"ENTSO-E actual solar: median offset {offsets.attrs['median_offset_h']:.3f} h, {len(days)} days relabelled")
    proxy, fits = solar_proxy.rolling_proxy(out[POINT_COLUMNS], solar_proxy.actual_solar(out), log=log)
    out["solar_proxy_mw"] = proxy
    weights = latest_weights(out, solar_proxy.actual_solar(out), solar_proxy.SOLAR)
    report = {
        "hour_labels": hour_labels.START_OF_HOUR,
        "relabelled_solar_days": solar_labels.day_ranges(days),
        "relabelled_solar_day_count": int(len(days)),
        "median_offset_h": round(offsets.attrs["median_offset_h"], 3),
        "solar_proxy_monthly_fits": fits,
    }
    return out, report, weights


def proxy_mae(table: pd.DataFrame, window=SELECTION) -> dict:
    """Solar proxy against actual solar of the same table, on the window's hours where both exist (MW)."""
    rows = (table.index >= pd.Timestamp(window[0], tz="UTC")) & (table.index < pd.Timestamp(window[1], tz="UTC"))
    both = table.loc[rows, ["solar_proxy_mw", "solar_mw"]].dropna()
    err = both["solar_mw"] - both["solar_proxy_mw"]
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
    # scored against the corrected actuals in both cases (the relabelled days end before the proxy's first month)
    scored_before = aligned.assign(solar_proxy_mw=before_proxy)
    report["stored_proxy_reproduced_max_abs_diff_mw"] = float(np.nanmax(np.abs((table["solar_proxy_mw"] - before_proxy).to_numpy())))
    report["proxy_mae_selection_before"] = proxy_mae(scored_before)
    report["proxy_mae_selection_after"] = proxy_mae(aligned)
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
    if weights is not None:
        (report_dir / "solar_proxy.json").write_text(json.dumps(weights, indent=2))
    audit["monthly"].round(3).to_csv(report_dir / "label_audit_monthly.csv", index_label="month")
    (report_dir / "align_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"corrected basis written to {out_path}; report in {report_dir}")
    return report
