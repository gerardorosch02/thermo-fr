"""Nuclear and hydro candidates on top of S2 (branch nuclear-hydro; pre-registration nuclear_hydro_prereg.md).

Reference S2: honest_v2 and LEAR combined by inverse 14-day EWMA-MAE weights (forecast/sprint.py), which must reproduce 14.3192.
Candidates, each S2 with the new features in both members:
- N1: nuclear availability as known at issue (GBM: nuc_avail_mw, nuc_avail_chg_mw, residual_avail_mw; LEAR: the 24-hour vector of
  nuc_avail_mw for D and the scalar change);
- N2: hydro (fr_hydro_fill_mwh, fr_hydro_chg4w_mwh, ch_hydro_anom_pct in both members);
- N3: N1 and N2 together.
Each candidate's weights are computed from its own two members, exactly as S2's. The rule (section 5): overall strict MAE at least
0.25 below S2's, no protocol slice worse than S2's by more than 1.0, and the upper end of the one-sided 98.33% moving-block
interval of the paired daily MAE difference against S2 below zero. The holdout is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import walk_forward
from .combine import TUESDAY_TO_SATURDAY, combine, daily_mae, inverse_mae_weights
from .features import HYDRO, NUCLEAR_AVAIL, build_features, feature_timings
from .shape import block_bootstrap_mean
from .sprint import S2_HALF_LIFE, S2_MIN_DAYS, SLICES, _strict, check_window, daily_abs_error, daily_mean_price, lag_diagnostic, metrics_for, regime_days
from .timing import check_point_in_time

VARIANTS = {"S2": ("honest_v2", "base"), "N1": ("honest_v2_n1", "n1"), "N2": ("honest_v2_n2", "n2"), "N3": ("honest_v2_n3", "n3")}
CANDIDATES = ("N1", "N2", "N3")
REFERENCE = "S2"
REFERENCE_MAE = 14.3192
MIN_GAIN = 0.25
SLICE_LIMIT = 1.0
ONE_SIDED_LEVEL = 1 - 0.05 / 3  # 98.33%, Bonferroni for three candidates
NUCLEAR_SHARE_TOP = 0.10  # top 10% of days by |nuc_avail_chg_mw|


def load_nh(directory) -> dict:
    """The cached inputs written by `nh-inputs`: hourly frame, nuclear daily table and hydro daily table."""
    d = Path(directory)
    hourly = pd.read_csv(d / "nh_hourly.csv", index_col=0)
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    nuclear = pd.read_csv(d / "nuclear_daily.csv", index_col=0, parse_dates=True)
    hydro = pd.read_csv(d / "hydro_daily.csv", index_col=0, parse_dates=True)
    return {"hourly": hourly, "nuclear_daily": nuclear, "hydro_daily": hydro}


def assert_gate(table, rows, feature_set: str) -> None:
    """The gate and the pre-market deadline, asserted for every forecast row (raises LookaheadError: the run stops)."""
    check_point_in_time(table.X.index[rows], feature_timings(feature_set), deadline="gate")
    check_point_in_time(table.X.index[rows], feature_timings(feature_set), deadline="premarket")


def gain_shares(model, columns) -> dict:
    gain = model.booster_.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    return {c: float(g) / total for c, g in zip(columns, gain)}


def gbm_stage(hourly: pd.DataFrame, nh: pd.DataFrame, test_start: str, test_end: str, sets=None, log=print) -> tuple[dict, list]:
    """GBM walk-forward for honest_v2 and the three new sets (gate asserted first), with each monthly fit's split-gain shares."""
    check_window(test_end)
    out, shares = {}, []
    for name, (feature_set, _) in VARIANTS.items():
        if sets is not None and name not in sets:
            continue
        table = build_features(hourly, feature_set, nh=nh if feature_set != "honest_v2" else None)
        days = table.info["delivery_day"]
        rows = (days >= pd.Timestamp(test_start)) & (days < pd.Timestamp(test_end))
        assert_gate(table, rows, feature_set)

        def on_fit(start, model_name, model, name=name, columns=list(table.X.columns)):
            shares.append({"variant": name, "fit_month": f"{start:%Y-%m}", **gain_shares(model, columns)})

        out[name] = walk_forward(table, test_start, test_end, models=("gbm",), log=log, on_fit=on_fit).predictions
    return out, shares


def assemble(gbm: dict, lear: dict) -> pd.DataFrame:
    """One hourly frame: for each variant v, gbm_v, lear_v, its S2-style combination (column v) and its LEAR weight w_lear_v."""
    base = gbm[REFERENCE]
    frame = base.drop(columns=["gbm"]).copy()
    for v in VARIANTS:
        frame[f"gbm_{v}"] = gbm[v]["gbm"].reindex(frame.index)
        frame[f"lear_{v}"] = lear[v]["lear"].reindex(frame.index)
        pair = frame[["actual", "delivery_day"]].assign(g=frame[f"gbm_{v}"], l=frame[f"lear_{v}"])
        errors = daily_mae(pair, ["g", "l"])
        days = pd.DatetimeIndex(sorted(frame["delivery_day"].unique()))
        weights = inverse_mae_weights(errors, days, half_life=S2_HALF_LIFE, min_days=S2_MIN_DAYS, update_weekdays=TUESDAY_TO_SATURDAY)
        frame[f"w_lear_{v}"] = weights["l"].reindex(pd.DatetimeIndex(frame["delivery_day"])).to_numpy()
        frame[v] = combine(pair, weights, ["g", "l"])
        frame[f"lear_missing_{v}"] = frame[f"lear_{v}"].isna()
    return frame


def paired(frame: pd.DataFrame, column: str, baseline: str) -> dict:
    diff = (daily_abs_error(frame, column) - daily_abs_error(frame, baseline)).dropna()
    low95, high95 = block_bootstrap_mean(diff.to_numpy(), block=7, resamples=5000, seed=0, level=0.95)
    _, high_one = block_bootstrap_mean(diff.to_numpy(), block=7, resamples=5000, seed=0, level=1 - 2 * (1 - ONE_SIDED_LEVEL))
    return {"days": int(len(diff)), "mean": round(float(diff.mean()), 4), "ci95": [round(low95, 4), round(high95, 4)],
            "upper_one_sided_98_33": round(high_one, 4), "share_won": round(float((diff < 0).mean()), 3)}


def season(month: int) -> str:
    return {12: "winter (DJF)", 1: "winter (DJF)", 2: "winter (DJF)", 3: "spring (MAM)", 4: "spring (MAM)", 5: "spring (MAM)",
            6: "summer (JJA)", 7: "summer (JJA)", 8: "summer (JJA)"}.get(month, "autumn (SON)")


def error_diagnostics(frame: pd.DataFrame, name: str, price: pd.Series, regime: set, nuc_top: set) -> dict:
    strict = _strict(frame)
    rows = strict.dropna(subset=[name])
    err = (rows[name] - rows["actual"]).abs()
    m = metrics_for(frame, name)
    day = rows["delivery_day"]
    winter = day.dt.month.isin([11, 12, 1, 2])
    return {
        "mae": round(float(err.mean()), 4), "rmse": round(float(np.sqrt(((rows[name] - rows["actual"]) ** 2).mean())), 4),
        "hours": int(len(rows)), "days": int(day.nunique()),
        "peak": m["slices"]["peak"]["gbm"]["mae"], "off_peak": m["slices"]["off_peak"]["gbm"]["mae"],
        "negative_hours": m["slices"]["negative_price_hours"]["gbm"]["mae"],
        "slices": {k: m["slices"][v]["gbm"]["mae"] for k, v in SLICES.items()},
        "winter_nov_feb": round(float(err[winter].mean()), 4), "not_winter": round(float(err[~winter].mean()), 4),
        "by_season": {s: round(float(p.mean()), 4) for s, p in err.groupby(day.dt.month.map(season))},
        "by_year": {int(y): round(float(p.mean()), 4) for y, p in err.groupby(day.dt.year)},
        "by_month": {str(k): round(float(p.mean()), 3) for k, p in err.groupby(day.dt.strftime("%Y-%m"))},
        "regime_change_days": round(float(err[day.isin(regime)].mean()), 4),
        "nuclear_news_top10_days": round(float(err[day.isin(nuc_top)].mean()), 4),
        "nuclear_news_other_days": round(float(err[~day.isin(nuc_top)].mean()), 4),
        "lag": lag_diagnostic(frame, name, price),
    }


def selection_report(frame: pd.DataFrame, hourly: pd.DataFrame, nuclear_daily: pd.DataFrame, shares: pd.DataFrame | None = None,
                     coefs: dict | None = None) -> dict:
    """The pre-registered rule for N1 to N3 against S2, and the section 5 diagnostics, on the strict rows of the frame."""
    price = daily_mean_price(hourly)
    strict = _strict(frame)
    strict_days = pd.DatetimeIndex(sorted(strict["delivery_day"].unique()))
    regime = regime_days(price, strict_days)
    chg = nuclear_daily["nuc_avail_chg_mw"].reindex(strict_days).abs()
    cut = float(chg.quantile(1 - NUCLEAR_SHARE_TOP))
    nuc_top = set(chg[chg >= cut].index)
    ref_mae = float((strict[REFERENCE] - strict["actual"]).abs().mean())
    out = {"reference": REFERENCE, "reference_mae": round(ref_mae, 4), "reference_reproduces": abs(round(ref_mae, 4) - REFERENCE_MAE) < 5e-5,
           "strict_hours": int(len(strict)), "strict_days": int(len(strict_days)),
           "first_day": str(strict_days.min().date()), "last_day": str(strict_days.max().date()),
           "nuclear_news_cut_mw": round(cut, 1), "nuclear_news_days": len(nuc_top), "candidates": {}, "diagnostics": {}}
    columns = list(VARIANTS) + [f"gbm_{v}" for v in VARIANTS] + [f"lear_{v}" for v in VARIANTS]
    for name in columns:
        d = error_diagnostics(frame, name, price, regime, nuc_top)
        if name in CANDIDATES:
            d["paired_vs_S2"] = paired(frame, name, REFERENCE)
        if name.startswith("gbm_") and name != "gbm_S2":
            d["paired_vs_honest_v2"] = paired(frame, name, "gbm_S2")
        if name.startswith("lear_") and name != "lear_S2":
            d["paired_vs_lear_base"] = paired(frame, name, "lear_S2")
        out["diagnostics"][name] = d
    ref_slices = out["diagnostics"][REFERENCE]["slices"]
    for name in CANDIDATES:
        d = out["diagnostics"][name]
        gain = ref_mae - d["mae"]
        worse = {k: round(d["slices"][k] - ref_slices[k], 4) for k in SLICES if d["slices"][k] is not None and ref_slices[k] is not None}
        conditions = {
            "gain_at_least_0.25": gain >= MIN_GAIN,
            "no_slice_worse_than_1.0": all(v <= SLICE_LIMIT for v in worse.values()),
            "one_sided_98.33_upper_below_zero": d["paired_vs_S2"]["upper_one_sided_98_33"] < 0,
        }
        out["candidates"][name] = {"gain": round(gain, 4), "slice_differences": worse, "conditions": conditions, "passes": all(conditions.values())}
    passing = [n for n in CANDIDATES if out["candidates"][n]["passes"]]
    out["to_shadow"] = min(passing, key=lambda n: out["diagnostics"][n]["mae"]) if passing else None
    out["finding"] = (f"{out['to_shadow']} passes and goes to the shadow (after S2)" if passing else
                      "no candidate is distinguishable from S2 on the selection window after correcting for three comparisons")
    out["ensemble"] = {}
    for v in VARIANTS:
        both = strict.dropna(subset=[f"lear_{v}"])
        e_g = (both[f"gbm_{v}"] - both["actual"]).groupby(both["delivery_day"]).mean()
        e_l = (both[f"lear_{v}"] - both["actual"]).groupby(both["delivery_day"]).mean()
        out["ensemble"][v] = {"daily_error_correlation_gbm_lear": round(float(e_g.corr(e_l)), 3),
                              "mean_lear_weight": round(float(frame[f"w_lear_{v}"].mean()), 3),
                              "lear_missing_hours": int(frame[f"lear_missing_{v}"].sum()),
                              "lear_missing_strict_hours": int(strict[f"lear_missing_{v}"].sum())}
    if shares is not None and len(shares):
        sel = shares[shares["fit_month"] >= f"{strict_days.min():%Y-%m}"]
        groups = {"nuclear_avail": NUCLEAR_AVAIL, "hydro": HYDRO}
        sg = {}
        for v in VARIANTS:
            part = sel[sel["variant"] == v]
            if part.empty:
                continue
            entry = {}
            for g, cols in groups.items():
                cols = [c for c in cols if c in part and part[c].notna().any()]
                if not cols:
                    continue
                total = part[cols].sum(axis=1)
                entry[g] = {"mean_share": round(float(total.mean()), 4),
                            "by_feature": {c: round(float(part[c].mean()), 4) for c in cols},
                            "by_year": {str(y): round(float(t.mean()), 4) for y, t in total.groupby(part["fit_month"].str[:4])}}
            sg[v] = entry
        out["split_gain_shares"] = sg
    if coefs:
        out["lear_coefficients"] = {}
        for v, table in coefs.items():
            if table is None or table.empty:
                continue
            t = table.copy()
            t["month"] = pd.to_datetime(t["day"]).dt.strftime("%Y-%m")
            entry = {}
            for c in [c for c in ("nuc_own_hour", "nuc_block_sum", "nuc_avail_chg", "fr_hydro_fill", "fr_hydro_chg4w", "ch_hydro_anom") if c in t]:
                g = t.groupby("month")[c]
                entry[c] = {"share_negative": round(float((t[c] < 0).mean()), 3), "share_zero": round(float((t[c] == 0).mean()), 3),
                            "share_positive": round(float((t[c] > 0).mean()), 3), "mean": float(t[c].mean()),
                            "by_month_mean": {k: float(x) for k, x in g.mean().items()},
                            "by_month_share_negative": {k: round(float(x), 3) for k, x in g.apply(lambda s: (s < 0).mean()).items()}}
            out["lear_coefficients"][v] = entry
    return out


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
