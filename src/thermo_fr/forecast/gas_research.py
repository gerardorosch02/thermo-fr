"""The ttf-gas research measurement (addendum v5, section 3), selection window only.

Research measurement, not a forecast model: what the D-2 TTF settlement would have been worth to the day-ahead forecast if
a feed had delivered it before 10:05 Paris on D-1. The DBnomics feed used here does not deliver it in time, so nothing here
is or can become the live model; the figures are an upper bound under an idealised timing assumption.

`thermo-fr gas-research` runs the matched control honest_v2_t23 and the candidates research_ttf_ideal_chg (R1) and
research_ttf_ideal_rel (R2) through the monthly walk-forward trained from 2023-01-01, and reports everything section 3 asks
for: MAE overall, by slice and by year; the paired daily difference against the control with a moving-block bootstrap
interval; the lag diagnostic; the mechanism checks (new-close days against the others, the top 10% of days by gas change);
the split-gain share; the share of next-contract substitutions; the control against 15.70; and the pre-registered reading
rule. No gas price is written or printed: the gas change enters only through the 90th-percentile cut, which is not shown.

Lag diagnostic, as logged since 2026-10-10 (it reproduces honest_v2's -0.053, -0.026 and 0.103 on the selection window):
for each strict delivery day, the mean signed error of the strict hours against the change in the daily mean price from D-3
to D-1; signed is their correlation, slope the least-squares slope, absolute the correlation of the absolute values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ
from .features import GAS, RESEARCH_LABEL, RESEARCH_SETS
from .models import GBM_PARAMS, make_model
from .shape import block_bootstrap_mean

SELECTION_END = "2026-07-01"  # exclusive: nothing from the holdout is ever forecast here
CONTROL = "honest_v2_t23"
LOGGED_HONEST_V2 = 15.70
SLICES = {"weekends": "weekends", "holidays": "holidays", "windiest 10%": "windiest_10pct_days", "sunniest 10%": "sunniest_10pct_days",
          "top 5% hours": "top_5pct_price_hours"}
MIN_GAIN = 0.25
SLICE_LIMIT = 1.0
LABELS = {"research_ttf_ideal_chg": "R1 research_ttf_ideal_chg", "research_ttf_ideal_rel": "R2 research_ttf_ideal_rel", CONTROL: "control honest_v2_t23"}


def daily_mean_price(hourly: pd.DataFrame) -> pd.Series:
    days = hourly.index.tz_convert(LOCAL_TZ).normalize().tz_localize(None)
    return hourly["price_eur_mwh"].groupby(days).mean()


def strict_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[predictions["strict"].astype(bool)].dropna(subset=["actual"])
    return frame


def daily_errors(predictions: pd.DataFrame, model: str = "gbm") -> pd.DataFrame:
    frame = strict_rows(predictions)
    err = frame[model] - frame["actual"]
    return pd.DataFrame({"err": err, "abs": err.abs(), "day": frame["delivery_day"]}).groupby("day").agg(err=("err", "mean"), mae=("abs", "mean"))


def lag_diagnostic(predictions: pd.DataFrame, price: pd.Series, model: str = "gbm") -> dict:
    daily = daily_errors(predictions, model)
    move = price.reindex(daily.index - pd.Timedelta(days=1)).to_numpy() - price.reindex(daily.index - pd.Timedelta(days=3)).to_numpy()
    move = pd.Series(move, index=daily.index)
    ok = move.notna() & daily["err"].notna()
    e, m = daily.loc[ok, "err"], move[ok]
    return {"signed": round(float(e.corr(m)), 3), "slope": round(float(np.cov(e, m)[0, 1] / m.var()), 3),
            "absolute": round(float(e.abs().corr(m.abs())), 3), "days": int(ok.sum())}


def mae(frame: pd.DataFrame, model: str = "gbm") -> float:
    return float((frame[model] - frame["actual"]).abs().mean())


def by_year(predictions: pd.DataFrame) -> dict:
    frame = strict_rows(predictions)
    return {int(y): round(mae(part), 2) for y, part in frame.groupby(frame["delivery_day"].dt.year)}


def paired_daily(candidate: pd.DataFrame, control: pd.DataFrame) -> dict:
    a, b = daily_errors(candidate)["mae"], daily_errors(control)["mae"]
    diff = (a - b.reindex(a.index)).dropna()
    low, high = block_bootstrap_mean(diff.to_numpy(), block=7, resamples=5000, seed=0)
    return {"days": int(len(diff)), "mean_difference": round(float(diff.mean()), 3), "ci95_low": round(low, 3), "ci95_high": round(high, 3),
            "share_candidate_better": round(float((diff < 0).mean()), 3), "distinguishable_from_zero": bool(low > 0 or high < 0)}


def mechanism(candidate: pd.DataFrame, control: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Gain (control MAE minus candidate MAE, strict hours) on new-close days (age 2) against days of age above 2, on days
    whose gas features are missing (scored with the control), and on the top 10% of days by |gas_chg_lag1_eur| against the rest
    (cut on the window's strict days with features; the cut itself is not reported)."""
    cand, ctrl = strict_rows(candidate), strict_rows(control)
    ctrl = ctrl.loc[cand.index]
    age = X.loc[cand.index, "gas_age_days"]
    chg = X.loc[cand.index, "gas_chg_lag1_eur"].abs()
    daily_chg = chg.groupby(cand["delivery_day"]).first()
    cut = daily_chg.dropna().quantile(0.9)
    big_days = set(daily_chg[daily_chg >= cut].index)
    big = cand["delivery_day"].isin(big_days)

    def gain(mask):
        if not mask.any():
            return {"hours": 0, "days": 0, "gain": None}
        return {"hours": int(mask.sum()), "days": int(cand.loc[mask, "delivery_day"].nunique()),
                "gain": round(mae(ctrl[mask]) - mae(cand[mask]), 3)}

    return {"age_2": gain(age == 2), "age_above_2": gain(age > 2), "features_missing": gain(age.isna()),
            "top10_gas_change_days": gain(big), "other_days": gain(~big & age.notna())}


def split_gain_share(table, train_start: str, month_start: str = "2026-06-01") -> dict:
    """LightGBM gain shares of the gas features in the walk-forward refit for the last selection month (trained on delivery days
    from train_start to the day before month_start, as in the backtest, relative target for R2)."""
    days = table.info["delivery_day"]
    train = (days >= pd.Timestamp(train_start)) & (days < pd.Timestamp(month_start))
    y = table.y[train] if table.target_offset is None else table.y[train] - table.target_offset[train]
    keep = y.notna()
    model = make_model("gbm")
    model.fit(table.X[train][keep], y[keep])
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=table.X.columns)
    share = gain / gain.sum()
    top = share.sort_values(ascending=False)
    return {"gas_total": round(float(share[GAS].sum()), 4), **{f: round(float(share[f]), 4) for f in GAS},
            "gas_ranks": {f: int(list(top.index).index(f)) + 1 for f in GAS}, "n_features": int(len(share)),
            "top5": {k: round(float(v), 3) for k, v in top.head(5).items()}}


def summarise(results: dict, hourly: pd.DataFrame, train_start: str) -> dict:
    price = daily_mean_price(hourly)
    control = results[CONTROL]
    out = {"label": RESEARCH_LABEL, "sets": {}}
    for name in (CONTROL,) + tuple(s for s in RESEARCH_SETS if s in results):
        r = results[name]
        preds = r["backtest"].predictions
        m = r["metrics_strict"]
        entry = {
            "rows": m["rows"], "first_day": m["first_day"], "last_day": m["last_day"],
            "mae": m["slices"]["all"]["gbm"]["mae"], "mae_unrounded": round(mae(strict_rows(preds)), 4),
            "naive_day": m["slices"]["all"]["naive_day"]["mae"],
            "peak": m["slices"]["peak"]["gbm"]["mae"], "off_peak": m["slices"]["off_peak"]["gbm"]["mae"],
            "slices": {label: m["slices"][key]["gbm"]["mae"] for label, key in SLICES.items()},
            "by_year": by_year(preds), "lag": lag_diagnostic(preds, price),
        }
        if name != CONTROL:
            strict = strict_rows(preds)
            entry["paired_vs_control"] = paired_daily(preds, control["backtest"].predictions)
            entry["mechanism"] = mechanism(preds, control["backtest"].predictions, r["table"].X)
            entry["split_gain"] = split_gain_share(r["table"], train_start)
            days = strict.groupby("delivery_day").agg(sub=("gas_next_contract", "max"), miss=("gas_features_missing", "max"))
            entry["strict_days"] = int(len(days))
            entry["substituted_days"] = int(days["sub"].astype(bool).sum())
            entry["features_missing_days"] = int(days["miss"].astype(bool).sum())
            entry["rows_scored_with_control"] = int(strict["gas_fallback"].astype(bool).sum())
        out["sets"][name] = entry
    out["control_minus_logged"] = round(out["sets"][CONTROL]["mae_unrounded"] - LOGGED_HONEST_V2, 3)
    out["reading_rule"] = reading_rule(out)
    return out


def reading_rule(summary: dict) -> dict:
    ctrl = summary["sets"][CONTROL]
    verdicts = {}
    for name, s in summary["sets"].items():
        if name == CONTROL:
            continue
        gain = ctrl["mae_unrounded"] - s["mae_unrounded"]
        worse = {k: round(s["slices"][k] - ctrl["slices"][k], 2) for k in SLICES}
        mech = s["mechanism"]
        g2, g3 = mech["age_2"]["gain"], mech["age_above_2"]["gain"]
        conditions = {
            f"1. overall MAE lower by at least {MIN_GAIN} (gain {gain:.3f})": gain >= MIN_GAIN,
            f"2. no slice worse by more than {SLICE_LIMIT} (differences {worse})": all(v <= SLICE_LIMIT for v in worse.values()),
            f"3. gain on new-close days {g2} larger than on days of age above 2 {g3}": g2 is not None and g3 is not None and g2 > g3,
            f"4. signed lag correlation {s['lag']['signed']} closer to zero than the control's {ctrl['lag']['signed']}":
                abs(s["lag"]["signed"]) < abs(ctrl["lag"]["signed"]),
        }
        verdicts[name] = {"conditions": {k: bool(v) for k, v in conditions.items()}, "meets_all": all(conditions.values()), "gain": round(gain, 3)}
    passing = {k: v for k, v in verdicts.items() if v["meets_all"]}
    if passing:
        best = max(passing, key=lambda k: passing[k]["gain"])
        text = f"a timely gas feed is judged worth pursuing: {LABELS[best]} meets all four conditions, gain {passing[best]['gain']:.2f} EUR/MWh"
    else:
        text = "no evidence that a timely gas feed would help on 2024 to 2026; the missing-gas explanation of the lag is weakened"
    return {"candidates": verdicts, "verdict": text, "worth_pursuing": bool(passing)}


def to_markdown(summary: dict) -> str:
    sets = summary["sets"]
    names = list(sets)
    lines = [f"> {summary['label']}", "", f"**Reading rule:** {summary['reading_rule']['verdict']}.", ""]
    lines += ["| | " + " | ".join(LABELS[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    lines.append("| Strict hours | " + " | ".join(f"{sets[n]['rows']:,}" for n in names) + " |")
    lines.append("| MAE all (unrounded) | " + " | ".join(f"{sets[n]['mae']:.2f} ({sets[n]['mae_unrounded']:.4f})" for n in names) + " |")
    lines.append("| Naive D-1 | " + " | ".join(f"{sets[n]['naive_day']:.2f}" for n in names) + " |")
    lines.append("| Peak / off-peak | " + " | ".join(f"{sets[n]['peak']:.2f} / {sets[n]['off_peak']:.2f}" for n in names) + " |")
    for label in SLICES:
        lines.append(f"| {label} | " + " | ".join(f"{sets[n]['slices'][label]:.2f}" for n in names) + " |")
    years = sorted({y for n in names for y in sets[n]["by_year"]})
    for y in years:
        lines.append(f"| {y} | " + " | ".join(f"{sets[n]['by_year'].get(y, float('nan')):.2f}" for n in names) + " |")
    lines.append("| Lag signed / slope / absolute | " + " | ".join(
        f"{sets[n]['lag']['signed']:+.3f} / {sets[n]['lag']['slope']:+.3f} / {sets[n]['lag']['absolute']:.3f}" for n in names) + " |")
    lines += ["", f"Control against the logged honest_v2 (trained from 2021, 15.70): {sets[CONTROL]['mae_unrounded']:.4f}, "
              f"difference {summary['control_minus_logged']:+.3f} EUR/MWh (the effect of training from 2023-01-01).", ""]
    for n in names[1:]:
        s = sets[n]
        p, m, g = s["paired_vs_control"], s["mechanism"], s["split_gain"]
        lines += [f"### {LABELS[n]}", "",
                  f"- Paired daily MAE difference against the control (candidate minus control), {p['days']} strict days: {p['mean_difference']:+.3f} "
                  f"EUR/MWh, 95% moving-block bootstrap interval [{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}] (blocks of 7 days, 5,000 resamples); "
                  f"candidate better on {p['share_candidate_better']:.1%} of days; distinguishable from zero: {p['distinguishable_from_zero']}.",
                  f"- Mechanism, gain = control MAE minus candidate MAE on strict hours: new-close days (age 2) {m['age_2']['gain']} over {m['age_2']['days']} days; "
                  f"age above 2 {m['age_above_2']['gain']} over {m['age_above_2']['days']} days; days without gas features (scored with the control) "
                  f"{m['features_missing']['days']}; top 10% of days by |gas change since the D-1 auction close| {m['top10_gas_change_days']['gain']} over "
                  f"{m['top10_gas_change_days']['days']} days, other days {m['other_days']['gain']} over {m['other_days']['days']} days.",
                  f"- Split-gain share of the gas features (refit for June 2026, trained 2023-01-01 to 2026-05-31): total {g['gas_total']:.2%}; "
                  + ", ".join(f"{f} {g[f]:.2%} (rank {g['gas_ranks'][f]} of {g['n_features']})" for f in GAS) + f"; top five {g['top5']}.",
                  f"- Strict days {s['strict_days']}: next-contract substitution on {s['substituted_days']} ({s['substituted_days'] / s['strict_days']:.1%}), "
                  f"gas features missing on {s['features_missing_days']} ({s['rows_scored_with_control']} strict hours scored with the control).",
                  "- Reading rule conditions: " + "; ".join(f"{k}: {'met' if v else 'not met'}" for k, v in summary["reading_rule"]["candidates"][n]["conditions"].items()),
                  ""]
    return "\n".join(lines)
