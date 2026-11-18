"""The risk band: an hourly predictive interval for the day-ahead price (docs/prereg/risk_band_prereg.md).

Three things, all from errors that were settled when the forecast was issued (the errors of delivery days up to D-1):

    existing   the logged quantile models of probabilistic.py, rerun to reproduce their coverage
    B          point forecast plus the empirical quantiles of its errors, same clock hour, previous 182 days
    T          errors scaled by a 28-day median absolute deviation per four-hour block, pooled over 364 days, with generalised
               Pareto tails (tail_model.py, the thesis code)

Stages:

    python -m thermo_fr.forecast.risk_band existing --out DIR              # quantile models, monthly walk-forward (resumable)
    python -m thermo_fr.forecast.risk_band bands    --out DIR --frame FRAME
    python -m thermo_fr.forecast.risk_band report   --out DIR --frame FRAME
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import lear_recipe, probabilistic as pb, tail_model
from .features import build_features
from .shape import block_bootstrap_mean

LEVELS = {"80": (0.10, 0.90), "95": (0.025, 0.975)}
QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)
NAMES = ("q025", "q10", "q50", "q90", "q975")
B_DAYS, MIN_DAYS = 182, 120
SCALE_DAYS, MIN_SCALE_DAYS, POOL_DAYS, BLOCK_HOURS = 28, 20, 364, 4
TAIL_MASS = 0.10
SCORED_START, SCORED_END = pd.Timestamp("2024-08-18"), pd.Timestamp("2026-06-30")
POINTS = ("S1", "honest_v2")
WALK_START, TEST_END = "2024-01-01", lear_recipe.TEST_END
LOGGED = {"model": {"pinball_mean": 4.881, "coverage_10_90": 0.802, "mean_width": 49.75},
          "bench_point": {"pinball_mean": 5.105, "coverage_10_90": 0.787, "mean_width": 48.35}}  # docs/experiments.md, 2026-10-11
LOGGED_FROM = pd.Timestamp("2024-02-17")
MAD_SCALE = 1.4826


# --- the two bands -----------------------------------------------------------------------------------------------------------

def error_table(frame: pd.DataFrame, point: str) -> pd.DataFrame:
    """Delivery days by clock hours 0 to 23 of the error (price minus point forecast); a repeated hour is averaged."""
    frame = frame[frame["strict"].astype(bool)] if "strict" in frame else frame
    err = frame["actual"] - frame[point]
    table = err.groupby([pd.DatetimeIndex(frame["delivery_day"]), frame["hour"].to_numpy()]).mean().unstack()
    return table.reindex(index=pd.date_range(table.index.min(), table.index.max(), freq="D"), columns=range(24))


def baseline_quantiles(errors: pd.DataFrame, days: int = B_DAYS, min_days: int = MIN_DAYS) -> dict:
    """{quantile name: day x hour table of the error quantile over the previous `days` days}; only days before D are read."""
    out = {name: pd.DataFrame(np.nan, index=errors.index, columns=errors.columns) for name in NAMES}
    values = errors.to_numpy()
    for i in range(len(errors)):
        past = values[max(0, i - days):i]
        if len(past) == 0:
            continue
        enough = np.sum(~np.isnan(past), axis=0) >= min_days
        if not enough.any():
            continue
        q = np.nanquantile(past[:, enough], QUANTILES, axis=0)
        for k, name in enumerate(NAMES):
            row = np.full(values.shape[1], np.nan)
            row[enough] = q[k]
            out[name].iloc[i] = row
    return out


def block_scales(errors: pd.DataFrame, days: int = SCALE_DAYS, min_days: int = MIN_SCALE_DAYS, block_hours: int = BLOCK_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(location, scale) per day and block of clock hours: median and 1.4826 x MAD of the block's errors over the previous `days`
    days. Tables are laid out day x hour so that they line up with the errors."""
    location = pd.DataFrame(np.nan, index=errors.index, columns=errors.columns)
    scale = location.copy()
    values = errors.to_numpy()
    blocks = [list(range(b, min(b + block_hours, 24))) for b in range(0, 24, block_hours)]
    for i in range(len(errors)):
        past = values[max(0, i - days):i]
        if len(past) == 0 or np.sum(~np.isnan(past).all(axis=1)) < min_days:
            continue
        for hours in blocks:
            sample = past[:, hours].ravel()
            sample = sample[~np.isnan(sample)]
            if len(sample) < min_days:
                continue
            m = float(np.median(sample))
            s = MAD_SCALE * float(np.median(np.abs(sample - m)))
            if s > 0:
                location.iloc[i, hours] = m
                scale.iloc[i, hours] = s
    return location, scale


def tail_quantiles(errors: pd.DataFrame, pool_days: int = POOL_DAYS, min_days: int = MIN_DAYS, tail_mass: float = TAIL_MASS) -> tuple[dict, pd.DataFrame]:
    """({quantile name: day x hour table}, daily table of the fitted tail parameters). For day D the pool is the standardised
    errors of the previous `pool_days` days, each standardised by its own day's location and scale, which were computed from
    earlier days only."""
    location, scale = block_scales(errors)
    z = ((errors - location) / scale).to_numpy()
    out = {name: pd.DataFrame(np.nan, index=errors.index, columns=errors.columns) for name in NAMES}
    params = {}
    loc, sc = location.to_numpy(), scale.to_numpy()
    for i in range(len(errors)):
        past = z[max(0, i - pool_days):i]
        if len(past) == 0 or np.sum(~np.isnan(past).all(axis=1)) < min_days or np.isnan(sc[i]).all():
            continue
        dist = tail_model.fit_spliced(past.ravel(), tail_mass)
        q = dist.ppf(np.array(QUANTILES))
        for k, name in enumerate(NAMES):
            out[name].iloc[i] = loc[i] + sc[i] * q[k]
        params[errors.index[i]] = {"xi_upper": dist.xi_upper, "beta_upper": dist.beta_upper, "xi_lower": dist.xi_lower, "beta_lower": dist.beta_lower,
                                   "u_upper": dist.u_upper, "u_lower": dist.u_lower, "pool": int(np.sum(~np.isnan(past)))}
    return out, pd.DataFrame(params).T


def lay_on_rows(frame: pd.DataFrame, tables: dict, point: str, method: str) -> pd.DataFrame:
    """The day x hour quantile tables as columns <point>_<method>_<name> on the hourly rows: point forecast plus error quantile."""
    keys = pd.MultiIndex.from_arrays([pd.DatetimeIndex(frame["delivery_day"]), frame["hour"].to_numpy()])
    out = pd.DataFrame(index=frame.index)
    for name, table in tables.items():
        out[f"{point}_{method}_{name}"] = frame[point].to_numpy() + table.stack(future_stack=True).reindex(keys).to_numpy()
    return out


def add_bands(frame: pd.DataFrame, points=POINTS) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    fitted = {}
    for point in points:
        errors = error_table(frame, point)
        frame = frame.join(lay_on_rows(frame, baseline_quantiles(errors), point, "B"))
        tables, params = tail_quantiles(errors)
        frame = frame.join(lay_on_rows(frame, tables, point, "T"))
        fitted[point] = params
    return frame, fitted


# --- scores ------------------------------------------------------------------------------------------------------------------

def interval_score(y, lo, hi, alpha: float) -> np.ndarray:
    """Width plus 2/alpha times the distance by which the outcome falls outside (Gneiting and Raftery 2007)."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def band_metrics(frame: pd.DataFrame, stem: str, level: str) -> dict:
    lo_q, hi_q = LEVELS[level]
    lo, hi = frame[f"{stem}_{NAMES[QUANTILES.index(lo_q)]}"].to_numpy(), frame[f"{stem}_{NAMES[QUANTILES.index(hi_q)]}"].to_numpy()
    y = frame["actual"].to_numpy()
    return {"hours": int(len(frame)), "coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo)),
            "interval_score": float(np.mean(interval_score(y, lo, hi, 1 - (hi_q - lo_q)))), "below": int(np.sum(y < lo)), "above": int(np.sum(y > hi)),
            "pinball_lo": pb.pinball(y, lo, lo_q), "pinball_hi": pb.pinball(y, hi, hi_q)}


def pinball_mean_10_50_90(frame: pd.DataFrame, lo: str, mid: str, hi: str) -> float:
    y = frame["actual"].to_numpy()
    return float(np.mean([pb.pinball(y, frame[lo].to_numpy(), 0.10), pb.pinball(y, frame[mid].to_numpy(), 0.50), pb.pinball(y, frame[hi].to_numpy(), 0.90)]))


def slices(frame: pd.DataFrame) -> dict:
    masks = dict(pb.slice_masks(frame))
    masks["negative_price_hours"] = (frame["actual"] < 0).to_numpy()
    days = pd.DatetimeIndex(frame["delivery_day"])
    for label in sorted({f"{d.year} H{1 if d.month <= 6 else 2}" for d in days.unique()}):
        year, half = int(label[:4]), int(label[-1])
        masks[label] = np.asarray((days.year == year) & ((days.month <= 6) == (half == 1)))
    for b in range(0, 24, BLOCK_HOURS):
        masks[f"hours_{b:02d}_{b + BLOCK_HOURS - 1:02d}"] = ((frame["hour"] >= b) & (frame["hour"] < b + BLOCK_HOURS)).to_numpy()
    spread = frame.groupby(days)["actual"].agg(lambda s: s.max() - s.min())
    masks["widest_range_10pct_days"] = days.isin(spread[spread >= spread.quantile(0.9)].index)
    return masks


def paired_interval_score(frame: pd.DataFrame, stem_a: str, stem_b: str, level: str = "80") -> dict:
    """Daily mean interval score of a minus b, with the moving-block bootstrap of the project (block 7, 5000 resamples, seed 0)."""
    lo_q, hi_q = LEVELS[level]
    alpha = 1 - (hi_q - lo_q)
    names = NAMES[QUANTILES.index(lo_q)], NAMES[QUANTILES.index(hi_q)]
    days = pd.DatetimeIndex(frame["delivery_day"])
    score = {s: pd.Series(interval_score(frame["actual"], frame[f"{s}_{names[0]}"], frame[f"{s}_{names[1]}"], alpha), index=frame.index).groupby(days).mean()
             for s in (stem_a, stem_b)}
    diff = (score[stem_a] - score[stem_b]).dropna().to_numpy()
    low, high = block_bootstrap_mean(diff, block=7, resamples=5000, seed=0, level=0.90)
    return {"days": int(len(diff)), "mean": float(diff.mean()), "one_sided_95_lower": float(low), "one_sided_95_upper": float(high),
            "share_days_better": float((diff < 0).mean())}


def scored_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = frame[frame["strict"].astype(bool) & (frame["delivery_day"] >= SCORED_START) & (frame["delivery_day"] <= SCORED_END)]
    return rows.dropna(subset=["actual", *columns])


def apply_rule(overall: dict, by_slice: dict, paired: dict) -> dict:
    t80, t95, b80 = overall["T"]["80"], overall["T"]["95"], overall["B"]["80"]
    conditions = {
        "i_interval_score_80_lower_and_secure": bool(t80["interval_score"] < b80["interval_score"] and paired["one_sided_95_upper"] < 0),
        "ii_coverage_80_within_77_83_and_weak_slices_hold": bool(
            0.77 <= t80["coverage"] <= 0.83
            and all(by_slice[s]["T"]["80"]["coverage"] >= by_slice[s]["B"]["80"]["coverage"] - 0.03 for s in ("windiest_10pct_days", "holidays"))),
        "iii_coverage_95_within_93_97": bool(0.93 <= t95["coverage"] <= 0.97),
    }
    return {"conditions": conditions, "working_band": "T" if all(conditions.values()) else "B"}


def report(frame: pd.DataFrame, existing: pd.DataFrame | None = None, points=POINTS) -> dict:
    out = {"scored_set": {"first_day": str(SCORED_START.date()), "last_day": str(SCORED_END.date())}}
    columns = [f"{p}_{m}_{n}" for p in points for m in ("B", "T") for n in NAMES]
    if existing is not None:
        frame = frame.join(existing[["lo", "q50", "hi", "q10", "q90"]].rename(columns={"lo": "E_q10", "hi": "E_q90", "q50": "E_q50", "q10": "Eraw_q10", "q90": "Eraw_q90"}))
        columns += ["E_q10", "E_q50", "E_q90"]
    rows = scored_rows(frame, columns)
    out["scored_set"].update({"hours": int(len(rows)), "days": int(rows["delivery_day"].nunique())})
    masks = slices(rows)
    for point in points:
        res = {"overall": {}, "by_slice": {}}
        for method in ("B", "T"):
            stem = f"{point}_{method}"
            res["overall"][method] = {level: band_metrics(rows, stem, level) for level in LEVELS}
            res["overall"][method]["pinball_mean_10_50_90"] = pinball_mean_10_50_90(rows, f"{stem}_q10", f"{stem}_q50", f"{stem}_q90")
        for name, mask in masks.items():
            part = rows[mask]
            res["by_slice"][name] = {method: {level: band_metrics(part, f"{point}_{method}", level) for level in LEVELS} for method in ("B", "T")}
        res["paired_T_minus_B_80"] = paired_interval_score(rows, f"{point}_T", f"{point}_B", "80")
        res["paired_T_minus_B_95"] = paired_interval_score(rows, f"{point}_T", f"{point}_B", "95")
        res["rule"] = apply_rule(res["overall"], res["by_slice"], res["paired_T_minus_B_80"])
        out[point] = res
    if existing is not None:
        y = rows["actual"].to_numpy()
        lo, hi = rows["E_q10"].to_numpy(), rows["E_q90"].to_numpy()
        out["existing_on_scored_set"] = {
            "hours": int(len(rows)), "coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo)),
            "interval_score": float(np.mean(interval_score(y, lo, hi, 0.2))), "pinball_mean_10_50_90": pinball_mean_10_50_90(rows, "E_q10", "E_q50", "E_q90"),
            "by_slice": {name: {"hours": int(mask.sum()), "coverage": float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask]))),
                                "mean_width": float(np.mean(hi[mask] - lo[mask])), "interval_score": float(np.mean(interval_score(y[mask], lo[mask], hi[mask], 0.2)))}
                         for name, mask in masks.items()}}
    return out


# --- the existing quantile models ---------------------------------------------------------------------------------------------

def existing_quantiles(hourly: pd.DataFrame, out: Path, point: pd.Series, log=print) -> pd.DataFrame:
    """q10, q50 and q90 of probabilistic.py, month by month as walk_forward_prob fits them (same features, same settings, same
    order of rows), without its two event classifiers, which do not enter the interval. One file per month, so a run resumes.
    `point` is the stored honest_v2 walk-forward forecast, which is the `point` column of the logged run."""
    hourly = lear_recipe.selection_inputs(hourly, TEST_END)
    table = build_features(hourly, "honest_v2")
    days = table.info["delivery_day"]
    folder = Path(out) / "quantiles"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for start in pd.date_range(pd.Timestamp(WALK_START), pd.Timestamp(TEST_END), freq="MS", inclusive="left"):
        path = folder / f"q_{start:%Y-%m}.csv"
        if not path.exists():
            end = start + pd.DateOffset(months=1)
            train = (days < start) & table.y.notna()
            test = (days >= start) & (days < end)
            part = table.info.loc[test, ["delivery_day", "hour", "is_peak", "dow", "holiday", "wind_mw", "solar_mw"]].copy()
            part["actual"] = table.y[test]
            part["naive_day"] = table.X.loc[test, "price_lag1"]
            for alpha, name in zip(pb.QUANTILES, ("q10", "q50", "q90")):
                model = pb.quantile_model(alpha, None)
                model.fit(table.X[train], table.y[train])
                part[name] = np.asarray(model.predict(table.X[test]), dtype=float)
            part["strict"] = table.info.loc[test, "weather_point_in_time"].to_numpy()
            tmp = path.with_suffix(".tmp")
            part.to_csv(tmp, index_label="timestamp_utc")
            tmp.replace(path)
            log(f"  quantiles: {start:%Y-%m} trained on {int(train.sum()):,} hours", flush=True)
        frames.append(lear_recipe.read_frame(path))
    pred = pb.sort_quantiles(pd.concat(frames).sort_index())
    pred["point"] = point.reindex(pred.index)
    ready = pb.trailing_error_quantiles(pb.trailing_error_quantiles(pb.conformal_interval(pred), "point"), "naive_day")
    return ready


def reproduce_existing(ready: pd.DataFrame) -> dict:
    """The logged row of 2026-10-11 on its own set: strict hours from 2024-02-17."""
    rows = ready[(ready["delivery_day"] >= LOGGED_FROM) & (ready["delivery_day"] < pd.Timestamp(TEST_END))]
    result = pb.evaluate_intervals(rows)
    overall = result["overall"]
    checks = {}
    for method, logged in LOGGED.items():
        got = overall[method]
        checks[method] = {"pinball_mean": got["pinball_mean"], "coverage_10_90": got["coverage_10_90"], "mean_width": got["mean_width"], "logged": logged,
                          "reproduced": bool(round(got["pinball_mean"], 3) == logged["pinball_mean"] and round(got["coverage_10_90"], 3) == logged["coverage_10_90"]
                                             and round(got["mean_width"], 2) == logged["mean_width"])}
    return {"hours": overall["model"]["hours"], "checks": checks, "raw_interval": overall["model_raw"], "by_slice": {k: v["model"] for k, v in result["by_slice"].items()},
            "ok": bool(all(c["reproduced"] for c in checks.values()))}


# --- command line -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    from . import neighbour_rl as nr
    from .inputs import load_inputs

    parser = argparse.ArgumentParser(description="The risk band (docs/prereg/risk_band_prereg.md)")
    parser.add_argument("stage", choices=["existing", "bands", "report"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--frame", default=None, help="frame.csv of the LEAR-recipe run (actual prices, honest_v2, L0, S1)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "existing":
        stored = lear_recipe.read_frame(Path(args.frame))
        ready = existing_quantiles(load_inputs(Path(args.data)), out, stored["honest_v2"])
        ready.to_csv(out / "existing_ready.csv", index_label="timestamp_utc")
        result = reproduce_existing(ready)
        (out / "existing_reproduction.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("hours", "checks", "ok")}, indent=1))
    elif args.stage == "bands":
        frame = nr.scored_frame(args.frame)
        banded, fitted = add_bands(frame)
        banded.to_csv(out / "bands.csv", index_label="timestamp_utc")
        for point, params in fitted.items():
            params.to_csv(out / f"tail_params_{point}.csv", index_label="delivery_day")
        print(f"bands written for {len(banded):,} rows")
    elif args.stage == "report":
        banded = lear_recipe.read_frame(out / "bands.csv")
        existing = lear_recipe.read_frame(out / "existing_ready.csv") if (out / "existing_ready.csv").exists() else None
        result = report(banded, existing)
        (out / "results.json").write_text(json.dumps(result, indent=1))
        for point in POINTS:
            print(point, json.dumps({m: {lv: {k: round(v, 4) for k, v in result[point]["overall"][m][lv].items()} for lv in LEVELS} for m in ("B", "T")}))
            print(point, "T minus B at 80%", result[point]["paired_T_minus_B_80"], result[point]["rule"])


if __name__ == "__main__":
    main()
