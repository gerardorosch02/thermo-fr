"""French residual load in LEAR, honest and idealised, and what the neighbours add to it (docs/prereg/french_rl_prereg.md).

Every candidate is the present LEAR (L0) with same-hour regressors added to each hourly model, fitted by
neighbour_rl.fit_predict_day:

    P   honest: load forecast minus the wind and solar proxies; zero on training days before the proxies exist
    Q   honest: the same from 2024-05-01, with the operator's forecast as a training stand-in before that
    F   idealised: load forecast minus the operator's wind and solar forecasts (window 728 only)
    FI  idealised: F plus the six neighbours' idealised residual load (window 728 only)
    M   idealised: F restricted to the days P has, handled as P (window 728 only)

F, FI and M read an input the timing rules place after the gate. They are references and are never adopted or published.

Stages (all resumable, one chunk per candidate, window and month):

    python -m thermo_fr.forecast.french_rl lear      --out DIR [--shard K/N] [--budget-minutes M]
    python -m thermo_fr.forecast.french_rl same-path --out DIR --frame FRAME
    python -m thermo_fr.forecast.french_rl scramble  --out DIR
    python -m thermo_fr.forecast.french_rl report    --out DIR --frame FRAME --neighbour-out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lear, lear_recipe, yardstick
from . import neighbour_rl as nr
from .features import build_features
from .sprint import SLICES, daily_mean_price
from .timing import LookaheadError, check_point_in_time, delivery_days

HONEST_START = pd.Timestamp("2024-05-01")  # first delivery day on which both generation proxies exist
RULE_START = pd.Timestamp("2024-05-29")  # first day on which P can have a forecast of its own (28 training days with the proxies)
MIN_PARTIAL_DAYS = 28
REFERENCE_WINDOW = 728
WINDOWS_OF = {"P": tuple(nr.WINDOWS), "Q": tuple(nr.WINDOWS), "F": (REFERENCE_WINDOW,), "FI": (REFERENCE_WINDOW,), "M": (REFERENCE_WINDOW,)}
CANDIDATES = tuple(WINDOWS_OF)
HONEST = ("P", "Q")
IDEALISED = ("F", "FI", "M")  # read the operators' wind and solar forecast of D: never adopted, never published
PARTIAL = {"P": ("fr",), "M": ("fr",)}
FIRST_DAY = {"P": HONEST_START, "Q": HONEST_START, "M": HONEST_START, "F": nr.SCORED_START, "FI": nr.SCORED_START}
ONE_SIDED_LEVEL = 1 - 0.05 / 3  # 98.33%, Bonferroni for three decisions (P, Q, the neighbours)
MIN_GAIN, SLICE_LIMIT, TOLERANCE = nr.MIN_GAIN, nr.SLICE_LIMIT, nr.TOLERANCE
SAME_PATH_DAYS = ("2024-02-18", "2024-05-10", "2024-10-27", "2025-03-30", "2026-06-24")
SCRAMBLE_NAMED = ("2024-02-18", "2024-05-29", "2024-10-27", "2025-03-30", "2025-10-26", "2026-03-29")
SCRAMBLE_DRAWN, SCRAMBLE_SEED = 14, 4
OPERATOR_COLUMNS = ("solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw")
HONEST_TIMINGS = {"load_fc_mw": "load_forecast", "wind_proxy_mw": "wind_proxy", "solar_proxy_mw": "solar_proxy"}


# --- the series ---------------------------------------------------------------------------------------------------------------

def french_tables(hourly: pd.DataFrame, info: pd.DataFrame, all_days: pd.DatetimeIndex) -> tuple[dict, dict]:
    """({"honest", "ideal", "q", "m": day x hour table}, {component: replaced values}).

    honest = load forecast - wind proxy - solar proxy (missing before the proxies exist).
    ideal  = load forecast - operator's solar, onshore and offshore wind forecasts (a missing offshore value counts as zero).
    q      = honest from HONEST_START, ideal on the days before it.
    m      = ideal on the days honest exists, missing elsewhere.
    """
    replaced = {}

    def lay(column: str) -> pd.DataFrame:
        table, replaced[column] = nr._filled(nr._day_table(hourly[column], info, all_days))
        return table

    load = lay("load_fc_mw")
    honest = load - lay("wind_proxy_mw") - lay("solar_proxy_mw")
    offshore = nr._day_table(hourly["wind_offshore_fc_mw"], info, all_days).fillna(0.0)
    ideal = load - lay("solar_fc_mw") - lay("wind_onshore_fc_mw") - offshore
    from_start = np.asarray(pd.DatetimeIndex(all_days) >= HONEST_START)
    q = ideal.copy()
    q.loc[from_start] = honest.loc[from_start]
    m = ideal.where(honest.notna().all(axis=1), axis=0)
    return {"honest": honest, "ideal": ideal, "q": q, "m": m}, replaced


def extra_for(candidate: str, french: dict, neighbours: dict) -> dict:
    if candidate == "P":
        return {"fr": french["honest"]}
    if candidate == "Q":
        return {"fr": french["q"]}
    if candidate == "F":
        return {"fr": french["ideal"]}
    if candidate == "FI":
        return {"fr": french["ideal"], **neighbours}
    if candidate == "M":
        return {"fr": french["m"]}
    raise ValueError(f"unknown candidate {candidate!r}")


def assert_gate(index: pd.DatetimeIndex, candidate: str) -> None:
    """The load forecast and the generation proxies must pass both deadlines. For an idealised candidate the operators' wind
    and solar forecast must FAIL the gate under the project's rule: that is what makes it idealised."""
    for deadline in ("gate", "premarket"):
        check_point_in_time(index, HONEST_TIMINGS if candidate in HONEST else {"load_fc_mw": "load_forecast"}, deadline=deadline)
    if candidate in IDEALISED:
        try:
            check_point_in_time(index, {"operator_wind_solar_fc": "wind_solar_forecast"}, deadline="gate")
        except LookaheadError:
            return
        raise AssertionError(f"the wind and solar forecast passes the gate under the timing rules: candidate {candidate} is no longer idealised")


def prepare(hourly: pd.DataFrame, neighbours: pd.DataFrame) -> dict:
    hourly = lear_recipe.selection_inputs(hourly)
    design, price, blocks = lear.lear_design(hourly)
    info = build_features(hourly, "honest_v2").info
    french, replaced = french_tables(hourly, info, design.index)
    zone, replaced_zone = nr.zone_tables(neighbours, info, design.index, "residual")
    return {"design": design, "price": price, "blocks": blocks, "info": info, "french": french, "neighbours": zone,
            "replaced": {"france": replaced, "neighbours": replaced_zone}}


def fit_day(p: dict, candidate: str, day: pd.Timestamp, window: int, min_partial_days: int = MIN_PARTIAL_DAYS) -> tuple[pd.Series, dict]:
    extra = extra_for(candidate, p["french"], p["neighbours"])
    return nr.fit_predict_day(p["design"], p["price"], p["blocks"], extra, day, window, partial=PARTIAL.get(candidate, ()),
                              min_partial_days=min_partial_days)


# --- chunks -------------------------------------------------------------------------------------------------------------------

def months(candidate: str) -> pd.PeriodIndex:
    return pd.period_range(FIRST_DAY[candidate], pd.Timestamp(nr.TEST_END) - pd.Timedelta(days=1), freq="M")


def chunk_days(candidate: str, month: pd.Period) -> pd.DatetimeIndex:
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    return days[(days >= FIRST_DAY[candidate]) & (days < pd.Timestamp(nr.TEST_END))]


def job_list() -> list[tuple]:
    return [(c, int(w), m) for c in CANDIDATES for w in WINDOWS_OF[c] for m in months(c)]


def run_jobs(hourly: pd.DataFrame, neighbours: pd.DataFrame, out: Path, shard: tuple[int, int] = (0, 1), budget_s: float | None = None,
             log=print) -> int:
    started = time.time()
    p = prepare(hourly, neighbours)
    mine = [job for k, job in enumerate(job_list()) if k % shard[1] == shard[0]]
    todo = [job for job in mine if not nr.chunk_path(out, *job).exists()]
    log(f"shard {shard[0]}/{shard[1]}: {len(todo)} of {len(mine)} chunks to run", flush=True)
    done = 0
    for candidate, window, month in todo:
        if budget_s is not None and time.time() - started > budget_s:
            break
        days = chunk_days(candidate, month)
        rows = p["info"]["delivery_day"].isin(days)
        lear.assert_gate(p["info"].index[rows], p["blocks"])
        assert_gate(p["info"].index[rows], candidate)
        t0 = time.time()
        preds, kepts = {}, {}
        for d in days:
            preds[d], kepts[d] = fit_day(p, candidate, d, window)
        names = list(extra_for(candidate, p["french"], p["neighbours"]))
        daily = pd.DataFrame(preds).T.reindex(columns=lear.HOURS)
        kept = pd.DataFrame(kepts).T.reindex(index=days, columns=names)
        path = nr.chunk_path(out, candidate, window, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        kept_tmp = path.with_suffix(".kept.tmp")
        kept.to_csv(kept_tmp, index_label="delivery_day")
        kept_tmp.replace(path.with_suffix(".kept.csv"))
        tmp = path.with_suffix(".tmp")
        daily.to_csv(tmp, index_label="delivery_day")
        tmp.replace(path)  # the forecast file is written last: its presence marks the chunk as done
        done += 1
        log(f"{candidate} w={window} {month}: {len(days)} days in {time.time() - t0:.0f} s", flush=True)
    left = len(todo) - done
    log(f"shard {shard[0]}/{shard[1]}: {done} chunks written, {left} left, {time.time() - started:.0f} s", flush=True)
    return left


def read_window(out: Path, candidate: str, window: int, suffix: str = ".csv") -> pd.DataFrame:
    frames = []
    for month in months(candidate):
        path = nr.chunk_path(out, candidate, window, month).with_suffix(suffix)
        if not path.exists():
            raise FileNotFoundError(f"chunk missing: {path}")
        frame = pd.read_csv(path, index_col=0, parse_dates=[0])
        if suffix == ".csv":
            frame.columns = [int(c) for c in frame.columns]
        frames.append(frame)
    return pd.concat(frames).sort_index()


# --- the report ---------------------------------------------------------------------------------------------------------------

def add_candidates(frame: pd.DataFrame, out: Path, neighbour_out: Path | None) -> dict:
    """Every candidate's window forecasts on the hourly frame, with L0's forecast of the same window where the candidate has
    none; P and Q as the four-window average; S1-P and S1-Q; I at the reference window from the neighbours test."""
    strict = frame["strict"].astype(bool)
    counts = {}
    for candidate in CANDIDATES:
        from_first = frame["delivery_day"] >= FIRST_DAY[candidate]
        counts[candidate] = {}
        for w in WINDOWS_OF[candidate]:
            value = lear_recipe.to_hourly(frame, read_window(out, candidate, w))
            missing = value.isna() & frame[f"lear_w{w}"].notna()
            counts[candidate][f"w{w}"] = {"by_construction_before_first_day": int((missing & strict & ~from_first).sum()),
                                           "fallbacks_from_first_day": int((missing & strict & from_first).sum())}
            frame[f"{candidate}_w{w}"] = value.where(~missing, frame[f"lear_w{w}"])
    for candidate in HONEST:
        frame[candidate] = sum(frame[f"{candidate}_w{w}"] for w in WINDOWS_OF[candidate]) / len(WINDOWS_OF[candidate])
        frame[f"S1_{candidate}"] = (frame["honest_v2"] + frame[candidate]) / 2
    if neighbour_out is not None:
        frame[f"I_w{REFERENCE_WINDOW}"] = lear_recipe.to_hourly(frame, nr.read_window(Path(neighbour_out), "I", REFERENCE_WINDOW))
    return counts


def own_days(out: Path, candidate: str) -> dict:
    """Days on which the candidate's regressor was used in every one of its windows (its kept count is recorded)."""
    used = None
    for w in WINDOWS_OF[candidate]:
        flag = read_window(out, candidate, w, ".kept.csv")["fr"].notna()
        used = flag if used is None else (used & flag.reindex(used.index, fill_value=False))
    days = used.index[used.to_numpy()]
    rule_days = pd.date_range(RULE_START, pd.Timestamp(nr.TEST_END) - pd.Timedelta(days=1), freq="D")
    return {"first_own_day": str(days.min().date()) if len(days) else None, "own_days": int(len(days)),
            "rule_days_without_own_forecast": int(len(rule_days.difference(days)))}


def kept_shares(out: Path) -> dict:
    result = {}
    for candidate in CANDIDATES:
        result[candidate] = {}
        for w in WINDOWS_OF[candidate]:
            kept = read_window(out, candidate, w, ".kept.csv")
            used = kept.dropna(how="all")
            result[candidate][f"w{w}"] = {name: float(used[name].sum() / (24.0 * len(used))) if len(used) else float("nan") for name in kept.columns}
    return result


def half_years(frame: pd.DataFrame, column: str) -> dict:
    rows = yardstick.strict_rows(frame)
    err = (rows[column] - rows["actual"]).abs()
    days = pd.DatetimeIndex(rows["delivery_day"])
    label = pd.Series([f"{y} H{1 if m <= 6 else 2}" for y, m in zip(days.year, days.month)], index=rows.index)
    return {k: float(v) for k, v in err.groupby(label).mean().items()}


def compare(frame: pd.DataFrame, column: str, baseline: str, one_sided: bool = False) -> dict:
    value, base = yardstick.mae(frame, column), yardstick.mae(frame, baseline)
    pair = yardstick.paired_daily(frame, column, baseline, one_sided_level=ONE_SIDED_LEVEL if one_sided else None)
    return {"mae": value, "baseline": baseline, "baseline_mae": base, "gain": base - value, "paired": pair}


def apply_rule(frame: pd.DataFrame, rule: pd.DataFrame) -> dict:
    """P and Q against L0 on the rule days; the neighbours (FI against F at the reference window) on all scored days."""
    base_mae = yardstick.mae(rule, "L0")
    base_slices, base_raw = lear_recipe.slice_maes(rule, "L0"), lear_recipe.slice_maes_unrounded(rule, "L0")
    out = {"baseline": "L0", "rule_days": int(yardstick.strict_rows(rule)["delivery_day"].nunique()),
           "rule_hours": int(len(yardstick.strict_rows(rule))), "baseline_mae": base_mae, "baseline_slices": base_slices,
           "one_sided_level": ONE_SIDED_LEVEL, "candidates": {}}
    for name in HONEST:
        own, raw = lear_recipe.slice_maes(rule, name), lear_recipe.slice_maes_unrounded(rule, name)
        value = yardstick.mae(rule, name)
        pair = yardstick.paired_daily(rule, name, "L0", one_sided_level=ONE_SIDED_LEVEL)
        worse = {k: round(own[k] - base_slices[k], 2) for k in SLICES}
        conditions = {"i_mae_at_least_0.25_below_L0": bool(base_mae - value >= MIN_GAIN),
                      "ii_no_slice_worse_by_more_than_1.0": bool(all(v <= SLICE_LIMIT for v in worse.values())),
                      "iii_one_sided_98.33_upper_below_zero": bool(pair["upper_one_sided"] < 0)}
        passes = bool(all(conditions.values()))
        out["candidates"][name] = {"mae": value, "gain": base_mae - value, "slices": own, "slice_differences": worse,
                                   "slice_differences_unrounded": {k: raw[k] - base_raw[k] for k in SLICES}, "paired_vs_L0": pair,
                                   "conditions": conditions, "passes": passes, "mae_all_864_days": yardstick.mae(frame, name),
                                   "verdict": "passes: forward live log of at least 90 days, pre-registered separately" if passes else "fails"}
    passing = [n for n in HONEST if out["candidates"][n]["passes"]]
    out["taken"] = min(passing, key=lambda n: out["candidates"][n]["mae"]) if passing else None
    f, fi = f"F_w{REFERENCE_WINDOW}", f"FI_w{REFERENCE_WINDOW}"
    nb = compare(frame, fi, f, one_sided=True)
    gain_ok, secure = bool(nb["gain"] >= MIN_GAIN), bool(nb["paired"]["upper_one_sided"] < 0)
    nb["conditions"] = {"gain_at_least_0.25_below_F": gain_ok, "one_sided_98.33_upper_below_zero": secure, "an_honest_candidate_passes": bool(passing)}
    nb["worth_building"] = bool(gain_ok and secure and passing)
    nb["verdict"] = ("worth building: own wind and solar estimates for the neighbours, under a new pre-registration" if nb["worth_building"]
                     else "parked: the neighbours' residual load is parked for LEAR at this gate")
    out["neighbours"] = nb
    if not passing:
        out["next"] = "null: neither honest candidate passes; the next item of the build list is the probabilistic layer"
    return out


def full_report(hourly: pd.DataFrame, neighbours: pd.DataFrame, frame: pd.DataFrame, out: Path, neighbour_out: Path | None) -> dict:
    hourly = lear_recipe.selection_inputs(hourly)
    report = {"reproduction": nr.reproduction(frame)}
    if not report["reproduction"]["ok"]:
        report["stopped"] = "a known figure did not reproduce: stop condition"
        return report
    report["forecasts_taken_from_L0"] = add_candidates(frame, out, neighbour_out)
    report["own_forecast_days"] = {c: own_days(out, c) for c in ("P", "M")}
    report["replaced_values"] = prepare(hourly, neighbours)["replaced"]
    rule = frame[frame["delivery_day"] >= RULE_START]
    report["rule"] = apply_rule(frame, rule)
    daily_price = daily_mean_price(hourly)
    report["extras_on_rule_days"] = {}
    for name in ("L0", *HONEST):
        windows = {w: (f"lear_w{w}" if name == "L0" else f"{name}_w{w}") for w in nr.WINDOWS}
        report["extras_on_rule_days"][name] = {**lear_recipe.extras(rule, name, daily_price, windows), "price_bands": nr.price_bands(rule, name),
                                               "half_years": half_years(rule, name)}
    w = REFERENCE_WINDOW
    l0, f, fi, m, pw = f"lear_w{w}", f"F_w{w}", f"FI_w{w}", f"M_w{w}", f"P_w{w}"
    reference = {"all_scored_days": {"mae": {c: yardstick.mae(frame, c) for c in (l0, f, fi)}, "F_vs_L0": compare(frame, f, l0),
                                     "F_half_years": half_years(frame, f), "L0_half_years": half_years(frame, l0)},
                 "rule_days": {"mae": {c: yardstick.mae(rule, c) for c in (l0, pw, f"Q_w{w}", m, f, fi)}, "P_vs_L0": compare(rule, pw, l0),
                               "M_vs_L0": compare(rule, m, l0), "F_vs_L0": compare(rule, f, l0), "P_vs_M": compare(rule, pw, m),
                               "M_vs_F": compare(rule, m, f)}}
    gain_f = reference["rule_days"]["F_vs_L0"]["gain"]
    reference["rule_days"]["share_of_F_gain_captured_by_P"] = float(reference["rule_days"]["P_vs_L0"]["gain"] / gain_f) if gain_f else float("nan")
    if f"I_w{w}" in frame:
        i = f"I_w{w}"
        reference["all_scored_days"]["mae"][i] = yardstick.mae(frame, i)
        reference["all_scored_days"]["I_vs_L0"] = compare(frame, i, l0)
        reference["all_scored_days"]["FI_vs_I"] = compare(frame, fi, i)
    report[f"window_{w}"] = reference
    report["S1_on_rule_days"] = {"S1_mae": yardstick.mae(rule, "S1"), "honest_v2_mae": yardstick.mae(rule, "honest_v2"),
                                 "S1_slices": lear_recipe.slice_maes(rule, "S1")}
    for name in HONEST:
        report["S1_on_rule_days"][f"S1_{name}"] = {"mae": yardstick.mae(rule, f"S1_{name}"), "paired_vs_S1": yardstick.paired_daily(rule, f"S1_{name}", "S1"),
                                                   "slices": lear_recipe.slice_maes(rule, f"S1_{name}")}
    report["kept_shares"] = kept_shares(out)
    return report


# --- checks -------------------------------------------------------------------------------------------------------------------

def same_path_check(hourly: pd.DataFrame, neighbours: pd.DataFrame, frame: pd.DataFrame, days=SAME_PATH_DAYS, log=print) -> dict:
    """Against the stored L0 windows on the pre-registered days: the fitting code with no added regressor, and candidate P
    with its regressor left out by the minimum-days rule (forced by an unreachable minimum where the proxies are plentiful)."""
    p = prepare(hourly, neighbours)
    result = {"days": {}, "max_abs_difference": 0.0}
    for d in pd.DatetimeIndex(days):
        rows = frame[frame["delivery_day"] == d]
        worst = {"no_regressor": 0.0, "regressor_left_out": 0.0}
        for w in nr.WINDOWS:
            stored = rows.groupby("hour")[f"lear_w{w}"].first()
            plain, _ = nr.fit_predict_day(p["design"], p["price"], p["blocks"], {}, d, w)
            left_out, kept = fit_day(p, "P", d, w, min_partial_days=10**9)
            if kept:
                raise AssertionError("the regressor was used although the minimum could not be met")
            worst["no_regressor"] = max(worst["no_regressor"], float((plain.reindex(stored.index) - stored).abs().max()))
            worst["regressor_left_out"] = max(worst["regressor_left_out"], float((left_out.reindex(stored.index) - stored).abs().max()))
        result["days"][f"{d:%Y-%m-%d}"] = worst
        result["max_abs_difference"] = max(result["max_abs_difference"], *worst.values())
        log(f"{d:%Y-%m-%d}: max abs difference from the stored L0 windows {max(worst.values()):.3e}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    return result


def scramble_days() -> pd.DatetimeIndex:
    named = pd.DatetimeIndex(SCRAMBLE_NAMED)
    pool = pd.date_range(nr.SCORED_START, pd.Timestamp(nr.TEST_END) - pd.Timedelta(days=1), freq="D").difference(named)
    drawn = np.random.default_rng(SCRAMBLE_SEED).choice(pool, size=SCRAMBLE_DRAWN, replace=False)
    return pd.DatetimeIndex(sorted(named.append(pd.DatetimeIndex(drawn))))


def keep_operator_forecast_of(scrambled: pd.DataFrame, base: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The scrambled inputs with the operator's wind and solar forecast of `day` put back: the declared idealisation of F, FI, M."""
    out = scrambled.copy()
    same = np.asarray(delivery_days(out.index) == day)
    for column in OPERATOR_COLUMNS:
        out.loc[same, column] = base.loc[same, column]
    return out


def scramble_check(hourly: pd.DataFrame, neighbours: pd.DataFrame, out: Path, days=None, log=print) -> dict:
    """For each day D every input dated D or later that is not known at the gate is replaced by noise (the operator's wind and
    solar forecast of D included for P and Q, kept for F, FI and M), the neighbours' values of later days too, and every
    stored forecast for D is refitted and compared."""
    base = lear_recipe.selection_inputs(hourly)
    stored = {(c, w): read_window(out, c, w) for c in CANDIDATES for w in WINDOWS_OF[c]}
    result = {"days": {}, "max_abs_difference": 0.0}
    for k, day in enumerate(scramble_days() if days is None else pd.DatetimeIndex(days)):
        scrambled = lear_recipe.scramble_from(base, day, seed=k)
        noisy = nr.scramble_neighbours(neighbours, day, seed=500 + k)
        prepared = {"honest": prepare(scrambled, noisy), "idealised": prepare(keep_operator_forecast_of(scrambled, base, day), noisy)}
        worst = {}
        for c in CANDIDATES:
            if day < FIRST_DAY[c]:
                continue
            p = prepared["idealised" if c in IDEALISED else "honest"]
            for w in WINDOWS_OF[c]:
                again, _ = fit_day(p, c, day, w)
                before = stored[(c, w)].loc[day]
                both_missing = again.isna() & before.isna()
                diff = float((again - before).abs().where(~both_missing, 0.0).fillna(np.inf).max())
                worst[c] = max(worst.get(c, 0.0), diff)
        a, b = scrambled.to_numpy(dtype=float), base.to_numpy(dtype=float)
        changed = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
        result["days"][f"{day:%Y-%m-%d}"] = {"scrambled_input_values": changed, **worst}
        result["max_abs_difference"] = max(result["max_abs_difference"], max(worst.values()))
        log(f"{day:%Y-%m-%d}: {changed} input values scrambled, max abs difference {max(worst.values()):.3e} over {sorted(worst)}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    return result


# --- command line -------------------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    from .inputs import load_inputs

    parser = argparse.ArgumentParser(description="French residual load in LEAR (docs/prereg/french_rl_prereg.md)")
    parser.add_argument("stage", choices=["lear", "same-path", "scramble", "report"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--neighbours", default="data/forecast/neighbour_fundamentals.csv")
    parser.add_argument("--frame", default=None, help="frame.csv of the LEAR-recipe run (honest_v2, L0, S1 and L0's windows)")
    parser.add_argument("--neighbour-out", default=None, help="output folder of the neighbours test (for I at the reference window)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", type=lear_recipe._shard)
    parser.add_argument("--budget-minutes", type=float, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hourly = load_inputs(Path(args.data))
    neighbours = nr.load_neighbours(Path(args.neighbours))
    if args.stage == "lear":
        left = run_jobs(hourly, neighbours, out, args.shard, None if args.budget_minutes is None else 60 * args.budget_minutes)
        print(f"chunks left in this shard: {left}")
    elif args.stage == "same-path":
        result = same_path_check(hourly, neighbours, nr.scored_frame(args.frame))
        (out / "same_path_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes")}))
    elif args.stage == "scramble":
        result = scramble_check(hourly, neighbours, out)
        (out / "scramble_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes")}))
    elif args.stage == "report":
        frame = nr.scored_frame(args.frame)
        report = full_report(hourly, neighbours, frame, out, args.neighbour_out)
        (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
        frame.to_csv(out / "frame.csv", index_label="timestamp_utc")
        print(json.dumps(report["reproduction"], indent=2))
        if "rule" in report:
            for name, c in report["rule"]["candidates"].items():
                print(name, round(c["mae"], 4), round(c["gain"], 4), c["conditions"], c["verdict"])
            nb = report["rule"]["neighbours"]
            print("neighbours", round(nb["gain"], 4), nb["conditions"], nb["verdict"])


if __name__ == "__main__":
    main()
