"""Neighbours' day-ahead load and residual load in LEAR: the experiment of docs/prereg/neighbour_rl_prereg.md.

Baseline L0 is LEAR as it is. Both candidates are L0 with one added regressor per neighbouring zone in each hourly model: the
model for hour h receives the zone's value for hour h of delivery day D.

    H  the neighbours' day-ahead load forecast (honest timing: due by 10:00 Paris on D-1, the French load forecast's rule)
    I  the neighbours' day-ahead residual load, load forecast minus the operators' wind and solar forecasts
       (IDEALISED timing: those forecasts are due by 18:00 on D-1, after the gate; I is a ceiling test and is never adopted)

L0, honest_v2 and S1 are read from the stored frame of the LEAR-recipe run (lear_recipe report), not refitted. The candidates
are fitted in resumable chunks:

    <out>/chunks/<candidate>/w<window>/<YYYY-MM>.csv          daily table of forecasts
    <out>/chunks/<candidate>/w<window>/<YYYY-MM>.kept.csv     per day and zone, the number of hourly fits that kept the zone

    python -m thermo_fr.forecast.neighbour_rl lear --out DIR --shard 0/2 --budget-minutes 110
    python -m thermo_fr.forecast.neighbour_rl same-path --out DIR --frame FRAME    # no added block reproduces the stored L0
    python -m thermo_fr.forecast.neighbour_rl scramble --out DIR                   # the look-ahead test
    python -m thermo_fr.forecast.neighbour_rl report --out DIR --frame FRAME

Nothing here reads a delivery day on or after the holdout start: lear_recipe.selection_inputs cuts the inputs first, and the
neighbours' file is cut the same way.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lear, lear_recipe, yardstick
from .features import build_features
from .sprint import SLICES, daily_mean_price
from .timing import check_point_in_time, delivery_days
from .transforms import asinh_forward, robust_scale

ZONES = ("de_lu", "be", "nl", "es", "it_north", "ch")
CANDIDATES = {"H": "load", "I": "residual"}
IDEALISED = ("I",)  # candidates that use an input the timing rules place after the gate; never adopted, never published
SCORED_START = pd.Timestamp("2024-02-18")  # first scored delivery day (2024-02-17 is excluded after the correction of 2026-11-17)
TEST_END = lear_recipe.TEST_END
WINDOWS = lear.LEAR_WINDOWS
FILL_DAYS = 7
MIN_GAIN = 0.25
SLICE_LIMIT = 1.0
ONE_SIDED_LEVEL = 1 - 0.05 / 2  # 97.5%, Bonferroni for two candidates
REPRODUCE = {"honest_v2": "15.7124", "L0": "15.3717", "S1": "14.3687"}
SAME_PATH_DAYS = ("2024-02-18", "2024-10-27", "2025-03-30", "2025-10-26", "2026-06-24")
SCRAMBLE_NAMED = ("2024-02-18", "2024-03-31", "2024-10-27", "2025-03-30", "2025-10-26", "2026-03-29")
SCRAMBLE_DRAWN, SCRAMBLE_SEED = 14, 2
KEPT_SAMPLE, KEPT_SEED = 60, 3
PRICE_BANDS = ((-np.inf, 0.0, "below 0"), (0.0, 20.0, "0 to 20"), (20.0, 50.0, "20 to 50"), (50.0, 80.0, "50 to 80"),
               (80.0, 133.0, "80 to 133"), (133.0, np.inf, "above 133"))
TOLERANCE = 1e-9


# --- the neighbours' series ---------------------------------------------------------------------------------------------------

def load_neighbours(path=Path("data/forecast/neighbour_fundamentals.csv"), end: str = TEST_END) -> pd.DataFrame:
    """The downloaded forecasts (hourly UTC), with every delivery day on or after `end` removed."""
    table = pd.read_csv(path, index_col=0)
    table.index = pd.to_datetime(table.index, utc=True)
    table = table.sort_index()
    return table[delivery_days(table.index) < pd.Timestamp(end)]


def _day_table(series: pd.Series, info: pd.DataFrame, all_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Paris delivery days by clock hours 0 to 23, laid out exactly as the LEAR blocks (lear.day_hour_table)."""
    days = pd.DatetimeIndex(info["delivery_day"])
    return lear.day_hour_table(series.reindex(info.index), days, info["hour"].to_numpy()).reindex(all_days)


def _filled(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A missing value takes the same clock hour of the most recent earlier day that has one, at most FILL_DAYS back."""
    out = table.ffill(limit=FILL_DAYS)
    return out, int((table.isna() & out.notna()).sum().sum())


def zone_tables(neighbours: pd.DataFrame, info: pd.DataFrame, all_days: pd.DatetimeIndex, kind: str) -> tuple[dict, dict]:
    """({zone: day x hour table}, {zone: replaced values}) for kind "load" (the load forecast) or "residual" (load forecast
    minus solar, onshore wind and offshore wind forecasts). Each component is laid out and filled on its own; a zone with no
    offshore series at all counts zero offshore wind."""
    if kind not in ("load", "residual"):
        raise ValueError(f"unknown kind {kind!r}")
    tables, replaced = {}, {}
    for z in ZONES:
        load, n = _filled(_day_table(neighbours[f"load_fc_{z}_mw"], info, all_days))
        if kind == "residual":
            for part in ("solar", "wind_onshore", "wind_offshore"):
                column = f"{part}_fc_{z}_mw"
                if part == "wind_offshore" and (column not in neighbours or neighbours[column].notna().sum() == 0):
                    continue  # no offshore fleet reported: zero
                piece, k = _filled(_day_table(neighbours[column], info, all_days))
                load, n = load - piece, n + k
        tables[z], replaced[z] = load, n
    return tables, replaced


def assert_gate(index: pd.DatetimeIndex, candidate: str) -> None:
    """The neighbours' load forecast follows the French load forecast's rule and must pass both deadlines. For an idealised
    candidate the wind and solar forecast must FAIL the gate under the project's rule: that is what makes it idealised, and if
    the rule ever changed this would say so instead of letting the label go stale."""
    for deadline in ("gate", "premarket"):
        check_point_in_time(index, {f"load_fc_{z}_mw": "load_forecast" for z in ZONES}, deadline=deadline)
    if candidate in IDEALISED:
        from .timing import LookaheadError

        try:
            check_point_in_time(index, {"neighbour_wind_solar_fc": "wind_solar_forecast"}, deadline="gate")
        except LookaheadError:
            return
        raise AssertionError("the wind and solar forecast passes the gate under the timing rules: candidate I is no longer idealised")


# --- fitting ------------------------------------------------------------------------------------------------------------------

def fit_predict_day(design: pd.DataFrame, price: pd.DataFrame, blocks: dict, extra: dict, day: pd.Timestamp, window: int,
                    start: pd.Timestamp = lear.LEAR_START, max_iter: int = 2500, partial: tuple = (),
                    min_partial_days: int = 28) -> tuple[pd.Series, dict]:
    """One window's forecast of `day`, refitted on the days ending D-1: L0's fit with, for the model of hour h, one more
    regressor per zone of `extra` (that zone's value for hour h). With an empty `extra` this is L0's own computation.

    A zone named in `partial` may lack values on some training days: those days stay in the fit and its scaled regressor is
    zero there (the median of the days that have it). If fewer than `min_partial_days` training days have it, or `day` itself
    lacks it, the zone is left out of this fit, which with no other zone is L0's own computation.

    Returns (24 forecasts, {zone used: number of hourly models that kept the zone's regressor})."""
    day = pd.Timestamp(day)
    out = pd.Series(np.nan, index=lear.HOURS, dtype=float)
    zones = list(extra)
    first, last = lear.window_days(day, window, start)
    train = (design.index >= first) & (design.index <= last)
    X, Y = design[train], price[train]
    E = {z: extra[z][train] for z in zones}
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    for z in zones:
        if z not in partial:
            ok &= E[z].notna().all(axis=1)
    X, Y = X[ok], Y[ok]
    rows_extra = {z: extra[z].loc[[day]] for z in zones}
    has = {z: E[z][ok].notna().all(axis=1).to_numpy() for z in zones}
    zones = [z for z in zones if z not in partial or (rows_extra[z].notna().all(axis=1).iloc[0] and int(has[z].sum()) >= min_partial_days)]
    if len(X) <= X.shape[1] + len(zones) + 1:
        return out, {}
    row = design.loc[[day]]
    if not (row.notna().all(axis=1).iloc[0] and all(rows_extra[z].notna().all(axis=1).iloc[0] for z in zones)):
        return out, {}
    scalers = lear.fit_scalers(X, Y, blocks)
    Xt = lear.transform(X, blocks, scalers)
    Xp = lear.transform(row, blocks, scalers)
    m, s = scalers["target"]
    Et, Ep = [], []
    for z in zones:
        values = E[z][ok].to_numpy()
        scale = robust_scale(values[has[z]])
        scaled = asinh_forward(values, *scale)
        scaled[~has[z]] = 0.0  # only a partial zone has such days
        Et.append(scaled)
        Ep.append(asinh_forward(rows_extra[z].to_numpy(), *scale))
    kept = {z: 0 for z in zones}
    for h in lear.HOURS:
        yt = lear.asinh_slope_forward(Y[h].to_numpy(), m, s, 1.0)
        if zones:
            Xh = np.hstack([Xt] + [e[:, [h]] for e in Et])
            xp = np.hstack([Xp] + [e[:, [h]] for e in Ep])
        else:
            Xh, xp = Xt, Xp
        model = lear.fit_hour(Xh, yt, max_iter)
        out[h] = float(lear.asinh_slope_inverse(model.predict(xp), m, s, 1.0)[0])
        for k, z in enumerate(zones):
            kept[z] += int(model.coef_[Xt.shape[1] + k] != 0.0)
    return out, kept


def forecast_days(design, price, blocks, extra, days, window) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily table of forecasts, daily table of kept counts per zone) for one window."""
    preds, kepts = {}, {}
    for d in pd.DatetimeIndex(days):
        preds[d], kepts[d] = fit_predict_day(design, price, blocks, extra, d, window)
    table = pd.DataFrame(preds).T.reindex(columns=lear.HOURS) if preds else pd.DataFrame(columns=lear.HOURS, dtype=float)
    kept = pd.DataFrame(kepts).T.reindex(columns=list(extra)) if kepts else pd.DataFrame(columns=list(extra), dtype=float)
    return table, kept


# --- chunks -------------------------------------------------------------------------------------------------------------------

def chunk_path(out: Path, candidate: str, window: int, month) -> Path:
    return Path(out) / "chunks" / candidate / f"w{int(window)}" / f"{month}.csv"


def months() -> pd.PeriodIndex:
    return pd.period_range(SCORED_START, pd.Timestamp(TEST_END) - pd.Timedelta(days=1), freq="M")


def chunk_days(month: pd.Period) -> pd.DatetimeIndex:
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    return days[(days >= SCORED_START) & (days < pd.Timestamp(TEST_END))]


def job_list() -> list[tuple]:
    return [(c, int(w), m) for m in months() for c in CANDIDATES for w in WINDOWS]


def prepare(hourly: pd.DataFrame, neighbours: pd.DataFrame) -> dict:
    """The L0 design and, per candidate, the neighbours' tables on the same days."""
    hourly = lear_recipe.selection_inputs(hourly)
    design, price, blocks = lear.lear_design(hourly)
    info = build_features(hourly, "honest_v2").info
    extra, replaced = {}, {}
    for candidate, kind in CANDIDATES.items():
        extra[candidate], replaced[candidate] = zone_tables(neighbours, info, design.index, kind)
    return {"design": design, "price": price, "blocks": blocks, "info": info, "extra": extra, "replaced": replaced}


def run_jobs(hourly: pd.DataFrame, neighbours: pd.DataFrame, out: Path, shard: tuple[int, int] = (0, 1), budget_s: float | None = None,
             log=print) -> int:
    started = time.time()
    p = prepare(hourly, neighbours)
    mine = [job for k, job in enumerate(job_list()) if k % shard[1] == shard[0]]
    todo = [job for job in mine if not chunk_path(out, *job).exists()]
    log(f"shard {shard[0]}/{shard[1]}: {len(todo)} of {len(mine)} chunks to run", flush=True)
    done = 0
    for candidate, window, month in todo:
        if budget_s is not None and time.time() - started > budget_s:
            break
        days = chunk_days(month)
        rows = p["info"]["delivery_day"].isin(days)
        lear.assert_gate(p["info"].index[rows], p["blocks"])
        assert_gate(p["info"].index[rows], candidate)
        t0 = time.time()
        daily, kept = forecast_days(p["design"], p["price"], p["blocks"], p["extra"][candidate], days, window)
        path = chunk_path(out, candidate, window, month)
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
    for month in months():
        path = chunk_path(out, candidate, window, month).with_suffix(suffix)
        if not path.exists():
            raise FileNotFoundError(f"chunk missing: {path}")
        frame = pd.read_csv(path, index_col=0, parse_dates=[0])
        if suffix == ".csv":
            frame.columns = [int(c) for c in frame.columns]
        frames.append(frame)
    return pd.concat(frames).sort_index()


# --- the report ---------------------------------------------------------------------------------------------------------------

def scored_frame(frame_path: Path) -> pd.DataFrame:
    """The stored evaluation frame of the LEAR-recipe run, from the first scored day."""
    frame = lear_recipe.read_frame(Path(frame_path))
    return frame[frame["delivery_day"] >= SCORED_START].copy()


def reproduction(frame: pd.DataFrame) -> dict:
    out, ok = {}, True
    rows = yardstick.strict_rows(frame)
    for name, printed in REPRODUCE.items():
        value = yardstick.mae(frame, name)
        same = f"{value:.4f}" == printed
        ok = ok and same
        out[name] = {"mae": value, "expected": printed, "printed": f"{value:.4f}", "reproduced": same}
    out["strict_hours"], out["strict_days"] = int(len(rows)), int(rows["delivery_day"].nunique())
    out["first_day"], out["last_day"] = str(rows["delivery_day"].min().date()), str(rows["delivery_day"].max().date())
    out["ok"] = bool(ok and out["strict_days"] == 864)
    return out


def add_candidates(frame: pd.DataFrame, out: Path) -> dict:
    """H and I (the four-window average, L0's forecast where a candidate has none on a scored hour), their windows, S1-H, S1-I."""
    fallbacks = {}
    for candidate in CANDIDATES:
        tables = {w: read_window(out, candidate, w) for w in WINDOWS}
        for w, table in tables.items():
            frame[f"{candidate}_w{w}"] = lear_recipe.to_hourly(frame, table)
        value = lear_recipe.to_hourly(frame, lear_recipe.window_average(list(tables.values())))
        missing = value.isna() & frame["L0"].notna()
        fallbacks[candidate] = int((missing & frame["strict"].astype(bool)).sum())
        frame[candidate] = value.where(~missing, frame["L0"])
        frame[f"S1_{candidate}"] = (frame["honest_v2"] + frame[candidate]) / 2
    return fallbacks


def price_bands(frame: pd.DataFrame, column: str) -> dict:
    rows = yardstick.strict_rows(frame)
    err = (rows[column] - rows["actual"]).abs()
    return {label: {"hours": int(((rows["actual"] >= lo) & (rows["actual"] < hi)).sum()),
                    "mae": float(err[(rows["actual"] >= lo) & (rows["actual"] < hi)].mean())} for lo, hi, label in PRICE_BANDS}


def apply_rule(frame: pd.DataFrame) -> dict:
    base = "L0"
    base_mae = yardstick.mae(frame, base)
    base_slices, base_raw = lear_recipe.slice_maes(frame, base), lear_recipe.slice_maes_unrounded(frame, base)
    out = {"baseline": base, "baseline_mae": base_mae, "baseline_slices": base_slices, "one_sided_level": ONE_SIDED_LEVEL, "candidates": {}}
    for name in CANDIDATES:
        own, raw = lear_recipe.slice_maes(frame, name), lear_recipe.slice_maes_unrounded(frame, name)
        value = yardstick.mae(frame, name)
        pair = yardstick.paired_daily(frame, name, base, one_sided_level=ONE_SIDED_LEVEL)
        worse = {k: round(own[k] - base_slices[k], 2) for k in SLICES}
        worse_raw = {k: raw[k] - base_raw[k] for k in SLICES}
        conditions = {"i_mae_at_least_0.25_below_L0": bool(base_mae - value >= MIN_GAIN),
                      "ii_no_slice_worse_by_more_than_1.0": bool(all(v <= SLICE_LIMIT for v in worse.values())),
                      "iii_one_sided_97.5_upper_below_zero": bool(pair["upper_one_sided"] < 0)}
        entry = {"mae": value, "gain": base_mae - value, "slices": own, "slice_differences": worse, "slice_differences_unrounded": worse_raw,
                 "paired_vs_L0": pair, "conditions": conditions, "idealised": name in IDEALISED}
        i, ii, iii = conditions.values()
        if name in IDEALISED:
            entry["verdict"] = ("ceiling met: build own wind and solar estimates for the neighbours, under a new pre-registration" if i and iii
                                else "inconclusive: gain without a secure interval, parked unless the owner decides otherwise" if i
                                else "ceiling not met: the neighbours' residual load is parked for LEAR at this gate")
        else:
            entry["passes"] = bool(i and ii and iii)
            entry["verdict"] = "passes: forward live log of at least 90 days, pre-registered separately" if entry["passes"] else "fails"
        out["candidates"][name] = entry
    return out


def kept_shares(out: Path) -> dict:
    """Per candidate and zone, the share of hourly fits that kept the zone's regressor: on the pre-registered sample of 60 scored
    days (seed 3) and on every scored day."""
    all_days = pd.date_range(SCORED_START, pd.Timestamp(TEST_END) - pd.Timedelta(days=1), freq="D")
    sample = pd.DatetimeIndex(sorted(np.random.default_rng(KEPT_SEED).choice(all_days, size=KEPT_SAMPLE, replace=False)))
    result = {}
    for candidate in CANDIDATES:
        tables = [read_window(out, candidate, w, ".kept.csv") for w in WINDOWS]
        total = sum(t.reindex(all_days).fillna(0.0) for t in tables)
        fits = 24.0 * len(WINDOWS)
        result[candidate] = {"sample_of_60_days": {z: float(total.loc[sample, z].sum() / (fits * len(sample))) for z in ZONES},
                             "all_scored_days": {z: float(total[z].sum() / (fits * len(all_days))) for z in ZONES}}
    return result


def full_report(hourly: pd.DataFrame, neighbours: pd.DataFrame, frame: pd.DataFrame, out: Path) -> dict:
    hourly = lear_recipe.selection_inputs(hourly)
    report = {"reproduction": reproduction(frame)}
    if not report["reproduction"]["ok"]:
        report["stopped"] = "a known figure did not reproduce: stop condition"
        return report
    report["fallbacks_to_L0_on_scored_hours"] = add_candidates(frame, out)
    p = prepare(hourly, neighbours)
    report["replaced_values"] = p["replaced"]
    report["rule"] = apply_rule(frame)
    daily_price = daily_mean_price(hourly)
    report["extras"] = {}
    for name in ("L0", *CANDIDATES):
        windows = {w: (f"lear_w{w}" if name == "L0" else f"{name}_w{w}") for w in WINDOWS}
        report["extras"][name] = {**lear_recipe.extras(frame, name, daily_price, windows), "price_bands": price_bands(frame, name)}
    report["S1"] = {"S1_mae": yardstick.mae(frame, "S1"), "honest_v2_mae": yardstick.mae(frame, "honest_v2")}
    for name in CANDIDATES:
        report["S1"][f"S1_{name}"] = {"mae": yardstick.mae(frame, f"S1_{name}"), "paired_vs_S1": yardstick.paired_daily(frame, f"S1_{name}", "S1"),
                                      "slices": lear_recipe.slice_maes(frame, f"S1_{name}")}
    report["S1"]["S1_slices"] = lear_recipe.slice_maes(frame, "S1")
    report["kept_shares"] = kept_shares(out)
    return report


# --- checks -------------------------------------------------------------------------------------------------------------------

def same_path_check(hourly: pd.DataFrame, frame: pd.DataFrame, days=SAME_PATH_DAYS, log=print) -> dict:
    """The new fitting code with no added block against the stored L0 windows, on the pre-registered days."""
    hourly = lear_recipe.selection_inputs(hourly)
    design, price, blocks = lear.lear_design(hourly)
    result = {"days": {}, "max_abs_difference": 0.0}
    for d in pd.DatetimeIndex(days):
        rows = frame[frame["delivery_day"] == d]
        worst = 0.0
        for w in WINDOWS:
            again, _ = fit_predict_day(design, price, blocks, {}, d, w)
            stored = rows.groupby("hour")[f"lear_w{w}"].first()
            worst = max(worst, float((again.reindex(stored.index) - stored).abs().max()))
        result["days"][f"{d:%Y-%m-%d}"] = worst
        result["max_abs_difference"] = max(result["max_abs_difference"], worst)
        log(f"{d:%Y-%m-%d}: max abs difference from the stored L0 windows {worst:.3e}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    return result


def scramble_days() -> pd.DatetimeIndex:
    named = pd.DatetimeIndex(SCRAMBLE_NAMED)
    pool = pd.date_range(SCORED_START, pd.Timestamp(TEST_END) - pd.Timedelta(days=1), freq="D").difference(named)
    drawn = np.random.default_rng(SCRAMBLE_SEED).choice(pool, size=SCRAMBLE_DRAWN, replace=False)
    return pd.DatetimeIndex(sorted(named.append(pd.DatetimeIndex(drawn))))


def scramble_neighbours(neighbours: pd.DataFrame, day: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Every neighbours' value of a delivery day after `day` replaced by noise. The values of `day` itself are left: the load
    forecast is known at the gate, and for candidate I the wind and solar forecast of D is the declared idealisation."""
    out = neighbours.copy()
    later = np.asarray(delivery_days(out.index) > day)
    rng = np.random.default_rng(seed)
    for column in out.columns:
        values = neighbours[column].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        spread = float(np.nanstd(values)) or 1.0
        out.loc[later, column] = float(np.nanmean(values)) + 5 * spread + rng.normal(0, 5 * spread, int(later.sum()))
    return out


def scramble_check(hourly: pd.DataFrame, neighbours: pd.DataFrame, out: Path, days=None, log=print) -> dict:
    """For each day D: the inputs scrambled from D onward (lear_recipe.scramble_from) and the neighbours' values of later days
    scrambled; every (candidate, window) forecast for D is refitted and compared with the stored chunk."""
    base = lear_recipe.selection_inputs(hourly)
    stored = {(c, w): read_window(out, c, w) for c in CANDIDATES for w in WINDOWS}
    result = {"days": {}, "max_abs_difference": 0.0}
    for k, day in enumerate(scramble_days() if days is None else pd.DatetimeIndex(days)):
        scrambled = lear_recipe.scramble_from(base, day, seed=k)
        noisy = scramble_neighbours(neighbours, day, seed=500 + k)
        p = prepare(scrambled, noisy)
        a, b = scrambled.to_numpy(dtype=float), base.to_numpy(dtype=float)
        changed = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
        n1, n0 = noisy.to_numpy(dtype=float), neighbours.to_numpy(dtype=float)
        changed_nb = int(((n1 != n0) & ~(np.isnan(n1) & np.isnan(n0))).sum())
        worst = {}
        for c in CANDIDATES:
            for w in WINDOWS:
                again, _ = fit_predict_day(p["design"], p["price"], p["blocks"], p["extra"][c], day, w)
                diff = float((again - stored[(c, w)].loc[day]).abs().max())
                worst[c] = max(worst.get(c, 0.0), diff)
        result["days"][f"{day:%Y-%m-%d}"] = {"scrambled_input_values": changed, "scrambled_neighbour_values": changed_nb, **worst}
        result["max_abs_difference"] = max(result["max_abs_difference"], max(worst.values()))
        log(f"{day:%Y-%m-%d}: {changed} input and {changed_nb} neighbour values scrambled, max abs difference {max(worst.values()):.3e}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    return result


# --- command line -------------------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    from .inputs import load_inputs

    parser = argparse.ArgumentParser(description="Neighbours' load and residual load in LEAR (docs/prereg/neighbour_rl_prereg.md)")
    parser.add_argument("stage", choices=["lear", "same-path", "scramble", "report"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--neighbours", default="data/forecast/neighbour_fundamentals.csv")
    parser.add_argument("--frame", default=None, help="frame.csv of the LEAR-recipe run (honest_v2, L0, S1 and L0's windows)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", type=lear_recipe._shard)
    parser.add_argument("--budget-minutes", type=float, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hourly = load_inputs(Path(args.data))
    neighbours = load_neighbours(Path(args.neighbours))
    if args.stage == "lear":
        left = run_jobs(hourly, neighbours, out, args.shard, None if args.budget_minutes is None else 60 * args.budget_minutes)
        print(f"chunks left in this shard: {left}")
    elif args.stage == "same-path":
        result = same_path_check(hourly, scored_frame(args.frame))
        (out / "same_path_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes")}))
    elif args.stage == "scramble":
        result = scramble_check(hourly, neighbours, out)
        (out / "scramble_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("max_abs_difference", "passes")}))
    elif args.stage == "report":
        frame = scored_frame(args.frame)
        report = full_report(hourly, neighbours, frame, out)
        (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
        frame.to_csv(out / "frame.csv", index_label="timestamp_utc")
        print(json.dumps(report["reproduction"], indent=2))
        if "rule" in report:
            for name, c in report["rule"]["candidates"].items():
                print(name, round(c["mae"], 4), round(c["gain"], 4), c["conditions"], c["verdict"])


if __name__ == "__main__":
    main()
