"""One development evaluation on 2026-07-01 to 2026-10-10 (docs/prereg/july_october_eval_prereg.md, approved by the owner on
2026-11-19). Development evidence, not a holdout test: the period's prices are known to the project and the tree model has
been scored on it before. After this run the period is spent for every model named here.

Stages, in the order they are run:

    trees        T300 and T800 month by month for June (the reproduction check) and July to October 2026
    check-trees  the stop conditions that need only the trees; prints nothing else
    lear         L0 (per window) and Q for each day through the live entry point's own fit (forward_q.fit_known), with the
                 distance of the day; P is not fitted
    scramble     three days of the period: inputs not known at the gate replaced by noise, forecasts compared with forecasts
    summary      every stop condition first; the comparisons and the diagnostic only if all of them pass

Every model reads one history: the frozen inputs with the generation proxies filled by the rule of amendment 1 of the forward
test (forward_q.fill_proxy_gaps). Q and P are never scored on a live day here: this module reads no live data at all.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import forward_q as fq
from . import lear, lear_recipe
from .features import build_features
from .models import GBM_PARAMS, make_model
from .refit import REFIT_PARAMS
from .shape import block_bootstrap_mean

FIRST, LAST = "2026-07-01", "2026-10-10"
CHECK_MONTH = "2026-06"
CHECK_DAYS = ("2026-06-03", "2026-06-10", "2026-06-17", "2026-06-24", "2026-06-30")  # five sampled June days
SCRAMBLE_DAYS = ("2026-07-15", "2026-08-26", "2026-10-06")
SETTINGS = {"T300": {**GBM_PARAMS, **REFIT_PARAMS}, "T800": dict(GBM_PARAMS)}
MONTHS = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")
BANDS = (1.0, 1.6)  # distance: low below 1.0, middle 1.0 to below 1.6, high 1.6 or more
HIGH_PRICE = 200.0  # the plain split: yesterday's mean price at or above this
OVERSHOOT = 5.0  # EUR/MWh: L0's predicted change this far below the realised one
MIN_HIGH_DAYS = 5
CRASH = ("2026-10-01", "2026-10-10")
EXPECTED_HOURS, EXPECTED_DAYS = 2448, 102
LOGGED_T800, LOGGED_TOLERANCE = 25.81, 0.10
TOLERANCE = 1e-9
COMPARISONS = (("T800", "T300"), ("S1", "T300"), ("S1-Q", "S1"))
ONE_SIDED = 1 - 0.05 / len(COMPARISONS)
BLOCK, RESAMPLES, SEED = 7, 5000, 0
INFO_COLUMNS = ["delivery_day", "hour", "dow", "holiday", "weather_point_in_time", "wind_mw", "solar_mw"]


def load_history(data: Path) -> tuple[pd.DataFrame, dict]:
    from .inputs import load_inputs

    return fq.fill_proxy_gaps(load_inputs(Path(data)))


def all_days() -> list[str]:
    return [f"{d:%Y-%m-%d}" for d in pd.date_range(FIRST, LAST, freq="D")]


# --- trees -------------------------------------------------------------------------------------------------------------------

def trees(history: pd.DataFrame, out: Path, log=print) -> None:
    """T300 and T800 exactly as forward_q_checks.gbm_production fits the production setting: refitted at the start of each
    calendar month on every earlier row with a price."""
    table = build_features(history, "honest_v2")
    days = table.info["delivery_day"]
    folder = Path(out) / "trees"
    folder.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        start = pd.Timestamp(month + "-01")
        train, test = np.asarray(days < start), np.asarray((days >= start) & (days < start + pd.DateOffset(months=1)))
        keep = table.y[train].notna()
        for name, params in SETTINGS.items():
            path = folder / f"{name}_{month}.csv"
            if path.exists():
                continue
            model = make_model("gbm", params)
            model.fit(table.X[train][keep], table.y[train][keep])
            part = table.info.loc[test, INFO_COLUMNS].copy()
            part[name] = np.asarray(model.predict(table.X[test]), dtype=float)
            tmp = path.with_suffix(".tmp")
            part.to_csv(tmp, index_label="timestamp_utc")
            tmp.replace(path)
            log(f"{name} {month}: trained on {int(keep.sum()):,} hours, {int(test.sum())} rows forecast", flush=True)


def read_trees(out: Path) -> pd.DataFrame:
    frames = []
    for month in MONTHS:
        parts = [pd.read_csv(Path(out) / "trees" / f"{name}_{month}.csv", index_col=0) for name in SETTINGS]
        frames.append(parts[0].join(parts[1][["T800"]]))
    frame = pd.concat(frames)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


# --- LEAR --------------------------------------------------------------------------------------------------------------------

def known_at_gate(history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The table forward_q.q_live hands to the fit: what may be used at 10:05 Paris on D-1, cut after the last hour of D."""
    hours = fq.day_hours(day)
    tau = fq.premarket_issue_for(hours[:1])[0]
    known = fq.as_known_q(history, tau)
    known = known[known.index < hours[-1] + pd.Timedelta(hours=1)]
    if known.reindex(hours)["load_fc_mw"].isna().any():
        raise fq.ForwardInputError(f"load forecast for {day:%Y-%m-%d} incomplete at the issue time")
    return known.sort_index()


def distance(known: pd.DataFrame, day: pd.Timestamp) -> dict:
    """The frozen definition: the 24 prices of D-1 on the scale each of L0's windows is fitted on for day D, averaged."""
    design, price, blocks = lear.lear_design(known)
    yesterday = design.loc[[day], blocks["price_d1"]].to_numpy(dtype=float)[0]
    out = {"yesterday_mean": float(np.mean(yesterday)), "windows": {}}
    for w in lear.LEAR_WINDOWS:
        first, last = lear.window_days(day, w)
        train = (design.index >= first) & (design.index <= last)
        X, Y = design[train], price[train]
        ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
        m, s = lear.fit_scalers(X[ok], Y[ok], blocks)["target"]
        out["windows"][str(w)] = {"distance": float(np.mean(lear.asinh_forward(yesterday, m, s))), "median": float(np.ravel(m)[0]),
                                  "scale": float(np.ravel(s)[0])}
    out["distance"] = float(np.mean([v["distance"] for v in out["windows"].values()]))
    return out


def lear_day(history: pd.DataFrame, day: str) -> dict:
    stamp = pd.Timestamp(day)
    known = known_at_gate(history, stamp)
    live = fq.fit_known(known, stamp, candidates=("L0", "Q"))
    hours = live["hour"]
    record = {"delivery_day": day, "index": [str(t) for t in live["index"]]}
    for name in ("L0", "Q"):
        record[name] = [float(x) for x in live[name]["hourly"].reindex(live["index"]).to_numpy()]
    record["L0_windows"] = {str(w): [float(live["L0"]["windows"][w].loc[h]) for h in hours] for w in lear.LEAR_WINDOWS}
    record["Q_windows"] = {str(w): [float(live["Q"]["windows"][w].loc[h]) for h in hours] for w in lear.LEAR_WINDOWS}
    record["Q_kept"] = {str(w): c for w, c in live["Q"]["kept"].items()}
    record.update(distance(known, stamp))
    return record


def lear_stage(history: pd.DataFrame, out: Path, shard=(0, 1), log=print) -> None:
    folder = Path(out) / "lear"
    folder.mkdir(parents=True, exist_ok=True)
    for k, day in enumerate(list(CHECK_DAYS) + all_days()):
        path = folder / f"{day}.json"
        if k % shard[1] != shard[0] or path.exists():
            continue
        t0 = time.time()
        record = lear_day(history, day)
        record["seconds"] = round(time.time() - t0, 1)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
        log(f"{day}: {record['seconds']} s", flush=True)


def read_lear(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(hourly frame with L0, Q and the per-window forecasts, one row per day with the distance)."""
    hourly, daily = [], []
    for path in sorted((Path(out) / "lear").glob("*.json")):
        r = json.loads(path.read_text())
        part = pd.DataFrame({"L0": r["L0"], "Q": r["Q"]}, index=pd.to_datetime(r["index"], utc=True))
        for w, values in r["L0_windows"].items():
            part[f"L0_w{w}"] = values
        for w, values in r["Q_windows"].items():
            part[f"Q_w{w}"] = values
        hourly.append(part)
        daily.append({"delivery_day": r["delivery_day"], "distance": r["distance"], "yesterday_mean": r["yesterday_mean"],
                      **{f"distance_w{w}": v["distance"] for w, v in r["windows"].items()}})
    return pd.concat(hourly).sort_index(), pd.DataFrame(daily).set_index("delivery_day").sort_index()


# --- scramble ----------------------------------------------------------------------------------------------------------------

def scramble(history: pd.DataFrame, out: Path, log=print) -> dict:
    """Forecasts against forecasts on three days of the period. The price of D is among the scrambled values; none is read."""
    base_table = build_features(history, "honest_v2")
    days = base_table.info["delivery_day"]
    result = {"days": {}, "max_abs_difference": 0.0}
    for k, d in enumerate(SCRAMBLE_DAYS):
        day = pd.Timestamp(d)
        noisy = lear_recipe.scramble_from(history, day, seed=k)
        a, b = noisy.to_numpy(dtype=float), history.to_numpy(dtype=float)
        changed = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).sum())
        worst = {}
        before, after = fq.fit_known(known_at_gate(history, day), day, candidates=("L0", "Q")), fq.fit_known(known_at_gate(noisy, day), day, candidates=("L0", "Q"))
        for name in ("L0", "Q"):
            worst[name] = max(float((before[name]["windows"][w] - after[name]["windows"][w]).abs().max()) for w in lear.LEAR_WINDOWS)
        noisy_table = build_features(noisy, "honest_v2")
        start = day.to_period("M").to_timestamp()
        train, rows = np.asarray(days < start), np.asarray(days == day)
        keep = base_table.y[train].notna()
        for name, params in SETTINGS.items():
            model = make_model("gbm", params)
            model.fit(base_table.X[train][keep], base_table.y[train][keep])
            worst[name] = float(np.abs(np.asarray(model.predict(base_table.X[rows])) - np.asarray(model.predict(noisy_table.X.loc[base_table.X.index[rows]]))).max())
        result["days"][d] = {"scrambled_input_values": changed, **worst}
        result["max_abs_difference"] = max(result["max_abs_difference"], *worst.values())
        log(f"{d}: {changed} input values scrambled, largest change {max(worst.values()):.3e}", flush=True)
    result["passes"] = bool(result["max_abs_difference"] <= TOLERANCE)
    (Path(out) / "scramble.json").write_text(json.dumps(result, indent=1))
    return result


# --- statistics --------------------------------------------------------------------------------------------------------------

def daily_error(frame: pd.DataFrame, name: str) -> pd.Series:
    return (frame[name] - frame["actual"]).abs().groupby(frame["delivery_day"]).mean().sort_index()


def paired(frame: pd.DataFrame, candidate: str, base: str) -> dict:
    """Candidate minus base, by delivery day. Supports if the one-sided upper end is below zero, contradicts if the mirrored
    lower end is above zero, undecided otherwise (the reading rule of the plan)."""
    diff = (daily_error(frame, candidate) - daily_error(frame, base)).to_numpy()
    if len(diff) == 0:
        return {"days": 0}
    lo, hi = block_bootstrap_mean(diff, block=BLOCK, resamples=RESAMPLES, seed=SEED, level=0.95)
    lower, upper = block_bootstrap_mean(diff, block=BLOCK, resamples=RESAMPLES, seed=SEED, level=1 - 2 * (1 - ONE_SIDED))
    verdict = "supports" if upper < 0 else "contradicts" if lower > 0 else "undecided"
    return {"days": int(len(diff)), "candidate_mae": float((frame[candidate] - frame["actual"]).abs().mean()),
            "base_mae": float((frame[base] - frame["actual"]).abs().mean()), "mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "one_sided_upper": float(upper), "mirrored_lower": float(lower), "share_days_better": float((diff < 0).mean()), "verdict": verdict}


def slices(frame: pd.DataFrame) -> dict:
    by_day = frame.groupby("delivery_day").agg(wind=("wind_mw", "mean"), solar=("solar_mw", "mean"))
    return {"weekends": frame["dow"] >= 5, "holidays": frame["holiday"].astype(bool),
            "windiest 10% of days": frame["delivery_day"].isin(by_day.index[by_day["wind"] >= by_day["wind"].quantile(0.9)]),
            "sunniest 10% of days": frame["delivery_day"].isin(by_day.index[by_day["solar"] >= by_day["solar"].quantile(0.9)]),
            "top 5% of hours by price": frame["actual"] >= frame["actual"].quantile(0.95)}


def spells(days: list[str]) -> list[list[str]]:
    """Maximal runs of consecutive calendar days."""
    out: list[list[str]] = []
    for d in sorted(days):
        if out and pd.Timestamp(d) - pd.Timestamp(out[-1][-1]) == pd.Timedelta(days=1):
            out[-1].append(d)
        else:
            out.append([d])
    return out


def band_report(frame: pd.DataFrame, daily: pd.DataFrame, days: list[str]) -> dict:
    """Changes of the daily mean from D-1 and the LEAR's error against the tree's on the given days."""
    rows = frame[frame["delivery_day"].isin(days)]
    if rows.empty:
        return {"days": 0, "spells": 0}
    mean = rows.groupby("delivery_day")[["actual", "L0", "T300", "S1"]].mean()
    yesterday = daily["yesterday_mean"].reindex(mean.index)
    runs = spells(days)
    return {"days": int(len(mean)), "spells": len(runs), "spell_lengths": [len(r) for r in runs], "spell_starts": [r[0] for r in runs],
            "mean_distance": float(daily["distance"].reindex(mean.index).mean()), "mean_yesterday_price": float(yesterday.mean()),
            "realised_change": float((mean["actual"] - yesterday).mean()), "L0_predicted_change": float((mean["L0"] - yesterday).mean()),
            "T300_predicted_change": float((mean["T300"] - yesterday).mean()),
            "L0_too_low_days": int(((mean["L0"] - mean["actual"]) < 0).sum()),
            "L0_minus_T300": paired(rows, "L0", "T300"), "S1_minus_T300": paired(rows, "S1", "T300")}


def recurrence(high: dict, without_longest: dict) -> str:
    """The rule written in the plan before the run."""
    if high["days"] < MIN_HIGH_DAYS or high["spells"] <= 1:
        return "too few days"

    def holds(r: dict) -> bool:
        return r["days"] > 0 and (r["L0_predicted_change"] < r["realised_change"] - OVERSHOOT) and r["L0_minus_T300"]["mean"] > 0

    return "recurs" if holds(high) and holds(without_longest) else "does not recur"


# --- stop conditions and summary ------------------------------------------------------------------------------------------------

def period_frame(history: pd.DataFrame, out: Path, with_lear: bool) -> pd.DataFrame:
    frame = read_trees(out)
    frame["actual"] = history["price_eur_mwh"].reindex(frame.index)
    if with_lear:
        hourly, _ = read_lear(out)
        frame = frame.join(hourly)
        frame["S1"] = (frame["T300"] + frame["L0"]) / 2
        frame["S1-Q"] = (frame["T300"] + frame["Q"]) / 2
        frame["S1-800"] = (frame["T800"] + frame["L0"]) / 2
    return frame


def strict_period(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[(frame["delivery_day"] >= FIRST) & (frame["delivery_day"] <= LAST) & frame["weather_point_in_time"].astype(bool)]
    return rows.dropna(subset=["actual"])


def tree_stops(history: pd.DataFrame, out: Path, stored_frame: Path, gbm_prod: Path) -> dict:
    """Stop conditions 1, 2 (trees) and 4. Condition 1 reads T800's error on the period, a figure the log already holds."""
    frame = period_frame(history, out, with_lear=False)
    june = frame[frame["delivery_day"].str.startswith(CHECK_MONTH)]
    stored = pd.read_csv(stored_frame, index_col=0, usecols=["timestamp_utc", "honest_v2"])
    stored.index = pd.to_datetime(stored.index, utc=True)
    prod = pd.read_csv(Path(gbm_prod) / f"honest_v2_prod_{CHECK_MONTH}.csv", index_col=0)
    prod.index = pd.to_datetime(prod.index, utc=True)
    period = strict_period(frame)
    t800 = float((period["T800"] - period["actual"]).abs().mean())
    result = {
        "june_T800_against_stored_research": float((june["T800"] - stored["honest_v2"].reindex(june.index)).abs().max()),
        "june_T300_against_stored_production_settings": float((june["T300"] - prod["honest_v2_prod"].reindex(june.index)).abs().max()),
        "june_rows_compared": int(stored["honest_v2"].reindex(june.index).notna().sum()),
        "strict_hours": int(len(period)), "strict_days": int(period["delivery_day"].nunique()),
        "T800_on_period": t800, "logged": LOGGED_T800}
    result["passes"] = {"1 logged figure": bool(abs(t800 - LOGGED_T800) <= LOGGED_TOLERANCE),
                        "2 trees reproduce June": bool(max(result["june_T800_against_stored_research"], result["june_T300_against_stored_production_settings"]) <= TOLERANCE),
                        "4 hours and days": bool(result["strict_hours"] == EXPECTED_HOURS and result["strict_days"] == EXPECTED_DAYS)}
    return result


def lear_stops(out: Path, stored_frame: Path) -> dict:
    hourly, _ = read_lear(out)
    columns = ["timestamp_utc", "Q"] + [f"lear_w{w}" for w in lear.LEAR_WINDOWS] + [f"Q_w{w}" for w in lear.LEAR_WINDOWS]
    stored = pd.read_csv(stored_frame, index_col=0, usecols=columns)
    stored.index = pd.to_datetime(stored.index, utc=True)
    worst = {}
    for d in CHECK_DAYS:
        rows = hourly[fq.delivery_days(hourly.index) == pd.Timestamp(d)]
        if rows.empty:
            worst[d] = None
            continue
        diffs = [float((rows[f"L0_w{w}"] - stored[f"lear_w{w}"].reindex(rows.index)).abs().max()) for w in lear.LEAR_WINDOWS]
        diffs += [float((rows[f"Q_w{w}"] - stored[f"Q_w{w}"].reindex(rows.index)).abs().max()) for w in lear.LEAR_WINDOWS]
        worst[d] = max(diffs)
    done = [v for v in worst.values() if v is not None]
    return {"days": worst, "max_abs_difference": max(done) if done else None, "passes": bool(len(done) == len(CHECK_DAYS) and max(done) <= TOLERANCE)}


def summary(history: pd.DataFrame, filled: dict, out: Path, stored_frame: Path, gbm_prod: Path) -> dict:
    stops = {"trees": tree_stops(history, out, stored_frame, gbm_prod), "lear": lear_stops(out, stored_frame),
             "scramble": json.loads((Path(out) / "scramble.json").read_text()) if (Path(out) / "scramble.json").exists() else {"passes": False}}
    passes = all(stops["trees"]["passes"].values()) and stops["lear"]["passes"] and stops["scramble"]["passes"]
    result = {"stops": stops, "stops_pass": bool(passes), "proxies_filled": filled}
    if not passes:
        return result  # nothing else is computed
    frame = strict_period(period_frame(history, out, with_lear=True))
    models = ["T300", "T800", "L0", "Q", "S1", "S1-Q", "S1-800"]
    same = frame.dropna(subset=models)
    result["hours"] = {"strict": int(len(frame)), "with_every_model": int(len(same)), "days": int(same["delivery_day"].nunique())}
    crash = (same["delivery_day"] >= CRASH[0]) & (same["delivery_day"] <= CRASH[1])
    parts = {"whole period": same, "without 1 to 10 October": same[~crash], "1 to 10 October": same[crash]}
    result["mean_price"] = {name: float(part["actual"].mean()) for name, part in parts.items()}
    result["models"] = {name: {m: {"mae": float((part[m] - part["actual"]).abs().mean()), "mean_error": float((part[m] - part["actual"]).mean())} for m in models}
                        for name, part in parts.items()}
    result["comparisons"] = {f"{c} against {b}": {name: paired(part, c, b) for name, part in parts.items()} for c, b in COMPARISONS}
    result["extra"] = {"S1-800 against T800": paired(same, "S1-800", "T800"), "L0 against T300": paired(same, "L0", "T300"), "Q against L0": paired(same, "Q", "L0")}
    result["slices"] = {label: {"hours": int(mask.sum()), **{f"{c} minus {b}": float((same[c] - same["actual"]).abs()[mask].mean() - (same[b] - same["actual"]).abs()[mask].mean())
                                                              for c, b in COMPARISONS}} for label, mask in slices(same).items()}
    result["windows"] = {f"L0_w{w}": {"mae": float((same[f"L0_w{w}"] - same["actual"]).abs().mean()), "mean_error": float((same[f"L0_w{w}"] - same["actual"]).mean())}
                         for w in lear.LEAR_WINDOWS}
    _, daily = read_lear(out)
    daily = daily.loc[[d for d in daily.index if FIRST <= d <= LAST]]
    band = pd.cut(daily["distance"], [-np.inf, BANDS[0], BANDS[1], np.inf], labels=["low", "middle", "high"], right=False)
    diagnostic = {"definition": "mean over 24 hours and L0's four windows of asinh((price of D-1 - m_w) / s_w), (m_w, s_w) the target scaler of window w for day D",
                  "bands": {name: band_report(same, daily, list(daily.index[band == name])) for name in ("low", "middle", "high")}}
    high_days = list(daily.index[band == "high"])
    runs = spells(high_days)
    longest = max(runs, key=len) if runs else []
    diagnostic["high_without_longest_spell"] = band_report(same, daily, [d for d in high_days if d not in longest])
    diagnostic["verdict"] = recurrence(diagnostic["bands"]["high"], diagnostic["high_without_longest_spell"])
    price_days = list(daily.index[daily["yesterday_mean"] >= HIGH_PRICE])
    price_runs = spells(price_days)
    price_longest = max(price_runs, key=len) if price_runs else []
    diagnostic["yesterday_200_or_more"] = band_report(same, daily, price_days)
    diagnostic["yesterday_200_or_more_without_longest_spell"] = band_report(same, daily, [d for d in price_days if d not in price_longest])
    diagnostic["other_days"] = band_report(same, daily, [d for d in daily.index if d not in price_days])
    diagnostic["distance_summary"] = {"min": float(daily["distance"].min()), "median": float(daily["distance"].median()), "max": float(daily["distance"].max())}
    result["distance"] = diagnostic
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="One development evaluation on July to 10 October 2026 (docs/prereg/july_october_eval_prereg.md)")
    parser.add_argument("stage", choices=["trees", "check-trees", "lear", "scramble", "summary"])
    parser.add_argument("--data", default="data/forecast/inputs.csv")
    parser.add_argument("--frame", default=None, help="frame.csv of the French residual load run (stored selection-window forecasts)")
    parser.add_argument("--gbm-prod", default=None, help="folder of the stored production-settings tree forecasts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", type=lear_recipe._shard)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, filled = load_history(Path(args.data))
    if args.stage == "trees":
        trees(history, out)
    elif args.stage == "check-trees":
        result = tree_stops(history, out, Path(args.frame), Path(args.gbm_prod))
        (out / "tree_stops.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    elif args.stage == "lear":
        lear_stage(history, out, shard=args.shard)
    elif args.stage == "scramble":
        print(json.dumps({k: v for k, v in scramble(history, out).items() if k != "days"}))
    else:
        result = summary(history, filled, out, Path(args.frame), Path(args.gbm_prod))
        (out / "results.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({"stops_pass": result["stops_pass"]}))


if __name__ == "__main__":
    main()
