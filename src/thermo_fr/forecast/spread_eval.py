"""One out-of-sample test of the France minus Germany day-ahead spread forecast with German weather-point inputs.

The frozen plan is docs/prereg/spread_eval_prereg.md. This module is its code and nothing else. Read the plan first.

    python -m thermo_fr.forecast.spread_eval dev      --out <folder>               development window, data to 2026-06-30 only
    python -m thermo_fr.forecast.spread_eval scramble --out <folder>               the two development scramble days
    python -m thermo_fr.forecast.spread_eval stops    --out <folder> --raw <xml>   after the fetch: every stop condition, no error computed
    python -m thermo_fr.forecast.spread_eval summary  --out <folder>               refuses unless every stop condition passed

Three models, each a gradient-boosted regression of the hourly spread at 300 trees of 31 leaves (GBM_PARAMS with REFIT_PARAMS),
refitted at the start of every calendar month on every earlier delivery day from 2021. They are the sets `honest`, `points_hist`
and `ceiling` of scripts/explore_fr_de_spread_points.py, column for column:
  H  the spread's own past, the calendar and the French inputs of the public model (honest_v2)
  G  H plus the German block: wind and solar proxies of D from weather points issued two days ahead, the ENTSO-E load forecast
     of D-1 and D-7, the ENTSO-E wind and solar forecasts of D-1, the proxies of D-1, a German holiday flag, and load of D-7
     minus the proxies. Everything is known at 10:05 Paris on D-1.
  C  G plus the ENTSO-E German load, wind and solar forecasts of D. Not available at 10:05. Diagnostic only.
The German price is the coupled auction (classification sequence 1). The French inputs are data/forecast/inputs.csv with the
wind and solar proxies filled by forward_q.fill_proxy_gaps, the history of the July to October evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import entsoe_rest as er
from . import forward_q as fq
from . import gen_proxy, lear_recipe
from .coverage import check_coverage, input_coverage
from .features import build_features
from .gen_proxy import ProxySpec
from .inputs import load_inputs
from .models import GBM_PARAMS
from .refit import REFIT_PARAMS
from .shape import block_bootstrap_mean
from .solar_proxy import pv_factor
from .timing import delivery_days
from .wind_proxy import power_curve

DEV_FIRST, DEV_LAST = "2024-05-01", "2026-06-30"
TEST_FIRST, TEST_LAST = "2026-07-01", "2026-10-10"  # the French inputs file ends on 2026-10-10
EXPECTED_DAYS, EXPECTED_HOURS = 102, 2448
OVERLAP = ("2026-06-01", "2026-06-30")  # the refetched files are compared with the stored ones on these delivery days
COUPLED = 0.005  # EUR/MWh: below this the two prices are the same
MODELS = ("H", "G", "C")
PARAMS = {**GBM_PARAMS, **REFIT_PARAMS}
MIN_GAIN, MAX_SLICE_LOSS = 0.25, 1.0  # the project's gain rule
ONE_SIDED = 0.95  # one decision, so no Bonferroni division
BLOCK, RESAMPLES, SEED = 7, 5000, 0
TOLERANCE = 1e-9
SCRAMBLE_DEV = ("2025-03-30", "2025-10-26")  # a 23-hour and a 25-hour day
SCRAMBLE_TEST = ("2026-07-01", "2026-08-15", "2026-08-26", "2026-09-30", "2026-10-10")  # 26 August lies in the reconstructed-proxy stretch
PRICE_MATCH, PRICE_SHARE = 0.01, 0.999  # overlap: prices within one cent on at least this share of hours
MW_MATCH, MW_SHARE = 1.0, 0.99  # overlap: forecasts within 1 MW on at least this share of hours
PROXY_FILL_FROM = "2026-08-20"  # from this day the French wind and solar proxies are the reconstructed ones (forward_q.fill_proxy_gaps)
MOVE = 5.0  # EUR/MWh: a day-on-day move of the base spread larger than this counts for the direction figure
TSO_COLUMNS = {"load": "load_fc_de_lu_mw", "onshore": "wind_onshore_fc_de_lu_mw", "offshore": "wind_offshore_fc_de_lu_mw", "solar": "solar_fc_de_lu_mw"}
DE_HOLIDAYS = holidays.Germany(years=range(2021, 2028))


@dataclass
class Data:
    french: pd.DataFrame  # the French inputs table (price, load forecast, weather points, proxies, ...), hourly UTC
    de: pd.Series  # German coupled day-ahead price (sequence 1), hourly UTC
    tso: pd.DataFrame  # ENTSO-E day-ahead forecasts for DE-LU: load, onshore, offshore, solar
    wind_pts: pd.DataFrame  # 100 m wind speed forecasts, as labelled by the provider
    solar_pts: pd.DataFrame  # radiation forecasts, as labelled by the provider (the mean of the hour ending at the label)
    last: str  # last delivery day any table may hold


def _hourly(path, **kw) -> pd.DataFrame:
    t = pd.read_csv(path, **kw)
    t.index = pd.to_datetime(t.pop("timestamp_utc"), utc=True)
    return t.sort_index()


def cut(obj, last: str):
    return obj[delivery_days(obj.index) <= pd.Timestamp(last)]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stored(inputs, neighbours, sequences, wind, solar) -> Data:
    """The stored tables. The German price and the ENTSO-E forecasts stop at DEV_LAST: nothing later is in the project's files.
    The French table runs to TEST_LAST and the weather points further; features() cuts both at the stage's last day."""
    history, filled = fq.fill_proxy_gaps(load_inputs(Path(inputs)))
    for column, info in filled.items():  # the fill may only touch days after the development window
        assert info["first_day"] is None or pd.Timestamp(info["first_day"]) > pd.Timestamp(DEV_LAST), (column, info)
    seq = _hourly(sequences)
    de = cut(seq["de_lu__PT60M__1"].where(seq["de_lu__PT60M__1"].notna(), seq["de_lu__PT15M__1"]), DEV_LAST)
    nf = _hourly(neighbours, usecols=["timestamp_utc", *TSO_COLUMNS.values()])
    tso = cut(nf.rename(columns={v: k for k, v in TSO_COLUMNS.items()})[list(TSO_COLUMNS)], DEV_LAST)
    return Data(french=cut(history, TEST_LAST), de=de, tso=tso, wind_pts=_hourly(wind).asfreq("h"), solar_pts=_hourly(solar).asfreq("h"), last=DEV_LAST)


# ---- the refetched raw files ---------------------------------------------------------------------------------------------

def _parts(raw: Path, stem: str) -> list:
    files = sorted(Path(raw).glob(f"{stem}_*.xml"))
    if not files:
        raise FileNotFoundError(f"no {stem}_*.xml in {raw}")
    return [p for f in files for p in er.parse_document(f.read_bytes())]


def coupled_german_price(parts: list) -> pd.Series:
    """Sequence 1 only, the hourly series where it exists and the mean of the quarter hours elsewhere: the rule of the stored
    extract (de_lu__PT60M__1, then de_lu__PT15M__1)."""
    wanted = er.coupled_sequence(q.sequence for q in parts)
    one = [p for p in parts if p.sequence == wanted]
    s60 = er.combine_resolutions([p for p in one if p.resolution == "PT60M"])
    s15 = er.combine_resolutions([p for p in one if p.resolution != "PT60M"])
    index = s60.index.union(s15.index)
    s60, s15 = s60.reindex(index), s15.reindex(index)
    return s60.where(s60.notna(), s15)


def parse_raw(raw: Path) -> dict:
    """The three refetched items as hourly UTC series, by the rules the stored tables were built with."""
    ws = _parts(raw, "wind_solar_forecast_DE_LU")
    tso = pd.DataFrame({name.replace("wind_", ""): er.combine_resolutions([p for p in ws if p.psr_type == psr]) for psr, name in er.PSR_NAMES.items()})
    tso["load"] = er.combine_resolutions(_parts(raw, "load_forecast_DE_LU"))
    return {"de": coupled_german_price(_parts(raw, "prices_DE_LU")), "tso": tso[list(TSO_COLUMNS)]}


def overlap_check(stored: Data, new: dict) -> dict:
    """Do the refetched series equal the stored ones on the June 2026 delivery days? Read before anything else."""
    out = {}

    def one(name, a, b, tol, share):
        days = delivery_days(a.index)
        a = a[(days >= pd.Timestamp(OVERLAP[0])) & (days <= pd.Timestamp(OVERLAP[1]))].dropna()
        b = b.reindex(a.index)
        close = ((a - b).abs() <= tol) & b.notna()
        out[name] = {"hours": int(len(a)), "missing_in_refetch": int(b.isna().sum()), "share_equal": float(close.mean()), "largest_difference": float((a - b).abs().max()),
                     "required_share": share, "pass": bool(len(a) > 0 and close.mean() >= share)}

    one("German coupled price", stored.de, new["de"], PRICE_MATCH, PRICE_SHARE)
    for column in TSO_COLUMNS:
        one(f"German {column} forecast", stored.tso[column], new["tso"][column], MW_MATCH, MW_SHARE)
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def extend(stored: Data, new: dict, last: str = TEST_LAST) -> Data:
    """Stored values to DEV_LAST, refetched values after it, nothing after `last`."""
    def join(a, b):
        b = b[(delivery_days(b.index) > pd.Timestamp(DEV_LAST)) & (delivery_days(b.index) <= pd.Timestamp(last))]
        return pd.concat([a, b]).sort_index()

    return replace(stored, de=join(stored.de, new["de"]), tso=join(stored.tso, new["tso"]), last=last)


# ---- proxies and features --------------------------------------------------------------------------------------------------

def proxies(data: Data) -> tuple[pd.DataFrame, dict, pd.Timestamp]:
    """German onshore, offshore and solar proxies, the French method: weights refitted each month by non-negative least squares
    on the trailing 365 days of ENTSO-E's day-ahead forecasts of past days, ending two days before the month."""
    wind = cut(data.wind_pts, data.last)
    solar = cut(data.solar_pts.shift(-1), data.last)  # radiation labelled T+1 is the mean of T to T+1: moved to period-start labels
    off = tuple(c for c in wind.columns if "_pt_Off" in c)
    on = tuple(c for c in wind.columns if c not in off)
    specs = {"onshore": (ProxySpec("wind", "de_onshore_px", on, power_curve, Path("unused")), wind),
             "offshore": (ProxySpec("wind", "de_offshore_px", off, power_curve, Path("unused")), wind),
             "solar": (ProxySpec("solar", "de_solar_px", tuple(solar.columns), pv_factor, Path("unused")), solar)}
    px, fits = pd.DataFrame(index=wind.index), {}
    for name, (spec, pts) in specs.items():
        px[name], fits[name] = gen_proxy.rolling_proxy(pts[list(spec.columns)], data.tso[name], spec)
        for f in fits[name]:  # the calibration of a month never reads a day inside or after it
            month = pd.Timestamp(f["month"] + "-01")
            assert pd.Timestamp(f["last_train_day"]) < month and pd.Timestamp(f["to"]).tz_convert("Europe/Paris").tz_localize(None) < month
    px["wind"] = px["onshore"] + px["offshore"]
    return px, fits, max(pd.Timestamp(fits[n][0]["month"] + "-01") for n in fits)


def features(data: Data) -> dict:
    """The frame of delivery hours, the target and the three input tables. Nothing here reads a value of D that is not known at
    10:05 Paris on D-1; the scramble check tests that claim on the output."""
    table_ = build_features(cut(data.french, data.last), "honest_v2")
    frame = table_.info[["delivery_day", "hour", "dow", "holiday", "weather_point_in_time"]].copy()
    frame["fr"], frame["de"] = data.french["price_eur_mwh"].reindex(frame.index), data.de.reindex(frame.index)
    frame["spread"] = frame["fr"] - frame["de"]
    frame = frame[frame["delivery_day"] <= pd.Timestamp(data.last)]
    key = pd.MultiIndex.from_arrays([frame["delivery_day"], frame["hour"]])

    def table(values) -> pd.DataFrame:  # one row per delivery day, one column per Paris hour
        t = pd.DataFrame({"d": frame["delivery_day"].to_numpy(), "h": frame["hour"].to_numpy(), "v": np.asarray(values)})
        return t.pivot_table(index="d", columns="h", values="v", dropna=False).sort_index().asfreq("D")

    def at(pivot) -> np.ndarray:  # the pivot's value at each row's (delivery day, hour)
        return pivot.stack(future_stack=True).reindex(key).to_numpy()

    def day_lag(series, k) -> np.ndarray:  # the value at the same Paris hour k delivery days earlier
        return at(table(series.reindex(frame.index)).shift(k))

    piv, dep = table(frame["spread"]), table(frame["de"])
    lags = pd.DataFrame(index=frame.index)
    for k in (1, 2, 7):
        lags[f"s_l{k}"] = at(piv.shift(k))
    lags["s_m7"] = at(piv.shift(1).rolling(7, min_periods=4).mean())
    daily, share, absd = piv.mean(axis=1), (piv.abs() > COUPLED).mean(axis=1), piv.abs().mean(axis=1)
    for name, s in (("s_base_l1", daily.shift(1)), ("s_base_m7", daily.shift(1).rolling(7, min_periods=4).mean()), ("decoupled_share_l1", share.shift(1)), ("s_abs_l1", absd.shift(1))):
        lags[name] = frame["delivery_day"].map(s).to_numpy()
    lags["de_l1"] = at(dep.shift(1))
    lags["hour"], lags["dow"], lags["holiday"] = frame["hour"], frame["dow"], frame["holiday"].astype(float)
    french = table_.X.reindex(frame.index)
    french = french[[c for c in french.columns if c not in lags.columns]]  # hour and calendar columns are already in the lags

    px, fits, first_month = proxies(data)
    tso_wind = data.tso["onshore"].fillna(0) + data.tso["offshore"].fillna(0)
    # days before the proxies exist: the ENTSO-E forecast of that day stands in. Those are training days only (asserted in scored()).
    early = delivery_days(px.index) < first_month
    wind = px["wind"].where(~early, tso_wind.reindex(px.index))
    solar = px["solar"].where(~early, data.tso["solar"].reindex(px.index))
    wind = pd.concat([tso_wind[tso_wind.index < wind.index[0]], wind])
    solar = pd.concat([data.tso["solar"][data.tso.index < solar.index[0]], solar])
    g = pd.DataFrame(index=frame.index)
    g["de_wind_px"], g["de_solar_px"] = wind.reindex(frame.index), solar.reindex(frame.index)
    g["de_load_l1"], g["de_load_l7"] = day_lag(data.tso["load"], 1), day_lag(data.tso["load"], 7)
    g["de_wind_fc_l1"], g["de_solar_fc_l1"] = day_lag(tso_wind, 1), day_lag(data.tso["solar"], 1)
    g["de_wind_px_l1"], g["de_solar_px_l1"] = day_lag(wind, 1), day_lag(solar, 1)
    g["de_holiday"] = frame["delivery_day"].map(lambda d: float(d in DE_HOLIDAYS)).to_numpy()
    g["de_residual_px"] = g["de_load_l7"] - g["de_wind_px"] - g["de_solar_px"]
    on, off = data.tso["onshore"].reindex(frame.index), data.tso["offshore"].reindex(frame.index)
    ideal = pd.DataFrame({"de_load_fc": data.tso["load"].reindex(frame.index), "de_solar_fc": data.tso["solar"].reindex(frame.index), "de_wind_fc": on.fillna(0) + off.fillna(0)})
    ideal["de_residual_fc"] = ideal["de_load_fc"] - ideal["de_solar_fc"] - ideal["de_wind_fc"]
    g_load = g.assign(de_load_fc=ideal["de_load_fc"])
    g_load["de_residual_px"] = g_load["de_load_fc"] - g_load["de_wind_px"] - g_load["de_solar_px"]
    c = pd.concat([g_load, ideal[["de_solar_fc", "de_wind_fc", "de_residual_fc"]]], axis=1)
    real_px = pd.DataFrame({"wind": px["wind"].where(~early).reindex(frame.index), "solar": px["solar"].where(~early).reindex(frame.index)})
    return {"frame": frame, "y": frame["spread"], "sets": {"H": pd.concat([lags, french], axis=1), "G": pd.concat([lags, french, g], axis=1), "C": pd.concat([lags, french, c], axis=1)},
            "first_month": first_month, "fits": fits, "px": real_px, "yesterday": lags["s_l1"]}


def fit_months(feat: dict, months, models=MODELS, log=lambda *_: None) -> pd.DataFrame:
    """Walk-forward forecasts: the model of a month is fitted on every earlier delivery day and applied to the month."""
    frame, y = feat["frame"], feat["y"]
    pred = pd.DataFrame(index=frame.index, columns=list(models), dtype=float)
    for start in pd.DatetimeIndex(months):
        if start < feat["first_month"]:
            raise ValueError(f"{start:%Y-%m} is before the German proxies exist")
        train = np.asarray((frame["delivery_day"] < start) & y.notna())
        test = np.asarray((frame["delivery_day"] >= start) & (frame["delivery_day"] < start + pd.DateOffset(months=1)))
        if not test.any():
            continue
        for name in models:
            X = feat["sets"][name]
            pred.loc[test, name] = lgb.LGBMRegressor(**PARAMS).fit(X[train], y[train]).predict(X[test])
        log(f"fitted {start:%Y-%m}")
    return pred


def scored(feat: dict, pred: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """The scored hours of a window: strict hours with a realised spread, yesterday's spread, both proxies and all three forecasts."""
    frame = feat["frame"]
    rows = ((frame["delivery_day"] >= pd.Timestamp(first)) & (frame["delivery_day"] <= pd.Timestamp(last)) & frame["weather_point_in_time"].astype(bool) & feat["y"].notna() & feat["yesterday"].notna()
            & feat["px"]["wind"].notna() & feat["px"]["solar"].notna() & pred[list(MODELS)].notna().all(axis=1))
    g = feat["sets"]["G"]
    assert (g.loc[rows, ["de_wind_px", "de_solar_px"]].to_numpy() == feat["px"].loc[rows, ["wind", "solar"]].to_numpy()).all()  # no scored hour uses a stand-in
    out = frame.loc[rows, ["delivery_day", "hour", "dow", "holiday", "spread"]].copy()
    out["de_holiday"] = g.loc[rows, "de_holiday"]
    out["wind_px"], out["solar_px"] = feat["px"].loc[rows, "wind"], feat["px"].loc[rows, "solar"]
    out["yesterday"] = feat["yesterday"][rows]
    for name in MODELS:
        out[name] = pred.loc[rows, name]
    return out


def months_of(first: str, last: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(first).replace(day=1), last, freq="MS")


# ---- the scramble check ----------------------------------------------------------------------------------------------------

def scrambled(data: Data, day: str, seed: int = 0) -> Data:
    """Every value that cannot be known at 10:05 Paris on the day before `day`, replaced by noise.

    Not known: any German price of `day` or later (the auction of `day` clears after the gate); any German ENTSO-E forecast of
    `day` or later (published on the eve, after the gate); every French input the project's own rule scrambles
    (lear_recipe.scramble_from, which includes the French price of `day`); any German weather point labelled after the last
    hour the features of `day` read. For
    wind that is the last hour of `day`; for radiation it is one label later, because the label of the first hour of the next
    day is the mean of the last hour of `day`. Those points were issued two days before their hour, so they are known."""
    rng = np.random.default_rng(seed)
    d = pd.Timestamp(day)

    def noise(obj, mask, low, high):
        obj = obj.copy()
        if isinstance(obj, pd.Series):
            obj[mask] = rng.uniform(low, high, int(mask.sum()))
        else:
            obj.loc[mask, :] = rng.uniform(low, high, (int(mask.sum()), obj.shape[1]))
        return obj

    return replace(data, french=lear_recipe.scramble_from(data.french, d, seed=seed), de=noise(data.de, delivery_days(data.de.index) >= d, -400, 900),
                   tso=noise(data.tso, delivery_days(data.tso.index) >= d, 0, 70000), wind_pts=noise(data.wind_pts, delivery_days(data.wind_pts.index) > d, 0, 30),
                   solar_pts=noise(data.solar_pts, delivery_days(data.solar_pts.index - pd.Timedelta(hours=1)) > d, 0, 900))


def scramble_check(data: Data, days, log=print) -> dict:
    """The forecasts of each day, from the data as they are and from the data with everything unknown at the gate scrambled."""
    out, base = {}, {}
    for day in days:
        month = pd.Timestamp(day).replace(day=1)
        if month not in base:
            feat = features(data)
            base[month] = (feat, fit_months(feat, [month]))
        feat, pred = base[month]
        rows = np.asarray(feat["frame"]["delivery_day"] == pd.Timestamp(day))
        feat_s = features(scrambled(data, day))
        pred_s = fit_months(feat_s, [month])
        rows_s = np.asarray(feat_s["frame"]["delivery_day"] == pd.Timestamp(day))
        a, b = pred.loc[rows, list(MODELS)], pred_s.loc[rows_s, list(MODELS)]
        same_inputs = all(feat["sets"][m].loc[rows].equals(feat_s["sets"][m].loc[rows_s]) for m in ("H", "G"))
        diff = float((a - b.reindex(a.index)).abs().max().max()) if len(a) and a.notna().all().all() else float("nan")
        changed = float((feat["y"][np.asarray(feat["frame"]["delivery_day"] >= pd.Timestamp(day))] - feat_s["y"][np.asarray(feat_s["frame"]["delivery_day"] >= pd.Timestamp(day))]).abs().min())
        diff_c = float((a["C"] - b["C"].reindex(a.index)).abs().max())
        lg = float((a[["H", "G"]] - b[["H", "G"]].reindex(a.index)).abs().max().max())
        out[day] = {"hours": int(rows.sum()), "largest_difference_H_G": lg, "inputs_identical_H_G": bool(same_inputs), "C_moved_by": diff_c,
                    "smallest_change_of_a_scrambled_target": changed, "pass": bool(rows.sum() >= 23 and lg <= TOLERANCE and same_inputs and changed > 0)}
        log(f"scramble {day}: H and G differ by at most {lg:.3g} over {int(rows.sum())} hours; C, which reads forecasts of D, moved by {diff_c:.3g}; overall {diff:.3g}")
    out["pass"] = all(v["pass"] for v in out.values())
    return out


# ---- scoring ---------------------------------------------------------------------------------------------------------------

def daily_error(frame: pd.DataFrame, name: str) -> pd.Series:
    return (frame[name] - frame["spread"]).abs().groupby(frame["delivery_day"]).mean().sort_index()


def paired(frame: pd.DataFrame, candidate: str, base: str) -> dict:
    """Candidate minus base by delivery day: two-sided 95% interval and the one-sided upper end at ONE_SIDED."""
    diff = (daily_error(frame, candidate) - daily_error(frame, base)).to_numpy()
    if len(diff) == 0:
        return {"days": 0}
    lo, hi = block_bootstrap_mean(diff, block=BLOCK, resamples=RESAMPLES, seed=SEED, level=0.95)
    lower, upper = block_bootstrap_mean(diff, block=BLOCK, resamples=RESAMPLES, seed=SEED, level=1 - 2 * (1 - ONE_SIDED))
    return {"days": int(len(diff)), "candidate_mae": float((frame[candidate] - frame["spread"]).abs().mean()), "base_mae": float((frame[base] - frame["spread"]).abs().mean()),
            "mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "one_sided_upper": float(upper), "mirrored_lower": float(lower), "share_days_better": float((diff < 0).mean())}


def slices(frame: pd.DataFrame) -> dict:
    by_day = frame.groupby("delivery_day").agg(wind=("wind_px", "mean"), solar=("solar_px", "mean"))
    return {"weekends": frame["dow"] >= 5, "holidays (French or German)": frame["holiday"].astype(bool) | frame["de_holiday"].astype(bool),
            "windiest 10% of days (German wind proxy)": frame["delivery_day"].isin(by_day.index[by_day["wind"] >= by_day["wind"].quantile(0.9)]),
            "sunniest 10% of days (German solar proxy)": frame["delivery_day"].isin(by_day.index[by_day["solar"] >= by_day["solar"].quantile(0.9)]),
            "top 5% of hours by absolute spread": frame["spread"].abs() >= frame["spread"].abs().quantile(0.95)}


def decide(frame: pd.DataFrame) -> dict:
    """The gain rule on G against H: a gain of at least MIN_GAIN, no slice worse by more than MAX_SLICE_LOSS, and the one-sided
    upper end of the paired daily difference below zero."""
    p = paired(frame, "G", "H")
    gain = p["base_mae"] - p["candidate_mae"]
    sl = {}
    for name, mask in slices(frame).items():
        s = frame[np.asarray(mask)]
        sl[name] = {"hours": int(len(s)), "days": int(s["delivery_day"].nunique()), "H": float((s["H"] - s["spread"]).abs().mean()) if len(s) else None,
                    "G": float((s["G"] - s["spread"]).abs().mean()) if len(s) else None}
        sl[name]["G_minus_H"] = None if not len(s) else sl[name]["G"] - sl[name]["H"]
    worst = max((v["G_minus_H"] for v in sl.values() if v["G_minus_H"] is not None), default=float("nan"))
    conditions = {"gain_at_least_0.25": bool(gain >= MIN_GAIN), "no_slice_worse_by_more_than_1.0": bool(worst <= MAX_SLICE_LOSS), "one_sided_upper_below_zero": bool(p["one_sided_upper"] < 0)}
    verdict = "passes" if all(conditions.values()) else "contradicts" if p["mirrored_lower"] > 0 else "fails"
    return {"paired": p, "gain": float(gain), "slices": sl, "worst_slice": float(worst), "conditions": conditions, "verdict": verdict}


def diagnostics(frame: pd.DataFrame) -> dict:
    """Reported with the decision, never part of it."""
    mae = {name: float((frame[name] - frame["spread"]).abs().mean()) for name in ("yesterday", *MODELS)}
    base = frame.groupby("delivery_day")[["spread", "yesterday", *MODELS]].mean()
    move = base["spread"] - base["yesterday"]
    big = move.abs() > MOVE
    direction = {name: float((np.sign(base.loc[big, name] - base.loc[big, "yesterday"]) == np.sign(move[big])).mean()) if big.any() else None for name in MODELS}
    month = frame["delivery_day"].dt.strftime("%Y-%m")
    by_month = {m: {"days": int(s["delivery_day"].nunique()), **{name: float((s[name] - s["spread"]).abs().mean()) for name in ("yesterday", *MODELS)}} for m, s in frame.groupby(month)}
    period = np.where(frame["delivery_day"] < pd.Timestamp(PROXY_FILL_FROM), f"before {PROXY_FILL_FROM}", f"from {PROXY_FILL_FROM}, French proxies reconstructed")
    by_period = {k: {"days": int(s["delivery_day"].nunique()), **{name: float((s[name] - s["spread"]).abs().mean()) for name in ("yesterday", *MODELS)}} for k, s in frame.groupby(period)}
    ceiling = mae["H"] - mae["C"]
    return {"hours": int(len(frame)), "days": int(frame["delivery_day"].nunique()), "mean_spread": float(frame["spread"].mean()), "mean_absolute_spread": float(frame["spread"].abs().mean()),
            "mae": mae, "base_spread_mae": {name: float((base[name] - base["spread"]).abs().mean()) for name in ("yesterday", *MODELS)},
            "direction_right": direction, "days_with_move_above_5": int(big.sum()), "by_month": by_month, "by_french_proxy_period": by_period,
            "G_against_yesterday": paired(frame, "G", "yesterday"), "C_against_G": paired(frame, "C", "G"),
            "share_of_ceiling_gain_recovered": float((mae["H"] - mae["G"]) / ceiling) if ceiling > 0 else None}


# ---- stages ----------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def stage_dev(data: Data, out: Path, log=print) -> dict:
    feat = features(data)
    pred = fit_months(feat, months_of(DEV_FIRST, DEV_LAST), log=log)
    frame = scored(feat, pred, DEV_FIRST, DEV_LAST)
    frame.to_csv(out / "dev_predictions.csv.gz")  # working file, never published
    result = {"window": [DEV_FIRST, DEV_LAST], "first_proxy_month": str(feat["first_month"].date()), "decision_rule_on_development": decide(frame), "diagnostics": diagnostics(frame),
              "predictions_sha256": sha256(out / "dev_predictions.csv.gz")}
    _write(out / "dev.json", result)
    return result


def stage_stops(stored: Data, raw: Path, out: Path, log=print) -> dict:
    """Every stop condition. Forecasts of the test window are produced and written, but no error against a realised spread of
    that window is computed here."""
    checks = {"raw_files": {f.name: sha256(f) for f in sorted(Path(raw).glob("*.xml"))}}
    new = parse_raw(raw)
    checks["overlap"] = overlap_check(stored, new)
    _write(out / "stop_checks.json", checks)
    if not checks["overlap"]["pass"]:
        log("STOP: the refetched series do not match the stored ones on the overlap"); return checks
    data = extend(stored, new)
    feat = features(data)
    window = (feat["frame"]["delivery_day"] >= pd.Timestamp(TEST_FIRST)) & (feat["frame"]["delivery_day"] <= pd.Timestamp(TEST_LAST))
    german = [c for c in feat["sets"]["C"].columns if c.startswith("de_")]
    report = input_coverage(pd.concat([feat["sets"]["C"][german], feat["sets"]["H"][["s_l1", "s_l7"]], feat["frame"][["fr", "de"]]], axis=1), feat["frame"]["delivery_day"], TEST_FIRST, TEST_LAST)
    french_report = input_coverage(feat["sets"]["H"], feat["frame"]["delivery_day"], TEST_FIRST, TEST_LAST)
    dev_report = input_coverage(feat["sets"]["H"], feat["frame"]["delivery_day"], "2025-07-01", "2025-10-10")
    checks["coverage"] = {"days_in_window": int(feat["frame"].loc[window, "delivery_day"].nunique()), "hours_in_window": int(window.sum()),
                          "hours_missing_by_input": {k: int(v) for k, v in report["hours_missing"].items() if v}, "expected_days": EXPECTED_DAYS, "expected_hours": EXPECTED_HOURS,
                          "french_inputs_with_a_fully_missing_day": {k: int(v) for k, v in french_report["days_fully_missing"].items() if v},
                          "the_same_inputs_a_year_earlier": {k: int(v) for k, v in dev_report["days_fully_missing"].items() if v}}
    try:
        check_coverage(report)
        # a French input may be missing in the window only if it was missing in the same way a year earlier (inputs absent by design)
        new_holes = set(checks["coverage"]["french_inputs_with_a_fully_missing_day"]) - set(checks["coverage"]["the_same_inputs_a_year_earlier"])
        checks["coverage"]["french_inputs_newly_missing"] = sorted(new_holes)
        checks["coverage"]["pass"] = checks["coverage"]["days_in_window"] == EXPECTED_DAYS and not new_holes
    except Exception as exc:  # CoverageError
        checks["coverage"].update({"pass": False, "error": str(exc)})
    _write(out / "stop_checks.json", checks)
    if not checks["coverage"]["pass"]:
        log("STOP: an input is missing inside the test window"); return checks
    pred = fit_months(feat, months_of(DEV_FIRST, TEST_LAST), log=log)
    dev = scored(feat, pred, DEV_FIRST, DEV_LAST)
    kept = pd.read_csv(out / "dev_predictions.csv.gz", index_col=0)
    kept.index = pd.to_datetime(kept.index, utc=True)
    common = kept.index.intersection(dev.index)
    largest = float((dev.loc[common, list(MODELS)] - kept.loc[common, list(MODELS)]).abs().max().max())
    checks["development_forecasts_reproduce"] = {"stored_hours": int(len(kept)), "hours_compared": int(len(common)), "largest_difference": largest, "tolerance": TOLERANCE,
                                                 "pass": bool(len(common) == len(kept) and largest <= TOLERANCE)}
    reference = json.loads((out / "dev.json").read_text())["diagnostics"]["mae"]
    again = {name: float((dev.loc[common, name] - dev.loc[common, "spread"]).abs().mean()) for name in ("yesterday", *MODELS)}
    checks["reference_figures_reproduce"] = {"logged": reference, "recomputed": again, "pass": bool(all(abs(reference[k] - again[k]) < 0.005 for k in reference))}
    checks["scramble"] = scramble_check(data, SCRAMBLE_TEST, log=log)
    test = scored(feat, pred, TEST_FIRST, TEST_LAST)
    checks["test_hours_scored"] = {"hours": int(len(test)), "days": int(test["delivery_day"].nunique()), "expected_hours": EXPECTED_HOURS, "expected_days": EXPECTED_DAYS,
                                   "pass": bool(test["delivery_day"].nunique() == EXPECTED_DAYS and len(test) >= EXPECTED_HOURS - 24)}
    test.to_csv(out / "test_predictions.csv.gz")  # working file, never published
    checks["test_predictions_sha256"] = sha256(out / "test_predictions.csv.gz")
    checks["all_pass"] = all(checks[k]["pass"] for k in ("overlap", "coverage", "development_forecasts_reproduce", "reference_figures_reproduce", "scramble", "test_hours_scored"))
    _write(out / "stop_checks.json", checks)
    log("stop conditions: " + ("all pass" if checks["all_pass"] else "FAILED, nothing is scored"))
    return checks


def stage_summary(out: Path) -> dict:
    checks = json.loads((out / "stop_checks.json").read_text())
    if not checks.get("all_pass"):
        raise SystemExit("a stop condition failed or was not run: nothing is scored")
    if sha256(out / "test_predictions.csv.gz") != checks["test_predictions_sha256"]:
        raise SystemExit("the forecasts on disk are not the ones the stop conditions were run on")
    frame = pd.read_csv(out / "test_predictions.csv.gz", index_col=0, parse_dates=["delivery_day"])
    result = {"window": [TEST_FIRST, TEST_LAST], "decision": decide(frame), "diagnostics": diagnostics(frame)}
    _write(out / "test_results.json", result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Out-of-sample test of the France to Germany spread forecast (docs/prereg/spread_eval_prereg.md)")
    parser.add_argument("stage", choices=["dev", "scramble", "stops", "summary"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs", default="data/forecast/inputs.csv")
    parser.add_argument("--neighbours", default="data/forecast/neighbour_fundamentals.csv")
    parser.add_argument("--sequences", help="per-sequence price extract (neighbour_sequences.csv.gz)")
    parser.add_argument("--wind", help="german_wind_points.csv.gz")
    parser.add_argument("--solar", help="german_solar_points.csv.gz")
    parser.add_argument("--raw", help="folder of the refetched ENTSO-E XML files (scripts/fetch_spread_test_raw.py)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        print(json.dumps(stage_summary(out), indent=1, default=str)); return
    stored = load_stored(args.inputs, args.neighbours, args.sequences, args.wind, args.solar)
    if args.stage == "dev":
        result = stage_dev(stored, out)
        print(json.dumps({k: result[k] for k in ("first_proxy_month", "decision_rule_on_development")}, indent=1)); print(json.dumps(result["diagnostics"], indent=1))
    elif args.stage == "scramble":
        result = scramble_check(stored, SCRAMBLE_DEV)
        _write(out / "scramble_dev.json", result); print(json.dumps(result, indent=1))
    else:
        stage_stops(stored, Path(args.raw), out)


if __name__ == "__main__":
    main()
