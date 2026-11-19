"""Exploratory: the France minus Germany day-ahead spread with German wind and solar forecasts built from weather points.
No registration, no decision. Every delivery day after 2026-06-30 is dropped on load.

    python scripts/explore_fr_de_spread_points.py data/forecast/inputs.csv data/forecast/neighbour_fundamentals.csv \
        <per-sequence extract> german_wind_points.csv.gz german_solar_points.csv.gz <out folder>

The earlier exploration (explore_fr_de_spread.py) showed that the ENTSO-E German load, wind and solar forecasts of the delivery
day cut the error of the spread forecast, but those are published after 10:05 Paris on D-1. This script replaces them with
inputs that are known at 10:05 and measures how much of that gain is left.

German proxies, built like the French ones (gen_proxy.rolling_proxy):
  points   Open-Meteo forecasts as issued two days ahead (scripts/fetch_german_points.py): 100 m wind speed at 14 onshore and
           3 offshore points, shortwave radiation at 13 points.
  curves   the French ones: cubic power curve (3, 12, 25 m/s) and radiation over 1,000 W/m2.
  labels   Open-Meteo radiation labelled T is the mean of the hour ending at T, so it is moved one hour earlier to match
           ENTSO-E's period-start labels. Decided from the documentation before any fit. Wind speed is left as labelled.
  target   the ENTSO-E day-ahead forecast of German onshore wind, offshore wind and solar on PAST days (no German actual
           generation is in the project's data). Onshore and offshore are fitted separately.
  refit    non-negative least squares at the start of every month on the trailing 365 days ending two days before the month,
           at least 60 days. The day-ahead forecast of a day is published the evening before it, so the last day used is
           public well before the first gate of the month.

Input sets, each refitted at the start of every month on all earlier days from 2021:
  lags         the spread's own past and the calendar
  honest       lags plus the French inputs of the public model (honest_v2)
  points       honest plus the German block: wind and solar proxies of D, the ENTSO-E load forecast of D-1 and D-7 at the
               same hour, the ENTSO-E wind and solar forecasts of D-1, the proxies of D-1, a German national holiday flag,
               and load of D-7 minus the proxies. All known at 10:05 Paris on D-1. Before the weather archive starts the
               proxy columns are empty, so the model has little history to learn them from in 2024.
  points_hist  the same, but on days before the proxies exist the proxy columns hold the ENTSO-E forecast of that day. Those
               days are only ever training days here (scoring starts when the proxies exist), and by then their forecasts
               are long public. Chosen before any result was seen. This is the main attainable set.
  points_load  points_hist plus the ENTSO-E German load forecast of D. NOT attainable: it isolates what the load forecast adds.
  idealised    honest plus the ENTSO-E German load, wind and solar forecasts of D. NOT attainable.
  ceiling      points_load plus the ENTSO-E German wind and solar forecasts of D: everything in the attainable set and everything
               in the idealised one. NOT attainable. Added after the reviewer noted that idealised lacks the lags of the German
               block, so it is not a true upper bound for it.
The lags of the German block are taken by delivery day and Paris hour. A first run lagged by 24 hours in UTC, which on the two
25-hour days put the last hour's lag inside D (2 scored hours of 18,981, found by the reviewer); its figures are superseded.
All sets are scored on the same hours, from the first month in which every German proxy exists.
The German price is the coupled auction (classification sequence 1)."""
import json
import sys
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from thermo_fr.forecast import gen_proxy, lear_recipe
from thermo_fr.forecast.features import build_features
from thermo_fr.forecast.gen_proxy import ProxySpec
from thermo_fr.forecast.inputs import load_inputs
from thermo_fr.forecast.models import GBM_PARAMS
from thermo_fr.forecast.refit import REFIT_PARAMS
from thermo_fr.forecast.shape import block_bootstrap_mean
from thermo_fr.forecast.solar_proxy import pv_factor
from thermo_fr.forecast.timing import delivery_days
from thermo_fr.forecast.wind_proxy import power_curve

LAST = "2026-06-30"
COUPLED = 0.005  # EUR/MWh: below this the two prices are the same
out = Path(sys.argv[6]); out.mkdir(parents=True, exist_ok=True)


def read_hourly(path, **kw):
    t = pd.read_csv(path, **kw); t.index = pd.to_datetime(t.pop("timestamp_utc"), utc=True)
    return t[delivery_days(t.index) <= pd.Timestamp(LAST)].sort_index()


# ---- 1. German proxies -------------------------------------------------------------------------------------------------
nf_all = read_hourly(sys.argv[2])
tso = pd.DataFrame({"onshore": nf_all["wind_onshore_fc_de_lu_mw"], "offshore": nf_all["wind_offshore_fc_de_lu_mw"], "solar": nf_all["solar_fc_de_lu_mw"],
                    "load": nf_all["load_fc_de_lu_mw"]})
wind_pts = read_hourly(sys.argv[4]).asfreq("h"); solar_pts = read_hourly(sys.argv[5]).asfreq("h").shift(-1)
off_cols = tuple(c for c in wind_pts.columns if "_pt_Off" in c); on_cols = tuple(c for c in wind_pts.columns if c not in off_cols)
specs = {"onshore": (ProxySpec("wind", "de_onshore_px", on_cols, power_curve, Path("unused")), wind_pts),
         "offshore": (ProxySpec("wind", "de_offshore_px", off_cols, power_curve, Path("unused")), wind_pts),
         "solar": (ProxySpec("solar", "de_solar_px", tuple(solar_pts.columns), pv_factor, Path("unused")), solar_pts)}
px, fits = pd.DataFrame(index=wind_pts.index), {}
for name, (spec, pts) in specs.items():
    px[name], fits[name] = gen_proxy.rolling_proxy(pts[list(spec.columns)], tso[name], spec)
    for f in fits[name]:  # the calibration of a month never reads a day inside or after it
        assert pd.Timestamp(f["last_train_day"]) < pd.Timestamp(f["month"] + "-01") and pd.Timestamp(f["to"]).tz_convert("Europe/Paris").tz_localize(None) < pd.Timestamp(f["month"] + "-01")
px["wind"] = px["onshore"] + px["offshore"]
first_month = max(pd.Timestamp(fits[n][0]["month"] + "-01") for n in fits)
print("GERMAN PROXIES (target: ENTSO-E day-ahead forecast of past days)")
for name in fits:
    print("  %-8s points %2d | first month %s | months %d | effective capacity first %s MW, last %s MW" % (
        name, len(specs[name][0].columns), fits[name][0]["month"], len(fits[name]), f"{fits[name][0]['capacity_mw']:,.0f}", f"{fits[name][-1]['capacity_mw']:,.0f}"))
q = px.join(tso.rename(columns=lambda c: c + "_fc"), how="inner"); q = q[delivery_days(q.index) >= first_month].dropna()
q["wind_fc"] = q["onshore_fc"] + q["offshore_fc"]; q["res"] = q["wind"] + q["solar"]; q["res_fc"] = q["wind_fc"] + q["solar_fc"]
qd = q.groupby(delivery_days(q.index)).mean()
print("  walk-forward fit against the ENTSO-E forecast of the same hour, %s to %s, %s hours" % (first_month.date(), LAST, f"{len(q):,}"))
for a, b in (("onshore", "onshore_fc"), ("offshore", "offshore_fc"), ("wind", "wind_fc"), ("solar", "solar_fc"), ("res", "res_fc")):
    e = q[a] - q[b]
    print("    %-9s mean %6.0f MW | MAE %5.0f MW (%2.0f%% of mean) | bias %+5.0f | hourly corr %.3f | daily-mean corr %.3f" % (
        a, q[b].mean(), e.abs().mean(), 100 * e.abs().mean() / q[b].mean(), e.mean(), np.corrcoef(q[a], q[b])[0, 1], np.corrcoef(qd[a], qd[b])[0, 1]))
pers = (q["res_fc"].shift(24) - q["res_fc"]).abs().mean()
print("    for scale: yesterday's ENTSO-E wind plus solar forecast as a forecast of today's has MAE %.0f MW" % pers)

# ---- 2. Spread, lags and input sets ------------------------------------------------------------------------------------
base = lear_recipe.selection_inputs(load_inputs(Path(sys.argv[1])))
table = build_features(base, "honest_v2")
seq = read_hourly(sys.argv[3])
de = seq["de_lu__PT60M__1"].where(seq["de_lu__PT60M__1"].notna(), seq["de_lu__PT15M__1"])
frame = table.info[["delivery_day", "hour", "dow", "holiday", "weather_point_in_time"]].copy()
frame["fr"], frame["de"] = base["price_eur_mwh"].reindex(frame.index), de.reindex(frame.index)
frame["spread"] = frame["fr"] - frame["de"]
frame = frame[frame["delivery_day"] <= LAST]
piv = frame.pivot_table(index="delivery_day", columns="hour", values="spread").sort_index().asfreq("D")
dep = frame.pivot_table(index="delivery_day", columns="hour", values="de").sort_index().asfreq("D")


def at(pivot):  # value of the pivot at each row's (delivery_day, hour)
    s = pivot.stack(future_stack=True); return s.reindex(pd.MultiIndex.from_arrays([frame["delivery_day"], frame["hour"]])).to_numpy()


lags = pd.DataFrame(index=frame.index)
for k in (1, 2, 7):
    lags[f"s_l{k}"] = at(piv.shift(k))
lags["s_m7"] = at(piv.shift(1).rolling(7, min_periods=4).mean())
daily = piv.mean(axis=1); share = (piv.abs() > COUPLED).mean(axis=1); absd = piv.abs().mean(axis=1)
for name, s in (("s_base_l1", daily.shift(1)), ("s_base_m7", daily.shift(1).rolling(7, min_periods=4).mean()), ("decoupled_share_l1", share.shift(1)), ("s_abs_l1", absd.shift(1))):
    lags[name] = frame["delivery_day"].map(s).to_numpy()
lags["de_l1"] = at(dep.shift(1)); lags["hour"] = frame["hour"]; lags["dow"] = frame["dow"]; lags["holiday"] = frame["holiday"].astype(float)
french = table.X.reindex(frame.index); french = french[[c for c in french.columns if c not in lags.columns]]


def day_lag(s, k):  # the value at the same Paris hour k delivery days earlier (a 24-hour lag in UTC reaches into D on 25-hour days)
    p = pd.DataFrame({"d": frame["delivery_day"], "h": frame["hour"], "v": s.reindex(frame.index)}).pivot_table(index="d", columns="h", values="v", dropna=False).sort_index().asfreq("D")
    return at(p.shift(k))


tso_wind = tso["onshore"].fillna(0) + tso["offshore"].fillna(0)
de_holidays = holidays.Germany(years=range(2021, 2027))


def german_block(fill_history: bool) -> pd.DataFrame:
    wind, solar = px["wind"].copy(), px["solar"].copy()
    if fill_history:  # days before the proxies exist: the ENTSO-E forecast of that day stands in (training days only)
        early = delivery_days(wind.index) < first_month
        wind = wind.where(~early, tso_wind.reindex(wind.index)); solar = solar.where(~early, tso["solar"].reindex(solar.index))
        wind = pd.concat([tso_wind[tso_wind.index < wind.index[0]], wind]); solar = pd.concat([tso["solar"][tso.index < solar.index[0]], solar])
    else:
        early = delivery_days(wind.index) < first_month; wind, solar = wind.where(~early), solar.where(~early)
    g = pd.DataFrame(index=frame.index)
    g["de_wind_px"], g["de_solar_px"] = wind.reindex(frame.index), solar.reindex(frame.index)
    g["de_load_l1"], g["de_load_l7"] = day_lag(tso["load"], 1), day_lag(tso["load"], 7)
    g["de_wind_fc_l1"], g["de_solar_fc_l1"] = day_lag(tso_wind, 1), day_lag(tso["solar"], 1)
    g["de_wind_px_l1"], g["de_solar_px_l1"] = day_lag(wind, 1), day_lag(solar, 1)
    g["de_holiday"] = frame["delivery_day"].map(lambda d: float(d in de_holidays)).to_numpy()
    g["de_residual_px"] = g["de_load_l7"] - g["de_wind_px"] - g["de_solar_px"]
    return g


nf = nf_all.reindex(frame.index)
ideal = pd.DataFrame({"de_load_fc": nf["load_fc_de_lu_mw"], "de_solar_fc": nf["solar_fc_de_lu_mw"], "de_wind_fc": nf["wind_onshore_fc_de_lu_mw"].fillna(0) + nf["wind_offshore_fc_de_lu_mw"].fillna(0)})
ideal["de_residual_fc"] = ideal["de_load_fc"] - ideal["de_solar_fc"] - ideal["de_wind_fc"]
g_plain, g_hist = german_block(False), german_block(True)
g_load = g_hist.assign(de_load_fc=ideal["de_load_fc"]); g_load["de_residual_px"] = g_load["de_load_fc"] - g_load["de_wind_px"] - g_load["de_solar_px"]
g_ceiling = pd.concat([g_load, ideal[["de_solar_fc", "de_wind_fc", "de_residual_fc"]]], axis=1)
sets = {"lags": lags, "honest": pd.concat([lags, french], axis=1), "points": pd.concat([lags, french, g_plain], axis=1),
        "points_hist": pd.concat([lags, french, g_hist], axis=1), "points_load": pd.concat([lags, french, g_load], axis=1),
        "idealised": pd.concat([lags, french, ideal], axis=1), "ceiling": pd.concat([lags, french, g_ceiling], axis=1)}
ATTAINABLE = ("lags", "honest", "points", "points_hist")
params = {**GBM_PARAMS, **REFIT_PARAMS}
y = frame["spread"]; dec = (y.abs() > COUPLED).astype(float)
pred = pd.DataFrame(index=frame.index); prob = pd.DataFrame(index=frame.index)
for start in pd.date_range(first_month, LAST, freq="MS"):
    train = np.asarray((frame["delivery_day"] < start) & y.notna()); test = np.asarray((frame["delivery_day"] >= start) & (frame["delivery_day"] < start + pd.DateOffset(months=1)))
    for name, X in sets.items():
        pred.loc[test, name] = lgb.LGBMRegressor(**params).fit(X[train], y[train]).predict(X[test])
        if name in ("honest", "points_hist", "idealised"):  # the decoupling question is asked of three sets only
            prob.loc[test, name] = lgb.LGBMClassifier(**params).fit(X[train], dec[train]).predict_proba(X[test])[:, 1]
    print("fitted", start.strftime("%Y-%m"), flush=True)
rows = (frame["delivery_day"] >= first_month) & frame["weather_point_in_time"].astype(bool) & y.notna() & lags["s_l1"].notna() & g_plain["de_wind_px"].notna() & g_plain["de_solar_px"].notna()
assert (g_hist.loc[rows, ["de_wind_px", "de_solar_px"]].to_numpy() == g_plain.loc[rows, ["de_wind_px", "de_solar_px"]].to_numpy()).all()  # no scored hour uses a stand-in
F = frame[rows].copy(); P = pred[rows]; Q = prob[rows]; L = lags[rows]
F["zero"], F["yesterday"] = 0.0, L["s_l1"]
for name in sets: F[name] = P[name]
F[["delivery_day", "hour", "spread", "yesterday", *sets]].to_csv(out / "spread_predictions.csv.gz")  # working file, not for publication
print(f"\nSCORED: hours {len(F):,}, days {F.delivery_day.nunique()}, {F.delivery_day.min():%Y-%m-%d} to {F.delivery_day.max():%Y-%m-%d}; mean spread {F.spread.mean():+.1f}, mean absolute spread {F.spread.abs().mean():.1f}")
yy = F.delivery_day.dt.year
cols = ("zero", "yesterday", *sets)


def daily_err(col, sub=None):
    S = F if sub is None else F[sub]; return (S[col] - S.spread).abs().groupby(S.delivery_day).mean()


def paired(a, b, sub=None):
    d = (daily_err(a, sub) - daily_err(b, sub)).to_numpy(); lo, hi = block_bootstrap_mean(d, block=7, resamples=5000, seed=0, level=0.95)
    return d.mean(), lo, hi


results = {"window": [str(F.delivery_day.min().date()), str(F.delivery_day.max().date())], "hours": int(len(F)), "days": int(F.delivery_day.nunique()), "mae": {}, "paired": {}}
print("\n2. FORECAST ERROR of the hourly spread, EUR/MWh (paired against yesterday's spread, 95% moving-block interval)")
for col in cols:
    m, lo, hi = paired(col, "yesterday"); big = F.spread.abs() > 5; sign = (np.sign(F.loc[big, col]) == np.sign(F.loc[big, "spread"])).mean()
    mae = (F[col] - F.spread).abs().mean(); by = (F[col] - F.spread).abs().groupby(yy).mean()
    results["mae"][col] = {"all": round(float(mae), 3), **{str(k): round(float(v), 3) for k, v in by.items()}}
    print("  %-11s %s MAE %.2f | ratio to yesterday %.2f | paired %+.2f [%+.2f, %+.2f] | corr %.2f | sign right when |spread| > 5: %.0f%% | by year %s" % (
        col, " " if col in ATTAINABLE + ("zero", "yesterday") else "*", mae, mae / (F.yesterday - F.spread).abs().mean(), m, lo, hi,
        np.corrcoef(F[col], F.spread)[0, 1] if F[col].std() > 0 else float("nan"), 100 * sign, " ".join("%d:%.2f" % (k, v) for k, v in by.items())))
print("  (* not available at 10:05 Paris on D-1)")
print("\n3. HOW MUCH OF THE GAIN SURVIVES (paired daily differences, EUR/MWh)")
for a, b in (("points_hist", "honest"), ("points", "honest"), ("points_hist", "points"), ("points_load", "points_hist"), ("idealised", "points_load"), ("idealised", "points_hist"), ("idealised", "honest"), ("ceiling", "points_hist"), ("ceiling", "idealised"), ("ceiling", "honest")):
    m, lo, hi = paired(a, b); results["paired"][f"{a} minus {b}"] = [round(float(v), 3) for v in (m, lo, hi)]
    print("  %-12s minus %-12s %+.2f [%+.2f, %+.2f]" % (a, b, m, lo, hi))
for label, sub in (("all scored days", None), *[(str(k), np.asarray(yy == k)) for k in sorted(yy.unique())]):
    gap = paired("idealised", "honest", sub)[0]; top = paired("ceiling", "honest", sub)[0]; got = paired("points_hist", "honest", sub)[0]; plain = paired("points", "honest", sub)[0]
    results["paired"][f"share recovered, {label}"] = {"of idealised": round(float(got / gap), 3), "of ceiling": round(float(got / top), 3), "without the stand-in, of ceiling": round(float(plain / top), 3)}
    print("  %-16s gain over honest: attainable %+.2f, idealised %+.2f, ceiling %+.2f | share recovered %.0f%% of idealised, %.0f%% of ceiling (%.0f%% of ceiling without the stand-in)" % (
        label, got, gap, top, 100 * got / gap, 100 * got / top, 100 * plain / top))
print("\n4. DAILY BASE SPREAD (mean of the day)")
B = F.groupby("delivery_day")[["spread", *cols]].mean(); move = B.spread - B.yesterday; bigm = move.abs() > 5; bigb = B.spread.abs() > 3
for col in cols:
    e = (B[col] - B.spread).abs(); line = "  %-11s MAE %.2f | corr %.2f | sign right when |base spread| > 3: %.0f%% of %d days" % (
        col, e.mean(), np.corrcoef(B[col], B.spread)[0, 1] if B[col].std() > 0 else float("nan"), 100 * (np.sign(B.loc[bigb, col]) == np.sign(B.loc[bigb, "spread"])).mean(), bigb.sum())
    if col in sets:
        pm = B[col] - B.yesterday
        line += " | direction of the move from yesterday right on %.0f%% of %d days with a move above 5, corr of moves %.2f, share captured %.2f" % (
            100 * (np.sign(pm[bigm]) == np.sign(move[bigm])).mean(), bigm.sum(), np.corrcoef(pm, move)[0, 1], np.polyfit(move, pm, 1)[0])
        results.setdefault("direction", {})[col] = round(float((np.sign(pm[bigm]) == np.sign(move[bigm])).mean()), 3)
    results.setdefault("base_mae", {})[col] = round(float(e.mean()), 3)
    print(line)
print("\n5. WILL THE HOUR DECOUPLE? (Brier score, lower is better)")
truth = (F.spread.abs() > COUPLED).astype(float); persist = (L["s_l1"].abs() > COUPLED).astype(float)
from sklearn.metrics import roc_auc_score
print("  share decoupled %.2f | constant rate %.3f | yesterday same hour %.3f" % (truth.mean(), ((truth.mean() - truth) ** 2).mean(), ((persist - truth) ** 2).mean()), end="")
for name in Q.columns: print(" | %s %.3f (AUC %.2f)" % (name, ((Q[name] - truth) ** 2).mean(), roc_auc_score(truth, Q[name])), end="")
print()
print("\n6. WHERE IT HELPS (hourly MAE of the spread)")
m_ = F.delivery_day.dt.month; season = np.where(m_.isin([11, 12, 1, 2]), "Nov-Feb", np.where(m_.isin([5, 6, 7, 8]), "May-Aug", "other"))
wind = ideal["de_wind_fc"].reindex(F.index).groupby(F.delivery_day).transform("mean"); wq = wind.groupby(F.delivery_day).first()
groups = {"Nov-Feb": season == "Nov-Feb", "May-Aug": season == "May-Aug", "other months": season == "other",
          "windiest tenth of days in Germany": (wind >= wq.quantile(0.9)).to_numpy(), "calmest tenth of days in Germany": (wind <= wq.quantile(0.1)).to_numpy(),
          "hours 10 to 15 Paris": F.hour.between(10, 15).to_numpy(), "hours 17 to 20 Paris": F.hour.between(17, 20).to_numpy()}
for name, mask in groups.items():
    s = F[np.asarray(mask)]
    print("  %-34s days %3d | mean spread %+6.1f | yesterday %5.1f | honest %5.1f | points_hist %5.1f | idealised* %5.1f" % (
        name, s.delivery_day.nunique(), s.spread.mean(), *[(s[c] - s.spread).abs().mean() for c in ("yesterday", "honest", "points_hist", "idealised")]))
print("\n7. WHAT DRIVES IT (gain share of the regression fitted on all days to %s)" % LAST)
tr = np.asarray(y.notna() & lags["s_l1"].notna())
for name in ("points_hist", "idealised"):
    mdl = lgb.LGBMRegressor(**params, importance_type="gain").fit(sets[name][tr], y[tr]); imp = pd.Series(mdl.feature_importances_, index=sets[name].columns); imp = (imp / imp.sum()).sort_values(ascending=False)
    print("  %s: %s" % (name, ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in imp.head(10).items())))
    if name == "points_hist": results["german_block_gain_share"] = round(float(imp[[c for c in g_hist.columns]].sum()), 3)
results["proxy_fit"] = {a: {"mae_mw": round(float((q[a] - q[b]).abs().mean()), 1), "mean_mw": round(float(q[b].mean()), 1), "corr": round(float(np.corrcoef(q[a], q[b])[0, 1]), 4)}
                        for a, b in (("onshore", "onshore_fc"), ("offshore", "offshore_fc"), ("wind", "wind_fc"), ("solar", "solar_fc"), ("res", "res_fc"))}
(out / "spread_points_results.json").write_text(json.dumps(results, indent=1))
