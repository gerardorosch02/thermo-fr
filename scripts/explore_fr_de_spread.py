"""Exploratory: the France minus Germany day-ahead spread by delivery hour, on the selection window. No registration, no
decision. Nothing after 2026-06-30 is read.

    python scripts/explore_fr_de_spread.py data/forecast/inputs.csv data/forecast/neighbour_fundamentals.csv <per-sequence extract>

Three input sets, each refitted at the start of every month on all earlier days from 2021:
  lags      the spread's own past and the calendar
  honest    lags plus the French inputs of the public model (honest_v2), all known at 10:05 Paris on D-1
  idealised honest plus the German load, wind and solar forecasts of D published by ENTSO-E. These are NOT available at 10:05:
            the set shows what a desk with its own German forecasts might reach, not what this project can issue.
The German price is the coupled auction (classification sequence 1)."""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from thermo_fr.forecast import lear_recipe
from thermo_fr.forecast.features import build_features
from thermo_fr.forecast.inputs import load_inputs
from thermo_fr.forecast.models import GBM_PARAMS
from thermo_fr.forecast.refit import REFIT_PARAMS
from thermo_fr.forecast.shape import block_bootstrap_mean

FIRST, LAST = "2024-02-18", "2026-06-30"
COUPLED = 0.005  # EUR/MWh: below this the two prices are the same

base = lear_recipe.selection_inputs(load_inputs(Path(sys.argv[1])))
table = build_features(base, "honest_v2")
seq = pd.read_csv(sys.argv[3]); seq.index = pd.to_datetime(seq.pop("timestamp_utc"), utc=True)
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
nf = pd.read_csv(sys.argv[2]); nf.index = pd.to_datetime(nf.pop("timestamp_utc"), utc=True); nf = nf.reindex(frame.index)
german = pd.DataFrame({"de_load_fc": nf["load_fc_de_lu_mw"], "de_solar_fc": nf["solar_fc_de_lu_mw"], "de_wind_fc": nf["wind_onshore_fc_de_lu_mw"].fillna(0) + nf["wind_offshore_fc_de_lu_mw"].fillna(0)})
german["de_residual_fc"] = german["de_load_fc"] - german["de_solar_fc"] - german["de_wind_fc"]
french = table.X.reindex(frame.index)
french = french[[c for c in french.columns if c not in lags.columns]]  # hour and calendar columns are already in the lags set
sets = {"lags": lags, "honest": pd.concat([lags, french], axis=1), "idealised": pd.concat([lags, french, german], axis=1)}
params = {**GBM_PARAMS, **REFIT_PARAMS}
y = frame["spread"]; dec = (y.abs() > COUPLED).astype(float)
pred = pd.DataFrame(index=frame.index); prob = pd.DataFrame(index=frame.index)
for start in pd.date_range("2024-02-01", LAST, freq="MS"):
    train = np.asarray((frame["delivery_day"] < start) & y.notna()); test = np.asarray((frame["delivery_day"] >= start) & (frame["delivery_day"] < start + pd.DateOffset(months=1)))
    for name, X in sets.items():
        pred.loc[test, name] = lgb.LGBMRegressor(**params).fit(X[train], y[train]).predict(X[test])
        prob.loc[test, name] = lgb.LGBMClassifier(**params).fit(X[train], dec[train]).predict_proba(X[test])[:, 1]
rows = (frame["delivery_day"] >= FIRST) & frame["weather_point_in_time"].astype(bool) & y.notna() & lags["s_l1"].notna()
F = frame[rows].copy(); P = pred[rows]; Q = prob[rows]; L = lags[rows]
F["zero"], F["yesterday"], F["week_mean"] = 0.0, L["s_l1"], L["s_m7"]
for name in sets: F[name] = P[name]
print(f"hours {len(F):,}, days {F.delivery_day.nunique()}, {F.delivery_day.min():%Y-%m-%d} to {F.delivery_day.max():%Y-%m-%d}")
yy = F.delivery_day.dt.year
print("\n1. THE SPREAD (France minus Germany, EUR/MWh)")
print(pd.DataFrame({"fr": F.fr.groupby(yy).mean(), "de": F.de.groupby(yy).mean(), "mean": F.spread.groupby(yy).mean(), "mean_abs": F.spread.abs().groupby(yy).mean(),
                    "decoupled": (F.spread.abs() > COUPLED).groupby(yy).mean(), "fr_above": (F.spread > COUPLED).groupby(yy).mean(), "fr_below": (F.spread < -COUPLED).groupby(yy).mean()}).round(2).to_string())
m = F.delivery_day.dt.month; season = np.where(m.isin([11, 12, 1, 2]), "Nov-Feb", np.where(m.isin([5, 6, 7, 8]), "May-Aug", "other"))
print(pd.DataFrame({"mean": F.spread.groupby(season).mean(), "mean_abs": F.spread.abs().groupby(season).mean(), "decoupled": (F.spread.abs() > COUPLED).groupby(season).mean()}).round(2).to_string())
def daily_err(col): return (F[col] - F.spread).abs().groupby(F.delivery_day).mean()
print("\n2. FORECAST ERROR of the hourly spread, EUR/MWh (paired against yesterday's spread, 95% moving-block interval)")
ref = daily_err("yesterday")
for col in ("zero", "yesterday", "week_mean", "lags", "honest", "idealised"):
    e = daily_err(col); diff = (e - ref).to_numpy(); lo, hi = block_bootstrap_mean(diff, block=7, resamples=5000, seed=0, level=0.95)
    big = F.spread.abs() > 5; sign = (np.sign(F.loc[big, col]) == np.sign(F.loc[big, "spread"])).mean()
    print("  %-10s MAE %.2f | ratio to yesterday %.2f | paired %+.2f [%+.2f, %+.2f] | corr %.2f | sign right when |spread| > 5: %.0f%% | by year %s" % (
        col, (F[col] - F.spread).abs().mean(), (F[col] - F.spread).abs().mean() / (F.yesterday - F.spread).abs().mean(), diff.mean(), lo, hi,
        np.corrcoef(F[col], F.spread)[0, 1] if F[col].std() > 0 else float("nan"), 100 * sign, " ".join("%d:%.2f" % (k, v) for k, v in (F[col] - F.spread).abs().groupby(yy).mean().items())))
d1, d2 = daily_err("honest"), daily_err("idealised"); lo, hi = block_bootstrap_mean((d2 - d1).to_numpy(), block=7, resamples=5000, seed=0, level=0.95)
print("  idealised minus honest: %+.2f [%+.2f, %+.2f]; honest minus lags: %+.2f" % ((d2 - d1).mean(), lo, hi, (d1 - daily_err("lags")).mean()))
print("\n3. DAILY BASE SPREAD (mean of the day), forecast error")
B = F.groupby("delivery_day")[["spread", "zero", "yesterday", "week_mean", "lags", "honest", "idealised"]].mean()
for col in ("zero", "yesterday", "week_mean", "lags", "honest", "idealised"):
    e = (B[col] - B.spread).abs(); big = B.spread.abs() > 3
    print("  %-10s MAE %.2f | corr %.2f | sign right when |base spread| > 3: %.0f%% of %d days" % (col, e.mean(), np.corrcoef(B[col], B.spread)[0, 1] if B[col].std() > 0 else float("nan"), 100 * (np.sign(B.loc[big, col]) == np.sign(B.loc[big, "spread"])).mean(), big.sum()))
print("\n4. WILL THE HOUR DECOUPLE? (probability, Brier score; lower is better)")
truth = (F.spread.abs() > COUPLED).astype(float); base_rate = truth.mean(); persist = (L["s_l1"].abs() > COUPLED).astype(float)
from sklearn.metrics import roc_auc_score
print("  share decoupled %.2f | Brier: constant rate %.3f, yesterday same hour %.3f" % (base_rate, ((base_rate - truth) ** 2).mean(), ((persist - truth) ** 2).mean()), end="")
for name in sets: print(" | %s %.3f (AUC %.2f)" % (name, ((Q[name] - truth) ** 2).mean(), roc_auc_score(truth, Q[name])), end="")
print()
print("\n5. WHAT DRIVES IT (gain share of the honest and idealised regressions fitted on all days to %s)" % LAST)
tr = np.asarray(y.notna() & lags["s_l1"].notna())
for name in ("honest", "idealised"):
    mdl = lgb.LGBMRegressor(**params, importance_type="gain").fit(sets[name][tr], y[tr]); imp = pd.Series(mdl.feature_importances_, index=sets[name].columns); imp = (imp / imp.sum()).sort_values(ascending=False)
    print("  %s: %s" % (name, ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in imp.head(10).items())))
print("\n6. BY SEASON AND ON COLD DAYS (hourly MAE of the spread)")
temp = french["temp_fc_c"].reindex(F.index) if "temp_fc_c" in french.columns else None
groups = {"Nov-Feb": season == "Nov-Feb", "May-Aug": season == "May-Aug", "other months": season == "other"}
if temp is not None:
    dt = temp.groupby(F.delivery_day).transform("mean"); groups["coldest tenth of days (forecast temperature)"] = (dt <= dt.groupby(F.delivery_day).first().quantile(0.1)).to_numpy()
wind = german["de_wind_fc"].reindex(F.index).groupby(F.delivery_day).transform("mean"); wq = wind.groupby(F.delivery_day).first()
groups["windiest tenth of days in Germany"] = (wind >= wq.quantile(0.9)).to_numpy(); groups["calmest tenth of days in Germany"] = (wind <= wq.quantile(0.1)).to_numpy()
for name, mask in groups.items():
    s = F[np.asarray(mask)]
    print("  %-46s days %3d | mean spread %+.1f | yesterday %.1f | lags %.1f | honest %.1f | idealised %.1f" % (name, s.delivery_day.nunique(), s.spread.mean(), *[(s[c] - s.spread).abs().mean() for c in ("yesterday", "lags", "honest", "idealised")]))
print("\n7. CHANGE FROM YESTERDAY: does the forecast get the direction of the daily base spread's move right?")
Bm = F.groupby("delivery_day")[["spread", "yesterday", "lags", "honest", "idealised"]].mean(); move = Bm.spread - Bm.yesterday; bigm = move.abs() > 5
for col in ("lags", "honest", "idealised"):
    pm = Bm[col] - Bm.yesterday; print("  %-10s direction right on %.0f%% of the %d days with a move above 5 | correlation of predicted and realised move %.2f | share of the move captured %.2f" % (col, 100 * (np.sign(pm[bigm]) == np.sign(move[bigm])).mean(), bigm.sum(), np.corrcoef(pm, move)[0, 1], np.polyfit(move, pm, 1)[0]))
