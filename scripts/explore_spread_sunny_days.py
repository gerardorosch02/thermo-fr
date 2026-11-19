"""Exploratory: why did the German inputs fail on sunny days in the out-of-sample test? Development window only.
No registration, no decision, no fit. Every table is cut at 2026-06-30 by spread_eval.load_stored; the forecasts are the stored
development forecasts of the frozen module (dev_predictions.csv.gz). Nothing of the test window is opened here: its registered
figures are quoted as constants in section 6 of part 3.

    python scripts/explore_spread_sunny_days.py <work folder holding dev_predictions.csv.gz> <sequences extract> <wind points> <solar points>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from thermo_fr.forecast import spread_eval as se

work = Path(sys.argv[1]); pd.set_option("display.width", 250)

# ---- part 1: assemble --------------------------------------------------------------
stored = se.load_stored("data/forecast/inputs.csv", "data/forecast/neighbour_fundamentals.csv", sys.argv[2], sys.argv[3], sys.argv[4])
assert stored.last == se.DEV_LAST
feat = se.features(stored)
P = pd.read_csv(work / "dev_predictions.csv.gz", index_col=0, parse_dates=["delivery_day"]); P.index = pd.to_datetime(P.index, utc=True)
C = feat["sets"]["C"].loc[P.index]; fr = feat["frame"].loc[P.index]
P["fr"], P["de"] = fr["fr"], fr["de"]
for c in ("de_solar_px", "de_wind_px", "de_solar_fc", "de_wind_fc", "de_load_fc", "de_solar_fc_l1", "de_solar_px_l1", "de_load_l7"): P[c] = C[c]
fx = [c for c in feat["sets"]["H"].columns if "solar" in c.lower() or "pv" in c.lower()]
print("French solar columns in H:", fx[:8])
for c in fx[:3]: P["fr_" + c] = feat["sets"]["H"].loc[P.index, c]
DEV = P.copy()
print(len(P), P.delivery_day.min().date(), P.delivery_day.max().date())

# ---- part 2: by sun, season, hour and proxy quality ---------------------------------
P = DEV.copy()
for m in ("yesterday", "H", "G", "C"): P["e_" + m] = (P[m] - P.spread).abs()
P["month"] = P.delivery_day.dt.month; P["year"] = P.delivery_day.dt.year
day = P.groupby("delivery_day").agg(solar=("de_solar_px", "mean"), solar_fc=("de_solar_fc", "mean"), wind=("de_wind_px", "mean"), eH=("e_H", "mean"), eG=("e_G", "mean"), eC=("e_C", "mean"),
                                    sp=("spread", "mean"), asp=("spread", lambda s: s.abs().mean()), de=("de", "mean"), fr=("fr", "mean"))
day["d"] = day.eG - day.eH; day["dC"] = day.eC - day.eH
print("1. G minus H by tenth of days ranked on the German solar proxy (all 791 days)")
q = pd.qcut(day.solar, 10, labels=False)
print(day.groupby(q).agg(days=("d", "size"), solar=("solar", "mean"), abs_spread=("asp", "mean"), H=("eH", "mean"), G=("eG", "mean"), G_minus_H=("d", "mean"), C_minus_H=("dC", "mean"), share_G_better=("d", lambda s: (s < 0).mean())).round(2).to_string())
print("\n2. The test's own slice, replicated: sunniest tenth of days within 1 July to 10 October of each earlier year, and within May to June")
for label, a, b in (("Jul-10 Oct 2024", "2024-07-01", "2024-10-10"), ("Jul-10 Oct 2025", "2025-07-01", "2025-10-10"), ("May-Jun 2024", "2024-05-01", "2024-06-30"), ("May-Jun 2025", "2025-05-01", "2025-06-30"), ("May-Jun 2026", "2026-05-01", "2026-06-30")):
    w = day[a:b]; top = w[w.solar >= w.solar.quantile(0.9)]
    print("  %-16s days %3d | all days G-H %+.2f C-H %+.2f | sunniest tenth (%d days): H %.2f G %.2f C %.2f, G-H %+.2f, C-H %+.2f | abs spread %.1f vs %.1f" % (
        label, len(w), w.d.mean(), w.dC.mean(), len(top), top.eH.mean(), top.eG.mean(), top.eC.mean(), top.d.mean(), top.dC.mean(), top.asp.mean(), w.asp.mean()))
print("\n3. By season and by sun (terciles of the daily solar proxy within the season)")
season = np.select([day.index.month.isin([5, 6, 7, 8]), day.index.month.isin([11, 12, 1, 2])], ["May-Aug", "Nov-Feb"], "Mar-Apr, Sep-Oct"); day["season"] = season
for s, w in day.groupby("season"):
    t = pd.qcut(w.solar, 3, labels=["dull", "middle", "sunny"])
    print(s); print(w.groupby(t, observed=True).agg(days=("d", "size"), abs_spread=("asp", "mean"), H=("eH", "mean"), G=("eG", "mean"), C=("eC", "mean"), G_minus_H=("d", "mean"), C_minus_H=("dC", "mean"), wind=("wind", "mean")).round(2).to_string())
print("\n4. May to August, sunny third of days: by Paris hour block")
S = P[P.delivery_day.isin(day.index[(day.season == "May-Aug")][pd.qcut(day[day.season == "May-Aug"].solar, 3, labels=False) == 2])]
blk = pd.cut(S.hour, [-1, 6, 9, 15, 19, 23], labels=["0-6", "7-9", "10-15", "16-19", "20-23"])
g = S.groupby(blk, observed=True)
print(pd.DataFrame({"hours": g.size(), "spread": g.spread.mean(), "abs_spread": g.spread.apply(lambda s: s.abs().mean()), "coupled": g.spread.apply(lambda s: (s.abs() < 0.005).mean()), "de<=0": g.de.apply(lambda s: (s <= 0).mean()),
                    "H": g.e_H.mean(), "G": g.e_G.mean(), "C": g.e_C.mean(), "bias_H": g.apply(lambda x: (x.H - x.spread).mean()), "bias_G": g.apply(lambda x: (x.G - x.spread).mean()), "bias_C": g.apply(lambda x: (x.C - x.spread).mean())}).round(2).to_string())
print("\n5. Solar proxy against ENTSO-E's forecast, hours 10 to 15 Paris, by half year")
M = P[P.hour.between(10, 15)]; half = M.delivery_day.dt.year.astype(str) + np.where(M.delivery_day.dt.month.between(4, 9), " Apr-Sep", " Oct-Mar")
g = M.groupby(half)
print(pd.DataFrame({"hours": g.size(), "forecast": g.de_solar_fc.mean(), "proxy": g.de_solar_px.mean(), "ratio": g.de_solar_px.mean() / g.de_solar_fc.mean(), "mae": g.apply(lambda x: (x.de_solar_px - x.de_solar_fc).abs().mean()),
                    "corr": g.apply(lambda x: np.corrcoef(x.de_solar_px, x.de_solar_fc)[0, 1])}).round(3).to_string())
dm = P.groupby("delivery_day").agg(px=("de_solar_px", "sum"), fc=("de_solar_fc", "sum")); dm["err"] = (dm.px - dm.fc) / dm.fc
top = dm[dm.index.month.isin([5, 6, 7, 8])]; t = pd.qcut(top.fc, 3, labels=["dull", "middle", "sunny"])
print("  May-Aug daily energy, proxy over forecast minus 1, by tercile of the forecast:", top.groupby(t, observed=True).err.agg(["mean", "std"]).round(3).to_dict())
day["perr"] = dm.err
w = day[day.season == "May-Aug"]; print("  May-Aug: corr of (G-H daily difference) with |proxy error| %.2f, with solar level %.2f, with wind level %.2f, with abs spread %.2f" % (
    np.corrcoef(w.d, w.perr.abs())[0, 1], np.corrcoef(w.d, w.solar)[0, 1], np.corrcoef(w.d, w.wind)[0, 1], np.corrcoef(w.d, w.asp)[0, 1]))

# ---- part 3: wind, spread size, and the frozen rule on 102-day windows ----------------
P = DEV.copy()
P["B"] = (P.H + P.G) / 2
for m in ("yesterday", "H", "G", "C", "B"): P["e_" + m] = (P[m] - P.spread).abs()
day = P.groupby("delivery_day").agg(solar=("de_solar_px", "mean"), wind=("de_wind_px", "mean"), eH=("e_H", "mean"), eG=("e_G", "mean"), eC=("e_C", "mean"), eB=("e_B", "mean"), asp=("spread", lambda s: s.abs().mean()), sp=("spread", "mean"))
day["d"] = day.eG - day.eH
print("1. G minus H by tenth of days ranked on the German wind proxy")
q = pd.qcut(day.wind, 10, labels=False)
print(day.groupby(q).agg(days=("d", "size"), wind=("wind", "mean"), solar=("solar", "mean"), abs_spread=("asp", "mean"), H=("eH", "mean"), G=("eG", "mean"), G_minus_H=("d", "mean"), better=("d", lambda s: (s < 0).mean())).round(2).to_string())
print("\n2. Gain against the size of the spread over the previous 30 days (known beforehand)")
day["asp30"] = day.asp.shift(1).rolling(30, min_periods=20).mean()
t = pd.qcut(day.asp30, 5)
g = day.groupby(t, observed=True).agg(days=("d", "size"), abs_spread=("asp", "mean"), H=("eH", "mean"), G=("eG", "mean"), G_minus_H=("d", "mean"))
g["gain_share_of_H"] = -g.G_minus_H / g.H; g["gain_share_of_spread"] = -g.G_minus_H / g.abs_spread
print(g.round(3).to_string())
print("  whole window: gain %.2f = %.1f%% of H's error, %.1f%% of the mean absolute spread" % (-day.d.mean(), -100 * day.d.mean() / day.eH.mean(), -100 * day.d.mean() / day.asp.mean()))
print("\n3. How often would the frozen rule fail on a 102-day window of the development period? (every window of 102 consecutive days)")
days = day.index; res = []
for i in range(0, len(days) - 101):
    w = P[(P.delivery_day >= days[i]) & (P.delivery_day <= days[i + 101])]
    by = w.groupby("delivery_day").agg(wind=("wind_px", "mean"), solar=("solar_px", "mean"))
    sl = {"weekends": w.dow >= 5, "holidays": w.holiday.astype(bool) | (w.de_holiday > 0), "windiest": w.delivery_day.isin(by.index[by.wind >= by.wind.quantile(0.9)]),
          "sunniest": w.delivery_day.isin(by.index[by.solar >= by.solar.quantile(0.9)]), "top5": w.spread.abs() >= w.spread.abs().quantile(0.95)}
    row = {"start": days[i], "gain": w.e_H.mean() - w.e_G.mean()}
    for k, m in sl.items():
        s = w[np.asarray(m)]; row[k] = (s.e_G.mean() - s.e_H.mean()) if len(s) else np.nan; row[k + "_days"] = s.delivery_day.nunique()
    for k, m in sl.items():
        s = w[np.asarray(m)]; row["B_" + k] = (s.e_B.mean() - s.e_H.mean()) if len(s) else np.nan
    row["B_gain"] = w.e_H.mean() - w.e_B.mean()
    res.append(row)
R = pd.DataFrame(res).set_index("start"); names = ["weekends", "holidays", "windiest", "sunniest", "top5"]
fail = (R[names] > 1.0)
print("  windows %d | gain: min %.2f, median %.2f, max %.2f | share of windows where a slice is worse by more than 1.0: %.0f%%" % (len(R), R.gain.min(), R.gain.median(), R.gain.max(), 100 * fail.any(axis=1).mean()))
print("  by slice:", {k: "%.0f%%" % (100 * fail[k].mean()) for k in names}, "| holiday days per window: median %d, windows with fewer than 5: %.0f%%" % (R.holidays_days.median(), 100 * (R.holidays_days < 5).mean()))
ind = R.iloc[::102]; fi = (ind[names] > 1.0)
print("  seven windows that do not overlap:", {str(k.date()): [n for n in names if fi.loc[k, n]] for k in ind.index})
summer = R[(R.index.month.isin([6, 7, 8]))]; print("  windows starting June to August (%d): slice failure %.0f%%, sunniest slice %.0f%%, median sunniest-slice difference %+.2f" % (len(summer), 100 * (summer[names] > 1).any(axis=1).mean(), 100 * (summer.sunniest > 1).mean(), summer.sunniest.median()))
Bn = ["B_" + n for n in names]
print("\n4. A half-and-half blend of H and G on the development window (not a proposal yet, a measurement)")
print("  hourly error: H %.2f, G %.2f, blend %.2f, C %.2f" % (P.e_H.mean(), P.e_G.mean(), P.e_B.mean(), P.e_C.mean()))
print("  102-day windows: blend gain min %.2f median %.2f | a slice worse by more than 1.0 in %.0f%% of windows (G: %.0f%%)" % (R.B_gain.min(), R.B_gain.median(), 100 * (R[Bn] > 1).any(axis=1).mean(), 100 * fail.any(axis=1).mean()))
print("\n5. Sunniest tenth of days inside each 102-day window: what those days look like (G against H)")
print("  median difference over all windows %+.2f; 10th to 90th percentile %+.2f to %+.2f" % (R.sunniest.median(), R.sunniest.quantile(0.1), R.sunniest.quantile(0.9)))
print("  windiest tenth: median %+.2f; 10th to 90th percentile %+.2f to %+.2f" % (R.windiest.median(), R.windiest.quantile(0.1), R.windiest.quantile(0.9)))
print("\n6. Where the test window's registered figures sit among the 690 development windows")
asp = pd.Series({days[i]: day.asp.iloc[i:i + 102].mean() for i in range(len(days) - 101)}); msp = pd.Series({days[i]: day.sp.iloc[i:i + 102].mean() for i in range(len(days) - 101)})
print("  gain: test 1.81; development windows below that: %d of %d (%.1f%%)" % ((R.gain < 1.814).sum(), len(R), 100 * (R.gain < 1.814).mean()))
print("  sunniest-tenth difference: test +1.70; development maximum %+.2f, share above 0: %.0f%%" % (R.sunniest.max(), 100 * (R.sunniest > 0).mean()))
print("  windiest-tenth difference: test -6.99; development median %+.2f" % R.windiest.median())
print("  mean absolute spread: test 14.5; development windows: min %.1f, median %.1f | mean spread: test -5.0; development max %.1f (closest to zero), median %.1f" % (asp.min(), asp.median(), msp.max(), msp.median()))
print("  holidays difference: test +1.81; development: share above 1.0 %.0f%%, 90th percentile %+.2f" % (100 * (R.holidays > 1).mean(), R.holidays.quantile(0.9)))
