import numpy as np, pandas as pd
P = pd.read_pickle("all.pkl"); pd.set_option("display.width", 260)
for m in ("yesterday", "H", "G", "C"): P["e_" + m] = (P[m] - P.spread).abs()
P["year"] = P.delivery_day.dt.year; md = P.delivery_day.dt.strftime("%m-%d")
W = P[(md >= "07-01") & (md <= "10-10")]
print("1. The same calendar window, 1 July to 10 October, in three years")
g = W.groupby("year")
print(pd.DataFrame({"days": g.delivery_day.nunique(), "FR": g.fr.mean(), "DE": g.de.mean(), "spread": g.spread.mean(), "abs_spread": g.spread.apply(lambda s: s.abs().mean()), "sd_spread": g.spread.std(),
    "coupled": g.spread.apply(lambda s: (s.abs() < 0.005).mean()), "FR>DE": g.spread.apply(lambda s: (s > 0.005).mean()), "de<=0": g.de.apply(lambda s: (s <= 0).mean()), "fr<=0": g.fr.apply(lambda s: (s <= 0).mean()),
    "wind_px": g.de_wind_px.mean(), "solar_px": g.de_solar_px.mean(), "de_load": g.de_load_fc.mean(), "yest": g.e_yesterday.mean(), "H": g.e_H.mean(), "G": g.e_G.mean(), "C": g.e_C.mean()}).round(2).to_string())
print("\n2. By Paris hour block, same window, 2025 against 2026")
blk = pd.cut(W.hour, [-1, 6, 9, 15, 19, 23], labels=["0-6", "7-9", "10-15", "16-19", "20-23"])
g = W[W.year >= 2025].groupby([W.year[W.year >= 2025], blk[W.year >= 2025]], observed=True)
print(pd.DataFrame({"FR": g.fr.mean(), "DE": g.de.mean(), "spread": g.spread.mean(), "abs": g.spread.apply(lambda s: s.abs().mean()), "FR>DE": g.spread.apply(lambda s: (s > 0.005).mean()), "H": g.e_H.mean(), "G": g.e_G.mean(), "C": g.e_C.mean(),
    "bias_H": g.apply(lambda x: (x.H - x.spread).mean()), "bias_G": g.apply(lambda x: (x.G - x.spread).mean()), "bias_C": g.apply(lambda x: (x.C - x.spread).mean())}).round(2).to_string())
print("\n3. 2026 by half month: level of the spread and the errors")
T = P[P.part == "test"]; hm = T.delivery_day.dt.strftime("%m") + np.where(T.delivery_day.dt.day <= 15, "a", "b")
g = T.groupby(hm)
print(pd.DataFrame({"days": g.delivery_day.nunique(), "FR": g.fr.mean(), "DE": g.de.mean(), "spread": g.spread.mean(), "abs": g.spread.apply(lambda s: s.abs().mean()), "wind": g.de_wind_px.mean(), "yest": g.e_yesterday.mean(), "H": g.e_H.mean(), "G": g.e_G.mean(), "C": g.e_C.mean(),
    "bias_H": g.apply(lambda x: (x.H - x.spread).mean()), "bias_G": g.apply(lambda x: (x.G - x.spread).mean())}).round(2).to_string())
print("\n4. The run-up: monthly mean spread and prices, January 2025 to October 2026")
A = P.copy(); g = A.groupby(A.delivery_day.dt.strftime("%Y-%m"))
print(pd.DataFrame({"FR": g.fr.mean(), "DE": g.de.mean(), "spread": g.spread.mean(), "abs": g.spread.apply(lambda s: s.abs().mean()), "H": g.e_H.mean(), "G": g.e_G.mean()}).round(1).loc["2025-01":].T.to_string())
