import numpy as np, pandas as pd
P = pd.read_pickle("all.pkl"); V = pd.read_pickle("variants_2.pkl"); P["W180"] = V["G_w180"]; pd.set_option("display.width", 260)
T = P[P.part == "test"].copy()
by = T.groupby("delivery_day").agg(s=("solar_px", "mean"), w=("wind_px", "mean")); sun = by.index[by.s >= by.s.quantile(0.9)]
S = T[T.delivery_day.isin(sun)]; R = T[~T.delivery_day.isin(sun)]
print("sunniest 11 days of the opened window: wind %.0f MW against %.0f on the other days; mean abs spread %.1f against %.1f; mean |day-on-day change of the hourly spread| %.1f against %.1f" % (
    by.w[sun].mean(), by.w.drop(sun).mean(), S.spread.abs().mean(), R.spread.abs().mean(), (S.spread - S.yesterday).abs().mean(), (R.spread - R.yesterday).abs().mean()))
blk = lambda X: pd.cut(X.hour, [-1, 6, 9, 15, 19, 23], labels=["0-6", "7-9", "10-15", "16-19", "20-23"])
g = S.groupby(blk(S), observed=True)
print(pd.DataFrame({"hours": g.size(), "FR": g.fr.mean(), "DE": g.de.mean(), "spread": g.spread.mean(), "abs": g.spread.apply(lambda s: s.abs().mean()), "yest": g.apply(lambda x: (x.yesterday - x.spread).abs().mean()),
    "H": g.apply(lambda x: (x.H - x.spread).abs().mean()), "G": g.apply(lambda x: (x.G - x.spread).abs().mean()), "C": g.apply(lambda x: (x.C - x.spread).abs().mean()), "W180": g.apply(lambda x: (x.W180 - x.spread).abs().mean()),
    "bias_H": g.apply(lambda x: (x.H - x.spread).mean()), "bias_G": g.apply(lambda x: (x.G - x.spread).mean()), "bias_C": g.apply(lambda x: (x.C - x.spread).mean()), "bias_W": g.apply(lambda x: (x.W180 - x.spread).mean()),
    "sd_H": g.H.std(), "sd_G": g.G.std(), "sd_real": g.spread.std()}).round(2).to_string())
print("\nsame table, the sunniest tenth of 1 July to 10 October 2025 (backtest)")
D = P[(P.delivery_day >= "2025-07-01") & (P.delivery_day <= "2025-10-10")]; b5 = D.groupby("delivery_day").solar_px.mean(); S5 = D[D.delivery_day.isin(b5.index[b5 >= b5.quantile(0.9)])]
g = S5.groupby(blk(S5), observed=True)
print(pd.DataFrame({"hours": g.size(), "FR": g.fr.mean(), "DE": g.de.mean(), "spread": g.spread.mean(), "abs": g.spread.apply(lambda s: s.abs().mean()), "yest": g.apply(lambda x: (x.yesterday - x.spread).abs().mean()),
    "H": g.apply(lambda x: (x.H - x.spread).abs().mean()), "G": g.apply(lambda x: (x.G - x.spread).abs().mean()), "bias_H": g.apply(lambda x: (x.H - x.spread).mean()), "bias_G": g.apply(lambda x: (x.G - x.spread).mean())}).round(2).to_string())
print("\nopened window, all days, first 20 days against the rest: bias and error (the break came at the start of July)")
for lab, X in (("1-20 July", T[T.delivery_day <= "2026-07-20"]), ("21 July-10 Oct", T[T.delivery_day > "2026-07-20"])):
    print("  %-15s days %d spread %.1f |" % (lab, X.delivery_day.nunique(), X.spread.mean()), " ".join("%s %.2f (bias %+.1f)" % (n, (X[n] - X.spread).abs().mean(), (X[n] - X.spread).mean()) for n in ("yesterday", "H", "G", "W180", "C")))
