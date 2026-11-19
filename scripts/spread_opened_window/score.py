import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/gerardorosch02/thermo-fr/src")
from thermo_fr.forecast.shape import block_bootstrap_mean
P = pd.read_pickle("all.pkl")
for f in sys.argv[1:]:
    V = pd.read_pickle(f)
    for c in V.columns: P[c] = V[c]
P["B"] = (P.H + P.G) / 2
models = [c for c in ("yesterday", "H", "G", "C", "B") if c in P] + [c for c in P.columns if c[:2] in ("G_", "H_", "N_", "B_")]
def table(S, label):
    de = lambda n: (S[n] - S.spread).abs().groupby(S.delivery_day).mean()
    print(f"\n{label}: {S.delivery_day.nunique()} days")
    for n in models:
        d = (de(n) - de("G")).to_numpy(); lo, hi = block_bootstrap_mean(d, block=7, resamples=2000, seed=0, level=0.95)
        by = S.groupby("delivery_day").agg(w=("wind_px", "mean"), s=("solar_px", "mean")); e = (S[n] - S.spread).abs(); eg = (S["H"] - S.spread).abs()
        sl = {"wkd": S.dow >= 5, "hol": S.holiday.astype(bool) | (S.de_holiday > 0), "wind": S.delivery_day.isin(by.index[by.w >= by.w.quantile(0.9)]), "sun": S.delivery_day.isin(by.index[by.s >= by.s.quantile(0.9)]), "top5": S.spread.abs() >= S.spread.abs().quantile(0.95)}
        base = S.groupby("delivery_day")[[n, "spread"]].mean()
        print("  %-10s MAE %6.2f | vs G %+5.2f [%+5.2f, %+5.2f] | bias %+5.2f | base %5.2f | vs H by slice: %s" % (n, e.mean(), d.mean(), lo, hi, (S[n] - S.spread).mean(), (base[n] - base.spread).abs().mean(),
              " ".join("%s %+5.2f" % (k, e[np.asarray(m)].mean() - eg[np.asarray(m)].mean()) for k, m in sl.items())))
table(P[P.part == "dev"], "BACKTEST 2024-05-01 to 2026-06-30")
T = P[P.part == "test"]; table(T, "OPENED WINDOW 2026-07-01 to 2026-10-10 (now design data)")
print("\nopened window by month, MAE"); print(pd.DataFrame({n: (T[n] - T.spread).abs().groupby(T.delivery_day.dt.strftime("%m")).mean() for n in models}).round(2).to_string())
md = P.delivery_day.dt.strftime("%m-%d"); 
for yr in (2024, 2025):
    S = P[(P.delivery_day.dt.year == yr) & (md >= "07-01") & (md <= "10-10")]; print(yr, "same window:", " ".join("%s %.2f" % (n, (S[n] - S.spread).abs().mean()) for n in models))
