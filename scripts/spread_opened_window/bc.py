"""Online correction from the model's own recent errors: forecast + shrink * mean error of the previous k days at the same Paris hour.
The spread of D-1 is known at 10:05 on D-1, so the error of the forecast for D-1 is known when D is forecast."""
import sys, numpy as np, pandas as pd
P = pd.read_pickle("all.pkl")
for f in sys.argv[1:]:
    V = pd.read_pickle(f)
    for c in V.columns: P[c] = V[c]
def corrected(name, k, shrink, by="hour"):
    res = (P.spread - P[name])
    if by == "hour":
        piv = pd.DataFrame({"d": P.delivery_day, "h": P.hour, "r": res}).pivot_table(index="d", columns="h", values="r").asfreq("D")
        c = piv.shift(1).rolling(k, min_periods=3).mean().stack(future_stack=True)
        add = c.reindex(pd.MultiIndex.from_arrays([P.delivery_day, P.hour])).to_numpy()
    else:
        daily = res.groupby(P.delivery_day).mean().asfreq("D"); add = P.delivery_day.map(daily.shift(1).rolling(k, min_periods=3).mean()).to_numpy()
    return P[name] + shrink * np.nan_to_num(add)
rows = []
for name in [c for c in ("H", "G", "G_w365", "G_w180", "G_w90", "G_wk", "G_w180_wk") if c in P]:
    for k, s, by in ((0, 0, "-"), (7, 0.5, "hour"), (7, 1.0, "hour"), (14, 0.5, "hour"), (14, 1.0, "hour"), (7, 0.5, "day"), (7, 1.0, "day"), (14, 1.0, "day")):
        f = P[name] if k == 0 else corrected(name, k, s, by); e = (f - P.spread).abs()
        rows.append({"model": name, "k": k, "shrink": s, "by": by, "backtest": e[P.part == "dev"].mean(), "window": e[P.part == "test"].mean(), "bias_window": (f - P.spread)[P.part == "test"].mean(),
                     "window_Jul": e[(P.part == "test") & (P.delivery_day < "2026-08-01")].mean(), "window_Oct": e[(P.part == "test") & (P.delivery_day >= "2026-10-01")].mean()})
pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(2).to_string(index=False))
