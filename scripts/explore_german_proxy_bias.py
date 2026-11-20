"""Where the German onshore wind proxy's low bias sits (development data to 2026-06-30, aggregates only)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/gerardorosch02/thermo-fr/src")
from thermo_fr.forecast import gen_proxy
from thermo_fr.forecast.gen_proxy import ProxySpec
from thermo_fr.forecast.timing import delivery_days
from thermo_fr.forecast.wind_proxy import power_curve
LAST = "2026-06-30"
def read(path):
    t = pd.read_csv(path); t.index = pd.to_datetime(t.pop("timestamp_utc"), utc=True)
    return t[delivery_days(t.index) <= pd.Timestamp(LAST)].sort_index()
nf = read("/home/claude/work/spread-live-root/static/neighbour_fundamentals.csv")
pts = read("/home/claude/work/german-points/german_wind_points.csv.gz").asfreq("h")
on = tuple(c for c in pts.columns if "_pt_Off" not in c)
tso = nf["wind_onshore_fc_de_lu_mw"]
spec = ProxySpec("wind", "px", on, power_curve, Path("unused"))
px, fits = gen_proxy.rolling_proxy(pts[list(on)], tso, spec)
q = pd.DataFrame({"px": px, "fc": tso}).dropna(); q = q[delivery_days(q.index) >= pd.Timestamp(fits[0]["month"] + "-01")]
print("hours", len(q), "mean fc", round(q.fc.mean()), "bias", round((q.px - q.fc).mean()), "corr", round(np.corrcoef(q.px, q.fc)[0, 1], 3), "rank corr", round(q.px.rank().corr(q.fc.rank()), 3))
print("in-sample bias of the monthly fits, MW: mean", round(np.mean([f["bias_mw"] for f in fits])), "min", round(min(f["bias_mw"] for f in fits)), "max", round(max(f["bias_mw"] for f in fits)), "| mean of training target", round(np.mean([f["mean_actual_mw"] for f in fits])))
q["decile"] = pd.qcut(q.fc, 10, labels=False)
g = q.groupby("decile").agg(fc=("fc", "mean"), px=("px", "mean"))
print("by tenth of the operator forecast: forecast mean / proxy mean"); print((g.round(0)).T.to_string())
by = q.groupby(q.index.year).apply(lambda d: pd.Series({"fc": d.fc.mean(), "bias": (d.px - d.fc).mean(), "corr": np.corrcoef(d.px, d.fc)[0, 1]}))
print(by.round(3).to_string())
# mean wind speed across points, and share of point-hours below cut-in
from thermo_fr.forecast import wind_proxy as wp
v = pts.loc[q.index, list(on)].to_numpy()
print("cut-in", wp.CUT_IN_MS, "rated", wp.RATED_MS, "cut-out", wp.CUT_OUT_MS, "hub", wp.HUB_HEIGHT_M, "| share of point-hours below cut-in", round(float((v < wp.CUT_IN_MS).mean()), 3), "| at or above rated", round(float((v >= wp.RATED_MS).mean()), 3), "| median speed", round(float(np.nanmedian(v)), 2))
# three cheap alternatives, each walk-forward on the same monthly schedule
def walk(transform, intercept=False):
    from scipy.optimize import nnls
    out = pd.Series(np.nan, index=pts.index); days = delivery_days(pts.index)
    for f in fits:
        month = pd.Timestamp(f["month"] + "-01"); last = pd.Timestamp(f["last_train_day"])
        tr = (days <= last) & (days > last - pd.Timedelta(days=365)) & pts[list(on)].notna().all(axis=1).to_numpy() & tso.reindex(pts.index).notna().to_numpy()
        A = transform(pts.loc[tr, list(on)].to_numpy()); b = tso.reindex(pts.index)[tr].to_numpy()
        if intercept: A = np.c_[A, np.ones(len(A))]
        w, _ = nnls(A, b)
        rows = (days >= month) & (days < month + pd.DateOffset(months=1))
        B = transform(pts.loc[rows, list(on)].to_numpy())
        if intercept: B = np.c_[B, np.ones(len(B))]
        out[rows] = B @ w
    d = pd.DataFrame({"px": out, "fc": tso}).reindex(q.index).dropna()
    e = d.px - d.fc
    return round(e.abs().mean()), round(e.mean()), round(np.corrcoef(d.px, d.fc)[0, 1], 3), round(d.px.rank().corr(d.fc.rank()), 3)
def smooth(sd):  # the cubic curve averaged over a normal spread of speeds around the point value: a fleet, not one turbine
    z = np.linspace(-2.5, 2.5, 11); w = np.exp(-z**2 / 2); w /= w.sum()
    return lambda v: sum(wi * power_curve(np.clip(v + sd * zi, 0, None)) for wi, zi in zip(w, z))
print("variant: MAE, bias, corr, rank corr")
print("  as built              ", walk(power_curve))
print("  as built + intercept  ", walk(power_curve, True))
for sd in (1.5, 2.5, 3.5):
    print(f"  curve smoothed sd {sd}  ", walk(smooth(sd)), "| + intercept", walk(smooth(sd), True))
# trailing ratio correction: scale by the median of forecast/proxy over the 60 days ending two days before each day
r = (q.fc / q.px.clip(lower=200)); dd = delivery_days(q.index)
daily = r.groupby(dd).median(); corr = daily.rolling(60, min_periods=20).median().shift(2)
adj = q.px * corr.reindex(dd).to_numpy(); e = (adj - q.fc).dropna()
print("  trailing median ratio ", round(e.abs().mean()), round(e.mean()), round(np.corrcoef(adj.dropna(), q.fc[adj.notna()])[0, 1], 3), "on", len(e), "hours")
