"""The benchmark the review asks for: add the median of (reference minus proxy) from the training window, walk-forward."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/gerardorosch02/thermo-fr/src")
from scipy.optimize import nnls
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
on = [c for c in pts.columns if "_pt_Off" not in c]
tso = nf["wind_onshore_fc_de_lu_mw"].reindex(pts.index)
_, fits = gen_proxy.rolling_proxy(pts[on], tso, ProxySpec("wind", "px", tuple(on), power_curve, Path("unused")))
days = delivery_days(pts.index); ok = pts[on].notna().all(axis=1).to_numpy() & tso.notna().to_numpy()
plain, add_med, add_hour = (pd.Series(np.nan, index=pts.index) for _ in range(3))
hour = pts.index.tz_convert("Europe/Paris").hour
for f in fits:
    month = pd.Timestamp(f["month"] + "-01"); last = pd.Timestamp(f["last_train_day"])
    tr = ok & (days <= last) & (days > last - pd.Timedelta(days=365))
    A = power_curve(pts.loc[tr, on].to_numpy()); b = tso[tr].to_numpy(); w, _ = nnls(A, b)
    resid = pd.Series(b - A @ w, index=pts.index[tr])
    rows = (days >= month) & (days < month + pd.DateOffset(months=1))
    base = power_curve(pts.loc[rows, on].to_numpy()) @ w
    plain[rows] = base; add_med[rows] = base + resid.median()
    by_hour = resid.groupby(hour[tr]).median()
    add_hour[rows] = base + by_hour.reindex(hour[rows]).to_numpy()
first = pd.Timestamp(fits[0]["month"] + "-01")
for name, s in (("as built", plain), ("plus median of the training error", add_med), ("plus median by hour of day", add_hour)):
    d = pd.DataFrame({"px": s, "fc": tso})[days >= first].dropna(); e = d.px - d.fc
    ramp = (d.px.diff() - d.fc.diff()).abs().mean()
    print(f"{name:36s} MAE {e.abs().mean():6.0f} bias {e.mean():6.0f} corr {np.corrcoef(d.px, d.fc)[0,1]:.3f} hourly ramp error {ramp:5.0f} hours {len(d)}")
