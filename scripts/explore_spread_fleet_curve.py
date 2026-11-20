"""Exploration on development data (to 2026-06-30): the frozen spread models G and W refitted with one change, the wind
power curve averaged over a spread of speeds (a fleet in place of one turbine). Nothing frozen is edited: the curve is
replaced in memory. Compared with the stored predictions of G and W."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/gerardorosch02/thermo-fr/src")
from thermo_fr.forecast import spread_eval as se, spread_w as sw
from thermo_fr.forecast.wind_proxy import power_curve
S = "/home/claude/work/spread-live-root/static/"
SD = float(sys.argv[1])
z = np.linspace(-2.5, 2.5, 11); w = np.exp(-z**2 / 2); w /= w.sum()
se.power_curve = lambda v: sum(wi * power_curve(np.clip(np.asarray(v, float) + SD * zi, 0, None)) for wi, zi in zip(w, z))
data = se.load_stored(S + "inputs.csv", S + "neighbour_fundamentals.csv", S + "neighbour_sequences.csv.gz", S + "german_wind_points.csv.gz", S + "german_solar_points.csv.gz")
t = time.time(); feat = se.features(data); print("features", round(time.time() - t), "s; first proxy month", feat["first_month"].date(), flush=True)
old = pd.read_csv("/home/claude/work/spread-w/w_predictions.csv.gz", index_col=0); old.index = pd.to_datetime(old.index, utc=True)
old = old[(old.delivery_day >= se.DEV_FIRST) & (old.delivery_day <= se.DEV_LAST)]
px = feat["px"].reindex(old.index)
print("wind proxy: stored mean", round(old.wind_px.mean()), "new mean", round(px["wind"].mean()), "corr", round(np.corrcoef(old.wind_px, px["wind"])[0, 1], 4), flush=True)
t = time.time(); g = se.fit_months(feat, se.months_of(se.DEV_FIRST, se.DEV_LAST), models=("G",))["G"].reindex(old.index); print("G fitted", round(time.time() - t), "s", flush=True)
t = time.time(); wv = sw.fit_w(feat, se.DEV_FIRST, se.DEV_LAST).reindex(old.index); print("W fitted", round(time.time() - t), "s", flush=True)
def paired(new, base):
    d = ((new - old.spread).abs() - (old[base] - old.spread).abs()).groupby(old.delivery_day).mean().to_numpy()
    rng = np.random.default_rng(0); n = len(d); k = int(np.ceil(n / 7)); means = np.empty(5000)
    for i in range(5000):
        starts = rng.integers(0, n - 6, k); means[i] = d[(starts[:, None] + np.arange(7)).ravel()[:n]].mean()
    return round(float(d.mean()), 2), [round(float(np.quantile(means, 0.025)), 2), round(float(np.quantile(means, 0.975)), 2)]
for name, new in (("G", g), ("W", wv)):
    print(name, "stored", round(float((old[name] - old.spread).abs().mean()), 2), "smoothed curve", round(float((new - old.spread).abs().mean()), 2), "paired difference and 95% interval", paired(new, name), "hours", int(new.notna().sum()), flush=True)
print("for reference: H", round(float((old.H - old.spread).abs().mean()), 2), "C", round(float((old.C - old.spread).abs().mean()), 2))
windy = old.groupby("delivery_day").wind_px.mean(); top = windy[windy >= windy.quantile(0.9)].index; low = windy[windy <= windy.quantile(0.1)].index
for label, days in (("windiest tenth", top), ("calmest tenth", low)):
    m = old.delivery_day.isin(days)
    print(label, {n: (round(float((old.loc[m, n] - old.loc[m, "spread"]).abs().mean()), 2), round(float((new[m] - old.loc[m, "spread"]).abs().mean()), 2)) for n, new in (("G", g), ("W", wv))})
