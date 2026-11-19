"""Opened window, design work for a NEW model (not H, G or C). Variants of G motivated by the level break of July 2026:
a target measured from the recent level, and recency weights. Walk-forward monthly refits as before. Development evidence only."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from thermo_fr.forecast import spread_eval as se
ROOT = Path("/home/claude/gerardorosch02/thermo-fr")
stored = se.load_stored(ROOT / "data/forecast/inputs.csv", ROOT / "data/forecast/neighbour_fundamentals.csv", "/home/claude/work/price-seq/neighbour_sequences.csv.gz",
                        "/home/claude/work/german-points/german_wind_points.csv.gz", "/home/claude/work/german-points/german_solar_points.csv.gz")
data = se.extend(stored, se.parse_raw(Path("/home/claude/work/spread-eval/raw")))
feat = se.features(data); frame, y = feat["frame"], feat["y"]; H, G = feat["sets"]["H"], feat["sets"]["G"]
VARIANTS = {  # name: (inputs, anchor column or None, half-life in days or None)
    "G_m7": (G, "s_m7", None), "H_m7": (H, "s_m7", None), "G_w365": (G, None, 365), "G_m7_w365": (G, "s_m7", 365), "G_l1": (G, "s_l1", None),
    "H_w365": (H, None, 365), "G_w180": (G, None, 180), "G_w730": (G, None, 730), "G_w365_wk": (G, None, 365), "H_w365_wk": (H, None, 365), "G_wk": (G, None, None), "G_w180_wk": (G, None, 180), "G_w90": (G, None, 90)}
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
pred = pd.DataFrame(index=frame.index, columns=names, dtype=float)
weekly = all(n.endswith("_wk") for n in names); assert weekly or not any(n.endswith("_wk") for n in names)
starts = pd.date_range(se.DEV_FIRST, se.TEST_LAST, freq="7D") if weekly else se.months_of(se.DEV_FIRST, se.TEST_LAST)  # weekly: refit every seventh day from 2024-05-01
for start in starts:
    test = np.asarray((frame["delivery_day"] >= start) & (frame["delivery_day"] < (start + pd.Timedelta(days=7) if weekly else start + pd.DateOffset(months=1))))
    for name in names:
        X, anchor, half = VARIANTS[name]
        a = X[anchor] if anchor else pd.Series(0.0, index=X.index)
        target = y - a
        train = np.asarray((frame["delivery_day"] < start) & target.notna())
        w = None if half is None else np.power(0.5, (start - frame["delivery_day"][train]).dt.days.to_numpy() / half)
        m = lgb.LGBMRegressor(**se.PARAMS).fit(X[train], target[train], sample_weight=w)
        pred.loc[test, name] = m.predict(X[test]) + a[test].to_numpy()
    print("fitted", start.strftime("%Y-%m-%d"), flush=True)
old = pd.read_pickle("/home/claude/work/open/all.pkl")
out = pred.reindex(old.index); out.to_pickle(f"/home/claude/work/open/variants_{sys.argv[2]}.pkl")
print(out.notna().mean().round(4).to_dict())
