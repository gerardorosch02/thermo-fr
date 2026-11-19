import pickle, numpy as np, pandas as pd
P = pd.read_pickle("all.pkl"); F = pickle.load(open("feat.pkl", "rb")); C, fr = F["C"], F["frame"]; pd.set_option("display.width", 260)
cols = [c for c in C.columns]
print(len(cols), cols)
d = fr["delivery_day"]; md = d.dt.strftime("%m-%d"); win = (md >= "07-01") & (md <= "10-10")
train = (d < "2026-07-01") & (d >= "2021-01-01"); test = (d >= "2026-07-01") & (d <= "2026-10-10")
rows = []
for c in cols:
    x = C[c]; tr = x[train].dropna(); te = x[test].dropna()
    if not len(te) or tr.std() == 0: continue
    lo, hi = tr.quantile(0.005), tr.quantile(0.995)
    rows.append({"feature": c, "mean_2024": x[win & (d.dt.year == 2024)].mean(), "mean_2025": x[win & (d.dt.year == 2025)].mean(), "mean_2026": te.mean(), "shift_in_sd": (te.mean() - x[win & (d.dt.year == 2025)].mean()) / tr.std(),
                 "above_train_99.5": (te > hi).mean(), "below_train_0.5": (te < lo).mean()})
R = pd.DataFrame(rows).set_index("feature")
print("\nFeatures whose window mean moved most against 2025 (in training standard deviations), or that leave the training range")
print(R.reindex(R.shift_in_sd.abs().sort_values(ascending=False).index).head(22).round(3).to_string())
print("\nOut of training range on more than 5% of test hours:"); print(R[(R["above_train_99.5"] > 0.05) | (R["below_train_0.5"] > 0.05)].round(3).to_string())
