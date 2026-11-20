"""Decode the forecast shards published on the PriceFM project page (third-party data, aggregates only) and compare the
France minus DE-LU spread implied by its median forecasts with the stored design-period predictions of the spread models.
Usage: python -I decode.py <Project_page dir> <comparator_predictions.csv.gz>"""
import gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

page, mine_path = Path(sys.argv[1]), sys.argv[2]
man = json.loads((page / "data" / "manifest.json").read_text())
R, M, C = man["regions"], man["models"], man["cases"]
MISSING = man["missing"]

def undelta(a):
    """Inverse of time_deltas along axis 2: a value is a difference when it and the decoded previous value are both present."""
    a = a.astype(np.int64)
    out = a.copy()
    for k in range(1, a.shape[2]):
        cur, prev = a[:, :, k], out[:, :, k - 1]
        valid = (cur != MISSING) & (prev != MISSING)
        out[:, :, k] = np.where(valid, prev + cur, cur)
    return out

def load(case):
    days, truth, fc = [], [], []
    for key, sh in sorted(man["shards"].items()):
        if not key.startswith(case + "/"):
            continue
        assert sh["dtype"] == "int16"
        raw = np.frombuffer(gzip.open(page / sh["file"]).read(), dtype="<i2")
        n = len(sh["dates"])
        t = raw[: n * 40 * 96].reshape(n, 40, 96)
        f = raw[n * 40 * 96:].reshape(n, 40, 96, len(M), 3)
        t, f = undelta(t).astype(float), undelta(f).astype(float)
        t[t == MISSING] = np.nan; f[f == MISSING] = np.nan
        days += sh["dates"]; truth.append(t / man["truth_scale"]); fc.append(f / man["forecast_scale"])
    return days, np.concatenate(truth), np.concatenate(fc)

out = {}
data = {case: load(case) for case in C}
days = data["Realistic"][0]
print("days", days[0], days[-1], len(days))
# 1. decoding check against the published table (all regions, all days): MAE of the median
pub = pd.read_csv(page / "data" / "results.csv").set_index(["model", "feature_case"])["MAE"]
for case in C:
    _, t, f = data[case]
    for model in ("PriceFM_Plus", "MLP", "Chronos_2", "Persistence_1"):
        mae = np.nanmean(np.abs(f[..., M.index(model), 1] - t))
        print(f"check {case:9s} {model:13s} decoded {mae:6.2f} published {pub[(model, case)]:6.2f}")
# 2. January to June 2026 only
half = np.array([d < "2026-07-01" for d in days])
fr, de = R.index("FR"), R.index("DE_LU")
stamps = pd.DatetimeIndex([pd.Timestamp(str(d), tz="UTC") + pd.Timedelta(minutes=15 * s) for d in np.array(days)[half] for s in range(96)])
def hourly(x):  # (days, 96) -> hourly mean series on UTC hours; an hour needs all four quarters
    s = pd.Series(x[half].reshape(-1), index=stamps)
    g = s.groupby(s.index.floor("h"))
    return g.mean().where(g.count() == 4)
rows = []
truth = data["Realistic"][1]
truth_spread = hourly(truth[:, fr]) - hourly(truth[:, de])
mine = pd.read_csv(mine_path, index_col=0)
mine.index = pd.to_datetime(mine.index, utc=True)
common = truth_spread.dropna().index.intersection(mine["spread"].dropna().index)
gap = (truth_spread[common] - mine.loc[common, "spread"]).abs()
print("hours in common", len(common), "first", common.min(), "last", common.max())
print("truth agreement: mean abs gap", round(float(gap.mean()), 4), "share of hours within 0.05:", round(float((gap < 0.05).mean()), 4), "max", round(float(gap.max()), 2))
y = mine.loc[common, "spread"]
for name in ("yesterday", "H", "HW", "G", "W", "C"):
    rows.append(("stored spread models", name, "", float((mine.loc[common, name] - y).abs().mean())))
for case in C:
    _, t, f = data[case]
    for model in ("PriceFM_Plus", "MLP", "Chronos_2", "TimesFM_3", "Persistence_1"):
        q = f[..., M.index(model), 1]
        spread = hourly(q[:, fr]) - hourly(q[:, de])
        rows.append(("spread from PriceFM page", model, case, float((spread[common] - y).abs().mean())))
        if model == "PriceFM_Plus":
            for reg, i in (("FR", fr), ("DE_LU", de)):
                rows.append((f"level {reg}, quarter-hours", model, case, float(np.nanmean(np.abs(q[half][:, i] - t[half][:, i])))))
            # by hour of the UTC day: is the start of the day easier than the rest?
            if case == "Realistic":
                e = np.abs(q[half] - t[half])
                for reg, i in (("FR", fr), ("DE_LU", de)):
                    by = np.nanmean(e[:, i].reshape(-1, 24, 4), axis=(0, 2))
                    print(f"PriceFM_Plus Realistic level MAE by UTC hour {reg}:", " ".join(f"{v:.0f}" for v in by))
res = pd.DataFrame(rows, columns=["what", "model", "case", "mae"])
print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
