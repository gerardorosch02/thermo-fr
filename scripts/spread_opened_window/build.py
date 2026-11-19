"""Opened test window (owner's approval 2026-11-19 18:35 London): assemble forecasts and inputs for diagnosis. No fit."""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from thermo_fr.forecast import spread_eval as se
stored = se.load_stored("data/forecast/inputs.csv", "data/forecast/neighbour_fundamentals.csv", "/home/claude/work/price-seq/neighbour_sequences.csv.gz",
                        "/home/claude/work/german-points/german_wind_points.csv.gz", "/home/claude/work/german-points/german_solar_points.csv.gz")
data = se.extend(stored, se.parse_raw(Path("/home/claude/work/spread-eval/raw")))
feat = se.features(data)
dev = pd.read_csv("/home/claude/work/spread-eval/dev_predictions.csv.gz", index_col=0, parse_dates=["delivery_day"]); test = pd.read_csv("/home/claude/work/spread-eval/test_predictions.csv.gz", index_col=0, parse_dates=["delivery_day"])
P = pd.concat([dev.assign(part="dev"), test.assign(part="test")]); P.index = pd.to_datetime(P.index, utc=True)
C = feat["sets"]["C"]; fr = feat["frame"]
P["fr"], P["de"] = fr["fr"].reindex(P.index), fr["de"].reindex(P.index)
for c in C.columns:
    if c not in P.columns: P[c] = C[c].reindex(P.index)
P.to_pickle("/home/claude/work/open/all.pkl")
pickle.dump({"C": C, "frame": fr, "y": feat["y"]}, open("/home/claude/work/open/feat.pkl", "wb"))
print(P.groupby("part").size().to_dict(), len(C.columns))
