"""Quantile sorting, conformal margins from past rows only, trailing benchmarks, pinball and Brier metrics, the walk-forward on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from thermo_fr.forecast import models, probabilistic as pb
from thermo_fr.forecast.features import build_features


def frame_for(days, hours=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range(days[0], days[1], freq="D"):
        for h in range(hours):
            actual = 50 + 20 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 10)
            rows.append({"delivery_day": d, "hour": h, "dow": d.dayofweek, "holiday": 0, "wind_mw": rng.uniform(0, 1), "solar_mw": rng.uniform(0, 1),
                         "actual": actual, "point": actual + rng.normal(0, 8), "naive_day": actual + rng.normal(0, 12),
                         "q10": actual - 5 + rng.normal(0, 2), "q50": actual + rng.normal(0, 2), "q90": actual + 5 + rng.normal(0, 2),
                         "strict": True})
    f = pd.DataFrame(rows)
    f.index = pd.date_range(f["delivery_day"].iloc[0], periods=len(f), freq="1h", tz="UTC")
    return f


def test_sorting_fixes_crossing_quantiles():
    f = pd.DataFrame({"q10": [3.0, 1.0], "q50": [2.0, 2.0], "q90": [1.0, 3.0]})
    s = pb.sort_quantiles(f)
    assert s["q10"].tolist() == [1.0, 1.0] and s["q50"].tolist() == [2.0, 2.0] and s["q90"].tolist() == [3.0, 3.0]


def test_spike_threshold_uses_only_earlier_days():
    days = pd.date_range("2024-01-01", "2024-03-10", freq="D")
    idx = pd.date_range(days[0], periods=len(days) * 24, freq="1h", tz="UTC")
    day_series = pd.Series(np.repeat(days, 24), index=idx)
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)  # strictly increasing: every day's prices exceed all earlier ones
    thr = pb.spike_thresholds(y, day_series, trailing_days=365)
    assert thr.iloc[:30 * 24].isna().all()  # fewer than 30 earlier days
    later = thr.iloc[40 * 24:40 * 24 + 24]
    assert (later < y.iloc[40 * 24:40 * 24 + 24].min()).all() and later.nunique() == 1  # below every price of the day itself
    labels = pb.event_labels(y, thr)
    assert labels["spike"].iloc[40 * 24:].eq(1.0).all() and labels["negative"].sum() == 0


def test_conformal_margin_comes_from_past_days_and_widens_too_narrow_intervals():
    f = frame_for(("2024-01-01", "2024-06-30"))
    # make the raw interval far too narrow: coverage well below 80%
    f["q10"], f["q90"] = f["actual"] - 1 + np.random.default_rng(1).normal(0, 3, len(f)), f["actual"] + 1 + np.random.default_rng(2).normal(0, 3, len(f))
    out = pb.conformal_interval(f, window_days=60, alpha=0.2, min_days=20)
    first = out[out["delivery_day"] < "2024-01-21"]
    assert (first["conformal_margin"] == 0).all()  # no margin before 20 past days exist
    later = out[out["delivery_day"] >= "2024-04-01"]
    raw_cov = ((later["actual"] >= later["q10"]) & (later["actual"] <= later["q90"])).mean()
    cal_cov = ((later["actual"] >= later["lo"]) & (later["actual"] <= later["hi"])).mean()
    assert raw_cov < 0.6 and 0.72 <= cal_cov <= 0.88 and (later["conformal_margin"] > 0).all()
    # changing the future does not change today's margin
    g = f.copy()
    g.loc[g["delivery_day"] >= "2024-05-01", "actual"] += 1000
    again = pb.conformal_interval(g, window_days=60, alpha=0.2, min_days=20)
    april = out["delivery_day"].between("2024-04-01", "2024-04-30")
    assert np.allclose(out.loc[april, "conformal_margin"], again.loc[april, "conformal_margin"])


def test_trailing_benchmarks_and_event_benchmarks_use_the_past_only():
    f = frame_for(("2024-01-01", "2024-04-30"))
    b = pb.trailing_error_quantiles(f, "point", trailing_days=365, min_days=30)
    early = b[b["delivery_day"] < "2024-01-31"]
    assert early["point_lo"].isna().all()
    late = b[b["delivery_day"] >= "2024-03-01"].dropna(subset=["point_lo"])
    assert (late["point_lo"] <= late["point_mid"]).all() and (late["point_mid"] <= late["point_hi"]).all()
    cov = ((late["actual"] >= late["point_lo"]) & (late["actual"] <= late["point_hi"])).mean()
    assert 0.65 < cov < 0.95
    f["is_negative"] = (f["hour"] == 3).astype(float)  # the event happens every day at hour 3 and never otherwise
    e = pb.event_benchmarks(f, "negative", trailing_days=365, last_days=7)
    late = e[e["delivery_day"] >= "2024-02-01"]
    assert late.loc[late["hour"] == 3, "clim_negative"].eq(1.0).all() and late.loc[late["hour"] != 3, "clim_negative"].eq(0.0).all()
    assert late.loc[late["hour"] == 3, "last7_negative"].eq(1.0).all()
    assert e[e["delivery_day"] == "2024-01-01"]["clim_negative"].isna().all()


def test_pinball_brier_reliability_and_acceptance_rules():
    y = np.array([10.0, 20.0, 30.0])
    assert pb.pinball(y, np.array([10.0, 20.0, 30.0]), 0.5) == 0.0
    assert pb.pinball(y, y - 10, 0.9) == pytest.approx(9.0) and pb.pinball(y, y - 10, 0.1) == pytest.approx(1.0)
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    assert pb.brier(labels, labels) == 0.0 and pb.brier(labels, np.full(10, 0.5)) == pytest.approx(0.25)
    table = pb.reliability_table(labels, np.linspace(0.05, 0.95, 10))
    assert len(table) == 10 and table["n"].sum() == 10 and list(table.columns) == ["bin", "n", "mean_forecast", "observed_frequency"]
    f = frame_for(("2024-01-01", "2024-03-31"))
    f["lo"], f["hi"] = f["q10"] - 4, f["q90"] + 4
    f["point_lo"], f["point_mid"], f["point_hi"] = f["point"] - 30, f["point"], f["point"] + 30
    f["naive_day_lo"], f["naive_day_mid"], f["naive_day_hi"] = f["naive_day"] - 30, f["naive_day"], f["naive_day"] + 30
    result = pb.evaluate_intervals(f)
    assert set(result["by_slice"]) >= {"all", "weekends", "top_5pct_price_hours"} and result["overall"]["model"]["hours"] == len(f)
    assert result["overall"]["model"]["pinball_mean"] < result["overall"]["bench_point"]["pinball_mean"]
    assert isinstance(result["accepted"], bool)
    rng = np.random.default_rng(5)
    f["is_negative"] = (rng.uniform(size=len(f)) < 0.1).astype(float)
    f["p_negative"] = np.where(f["is_negative"] == 1, 0.6, 0.05)
    f["clim_negative"], f["last7_negative"] = 0.1, 0.1
    f["is_spike"], f["p_spike"], f["clim_spike"], f["last7_spike"] = 0.0, 0.01, 0.05, 0.0
    events = pb.evaluate_events(f)
    assert events["negative"]["accepted"] and events["negative"]["bss_vs_clim"] > 0 and events["negative"]["events"] == int(f["is_negative"].sum())
    assert events["spike"]["events"] == 0 and not events["spike"]["accepted"]


def test_walk_forward_prob_on_synthetic_data(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 30)
    inputs = synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")
    table = build_features(inputs, "honest_v2")
    bt = pb.walk_forward_prob(table, "2024-02-01", "2024-05-01", log=lambda *_: None, classifier_params={"n_estimators": 30})
    pred = bt.predictions
    assert bt.months == ["2024-02", "2024-03", "2024-04"]
    assert {"point", "q10", "q50", "q90", "p_negative", "p_spike", "is_negative", "is_spike", "spike_threshold", "strict"} <= set(pred.columns)
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()
    assert pred["p_spike"].between(0, 1).all()
    ready = pb.prepare(pred)
    result = pb.evaluate_intervals(ready)
    assert result["overall"]["model"]["hours"] > 1000 and 0 < result["overall"]["model"]["coverage_10_90"] <= 1
    events = pb.evaluate_events(ready)
    assert "spike" in events and events["spike"]["hours"] > 0
