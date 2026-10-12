"""Isotonic recalibration: walk-forward without leakage, breakpoints, the acceptance metrics, the yearly diagnostic."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import calibration as cal


def synthetic_raw(start="2023-02-01", months=30, seed=0, overconfident=True):
    """Hourly out-of-sample predictions whose true frequency is a monotone distortion of the raw probability."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=months * 30, freq="D")
    idx = pd.date_range(days[0], periods=len(days) * 24, freq="1h", tz="UTC")
    p = np.clip(rng.beta(0.3, 6, len(idx)), 0, 1)
    truth = np.clip(p ** 0.6 if overconfident else p, 0, 1)  # a raw 1% is really about 6%, a raw 10% about 25%
    y = (rng.uniform(size=len(idx)) < truth).astype(float)
    frame = pd.DataFrame({"delivery_day": np.repeat(days, 24), "p_negative": p, "is_negative": y, "strict": True}, index=idx)
    frame["clim_negative"] = float(y.mean())
    frame["last7_negative"] = pd.Series(y, index=idx).rolling(24 * 7, min_periods=24).mean().shift(24).fillna(float(y.mean())).to_numpy()
    return frame


def test_walk_forward_uses_only_earlier_months_and_respects_the_minimum_history():
    raw = synthetic_raw(months=16)
    months = sorted(set(raw["delivery_day"].dt.to_period("M")))
    calibrated, fits = cal.walk_forward_calibration(raw, months, min_history_months=12)
    first_fitted = next(m for m, f in fits.items() if f["fitted"])
    assert fits[str(months[0])]["fitted"] is False and fits[str(months[11])]["fitted"] is False and first_fitted == str(months[12])
    assert calibrated[raw["delivery_day"].dt.to_period("M") < months[12]].isna().all()
    assert calibrated[raw["delivery_day"].dt.to_period("M") >= months[12]].notna().all()
    # changing the outcomes of a later month does not change an earlier month's calibration
    later = raw.copy()
    later.loc[later["delivery_day"].dt.to_period("M") == months[14], "is_negative"] = 1.0
    again, _ = cal.walk_forward_calibration(later, months, min_history_months=12)
    mask = raw["delivery_day"].dt.to_period("M") == months[12]
    assert np.allclose(calibrated[mask], again[mask])
    # the calibration is monotone in the raw probability within a month
    month_rows = raw[mask].assign(cal=calibrated[mask]).sort_values("p_negative")
    assert (np.diff(month_rows["cal"].to_numpy()) >= -1e-12).all()


def test_breakpoints_reproduce_the_fitted_model_without_scikit_learn():
    raw = synthetic_raw(months=14)
    model = cal.fit_isotonic(raw["p_negative"], raw["is_negative"])
    bp = cal.breakpoints(model)
    probe = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.9, 1.0])
    assert np.allclose(cal.apply_breakpoints(bp, probe), model.predict(probe), atol=1e-5)
    assert (np.diff(cal.apply_breakpoints(bp, np.linspace(0, 1, 50))) >= 0).all()
    assert np.isnan(cal.apply_breakpoints({"x": [], "y": []}, probe)).all()


def test_evaluation_accepts_a_recalibration_that_fixes_an_overconfident_model_and_not_one_that_adds_noise():
    raw = synthetic_raw(months=30, seed=1)
    months = sorted(set(raw["delivery_day"].dt.to_period("M")))
    calibrated, _ = cal.walk_forward_calibration(raw, months)
    scored = raw.join(calibrated)
    result = cal.evaluate_recalibration(scored)
    assert result["accepted"] and result["calibrated"]["brier"] < result["raw"]["brier"]
    assert result["calibrated"]["mace"]["unweighted"] < result["raw"]["mace"]["unweighted"] and result["calibrated"]["bss_vs_clim"] > 0
    assert result["hours"] == int(scored["p_negative_cal"].notna().sum()) and len(result["calibrated"]["low_end_bins"]) == 3
    noisy = scored.copy()
    noisy["p_negative_cal"] = np.clip(noisy["p_negative"] + np.random.default_rng(2).normal(0, 0.3, len(noisy)), 0, 1)
    assert not cal.evaluate_recalibration(noisy)["accepted"]
    m = cal.mace(np.array([0, 1, 1, 0.0]), np.array([0.05, 0.95, 0.15, 0.85]))
    assert m["bins_used"] == 4 and m["unweighted"] == pytest.approx(np.mean([0.05, 0.05, 0.85, 0.85]), abs=1e-6)


def test_pairs_aggregate_and_the_weighted_fit_equals_the_row_fit():
    raw = synthetic_raw(months=14, seed=3)
    pairs = cal.aggregate_pairs(raw["p_negative"], raw["is_negative"])
    assert pairs["n"].sum() == len(raw) and pairs["k"].sum() == raw["is_negative"].sum() and pairs["p"].is_monotonic_increasing
    rows_model = cal.fit_isotonic(np.round(raw["p_negative"], 4), raw["is_negative"])
    pairs_model = cal.fit_from_pairs(pairs)
    probe = np.linspace(0, 0.6, 61)
    assert np.allclose(rows_model.predict(probe), pairs_model.predict(probe), atol=1e-6)
    merged = cal.merge_pairs(pairs, pairs)
    assert merged["n"].sum() == 2 * len(raw) and len(merged) == len(pairs)


def test_build_calibration_from_archives_then_extends_with_settled_live_versions(tmp_path):
    raw = synthetic_raw(months=14, seed=4)
    archive = tmp_path / "prob_raw_selection.parquet"
    raw.to_parquet(archive)
    model_dir = tmp_path / "published" / "model"
    meta = cal.build_calibration(model_dir, "honest_v2", archives=[archive], published_dir=None, now=pd.Timestamp("2026-10-12T15:00Z"), log=lambda *_: None)
    assert meta["pairs_hours"] == len(raw) and meta["last_day"] == str(raw["delivery_day"].max().date()) and len(meta["breakpoints"]["x"]) > 2
    paths = cal.calibration_paths(model_dir, "honest_v2")
    assert paths["meta"].exists() and paths["pairs"].exists() and cal.load_calibration(model_dir, "honest_v2")["pairs_hours"] == len(raw)
    calibrated = cal.apply_calibration(meta, np.array([0.0, 0.01, 0.1, 0.5, 1.0]))
    assert (np.diff(calibrated) >= -1e-12).all() and 0 <= calibrated.min() and calibrated.max() <= 1
    # a published dataset with two settled days after the archive's last day: the pre-market version of each enters the pairs once
    published = tmp_path / "published"
    day1, day2 = "2026-10-20", "2026-10-21"
    rows = []
    for day, issued, p in ((day1, "2026-10-19T08:10:00Z", 0.02), (day1, "2026-10-19T10:30:00Z", 0.9), (day2, "2026-10-20T08:10:00Z", 0.03)):
        for h in range(24):
            ts = (pd.Timestamp(day, tz="Europe/Paris") + pd.Timedelta(hours=h)).tz_convert("UTC")
            rows.append({"delivery_day": day, "issued_at_utc": issued, "feature_set": "honest_v2", "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "p_negative": p + 0.5, "p_negative_raw": p, "calibrated": 1, "hour": h})
    pd.DataFrame(rows).to_csv(published / "probabilistic.csv", index=False)
    actual_rows = [{"delivery_day": r["delivery_day"], "timestamp_utc": r["timestamp_utc"], "hour": r["hour"], "price": -1.0 if r["hour"] == 13 else 40.0}
                   for r in rows if r["issued_at_utc"].endswith("08:10:00Z")]
    pd.DataFrame(actual_rows).to_csv(published / "actuals.csv", index=False)
    live, last = cal.live_pairs_from_published(published, "honest_v2", meta["last_day"])
    assert last == day2 and live["n"].sum() == 48 and live["k"].sum() == 2 and set(live["p"]) == {0.02, 0.03}  # the late 0.9 version is not used
    extended = cal.build_calibration(model_dir, "honest_v2", archives=(), published_dir=published, now=pd.Timestamp("2026-11-02T03:00Z"), log=lambda *_: None)
    assert extended["pairs_hours"] == len(raw) + 48 and extended["pairs_events"] == int(raw["is_negative"].sum()) + 2 and extended["last_day"] == day2
    again = cal.build_calibration(model_dir, "honest_v2", archives=(), published_dir=published, now=pd.Timestamp("2026-12-01T03:00Z"), log=lambda *_: None)
    assert again["pairs_hours"] == extended["pairs_hours"]  # the same days are not added twice
    assert cal.public_calibration(again)["applied"] and "breakpoints" not in cal.public_calibration(again)
    assert cal.load_calibration(tmp_path / "nowhere", "honest_v2") is None and not cal.public_calibration(None)["applied"]


def test_yearly_diagnostic_counts_negative_hours_per_paris_year_and_joins_the_raw_forecasts():
    idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="1h", tz="UTC")
    prices = pd.Series(50.0, index=idx)
    prices[(idx.year == 2022) & (idx.hour == 13)] = -1.0  # 365 negative hours in 2022
    raw = synthetic_raw(start="2023-02-01", months=11)
    table = cal.yearly_diagnostic(prices, raw)
    by = table.set_index("year")
    assert by.loc[2021, "negative_hours"] == 0 and by.loc[2022, "negative_hours"] == 365 and by.loc[2022, "share_negative"] == pytest.approx(365 / 8760, abs=1e-4)
    assert pd.isna(by.loc[2021, "mean_raw_forecast"]) and by.loc[2023, "predicted_hours"] == len(raw)
    assert list(table.columns) == ["year", "hours", "negative_hours", "share_negative", "predicted_hours", "mean_raw_forecast", "share_negative_predicted_rows"]
