"""Shape target, benchmarks, metrics, the battery schedule and the live record."""

import numpy as np
import pandas as pd
import pytest

from test_forecast_features import synthetic_inputs
from test_store import curve_for
from thermo_fr.forecast import models
from thermo_fr.forecast.features import build_features
from thermo_fr.forecast.shape import (
    battery_backtest,
    battery_day,
    benchmark_shapes,
    best_blocks,
    evaluate_shape,
    extreme_hit_rate,
    live_frame,
    live_shape_record,
    shape_metrics,
    shape_of,
    spread_error,
    walk_forward_shape,
)
from thermo_fr.forecast.store import Store, score_curve


def day_frame(actual, model=None, d1=None, day="2026-10-13"):
    """One delivery day as the scoring frame: shapes are prices minus the day's mean."""
    actual = np.asarray(actual, dtype=float)
    hours = pd.date_range(f"{day}T00:00", periods=len(actual), freq="1h", tz="Europe/Paris").tz_convert("UTC")
    frame = pd.DataFrame(index=hours)
    frame["delivery_day"] = pd.Timestamp(day)
    frame["hour"] = range(len(actual))
    frame["actual"] = actual
    frame["actual_base"] = actual.mean()
    frame["actual_shape"] = actual - actual.mean()
    for name, values in (("model", model), ("d1", d1)):
        values = np.asarray(values if values is not None else actual, dtype=float)
        frame[f"base_{name}"] = values.mean()
        frame[f"shape_{name}"] = values - values.mean()
    return frame


def test_shape_removes_the_daily_mean_and_benchmarks_come_from_the_lags():
    inputs = synthetic_inputs("2023-09-01", "2024-03-01", issued_from="2023-09-01")
    table = build_features(inputs, "honest_v2")
    shape = shape_of(table.y, table.info["delivery_day"])
    assert shape.groupby(table.info["delivery_day"].to_numpy()).mean().abs().max() < 1e-9
    bench = benchmark_shapes(table)
    ts = pd.Timestamp("2024-02-14T17:00Z")
    day = table.info["delivery_day"] == pd.Timestamp("2024-02-14")
    assert bench.loc[ts, "shape_d1"] == pytest.approx(table.X.loc[ts, "price_lag1"] - table.X.loc[day, "price_lag1"].mean())
    assert bench.loc[ts, "base_d1"] == pytest.approx(table.X.loc[day, "price_lag1"].mean())
    assert bench.loc[ts, "shape_same_type"] == pytest.approx(table.X.loc[ts, "price_lag_same_type"] - table.X.loc[day, "price_lag_same_type"].mean())


def test_metrics_on_a_known_day():
    actual = [10, 5, 0, 0, 5, 10, 20, 40, 60, 60, 50, 45, 40, 40, 45, 50, 60, 80, 90, 80, 60, 40, 30, 20]
    model = [v + 3 for v in actual]  # same shape, different level: shape error zero
    d1 = actual[::-1]  # reversed day: wrong shape
    frame = day_frame(actual, model, d1)
    metrics = shape_metrics(frame, ("model", "d1"))
    assert metrics["model"]["shape_mae"] == 0.0 and metrics["model"]["spread_error"] == 0.0
    assert metrics["model"]["cheapest2_hit_rate"] == 1.0 and metrics["model"]["dearest2_hit_rate"] == 1.0
    assert metrics["d1"]["shape_mae"] > 10 and metrics["d1"]["cheapest2_hit_rate"] == 0.0
    assert spread_error(frame, "shape_model") == 0.0 and extreme_hit_rate(frame, "shape_d1", cheapest=False) == 0.0


def test_best_blocks_and_the_skip_rule():
    prices = np.array([10.0, 5.0, 0.0, 0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 50.0, 45.0, 40.0, 40.0, 45.0, 50.0, 60.0, 80.0, 90.0, 80.0, 60.0, 40.0, 30.0, 20.0])
    c, d, value = best_blocks(prices)
    assert (c, d) == (2, 17) and value == pytest.approx(0.88 * (80 + 90) - 0)  # charge at the two zero hours, discharge at 17 and 18
    flat = np.full(24, 50.0)
    c, d, value = best_blocks(flat)
    assert value == pytest.approx(0.88 * 100 - 100) and value < 0
    assert battery_day(flat, flat)["traded"] is False and battery_day(flat, flat)["value_eur"] == 0.0
    small = np.array([50.0] * 22 + [55.0, 55.0])  # a 5 EUR spread does not cover the 12% loss on a 50 EUR level
    assert battery_day(small, small)["traded"] is False
    # chosen on the forecast, settled on the actual: a wrong forecast can lose money
    wrong = prices[::-1]
    result = battery_day(wrong, prices)
    assert result["traded"] and result["value_eur"] < 0
    # the charge block always ends before the discharge block starts
    for _ in range(20):
        p = np.random.default_rng(0).normal(50, 20, 24)
        c, d, _ = best_blocks(p)
        assert c + 2 <= d
    # the autumn day has 25 hours and the spring day 23: both schedule
    assert best_blocks(np.random.default_rng(1).normal(50, 20, 25))[1] <= 23
    assert np.isnan(best_blocks(np.array([1.0, 2.0, 3.0]))[2])


def test_battery_backtest_totals_and_perfect_foresight():
    actual = [10, 5, 0, 0, 5, 10, 20, 40, 60, 60, 50, 45, 40, 40, 45, 50, 60, 80, 90, 80, 60, 40, 30, 20]
    frames = [day_frame(actual, model=[v + 3 for v in actual], d1=actual[::-1], day="2026-10-13"),
              day_frame([50.0] * 24, model=[50.0] * 24, d1=[50.0] * 24, day="2026-10-14")]  # a flat day: everyone skips
    result = battery_backtest(pd.concat(frames), ("model", "d1"))
    s = result["summary"]
    assert s["days"] == 2 and s["perfect_eur_per_day"] == pytest.approx(0.88 * 170 / 2)
    assert s["model"]["eur_per_day"] == pytest.approx(0.88 * 170 / 2) and s["model"]["share_of_perfect"] == 1.0
    assert s["model"]["days_traded"] == 1 and s["model"]["days_skipped"] == 1 and s["model"]["losing_days"] == 0
    assert s["d1"]["eur_per_day"] < 0 and s["d1"]["losing_days"] == 1 and s["d1"]["share_of_perfect"] < 0
    daily = result["daily"]
    assert daily["value_perfect"].tolist() == pytest.approx([0.88 * 170, 0.0])


def test_walk_forward_shape_beats_a_reversed_benchmark_on_synthetic_data(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 40)
    inputs = synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")
    table = build_features(inputs, "honest_v2")
    bt = walk_forward_shape(table, "2024-02-01", "2024-05-01", log=lambda *_: None)
    assert bt.months == ["2024-02", "2024-03", "2024-04"]
    pred = bt.predictions
    assert {"actual_shape", "shape_model", "shape_d1", "shape_same_type", "base_model", "base_d1", "strict"} <= set(pred.columns)
    assert pred.groupby("delivery_day")["actual_shape"].mean().abs().max() < 1e-9
    result = evaluate_shape(pred)
    assert result["shape"]["model"]["shape_mae"] < 3 * result["shape"]["d1"]["shape_mae"]  # the synthetic shape is learnable
    assert set(result["battery"]) >= {"model", "d1", "same_type", "perfect_eur_per_day", "days"}
    assert result["battery"]["model"]["share_of_perfect"] is None or result["battery"]["model"]["share_of_perfect"] <= 1.0
    assert len(result["battery_daily"]) == result["battery"]["days"]
    with_level = walk_forward_shape(table, "2024-02-01", "2024-03-01", level=table.X["price_lag7"], log=lambda *_: None)
    day = with_level.predictions["delivery_day"] == pd.Timestamp("2024-02-14")
    assert with_level.predictions.loc[day, "base_model"].iloc[0] == pytest.approx(table.X.loc[with_level.predictions.index[day], "price_lag7"].mean())


def test_paired_comparison_and_block_bootstrap():
    from thermo_fr.forecast.shape import block_bootstrap_mean, daily_shape_mae, paired_comparison, paired_report

    rng = np.random.default_rng(3)
    model = rng.normal(10, 5, 200)
    daily = pd.DataFrame({"value_model": model, "value_d1": model - 2 + rng.normal(0, 1, 200), "value_same_type": model + rng.normal(0, 3, 200)})
    clear = paired_comparison(daily, "value_model", "value_d1", resamples=2000)
    assert clear["days"] == 200 and 1.5 < clear["mean_difference"] < 2.5 and clear["ci95_low"] > 0 and clear["distinguishable_from_zero"]
    assert clear["share_model_wins"] > 0.9
    noisy = paired_comparison(daily, "value_model", "value_same_type", resamples=2000)
    assert noisy["ci95_low"] < 0 < noisy["ci95_high"] and not noisy["distinguishable_from_zero"]
    low, high = block_bootstrap_mean(np.zeros(30))
    assert low == 0.0 and high == 0.0
    assert np.isnan(block_bootstrap_mean(np.array([]))[0])
    # a block bootstrap on an autocorrelated series is wider than an i.i.d. one
    series = np.cumsum(rng.normal(0, 1, 300)) * 0.1 + rng.normal(0, 1, 300)
    wide = block_bootstrap_mean(series, block=7, resamples=2000)
    narrow = block_bootstrap_mean(series, block=1, resamples=2000)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    frame = pd.concat([day_frame([10, 5, 0, 0, 5, 10, 20, 40, 60, 60, 50, 45, 40, 40, 45, 50, 60, 80, 90, 80, 60, 40, 30, 20],
                                 model=[v + 3 for v in range(24)], d1=list(range(24)), day=d) for d in ("2026-10-13", "2026-10-14")])
    frame["shape_same_type"] = frame["shape_d1"]
    shape_daily = daily_shape_mae(frame)
    assert list(shape_daily.columns) == ["delivery_day", "shape_mae_model", "shape_mae_d1", "shape_mae_same_type"] and len(shape_daily) == 2
    report = paired_report(pd.DataFrame({"value_model": [5.0, 6.0], "value_d1": [4.0, 7.0], "value_same_type": [1.0, 1.0]}), shape_daily, resamples=200)
    assert set(report) == {"battery_value", "shape_mae"} and set(report["battery_value"]) == {"d1", "same_type"}
    assert report["battery_value"]["same_type"]["mean_difference"] == 4.5 and report["shape_mae"]["d1"]["days"] == 2


def test_live_record_from_the_store(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"
    curve = curve_for(day)  # forecast = 100 + local hour, naive_day = same shape shifted
    store.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:00Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve)
    actual = pd.Series(90 + curve["hour"].to_numpy(), index=curve.index)  # the same ramp, lower level: a perfect shape
    store.save_actuals(actual, pd.Series([day] * len(curve)))
    store.save_score(1, day, "honest_v2", "2026-10-12T08:00:00Z", score_curve(curve, actual))
    record = live_shape_record(store, "honest_v2")
    assert record["days"] == 1 and record["shape"]["model"]["shape_mae"] == pytest.approx(0.0)
    assert record["battery"]["model"]["share_of_perfect"] == 1.0 and len(record["daily"]) == 1
    assert record["daily"].loc[0, "shape_mae_model"] == pytest.approx(0.0) and record["daily"].loc[0, "delivery_day"] == day
    frame = live_frame(curve, actual, day)
    assert frame["actual_shape"].mean() == pytest.approx(0.0) and (frame["shape_model"] == frame["actual_shape"]).all()
    assert live_shape_record(store, "honest")["days"] == 0
    store.close()
