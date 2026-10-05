"""Feature building, benchmarks, models and the walk-forward on a synthetic inputs table."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import models
from thermo_fr.forecast.backtest import evaluate, run_backtest, walk_forward, worst_days
from thermo_fr.forecast.features import EXTENDED, HONEST, FEATURE_TIMINGS, build_features, feature_timings
from thermo_fr.forecast.inputs import INPUT_COLUMNS
from thermo_fr.forecast.models import benchmark_predictions, fit_predict
from thermo_fr.forecast.timing import LookaheadError, check_point_in_time


def synthetic_inputs(start="2023-09-01", end="2024-05-01", issued_from="2024-01-10", seed=0) -> pd.DataFrame:
    """An inputs table with a simple price structure: hour shape + weekly cycle + residual load + noise."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    local = index.tz_convert("Europe/Paris")
    hours = np.asarray(local.hour)
    load = 50_000 + 8_000 * np.sin((hours - 6) / 24 * 2 * np.pi) + rng.normal(0, 1_500, len(index))
    wind = np.clip(5_000 + 4_000 * rng.standard_normal(len(index)).cumsum() / 50, 500, 15_000)
    solar = np.clip(6_000 * np.sin((hours - 6) / 12 * np.pi), 0, None) * (local.month.isin([4, 5, 6, 7, 8]) * 0.5 + 0.5)
    residual = load - wind - solar
    price = 20 + residual / 500 + 10 * (np.asarray(local.dayofweek) < 5) + rng.normal(0, 5, len(index))
    temp = 10 + 8 * np.sin((np.asarray(local.dayofyear) - 100) / 365 * 2 * np.pi) + rng.normal(0, 2, len(index))
    frame = pd.DataFrame(index=index, columns=INPUT_COLUMNS, dtype=float)
    frame["price_eur_mwh"] = price
    frame["load_fc_mw"] = load
    frame["solar_fc_mw"] = solar
    frame["wind_onshore_fc_mw"] = wind * 0.9
    frame["wind_offshore_fc_mw"] = wind * 0.1
    frame["temp_proxy_c"] = temp
    frame["wind100_proxy_ms"] = wind / 1000
    frame["radiation_proxy_wm2"] = solar / 10
    issued = index >= pd.Timestamp(issued_from, tz="UTC")
    frame.loc[issued, "temp_fc_c"] = temp[issued] + rng.normal(0, 1, issued.sum())
    frame.loc[issued, "wind100_fc_ms"] = wind[issued] / 1000 + rng.normal(0, 0.5, issued.sum())
    frame.loc[issued, "radiation_fc_wm2"] = solar[issued] / 10
    return frame


@pytest.fixture
def inputs():
    return synthetic_inputs()


def test_feature_lists_and_timings_are_consistent():
    assert set(HONEST) <= set(EXTENDED)
    assert all(f in FEATURE_TIMINGS for f in EXTENDED)
    assert {"solar_fc_mw", "wind_fc_mw", "residual_load_fc_mw"} == set(EXTENDED) - set(HONEST)
    honest = feature_timings("honest")
    assert "wind_solar_forecast" not in honest.values()
    assert "wind_solar_forecast" in feature_timings("extended").values()


def test_honest_features_pass_the_gate_and_extended_do_not(inputs):
    table = build_features(inputs, "honest")
    check_point_in_time(table.X.index, feature_timings("honest"))
    with pytest.raises(LookaheadError):
        check_point_in_time(table.X.index, feature_timings("extended"))


def test_rows_are_paris_delivery_hours(inputs):
    table = build_features(inputs, "honest")
    info = table.info
    spring = info[info["delivery_day"] == pd.Timestamp("2024-03-31")]
    autumn = info[info["delivery_day"] == pd.Timestamp("2023-10-29")]
    assert len(spring) == 23 and len(autumn) == 25
    assert sorted(autumn["hour"].tolist()).count(2) == 2 and 2 not in spring["hour"].tolist()
    assert table.X.loc[spring.index, "hour"].tolist() == spring["hour"].tolist()
    assert set(table.X["dow"].unique()) == set(range(7)) and table.X["holiday"].max() == 1
    christmas = info[info["delivery_day"] == pd.Timestamp("2023-12-25")].index
    assert table.X.loc[christmas, "holiday"].eq(1).all()


def test_price_lags_match_same_local_hour(inputs):
    table = build_features(inputs, "honest")
    X, y = table.X, table.y
    ts = pd.Timestamp("2024-02-14T17:00Z")  # a plain winter day
    assert X.loc[ts, "price_lag1"] == pytest.approx(y[ts - pd.Timedelta(days=1)])
    assert X.loc[ts, "price_lag7"] == pytest.approx(y[ts - pd.Timedelta(days=7)])
    day_before = table.info["delivery_day"] == pd.Timestamp("2024-02-13")
    assert X.loc[ts, "price_lag1_mean"] == pytest.approx(y[day_before].mean())
    assert X.loc[ts, "price_lag1_max"] == pytest.approx(y[day_before].max())
    # Across the spring switch the "same hour" is the same local clock hour, 23 real hours earlier.
    after = pd.Timestamp("2024-04-01T15:00Z")  # 17:00 Paris on 1 April
    before = pd.Timestamp("2024-03-31T15:00Z")  # 17:00 Paris on 31 March
    assert X.loc[after, "price_lag1"] == pytest.approx(y[before])
    # First week has no lag-7 values.
    assert X["price_lag7"].iloc[:24 * 7].isna().all() and X["price_lag7"].iloc[24 * 7 + 1:].notna().all()


def test_weather_uses_issued_when_available_else_proxy(inputs):
    table = build_features(inputs, "honest")
    early = pd.Timestamp("2023-12-01T12:00Z")
    late = pd.Timestamp("2024-02-01T12:00Z")
    assert table.X.loc[early, "temp_c"] == inputs.loc[early, "temp_proxy_c"]
    assert table.X.loc[late, "temp_c"] == inputs.loc[late, "temp_fc_c"]
    assert not table.info.loc[early, "weather_point_in_time"] and table.info.loc[late, "weather_point_in_time"]


def test_extended_adds_residual_load(inputs):
    table = build_features(inputs, "extended")
    ts = pd.Timestamp("2024-02-14T17:00Z")
    wind = inputs.loc[ts, "wind_onshore_fc_mw"] + inputs.loc[ts, "wind_offshore_fc_mw"]
    assert table.X.loc[ts, "wind_fc_mw"] == pytest.approx(wind)
    assert table.X.loc[ts, "residual_load_fc_mw"] == pytest.approx(inputs.loc[ts, "load_fc_mw"] - inputs.loc[ts, "solar_fc_mw"] - wind)


def test_benchmarks_are_the_lagged_prices(inputs):
    table = build_features(inputs, "honest")
    bench = benchmark_predictions(table.X)
    assert list(bench.columns) == ["naive_day", "naive_week"]
    assert bench["naive_day"].equals(table.X["price_lag1"]) and bench["naive_week"].equals(table.X["price_lag7"])


@pytest.fixture
def fast_gbm(monkeypatch):
    monkeypatch.setitem(models.GBM_PARAMS, "n_estimators", 60)


def test_models_beat_the_naive_benchmark_on_synthetic_data(inputs, fast_gbm):
    table = build_features(inputs, "extended")
    cut = table.info["delivery_day"] < pd.Timestamp("2024-03-01")
    for name in ("gbm", "linear"):
        pred = fit_predict(name, table.X[cut], table.y[cut], table.X[~cut])
        mae = np.nanmean(np.abs(pred - table.y[~cut].to_numpy()))
        naive = np.nanmean(np.abs(table.X.loc[~cut, "price_lag1"] - table.y[~cut]))
        assert mae < naive, (name, mae, naive)


def test_walk_forward_never_trains_on_the_month_it_forecasts(inputs, fast_gbm):
    table = build_features(inputs, "honest")
    bt = walk_forward(table, "2024-02-01", "2024-04-01", models=("linear",), log=lambda *_: None)
    preds = bt.predictions
    assert bt.months == ["2024-02", "2024-03"]
    assert preds["delivery_day"].min() == pd.Timestamp("2024-02-01") and preds["delivery_day"].max() == pd.Timestamp("2024-03-31")
    february = preds[preds["delivery_day"] < "2024-03-01"]
    march = preds[preds["delivery_day"] >= "2024-03-01"]
    expected_feb = int((table.info["delivery_day"] < pd.Timestamp("2024-02-01")).sum())
    assert february["train_rows"].iloc[0] == expected_feb
    assert march["train_rows"].iloc[0] == expected_feb + len(february)
    assert preds["strict"].all()  # issued weather from 10 January, honest features


def test_strict_flag_marks_proxy_weather_and_extended_rows(inputs, fast_gbm):
    honest = walk_forward(build_features(inputs, "honest"), "2024-01-01", "2024-02-01", models=("linear",), log=lambda *_: None)
    flags = honest.predictions.groupby("delivery_day")["strict"].all()
    assert not flags[pd.Timestamp("2024-01-05")] and flags[pd.Timestamp("2024-01-20")]
    extended = walk_forward(build_features(inputs, "extended"), "2024-02-01", "2024-03-01", models=("linear",), log=lambda *_: None)
    assert not extended.predictions["strict"].any()


def test_evaluate_reports_slices_hours_and_improvements(inputs, fast_gbm):
    table = build_features(inputs, "honest")
    bt = walk_forward(table, "2024-02-01", "2024-04-01", log=lambda *_: None)
    metrics = evaluate(bt.predictions)
    assert set(metrics["slices"]) == {"all", "peak", "off_peak", "top_5pct_price_hours", "negative_price_hours"}
    assert metrics["slices"]["all"]["gbm"]["hours"] == len(bt.predictions)
    assert metrics["slices"]["peak"]["gbm"]["hours"] + metrics["slices"]["off_peak"]["gbm"]["hours"] == len(bt.predictions)
    assert metrics["slices"]["negative_price_hours"]["gbm"]["mae"] is None  # synthetic prices stay positive
    assert sorted(metrics["by_hour"]) == list(range(24))
    assert set(metrics["by_month"]) == {"2024-02", "2024-03"}
    assert set(metrics["improvement_pct"]["gbm"]) == {"naive_day", "naive_week"}
    worst = worst_days(bt.predictions, table, inputs, n=5)
    assert len(worst) == 5 and worst["mae"].is_monotonic_decreasing
    assert {"holiday", "temp_anomaly_c", "negative_hours", "price_jump_vs_prev_day"} <= set(worst.columns)


def test_run_backtest_covers_both_feature_sets(inputs, fast_gbm):
    results = run_backtest(inputs, "2024-03-01", "2024-04-01", log=lambda *_: None)
    assert set(results) == {"honest", "extended"}
    assert results["honest"]["metrics_strict"]["rows"] > 0
    assert results["extended"]["metrics_strict"]["rows"] == 0  # nothing in the extended set is point in time
    assert results["extended"]["metrics_all"]["rows"] == results["honest"]["metrics_all"]["rows"]
