"""Feature building, benchmarks, models and the walk-forward on a synthetic inputs table."""

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import models
from thermo_fr.forecast.backtest import evaluate, run_backtest, walk_forward, worst_days
from thermo_fr.forecast.features import (
    CALENDAR_STRUCTURE,
    EXTENDED,
    FEATURE_TIMINGS,
    HONEST,
    HONEST_BASE,
    HONEST_CALENDAR,
    HONEST_SOLAR,
    HONEST_WIND,
    SAME_TYPE_LAGS,
    build_features,
    feature_timings,
)
from thermo_fr.data.solar_points import POINT_COLUMNS as SOLAR_POINT_COLUMNS
from thermo_fr.data.wind_points import POINT_COLUMNS
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
    cloud = pd.Series(rng.uniform(0.4, 1.0, len(index)), index=index).groupby(local.normalize()).transform("first").to_numpy()
    solar = np.clip(6_000 * np.sin((hours - 6) / 12 * np.pi), 0, None) * (local.month.isin([4, 5, 6, 7, 8]) * 0.5 + 0.5) * cloud
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
    # actual wind generation (calibration target), hub-height speeds at the points that imply it, and the proxy
    frame["wind_onshore_mw"] = wind * 0.9
    frame["wind_offshore_mw"] = wind * 0.1
    speed = 3 + 9 * np.cbrt(wind / 15_000)  # inverse of the proxy's power curve at full capacity
    for column in POINT_COLUMNS:
        frame.loc[issued, column] = np.clip(speed[issued] + rng.normal(0, 0.8, issued.sum()), 0, None)
    frame.loc[issued, "wind_proxy_mw"] = wind[issued] + rng.normal(0, 400, issued.sum())
    # actual solar generation, the radiation at the points that implies it (21 points of 1,000 MW each at 1,000 W/m2), and the proxy
    frame["solar_mw"] = solar
    radiation = solar / len(SOLAR_POINT_COLUMNS) / 1000.0 * 1000.0  # W/m2 per point when every point carries 1,000 MW
    for column in SOLAR_POINT_COLUMNS:
        frame.loc[issued, column] = np.clip(radiation[issued] * rng.uniform(0.8, 1.2, issued.sum()), 0, None)
    frame.loc[issued, "solar_proxy_mw"] = np.clip(solar[issued] + rng.normal(0, 200, issued.sum()), 0, None)
    return frame


@pytest.fixture
def inputs():
    return synthetic_inputs()


def test_feature_lists_and_timings_are_consistent():
    assert set(HONEST_BASE) < set(HONEST_WIND) < set(HONEST) < set(EXTENDED)
    assert all(f in FEATURE_TIMINGS for f in EXTENDED)
    assert {"solar_fc_mw", "wind_fc_mw", "residual_load_fc_mw"} == set(EXTENDED) - set(HONEST)
    assert set(HONEST_WIND) - set(HONEST_BASE) == {"wind_proxy_mw"} and FEATURE_TIMINGS["wind_proxy_mw"] == "wind_proxy"
    assert set(HONEST_SOLAR) - set(HONEST_WIND) == {"solar_proxy_mw"} and FEATURE_TIMINGS["solar_proxy_mw"] == "solar_proxy"
    assert set(HONEST_CALENDAR) - set(HONEST_WIND) == set(CALENDAR_STRUCTURE) | set(SAME_TYPE_LAGS)
    assert set(HONEST) == set(HONEST_SOLAR) | set(HONEST_CALENDAR)
    assert all(FEATURE_TIMINGS[f] == "calendar" for f in CALENDAR_STRUCTURE)
    assert all(FEATURE_TIMINGS[f] == "price_lag_same_type" for f in SAME_TYPE_LAGS)
    for name in ("honest", "honest_wind", "honest_base", "honest_solar", "honest_calendar"):
        assert "wind_solar_forecast" not in feature_timings(name).values()
    assert "wind_solar_forecast" in feature_timings("extended").values()


def test_feature_set_variants_drop_their_parts_and_old_tables_get_nan(inputs):
    table = build_features(inputs, "honest")
    wind = build_features(inputs, "honest_wind")
    base = build_features(inputs, "honest_base")
    assert {"wind_proxy_mw", "solar_proxy_mw", "day_type", "price_lag_same_type"} <= set(table.X)
    assert "wind_proxy_mw" in wind.X and not {"solar_proxy_mw", "day_type", "price_lag_same_type"} & set(wind.X)
    assert "wind_proxy_mw" not in base.X
    ts = pd.Timestamp("2024-02-14T17:00Z")
    assert table.X.loc[ts, "wind_proxy_mw"] == inputs.loc[ts, "wind_proxy_mw"]
    assert table.X.loc[ts, "solar_proxy_mw"] == inputs.loc[ts, "solar_proxy_mw"]
    assert table.info.loc[ts, "wind_mw"] == pytest.approx(inputs.loc[ts, "wind_onshore_mw"] + inputs.loc[ts, "wind_offshore_mw"])
    assert table.info.loc[ts, "solar_mw"] == inputs.loc[ts, "solar_mw"]
    old = inputs.drop(columns=["wind_proxy_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw", "solar_proxy_mw"] + POINT_COLUMNS
                      + SOLAR_POINT_COLUMNS)
    legacy = build_features(old, "honest")
    assert legacy.X["wind_proxy_mw"].isna().all() and legacy.X["solar_proxy_mw"].isna().all()
    assert legacy.info["wind_mw"].isna().all() and legacy.info["solar_mw"].isna().all()
    assert legacy.X["day_type"].notna().all()  # the calendar needs no inputs
    with pytest.raises(ValueError, match="feature_set must be"):
        build_features(inputs, "secret")


def test_calendar_structure_and_same_type_lag(inputs):
    table = build_features(inputs, "honest")
    X, y, info = table.X, table.y, table.info
    monday = pd.Timestamp("2024-02-12T16:00Z")  # 17:00 Paris on a Monday
    friday = pd.Timestamp("2024-02-09T16:00Z")
    saturday, saturday_before = pd.Timestamp("2024-02-10T16:00Z"), pd.Timestamp("2024-02-03T16:00Z")
    assert X.loc[monday, "price_lag_same_type"] == pytest.approx(y[friday]) and X.loc[monday, "same_type_lag_days"] == 3
    assert X.loc[saturday, "price_lag_same_type"] == pytest.approx(y[saturday_before]) and X.loc[saturday, "same_type_lag_days"] == 7
    assert X.loc[saturday, "price_lag_same_type"] == pytest.approx(X.loc[saturday, "price_lag7"])
    day_before = info["delivery_day"] == pd.Timestamp("2024-02-09")
    assert X.loc[monday, "price_lag_same_type_mean"] == pytest.approx(y[day_before].mean())
    tuesday = pd.Timestamp("2024-02-13T16:00Z")
    assert X.loc[tuesday, "price_lag_same_type"] == pytest.approx(X.loc[tuesday, "price_lag1"]) and X.loc[tuesday, "same_type_lag_days"] == 1
    pont = info["delivery_day"] == pd.Timestamp("2024-05-10")  # Friday after Ascension
    assert X.loc[pont, "bridge_day"].eq(1).all() and X.loc[pont, "post_holiday"].eq(1).all() and X.loc[pont, "day_type"].eq(0).all()
    ascension = info["delivery_day"] == pd.Timestamp("2024-05-09")
    assert X.loc[ascension, "day_type"].eq(2).all() and X.loc[ascension, "holiday"].eq(1).all()
    christmas = info["delivery_day"] == pd.Timestamp("2023-12-25")
    assert X.loc[christmas, "year_end_break"].eq(1).all() and info.loc[christmas, "holiday"].eq(1).all()
    assert set(X["day_type"].unique()) == {0, 1, 2} and (X["same_type_lag_days"] >= 1).all()


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


def test_strict_flag_marks_proxy_weather_and_the_gate_check_marks_the_set(inputs, fast_gbm):
    honest = walk_forward(build_features(inputs, "honest"), "2024-01-01", "2024-02-01", models=("linear",), log=lambda *_: None)
    flags = honest.predictions.groupby("delivery_day")["strict"].all()
    assert not flags[pd.Timestamp("2024-01-05")] and flags[pd.Timestamp("2024-01-20")]
    assert honest.point_in_time
    extended = walk_forward(build_features(inputs, "extended"), "2024-02-01", "2024-03-01", models=("linear",), log=lambda *_: None)
    assert extended.predictions["strict"].all() and not extended.point_in_time


def test_evaluate_reports_slices_hours_and_improvements(inputs, fast_gbm):
    table = build_features(inputs, "honest")
    bt = walk_forward(table, "2024-02-01", "2024-05-01", log=lambda *_: None)
    metrics = evaluate(bt.predictions)
    assert set(metrics["slices"]) == {"all", "peak", "off_peak", "top_5pct_price_hours", "negative_price_hours", "weekends", "holidays",
                                      "windiest_10pct_days", "sunniest_10pct_days"}
    windy = metrics["slices"]["windiest_10pct_days"]["gbm"]["hours"]
    assert 0 < windy <= 0.11 * metrics["rows"] + 48 and metrics["windy_day_cut_mw"] > 0
    sunny = metrics["slices"]["sunniest_10pct_days"]["gbm"]["hours"]
    assert 0 < sunny <= 0.11 * metrics["rows"] + 48 and metrics["sunny_day_cut_mw"] > 0
    assert metrics["slices"]["weekends"]["gbm"]["hours"] == int((bt.predictions["dow"] >= 5).sum()) > 0
    assert metrics["slices"]["holidays"]["gbm"]["hours"] == 24  # Easter Monday, 1 April 2024, is the only holiday in the window
    assert metrics["slices"]["all"]["gbm"]["hours"] == len(bt.predictions)
    assert metrics["slices"]["peak"]["gbm"]["hours"] + metrics["slices"]["off_peak"]["gbm"]["hours"] == len(bt.predictions)
    assert metrics["slices"]["negative_price_hours"]["gbm"]["mae"] is None  # synthetic prices stay positive
    assert sorted(metrics["by_hour"]) == list(range(24))
    assert set(metrics["by_month"]) == {"2024-02", "2024-03", "2024-04"}
    assert set(metrics["improvement_pct"]["gbm"]) == {"naive_day", "naive_week"}
    worst = worst_days(bt.predictions, table, inputs, n=5)
    assert len(worst) == 5 and worst["mae"].is_monotonic_decreasing
    assert {"holiday", "temp_anomaly_c", "negative_hours", "price_jump_vs_prev_day"} <= set(worst.columns)


def test_run_backtest_covers_both_feature_sets(inputs, fast_gbm):
    results = run_backtest(inputs, "2024-03-01", "2024-04-01", log=lambda *_: None)
    assert set(results) == {"honest", "honest_wind", "honest_base", "extended"}
    assert results["honest_wind"]["backtest"].point_in_time and results["honest_base"]["backtest"].point_in_time
    assert results["honest"]["metrics_strict"]["rows"] > 0
    assert results["extended"]["metrics_strict"]["rows"] == results["honest"]["metrics_strict"]["rows"]
    assert results["honest"]["backtest"].point_in_time and not results["extended"]["backtest"].point_in_time
