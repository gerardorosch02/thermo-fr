"""The morning brief: LightGBM contributions grouped into a waterfall that sums to the change in the model forecast, the fundamentals table, the
one-line summary, the record strip, the early-trade line (local only), the drivers' round trip through the store and the public dataset,
and the public app's copy of the arithmetic kept equal to the package's."""

import json

import numpy as np
import pandas as pd
import pytest

import streamlit_app as public
from test_forecast_features import synthetic_inputs
from test_jobs import DAY, NOW, FakeEntsoe, history, run, solar_weights, store, table, weights  # noqa: F401  fixtures reused
from test_store import curve_for
from thermo_fr.dashboard import brief as bq
from thermo_fr.forecast import brief as b
from thermo_fr.forecast.features import FEATURES, build_features
from thermo_fr.forecast.models import contributions_with, predict_with
from thermo_fr.forecast.publish import export_published, import_published
from thermo_fr.forecast.refit import refit_model
from thermo_fr.forecast.store import Store


def test_every_feature_of_the_published_sets_has_a_group():
    for feature_set in ("honest_v2", "honest", "extended"):
        assert all(b.group_of(f) != "other" for f in FEATURES[feature_set]), feature_set
    assert b.group_of("expected_value") == b.AVERAGE_GROUP and b.group_of("something_new") == "other"
    assert b.group_of("residual_v2_mw") == "load and temperature" and b.group_of("spread_ch_lag1") == "neighbour prices"
    assert public.GROUP_ORDER == b.GROUP_ORDER and public.FUNDAMENTALS == b.FUNDAMENTALS and public.NEIGHBOUR_ROWS == b.NEIGHBOUR_ROWS


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Two stored models (a refit between them) and the honest_v2 feature rows of two consecutive delivery days."""
    out = tmp_path_factory.mktemp("model")
    inputs = synthetic_inputs("2023-09-01", "2024-05-01", issued_from="2024-01-10")
    before = inputs[inputs.index < "2024-04-01"]
    refit_model(before, out_dir=out / "a", feature_set="honest_v2", holdout_days=0, log=lambda *_: None, params={"n_estimators": 25}, save_proxies=False)
    refit_model(before, out_dir=out / "b", feature_set="honest_v2", holdout_days=0, log=lambda *_: None, params={"n_estimators": 40}, save_proxies=False)
    features = build_features(inputs, "honest_v2")
    days = {day: features.X[features.info["delivery_day"] == pd.Timestamp(day)] for day in ("2024-04-09", "2024-04-10")}
    return {"model_a": out / "a" / "honest_v2.txt", "model_b": out / "b" / "honest_v2.txt", "rows": days}


def test_contributions_sum_to_the_prediction_hour_by_hour(fitted):
    rows = fitted["rows"]["2024-04-10"]
    contributions = contributions_with(fitted["model_a"], rows)
    assert list(contributions.columns) == list(FEATURES["honest_v2"]) + ["expected_value"] and len(contributions) == 24
    assert np.allclose(contributions.sum(axis=1), predict_with(fitted["model_a"], rows), atol=1e-6)
    assert contributions["expected_value"].nunique() == 1


def test_group_steps_sum_to_the_change_in_base_prediction(fitted):
    """The waterfall from yesterday's base model forecast to today's: the group steps add up exactly, with and without a refit in between."""
    rows_1, rows_2 = fitted["rows"]["2024-04-09"], fitted["rows"]["2024-04-10"]
    previous = b.driver_table(rows_1, contributions_with(fitted["model_a"], rows_1))
    current = b.driver_table(rows_2, contributions_with(fitted["model_a"], rows_2))
    assert list(previous.columns) == ["feature", "value", "contribution"] and len(previous) == len(FEATURES["honest_v2"]) + 1
    assert previous.loc[previous["feature"] == "load_fc_mw", "value"].iloc[0] == pytest.approx(rows_1["load_fc_mw"].mean())
    result = b.waterfall(previous, current)
    base_1, base_2 = predict_with(fitted["model_a"], rows_1).mean(), predict_with(fitted["model_a"], rows_2).mean()
    assert result["complete"] and result["start"] == pytest.approx(base_1, abs=1e-6) and result["end"] == pytest.approx(base_2, abs=1e-6)
    assert sum(s["change"] for s in result["steps"]) == pytest.approx(base_2 - base_1, abs=1e-6)
    assert [s["group"] for s in result["steps"]] == [g for g in b.GROUP_ORDER if g != b.AVERAGE_GROUP]  # same model: no refit step
    # the model was refitted between the two versions: the expected values differ and the difference is its own step; the sum still holds
    refitted = b.driver_table(rows_2, contributions_with(fitted["model_b"], rows_2))
    result = b.waterfall(previous, refitted)
    base_b = predict_with(fitted["model_b"], rows_2).mean()
    assert result["steps"][-1]["group"] == b.REFIT_STEP and abs(result["steps"][-1]["change"]) > 0
    assert sum(s["change"] for s in result["steps"]) == pytest.approx(base_b - base_1, abs=1e-6)
    # the public app's copy of the arithmetic, on the published form of the same tables (with the group column), agrees step for step
    pub_prev, pub_cur = previous.assign(group=previous["feature"].map(b.group_of)), refitted.assign(group=refitted["feature"].map(b.group_of))
    assert public.waterfall(pub_prev, pub_cur) == result
    assert public.summary_line(result, pub_cur, pub_prev, ("Tue", "Wed")) == b.summary_line(result, refitted, previous, ("Tue", "Wed"))
    assert public.fundamentals_table(pub_cur, pub_prev).equals(b.fundamentals_table(refitted, previous))
    # an incomplete pair gives no steps and an empty summary
    assert not b.waterfall(None, current)["complete"] and b.summary_line(b.waterfall(None, current), current, None) == ""


def test_model_forecast_splits_peak_and_off_peak_and_negative_hours_use_the_calibrated_probability():
    curve = curve_for("2026-10-14").reset_index()  # forecast = 100 + hour
    level = b.model_forecast(curve)
    assert level["base"] == pytest.approx(111.5) and level["peak"] == pytest.approx(113.5) and level["offpeak"] == pytest.approx(109.5) and level["hours"] == 24
    assert public.model_forecast(curve) == level
    by_hour = pd.DataFrame({"hour": range(24), "p_negative": 0.2, "p_negative_cal": [0.01] * 12 + [0.05, 0.12] + [0.0] * 10})
    assert b.negative_hours(by_hour) == [{"hour": 12, "p": 0.05}, {"hour": 13, "p": 0.12}]  # at or above 5%, calibrated, not the raw 20%
    by_hour["p_negative_cal"] = float("nan")
    assert len(b.negative_hours(by_hour)) == 24  # no calibrated value: the raw one
    published = pd.DataFrame({"hour": range(24), "p_negative": [0.0] * 23 + [0.07]})  # the public file carries the calibrated value as p_negative
    assert public.negative_hours(published) == [{"hour": 23, "p": 0.07}]
    assert b.negative_hours(pd.DataFrame()) == [] and b.model_forecast(pd.DataFrame())["hours"] == 0


def drivers_from(values: dict, contributions: dict) -> pd.DataFrame:
    features = sorted(set(values) | set(contributions))
    return pd.DataFrame({"feature": features, "value": [values.get(f, float("nan")) for f in features],
                         "contribution": [contributions.get(f, 0.0) for f in features]})


def test_summary_line_names_the_two_largest_groups_with_their_fundamental():
    previous = drivers_from({"wind_proxy_mw": 2200.0, "price_lag1_mean": 95.0, "load_fc_mw": 50000.0, "temp_c": 12.0},
                            {"wind_proxy_mw": 4.0, "price_lag1": 30.0, "load_fc_mw": 10.0, "expected_value": 60.0})
    current = drivers_from({"wind_proxy_mw": 8000.0, "price_lag1_mean": 86.0, "load_fc_mw": 50200.0, "temp_c": 12.5},
                           {"wind_proxy_mw": -7.0, "price_lag1": 26.0, "load_fc_mw": 11.0, "expected_value": 60.0})
    result = b.waterfall(previous, current)
    assert result["change"] == pytest.approx(-14.0)
    assert b.summary_line(result, current, previous) == "Model forecast down 14 EUR/MWh: wind -11, recent French prices -4."
    # the French price lags as the largest group: the line says plainly that the model is following the market
    following = drivers_from({"wind_proxy_mw": 8000.0, "price_lag1_mean": 86.0, "load_fc_mw": 50200.0, "temp_c": 12.5},
                             {"wind_proxy_mw": 0.0, "price_lag1": 19.0, "load_fc_mw": 11.0, "expected_value": 60.0})
    result_f = b.waterfall(previous, following)
    assert result_f["steps"][0]["group"] == b.FRENCH_GROUP == "recent French prices (model following the market)"
    assert b.summary_line(result_f, following, previous) == ("Model forecast down 14 EUR/MWh, mostly the model following the market: "
                                                             "recent French prices -11, wind -4.")
    pub_prev, pub_fol = previous.assign(group=previous["feature"].map(b.group_of)), following.assign(group=following["feature"].map(b.group_of))
    assert public.summary_line(public.waterfall(pub_prev, pub_fol), pub_fol, pub_prev) == b.summary_line(result_f, following, previous)
    # a drivers.csv exported before the rename still lands in the renamed group, on both sides
    legacy = pub_fol.assign(group=pub_fol["group"].replace({b.FRENCH_GROUP: "recent French prices"}))
    assert public.grouped(legacy)[b.FRENCH_GROUP] == pytest.approx(19.0) and b.grouped(legacy)[b.FRENCH_GROUP] == pytest.approx(19.0)
    assert "recent French prices" not in public.grouped(legacy).index
    assert b.step_phrase("wind", current, previous) == "wind 5.8 GW higher" and b.step_phrase(b.FRENCH_GROUP, current, previous) == "French base price 9 EUR/MWh lower"
    assert b.step_phrase("load and temperature", current, previous) == "temperature 0.5 C higher"  # the load moved less than 0.3 GW
    assert b.step_phrase("calendar", current, previous, ("Fri", "Sat")) == "calendar, Fri to Sat"
    assert b.step_phrase("neighbour prices", current, previous) == "German base price"  # no DE value in these tables
    flat = b.waterfall(previous, previous)
    assert b.summary_line(flat, previous, previous) == "Model forecast unchanged."


def test_fundamentals_table_rows_units_and_spreads():
    current = drivers_from({"load_fc_mw": 45240.7, "temp_c": 16.3, "wind_proxy_mw": 830.7, "solar_proxy_mw": 3134.3, "nuclear_d1_early_mw": 41000.0,
                            "residual_v2_mw": 275.7, "price_lag1_mean": 80.0, "price_de_lu_lag1": 90.0, "spread_de_lu_lag1": -10.0,
                            "price_ch_lag1": 85.0, "spread_ch_lag1": -5.0}, {})
    previous = drivers_from({"load_fc_mw": 46240.7, "temp_c": 15.0, "wind_proxy_mw": 2830.7}, {})
    table = b.fundamentals_table(current, previous)
    assert table["quantity"].tolist() == ["French base, previous day", "Load forecast", "Temperature", "Wind estimate",
                                          "Solar estimate", "Latest nuclear output", "Load minus wind, solar and nuclear", "DE D-1 base", "BE D-1 base", "ES D-1 base", "IT-North D-1 base", "CH D-1 base"]
    row = table.set_index("quantity")
    assert row.loc["French base, previous day", "tomorrow"] == 80.0 and np.isnan(row.loc["French base, previous day", "change"])  # the spreads are measured against it
    assert row.loc["Load forecast", "tomorrow"] == pytest.approx(45.2407) and row.loc["Load forecast", "change"] == pytest.approx(-1.0)
    assert row.loc["Wind estimate", "change"] == pytest.approx(-2.0) and np.isnan(row.loc["Solar estimate", "change"])
    assert row.loc["DE D-1 base", "fr_spread"] == -10.0 and row.loc["CH D-1 base", "tomorrow"] == 85.0 and np.isnan(row.loc["BE D-1 base", "tomorrow"])
    # the table shows the seven fundamentals with the French D-1 base on top; the neighbours are the two-line block
    shown = bq.format_fundamentals(table)
    assert list(shown.columns) == ["Quantity", "Tomorrow", "Previous day", "Change"] and len(shown) == 7
    assert shown["Quantity"].iloc[0] == "French base, previous day (EUR/MWh)" and shown["Tomorrow"].iloc[0] == "80" and shown["Change"].iloc[0] == "n/a"
    assert shown["Quantity"].iloc[1] == "Load forecast (GW)" and shown["Change"].iloc[1] == "-1.0" and shown["Tomorrow"].iloc[2] == "16.3"
    assert shown["Previous day"].iloc[4] == "n/a" and shown["Change"].iloc[4] == "n/a"
    block = b.neighbour_block(table)
    assert list(block.columns) == ["DE", "BE", "ES", "IT-North", "CH"] and list(block.index) == ["D-1 base", "FR spread"]
    assert block.loc["D-1 base"].tolist() == ["90", "n/a", "n/a", "n/a", "85"] and block.loc["FR spread"].tolist() == ["-10", "n/a", "n/a", "n/a", "-5"]
    with_change = b.fundamentals_table(current, drivers_from({"price_de_lu_lag1": 87.0, "price_ch_lag1": 85.0}, {}))
    assert b.neighbour_block(with_change).loc["D-1 base"].tolist() == ["90 (+3)", "n/a", "n/a", "n/a", "85 (+0)"]
    assert np.isnan(b.fundamentals_table(current, None)["change"]).all()


def test_record_strip_shows_the_count_always_and_the_aggregates_from_twenty_windows():
    few = b.record_strip({"scored_days": 4, "hit_rate": 0.5, "mean_pnl_per_mwh": -2.99, "model_mae": 19.68, "market_mae": 3.46})
    assert few["windows"] == 4 and not few["show"] and "4 of 20 windows" in few["text"] and "hit_rate" not in few
    many = b.record_strip({"scored_windows": 23, "hit_rate": 0.43, "mean_pnl_per_mwh": -1.2, "model_mae": 18.0, "market_mae": 4.0})
    assert many["show"] and many["hit_rate"] == 0.43 and "23 windows: hit rate 43%" in many["text"] and "18.0 against the market's 4.0" in many["text"]
    assert b.record_strip(None)["windows"] == 0 and not b.record_strip({"scored_windows": 25})["show"]  # a count without aggregates (collecting)


def test_early_trade_prefers_the_snapshot_before_the_window_and_labels_the_window_close_as_a_check():
    """Every price here is made up."""
    windows = pd.DataFrame([
        {"delivery_date": "2026-10-14", "product": "base", "trade_date": "2026-10-13", "window_start": "08:00", "window_end": "09:00", "close": 101.0},
        {"delivery_date": "2026-10-14", "product": "base", "trade_date": "2026-10-13", "window_start": "11:15", "window_end": "12:00", "close": 104.0},
        {"delivery_date": "2026-10-14", "product": "peak", "trade_date": "2026-10-13", "window_start": "11:15", "window_end": "12:00", "close": 130.0},
        {"delivery_date": "2026-10-15", "product": "base", "trade_date": "2026-10-14", "window_start": "11:15", "window_end": "12:00", "close": 90.0},
    ])
    snapshots = pd.DataFrame([
        {"delivery_date": "2026-10-14", "product": "base", "trade_date": "2026-10-13", "snapshot_at_utc": "2026-10-13T09:20:00Z",
         "trade_time_utc": "2026-10-13T09:04:00Z", "trade_time_paris": "11:04", "price": 102.0, "volume_mwh": 24.0, "trades_before_window": 7},
        {"delivery_date": "2026-10-14", "product": "base", "trade_date": "2026-10-13", "snapshot_at_utc": "2026-10-13T08:50:00Z",
         "trade_time_utc": "2026-10-13T08:31:00Z", "trade_time_paris": "10:31", "price": 100.0, "volume_mwh": 24.0, "trades_before_window": 5},
    ])
    record = {"scored_days": 4, "hit_rate": 0.5}
    # the snapshot wins: the latest trade before the window, with its time in the label; the gap is the model forecast against that trade
    early = b.early_trade(windows, "2026-10-14", 96.5, record, snapshots)
    assert early["kind"] == "before_window" and early["label"] == "Latest EEX base trade before the window (11:04 Paris)"
    assert early["price"] == 102.0 and early["gap"] == pytest.approx(-5.5) and early["direction"] == "below"
    assert early["text"] == ("Model forecast 5.5 EUR/MWh below that trade (tape read at 11:20 Paris, 7 trades before the window); "
                             "the model's live hit rate is 50% over 4 windows.")
    # only the window close: labelled as such, after the forecast, and not framed as a gap to trade
    close = b.early_trade(windows, "2026-10-14", 96.5, record, None)
    assert close["kind"] == "window_close" and close["label"] == "EEX window close, after the forecast was issued"
    assert close["price"] == 104.0 and close["gap"] == pytest.approx(-7.5) and close["window"] == "11:15-12:00"
    assert close["text"] == ("The 11:15-12:00 Paris window closed 7.5 EUR/MWh above the model forecast. It traded after the forecast was issued, "
                             "so this checks the forecast and is not a gap to trade; the model's live hit rate is 50% over 4 windows.")
    assert b.early_trade(windows, "2026-10-14", 96.5, record, snapshots[snapshots["delivery_date"] == "2026-10-15"])["kind"] == "window_close"
    assert "no scored window yet" in b.early_trade(windows, "2026-10-15", 96.5, {})["text"]
    assert b.early_trade(windows, "2026-10-16", 96.5, {}, snapshots) is None and b.early_trade(pd.DataFrame(), "2026-10-14", 96.5, {}) is None
    assert b.early_trade(windows, "2026-10-14", float("nan"), {}, snapshots) is None
    assert not hasattr(public, "early_trade") and not hasattr(public, "BEFORE_WINDOW_LABEL")  # the public app has no such thing


def test_morning_run_stores_the_drivers_and_the_public_dataset_carries_them(store, table, history, tmp_path, weights, solar_weights):  # noqa: F811
    before = table[table.index < "2024-04-01"]
    refit_model(before, out_dir=tmp_path / "model", feature_set="honest_v2", holdout_days=0, log=lambda *_: None, params={"n_estimators": 20})
    v2 = tmp_path / "model" / "honest_v2.txt"
    previous_day = (pd.Timestamp(DAY) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    from thermo_fr.forecast.jobs import morning_run
    from test_jobs import FakeSolarPoints, FakeWeather, FakeWindPoints

    def run_for(day, now):
        return morning_run(store, day, kind="test", inputs_path=history, entsoe=FakeEntsoe(table), weather=FakeWeather(table), now=now,
                           reports_dir=tmp_path / "none", wind_points=FakeWindPoints(table), wind_weights=weights, solar_points=FakeSolarPoints(table),
                           solar_weights=tmp_path / "model" / "solar_proxy.json", feature_sets=("honest_v2",), model_file=v2, fallback_model_file=None)

    assert run_for(previous_day, NOW - pd.Timedelta(days=1))["status"] == "ok" and run_for(DAY, NOW)["status"] == "ok"
    meta, curve, _ = store.headline_forecast(DAY, "honest_v2")
    drivers = store.drivers(int(meta["forecast_id"]))
    assert len(drivers) == len(FEATURES["honest_v2"]) + 1 and b.base_of(drivers) == pytest.approx(curve["forecast"].mean(), abs=0.01)  # the curve is rounded
    live_fit = store.save_forecast(99, DAY, "honest", pd.Timestamp("2024-04-09T07:40Z"), "gbm", "test", 500, True, curve_for(DAY))
    assert store.drivers(live_fit).empty  # a model fitted live stores no drivers
    # the local brief panel on this store
    panel = bq.brief_panel(store.path, DAY, previous_day, "honest_v2", market_path=tmp_path / "no_market.csv", early_path=tmp_path / "no_early.csv")
    assert panel["level"]["base"] == pytest.approx(curve["forecast"].mean()) and panel["current_note"] == "headline"
    assert panel["previous_source"] == "the previous day's pre-market version" and panel["waterfall"]["complete"]
    assert sum(s["change"] for s in panel["waterfall"]["steps"]) == pytest.approx(panel["waterfall"]["change"], abs=1e-6)
    assert panel["summary"].startswith("Model forecast") and panel["record"]["windows"] == 0 and panel["early_trade"] is None
    assert not np.isnan(panel["fundamentals"].set_index("quantity").loc["DE D-1 base", "fr_spread"])
    # export, re-import into an empty store, export again: drivers.csv is there, grouped, byte for byte, and free of anything from EEX
    out = tmp_path / "published"
    export_published(store, out, days=90, now=NOW + pd.Timedelta(hours=1), market_path=tmp_path / "no_market.csv")
    published = pd.read_csv(out / "drivers.csv")
    assert list(published.columns) == ["delivery_day", "issued_at_utc", "feature_set", "feature", "group", "value", "contribution"]
    assert set(published["delivery_day"]) == {previous_day, DAY} and set(published["group"]) <= set(b.GROUP_ORDER)
    assert not any(c in published.columns for c in ("vwap", "close", "entry", "auction", "pnl_per_mwh"))
    fresh = Store(tmp_path / "fresh.sqlite")
    summary = import_published(fresh, out)
    assert summary["driver_versions"] == 2 and import_published(fresh, out)["driver_versions"] == 0
    again = tmp_path / "again"
    export_published(fresh, again, days=90, now=NOW + pd.Timedelta(hours=2), market_path=tmp_path / "no_market.csv")
    fresh.close()
    assert (out / "drivers.csv").read_bytes() == (again / "drivers.csv").read_bytes()
    # the public app builds the same brief from the published files
    data = {"status": json.loads((out / "status.json").read_text()), "forecasts": pd.read_csv(out / "forecasts.csv"), "tomorrow": pd.read_csv(out / "tomorrow.csv"),
            "prob_tomorrow": pd.read_csv(out / "tomorrow_probabilistic.csv"), "drivers": published, "actuals": pd.DataFrame(), "band": pd.DataFrame()}
    data["forecasts"] = public.default_set_rows(data["forecasts"], data["status"])
    brief = public.brief_from_published(data)
    assert brief["day"] == DAY and brief["previous_day"] == previous_day and brief["premarket"]
    assert brief["level"]["base"] == pytest.approx(panel["level"]["base"], abs=1e-9)
    assert brief["waterfall"]["complete"] and brief["summary"] == panel["summary"]
    for mine, theirs in zip(brief["waterfall"]["steps"], panel["waterfall"]["steps"]):
        assert mine["group"] == theirs["group"] and mine["change"] == pytest.approx(theirs["change"], abs=1e-9)
    assert brief["market"]["n"] == 0 and not brief["market"]["ready"]


def test_local_brief_falls_back_to_the_latest_version_with_drivers_and_to_the_stored_inputs(tmp_path):
    """A headline version issued before drivers existed, and a previous day whose versions have none."""
    s = Store(tmp_path / "db.sqlite")
    day, previous = "2026-10-14", "2026-10-13"
    head = s.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-13T08:07Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve_for(day, 100))
    late = s.save_forecast(2, day, "honest_v2", pd.Timestamp("2026-10-13T15:00Z"), "gbm:honest_v2.txt", "manual", 0, True, curve_for(day, 104))
    s.save_drivers(late, drivers_from({"load_fc_mw": 45000.0, "wind_proxy_mw": 900.0}, {"load_fc_mw": 20.0, "expected_value": 95.5}))
    index = curve_for(previous).index
    s.save_input_values(1, previous, pd.DataFrame({"load_fc_mw": 47000.0, "temp_fc_c": 14.0, "wind_proxy_mw": 2900.0, "solar_proxy_mw": 3000.0}, index=index))
    panel = bq.brief_panel(s.path, day, previous, "honest_v2", market_path=tmp_path / "no_market.csv", early_path=tmp_path / "no_early.csv")
    s.close()
    assert panel["meta"]["forecast_id"] == head and panel["level"]["base"] == pytest.approx(111.5)  # the model forecast stays the headline's
    assert panel["current_note"].startswith("from the version issued 17:00 Paris")
    assert panel["previous_source"].startswith("the inputs stored for the previous day")
    row = panel["fundamentals"].set_index("quantity")
    assert row.loc["Load forecast", "change"] == pytest.approx(-2.0) and row.loc["Wind estimate", "change"] == pytest.approx(-2.0)
    assert not panel["waterfall"]["complete"] and panel["summary"] == ""  # inputs carry no contributions: no waterfall from them
