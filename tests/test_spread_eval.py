"""The spread test's own pieces on synthetic data: reading the refetched files, joining them to the stored ones, what the
scramble leaves alone, and the decision rule. No real price is read here."""

import numpy as np
import pandas as pd

from test_entsoe_rest import _hour, _quarter, gl_document, gl_timeseries, period_xml, sequence_price_document
from thermo_fr.forecast import spread_eval as se
from thermo_fr.forecast.timing import delivery_days


def hourly(start, end, value=1.0):
    index = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    return pd.Series(float(value), index=index)


def toy_data(last="2026-07-05"):
    index = pd.date_range("2026-06-25", "2026-07-08", freq="h", tz="UTC", inclusive="left")
    french = pd.DataFrame({"price_eur_mwh": 50.0, "load_fc_mw": 40000.0, "nuclear_mw": 30000.0, "wind_pt_a_ms": 6.0}, index=index)
    tso = pd.DataFrame({"load": 50000.0, "onshore": 9000.0, "offshore": 2000.0, "solar": 5000.0}, index=index)
    wind = pd.DataFrame({"de_wind_pt_A_ms": 7.0, "de_wind_pt_OffB_ms": 9.0}, index=index)
    solar = pd.DataFrame({"de_solar_pt_A_wm2": 100.0}, index=index)
    return se.Data(french=french, de=pd.Series(80.0, index=index), tso=tso, wind_pts=wind, solar_pts=solar, last=last)


def test_the_german_price_is_sequence_one_hourly_where_it_exists_and_the_quarter_hour_mean_elsewhere(tmp_path):
    late = period_xml("2026-03-01T02:00Z", "2026-03-01T03:00Z", "PT15M", {1: 4, 2: 8, 3: 12, 4: 16}, "price.amount")
    document = sequence_price_document([("2", [_quarter(99)]), ("1", [_hour(10)]), ("1", [late])])
    (tmp_path / "prices_DE_LU_a_b.xml").write_bytes(document)
    price = se.coupled_german_price(se._parts(tmp_path, "prices_DE_LU"))
    assert list(price.to_numpy()) == [10.0, 10.0, 10.0]  # two hourly values, then the mean of 4, 8, 12, 16; sequence 2 is never read
    assert price.index[-1] == pd.Timestamp("2026-03-01T02:00Z")


def test_the_refetched_forecasts_come_out_under_the_stored_names(tmp_path):
    period = lambda v: period_xml("2026-03-01T00:00Z", "2026-03-01T02:00Z", "PT60M", {1: v, 2: v})  # noqa: E731
    (tmp_path / "prices_DE_LU_a_b.xml").write_bytes(sequence_price_document([("1", [_hour(10)])]))
    (tmp_path / "load_forecast_DE_LU_a_b.xml").write_bytes(gl_document([gl_timeseries([period(50000)], curve="A01")]))
    (tmp_path / "wind_solar_forecast_DE_LU_a_b.xml").write_bytes(gl_document([gl_timeseries([period(5)], "B16", "A01"), gl_timeseries([period(9)], "B19", "A01"),
                                                                              gl_timeseries([period(2)], "B18", "A01")]))
    new = se.parse_raw(tmp_path)
    assert list(new["tso"].columns) == ["load", "onshore", "offshore", "solar"]
    assert new["tso"].iloc[0].to_dict() == {"load": 50000.0, "onshore": 9.0, "offshore": 2.0, "solar": 5.0}


def test_stored_values_are_kept_and_refetched_ones_start_after_the_development_window():
    data = toy_data(last=se.DEV_LAST)
    stored = se.replace(data, de=se.cut(data.de, se.DEV_LAST), tso=se.cut(data.tso, se.DEV_LAST))
    new = {"de": data.de * 0 + 123.0, "tso": data.tso * 0 + 7.0}
    joined = se.extend(stored, new, last="2026-07-05")
    days = delivery_days(joined.de.index)
    assert (joined.de[days <= pd.Timestamp(se.DEV_LAST)] == 80.0).all() and (joined.de[days > pd.Timestamp(se.DEV_LAST)] == 123.0).all()
    assert days.max() == pd.Timestamp("2026-07-05") and joined.last == "2026-07-05"
    assert joined.de.index.is_unique and (joined.tso[delivery_days(joined.tso.index) > pd.Timestamp(se.DEV_LAST)] == 7.0).all().all()


def test_the_overlap_check_fails_when_the_refetched_price_differs():
    index = pd.date_range("2026-05-31T22:00Z", "2026-06-30T22:00Z", freq="h", inclusive="left")
    stored = se.Data(french=pd.DataFrame(index=index), de=pd.Series(80.0, index=index), tso=pd.DataFrame({k: 1000.0 for k in se.TSO_COLUMNS}, index=index),
                     wind_pts=pd.DataFrame(), solar_pts=pd.DataFrame(), last=se.DEV_LAST)
    same = {"de": stored.de.copy(), "tso": stored.tso + 0.4}
    assert se.overlap_check(stored, same)["pass"]
    shifted = {"de": stored.de.shift(1), "tso": stored.tso.copy()}  # one hour off: a different series under the same name
    shifted["de"].iloc[0] = 80.0
    off = {"de": stored.de + np.where(np.arange(len(index)) % 2, 0.5, 0.0), "tso": stored.tso.copy()}
    assert se.overlap_check(stored, shifted)["pass"]  # a constant series cannot show a shift; the next case can
    assert not se.overlap_check(stored, off)["pass"]
    assert not se.overlap_check(stored, {"de": stored.de.iloc[:100], "tso": stored.tso})["pass"]  # hours missing in the refetch


def test_the_scramble_leaves_alone_exactly_what_is_known_at_the_gate():
    data = toy_data()
    day = pd.Timestamp("2026-07-02")
    noisy = se.scrambled(data, "2026-07-02", seed=1)
    days = delivery_days(data.de.index)
    assert (noisy.de[days < day] == data.de[days < day]).all() and (noisy.de[days >= day] != data.de[days >= day]).all()
    assert (noisy.tso[days < day] == data.tso[days < day]).all().all() and (noisy.tso[days >= day] != data.tso[days >= day]).all().all()
    # French table: the price of D is scrambled, its load forecast and point forecasts are not (the project's own rule)
    assert (noisy.french.loc[days == day, "price_eur_mwh"] != 50.0).all() and (noisy.french.loc[days == day, "load_fc_mw"] == 40000.0).all()
    assert (noisy.french.loc[days < day, "price_eur_mwh"] == 50.0).all()
    # wind points: every label of D is kept, the first label of D+1 is not
    wind_days = delivery_days(data.wind_pts.index)
    assert (noisy.wind_pts[wind_days <= day] == data.wind_pts[wind_days <= day]).all().all() and (noisy.wind_pts[wind_days > day] != data.wind_pts[wind_days > day]).all().all()
    # radiation: the first label of D+1 is the mean of the last hour of D, so it is kept; the one after it is scrambled
    first_next = data.solar_pts.index[delivery_days(data.solar_pts.index) > day][0]
    assert noisy.solar_pts.loc[first_next].iloc[0] == 100.0
    assert noisy.solar_pts.loc[first_next + pd.Timedelta(hours=1)].iloc[0] != 100.0
    assert (noisy.solar_pts[data.solar_pts.index <= first_next] == 100.0).all().all()


def scored_frame(gain, days=60, slice_loss=0.0, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-07-01", periods=days * 24, freq="h", tz="UTC")
    day = pd.DatetimeIndex(index.tz_localize(None).normalize())
    frame = pd.DataFrame({"delivery_day": day, "hour": index.hour, "dow": day.dayofweek, "holiday": False, "de_holiday": 0.0,
                          "spread": rng.normal(-30, 20, len(index)), "wind_px": rng.uniform(0, 40000, len(index)), "solar_px": rng.uniform(0, 30000, len(index))}, index=index)
    error = rng.normal(0, 15, len(index))
    frame["yesterday"] = frame["spread"] + 2 * error
    frame["H"] = frame["spread"] + np.sign(error) * (np.abs(error) + gain)
    frame["G"] = frame["spread"] + error
    frame["C"] = frame["spread"] + 0.9 * error
    weekend = (frame["dow"] >= 5).to_numpy()
    frame.loc[weekend, "G"] = frame.loc[weekend, "spread"] + np.sign(error[weekend]) * (np.abs(error[weekend]) + gain + slice_loss)
    return frame


def test_the_rule_passes_on_a_clear_gain_and_names_each_condition():
    result = se.decide(scored_frame(gain=3.0))
    assert result["verdict"] == "passes" and all(result["conditions"].values())
    assert abs(result["paired"]["mean"] + result["gain"]) < 0.05 and result["paired"]["one_sided_upper"] < 0


def test_a_gain_below_a_quarter_fails_even_when_the_interval_is_below_zero():
    result = se.decide(scored_frame(gain=0.1, days=100))
    assert result["conditions"]["one_sided_upper_below_zero"] and not result["conditions"]["gain_at_least_0.25"]
    assert result["verdict"] == "fails"


def test_a_slice_worse_by_more_than_one_fails_the_rule():
    result = se.decide(scored_frame(gain=3.0, slice_loss=1.5))
    assert not result["conditions"]["no_slice_worse_by_more_than_1.0"] and result["verdict"] == "fails"
    assert result["slices"]["weekends"]["G_minus_H"] > 1.0


def test_a_worse_candidate_contradicts():
    assert se.decide(scored_frame(gain=-3.0))["verdict"] == "contradicts"


def test_the_summary_refuses_without_passed_stop_conditions(tmp_path):
    import json

    import pytest

    (tmp_path / "stop_checks.json").write_text(json.dumps({"all_pass": False}))
    with pytest.raises(SystemExit):
        se.stage_summary(tmp_path)
    scored_frame(gain=3.0).to_csv(tmp_path / "test_predictions.csv.gz")
    (tmp_path / "stop_checks.json").write_text(json.dumps({"all_pass": True, "test_predictions_sha256": "not the file"}))
    with pytest.raises(SystemExit):
        se.stage_summary(tmp_path)
    (tmp_path / "stop_checks.json").write_text(json.dumps({"all_pass": True, "test_predictions_sha256": se.sha256(tmp_path / "test_predictions.csv.gz")}))
    assert se.stage_summary(tmp_path)["decision"]["verdict"] == "passes"
