"""The negative-price panel's summary line, the lowest-bin label read from the backtest table, and the adaptive axis; published values untouched."""

import math

import pandas as pd
import pytest

from thermo_fr.dashboard import probability_text as pt

BACKTEST = {"selection": {"events": {"negative": {"reliability": [{"bin": "0.0-0.1", "n": 19354, "mean_forecast": 0.0023, "observed_frequency": 0.0204},
                                                                  {"bin": "0.1-0.2", "n": 231, "mean_forecast": 0.1417, "observed_frequency": 0.3723}]}}},
            "holdout": {"events": {"negative": {"reliability": [{"bin": "0.0-0.1", "n": 2305, "mean_forecast": 0.0032, "observed_frequency": 0.0239},
                                                                {"bin": "0.1-0.2", "n": 34, "mean_forecast": 0.1443, "observed_frequency": 0.3529}]}}}}


def hourly(values):
    return pd.DataFrame({"hour": range(len(values)), "p_negative": values})


def test_lowest_bin_is_read_from_the_holdout_table_not_hard_coded():
    info = pt.lowest_bin(BACKTEST)
    assert info == {"upper": 0.1, "observed": 0.0239, "mean_forecast": 0.0032, "source": "holdout"}
    assert pt.bin_label(info) == "low (about 2% historically)"
    only_selection = pt.lowest_bin({"selection": BACKTEST["selection"]})
    assert only_selection["source"] == "selection" and pt.bin_label(only_selection) == "low (about 2% historically)"
    other = {"holdout": {"events": {"negative": {"reliability": [{"bin": "0.0-0.05", "n": 10, "mean_forecast": 0.01, "observed_frequency": 0.0351}]}}}}
    assert pt.lowest_bin(other)["upper"] == 0.05 and pt.bin_label(pt.lowest_bin(other)) == "low (about 4% historically)"
    assert pt.lowest_bin({}) == pt.FALLBACK_BIN and pt.bin_label(pt.lowest_bin({})) == "low"


def test_hours_in_the_lowest_bin_carry_its_label_never_a_tiny_percentage():
    info = pt.lowest_bin(BACKTEST)
    for p in (0.0, 0.0001, 0.004, 0.034, 0.0999):  # the holdout bin spans forecasts from 0 to 10%
        assert pt.format_probability(p, info) == "low (about 2% historically)"
    assert pt.format_probability(0.10, info) == "10%" and pt.format_probability(0.17, info) == "17%" and pt.format_probability(1.0, info) == "100%"
    assert pt.format_probability(float("nan"), info) == "n/a" and pt.format_probability(None, info) == "n/a"
    # without a backtest file the label falls back to a plain "low" under 1% and the model's figure above
    assert pt.format_probability(0.004) == "low" and pt.format_probability(0.034) == "3.4%" and pt.format_probability(0.5) == "50%"
    for p in (0.0, 0.0001, 0.004):
        assert "0.0%" not in pt.format_probability(p, info) and "0.01%" not in pt.format_probability(p, info)


def test_summary_line_uses_the_lowest_band_sentence_or_names_the_peak_hour():
    info = pt.lowest_bin(BACKTEST)
    quiet = pt.negative_summary(hourly([0.0001] * 18 + [0.0] * 6), info)
    assert quiet["all_in_lowest_band"] and quiet["hours_above_5pct"] == 0
    assert quiet["text"] == "Negative prices unlikely tomorrow (every hour in the lowest band, where about 2% of hours historically went negative)."
    values = [0.001] * 24
    values[14], values[15], values[16], values[9] = 0.23, 0.08, 0.051, 0.05  # 0.05 itself is not above 5%
    s = pt.negative_summary(hourly(values), info)
    assert s["max_p"] == 0.23 and s["max_hour"] == 14 and s["hours_above_5pct"] == 3 and not s["all_in_lowest_band"]
    assert s["text"] == "Highest hourly probability of a negative price tomorrow: 23% at 14:00 Paris; 3 hour(s) above 5%."
    edge = pt.negative_summary(hourly([0.0001] * 23 + [0.10]), info)  # one hour at the bin's upper edge is no longer inside it
    assert not edge["all_in_lowest_band"] and edge["text"].startswith("Highest hourly probability of a negative price tomorrow: 10% at 23:00 Paris; 1 hour(s)")
    assert pt.negative_summary(hourly([0.0001] * 24))["text"] == "Negative prices unlikely tomorrow (every hour below 1%)."  # no table: 1% fallback
    assert pt.negative_summary(pd.DataFrame(), info)["text"] == "" and pt.negative_summary(hourly([float("nan")] * 3), info)["text"] == ""


def test_axis_adapts_and_the_half_line_appears_only_when_reached():
    assert pt.probability_axis_top(0.0002) == 0.10 and pt.probability_axis_top(None) == 0.10 and pt.probability_axis_top(0.05) == 0.10
    assert pt.probability_axis_top(0.23) == pytest.approx(0.276) and pt.probability_axis_top(0.9) == 1.0
    assert not pt.show_half_line(0.10) and not pt.show_half_line(0.276) and pt.show_half_line(0.5) and pt.show_half_line(1.0)


def test_calibration_sentence_quotes_the_holdout_bin_and_is_empty_without_a_table():
    sentence = pt.lowest_bin_sentence(BACKTEST)
    assert sentence.startswith("In the lowest reliability bin (forecasts below 10%) the model is overconfident: on the holdout window it gave 0.3% on "
                               "average where 2.4% of those hours went negative (selection: 0.2% against 2.0%).")
    assert 'labelled "low (about 2% historically)"' in sentence and "full values" in sentence
    assert pt.lowest_bin_sentence({}) == "" and pt.lowest_bin_sentence({"x": {"events": {}}}) == ""


def test_public_app_carries_the_same_display_functions():
    import streamlit_app as app

    info = app.lowest_bin(BACKTEST)
    assert info == pt.lowest_bin(BACKTEST)
    values = [0.0001] * 24
    values[7] = 0.17
    frame = hourly(values)
    assert app.negative_summary(frame, info) == pt.negative_summary(frame, info)
    assert app.negative_summary(hourly([0.0] * 24), info) == pt.negative_summary(hourly([0.0] * 24), info)
    for p in (0.0, 0.004, 0.01, 0.034, 0.17, 0.5, math.nan):
        assert app.format_probability(p, info) == pt.format_probability(p, info)
    assert app.probability_axis_top(0.17) == pt.probability_axis_top(0.17) and app.show_half_line(0.6) == pt.show_half_line(0.6)
    assert app.lowest_bin_sentence(BACKTEST) == pt.lowest_bin_sentence(BACKTEST)


def test_published_probabilities_keep_full_precision(tmp_path):
    """The display relabels; the file does not."""
    from test_store import curve_for
    from thermo_fr.forecast.publish import export_published
    from thermo_fr.forecast.store import Store

    store = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"
    curve = curve_for(day)
    fid = store.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:00Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve)
    prob = pd.DataFrame(index=curve.index)
    prob["q10"], prob["q50"], prob["q90"], prob["lo"], prob["hi"] = curve["forecast"] - 6, curve["forecast"], curve["forecast"] + 6, curve["forecast"] - 7, curve["forecast"] + 7
    prob["p_negative"], prob["p_spike"], prob["spike_threshold"], prob["conformal_margin"] = 0.0004, 0.3, 140.0, 1.0
    store.save_probabilistic(fid, prob)
    export_published(store, tmp_path / "published", days=90, now=pd.Timestamp("2026-10-12T12:00Z"), market_path=tmp_path / "none.csv")
    store.close()
    tomorrow = pd.read_csv(tmp_path / "published" / "tomorrow_probabilistic.csv")
    assert tomorrow["p_negative"].eq(0.0004).all()
