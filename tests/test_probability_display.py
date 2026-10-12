"""The negative-price panel's summary line, the 'below 1%' formatting and the adaptive axis; published values stay at full precision."""

import math

import pandas as pd
import pytest

from thermo_fr.dashboard import probability_text as pt


def hourly(values):
    return pd.DataFrame({"hour": range(len(values)), "p_negative": values})


def test_probabilities_under_one_percent_are_never_shown_as_zero():
    assert pt.format_probability(0.0) == "below 1%" and pt.format_probability(0.0001) == "below 1%" and pt.format_probability(0.0099) == "below 1%"
    assert pt.format_probability(0.01) == "1.0%" and pt.format_probability(0.034) == "3.4%" and pt.format_probability(0.099) == "9.9%"
    assert pt.format_probability(0.10) == "10%" and pt.format_probability(0.5) == "50%" and pt.format_probability(1.0) == "100%"
    assert pt.format_probability(float("nan")) == "n/a" and pt.format_probability(None) == "n/a"
    for p in (0.0, 0.0001, 0.004):
        assert "0.0%" not in pt.format_probability(p) and "0.01%" not in pt.format_probability(p)


def test_summary_line_names_the_peak_hour_and_the_hours_above_five_percent():
    values = [0.001] * 24
    values[14], values[15], values[16], values[9] = 0.23, 0.08, 0.051, 0.05  # 0.05 itself is not above 5%
    s = pt.negative_summary(hourly(values))
    assert s["max_p"] == 0.23 and s["max_hour"] == 14 and s["hours_above_5pct"] == 3 and not s["all_below_1pct"]
    assert s["text"] == "Highest hourly probability of a negative price tomorrow: 23% at 14:00 Paris; 3 hour(s) above 5%."
    quiet = pt.negative_summary(hourly([0.0001] * 18 + [0.0] * 6))
    assert quiet["all_below_1pct"] and quiet["hours_above_5pct"] == 0 and quiet["text"] == "Negative prices very unlikely tomorrow (every hour below 1%)."
    edge = pt.negative_summary(hourly([0.0001] * 23 + [0.012]))  # one hour at 1.2%: not every hour is below 1%
    assert not edge["all_below_1pct"] and edge["text"].startswith("Highest hourly probability of a negative price tomorrow: 1.2% at 23:00 Paris; 0 hour(s)")
    assert pt.negative_summary(pd.DataFrame())["text"] == "" and pt.negative_summary(hourly([float("nan")] * 3))["text"] == ""


def test_axis_adapts_and_the_half_line_appears_only_when_reached():
    assert pt.probability_axis_top(0.0002) == 0.10 and pt.probability_axis_top(None) == 0.10 and pt.probability_axis_top(0.05) == 0.10
    assert pt.probability_axis_top(0.23) == pytest.approx(0.276) and pt.probability_axis_top(0.9) == 1.0
    assert not pt.show_half_line(0.10) and not pt.show_half_line(0.276) and pt.show_half_line(0.5) and pt.show_half_line(1.0)


def test_lowest_bin_sentence_quotes_the_backtest_and_is_empty_without_one():
    backtest = {"selection": {"events": {"negative": {"reliability": [{"bin": "0.0-0.1", "n": 19354, "mean_forecast": 0.0023, "observed_frequency": 0.0204}]}}},
                "holdout": {"events": {"negative": {"reliability": [{"bin": "0.0-0.1", "n": 2305, "mean_forecast": 0.0032, "observed_frequency": 0.0239}]}}}}
    sentence = pt.lowest_bin_sentence(backtest)
    assert "selection: forecast about 0.2%, observed 2.0%" in sentence and "holdout: forecast about 0.3%, observed 2.4%" in sentence
    assert "below 1%" in sentence and "full values" in sentence
    assert pt.lowest_bin_sentence({}) == "" and pt.lowest_bin_sentence({"x": {"events": {}}}) == ""


def test_public_app_carries_the_same_display_functions():
    import streamlit_app as app

    values = [0.0001] * 24
    values[7] = 0.17
    frame = hourly(values)
    assert app.negative_summary(frame) == pt.negative_summary(frame)
    for p in (0.0, 0.004, 0.01, 0.034, 0.17, 0.5, math.nan):
        assert app.format_probability(p) == pt.format_probability(p)
    assert app.probability_axis_top(0.17) == pt.probability_axis_top(0.17) and app.show_half_line(0.6) == pt.show_half_line(0.6)
    assert app.VERY_UNLIKELY == pt.VERY_UNLIKELY


def test_published_probabilities_keep_full_precision(tmp_path):
    """The display rounds; the file does not."""
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
