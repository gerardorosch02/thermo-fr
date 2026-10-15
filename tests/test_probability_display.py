"""The negative-price panel shows the calibrated probability directly: formatting, the summary line, the axis, the calibration note."""

import math

import pandas as pd
import pytest

from thermo_fr.dashboard import probability_text as pt


def hourly(values, column="p_negative"):
    return pd.DataFrame({"hour": range(len(values)), column: values})


def test_probabilities_are_shown_as_themselves_with_a_floor_under_a_twentieth_of_a_percent():
    assert pt.format_probability(0.0) == "<0.1%" and pt.format_probability(0.0004) == "<0.1%"
    assert pt.format_probability(0.0012) == "0.1%" and pt.format_probability(0.004) == "0.4%" and pt.format_probability(0.034) == "3.4%"
    assert pt.format_probability(0.0999) == "10.0%" and pt.format_probability(0.10) == "10%" and pt.format_probability(0.234) == "23%"
    assert pt.format_probability(1.0) == "100%" and pt.format_probability(float("nan")) == "n/a" and pt.format_probability(None) == "n/a"


def test_summary_line_names_the_highest_hour_and_the_hours_above_five_percent():
    values = [0.0012] * 24
    values[16] = 0.0021
    s = pt.negative_summary(hourly(values))
    assert s["max_hour"] == 16 and s["label"] == "0.2%" and s["hours_above_5pct"] == 0 and s["text"] == "Highest hour 16:00 Paris, 0.2%."
    values[14], values[15], values[9] = 0.23, 0.08, 0.051
    assert pt.negative_summary(hourly(values))["text"] == "Highest hour 14:00 Paris, 23%; 3 hour(s) above 5%."
    assert pt.negative_summary(hourly([0.0001] * 24))["text"] == "Highest hour 00:00 Paris, <0.1%."
    assert pt.negative_summary(hourly([0.0001] * 23 + [0.07], "p_negative_cal"), "p_negative_cal")["text"] == "Highest hour 23:00 Paris, 7.0%; 1 hour(s) above 5%."
    assert pt.negative_summary(pd.DataFrame())["text"] == "" and pt.negative_summary(hourly([float("nan")] * 3))["text"] == ""


def test_axis_adapts_and_the_half_line_appears_only_when_reached():
    assert pt.probability_axis_top(0.0002) == 0.10 and pt.probability_axis_top(None) == 0.10 and pt.probability_axis_top(0.05) == 0.10
    assert pt.probability_axis_top(0.23) == pytest.approx(0.276) and pt.probability_axis_top(0.9) == 1.0
    assert not pt.show_half_line(0.10) and not pt.show_half_line(0.276) and pt.show_half_line(0.5) and pt.show_half_line(1.0)


BACKTEST = {"selection": {"recalibration": {"first_day": "2024-02-17", "last_day": "2026-06-30", "events": 1264, "accepted": True,
                                             "raw": {"brier": 0.03283, "bss_vs_clim": 0.2256, "mace": {"unweighted": 0.1177}},
                                             "calibrated": {"brier": 0.03059, "bss_vs_clim": 0.2784, "mace": {"unweighted": 0.0292}}}},
            "holdout": {"recalibration": {"first_day": "2026-07-01", "last_day": "2026-10-10", "events": 158, "accepted": True,
                                           "raw": {"brier": 0.03374, "bss_vs_clim": 0.367, "mace": {"unweighted": 0.2028}},
                                           "calibrated": {"brier": 0.03051, "bss_vs_clim": 0.4276, "mace": {"unweighted": 0.1187}}}}}
CALIBRATION = {"applied": True, "pairs_hours": 32351, "pairs_events": 1770, "last_day": "2026-10-10", "fitted_at_utc": "2026-10-12T15:00:00Z"}


def test_calibration_note_states_the_fit_and_both_windows_or_the_raw_state():
    note = pt.calibration_sentence(CALIBRATION, BACKTEST)
    assert note.startswith("The probability shown is calibrated: an isotonic (monotone) map from the classifier's raw output to the observed frequency, "
                           "fitted on 32,351 out-of-sample hours (1770 negative) through 2026-10-10, refitted monthly")
    assert "selection window (2024-02-17 to 2026-06-30, 1264 negative hours): Brier 0.0328 raw to 0.0306 calibrated, calibration error 0.118 to 0.029, skill against climatology +0.23 to +0.28" in note
    assert "holdout window (2026-07-01 to 2026-10-10, 158 negative hours): Brier 0.0337 raw to 0.0305 calibrated" in note and "p_negative_raw" in note
    raw_note = pt.calibration_sentence({"applied": False}, BACKTEST)
    assert raw_note.startswith("The probability shown is the raw classifier output") and pt.calibration_sentence(None, {}) == raw_note
    assert "Pre-registered" not in pt.calibration_sentence(CALIBRATION, {})


def test_public_app_carries_the_same_display_functions():
    import streamlit_app as app

    values = [0.0001] * 24
    values[7] = 0.17
    for f in (hourly(values), hourly([0.0] * 24), hourly([0.0001] * 23 + [0.07])):
        assert app.negative_summary(f) == pt.negative_summary(f)
    for p in (0.0, 0.0004, 0.004, 0.01, 0.034, 0.07, 0.17, 0.5, math.nan):
        assert app.format_probability(p) == pt.format_probability(p)
    assert app.probability_axis_top(0.17) == pt.probability_axis_top(0.17) and app.show_half_line(0.6) == pt.show_half_line(0.6)
    assert not hasattr(app, "calibration_sentence")  # the public page states the calibration in its own words, from the same files


def test_published_probabilities_keep_full_precision_and_carry_the_raw_value(tmp_path):
    from test_store import curve_for
    from thermo_fr.forecast.publish import export_published
    from thermo_fr.forecast.store import Store

    store = Store(tmp_path / "db.sqlite")
    day = "2026-10-13"
    curve = curve_for(day)
    fid = store.save_forecast(1, day, "honest_v2", pd.Timestamp("2026-10-12T08:00Z"), "gbm:honest_v2.txt", "scheduled", 0, True, curve)
    prob = pd.DataFrame(index=curve.index)
    prob["q10"], prob["q50"], prob["q90"], prob["lo"], prob["hi"] = curve["forecast"] - 6, curve["forecast"], curve["forecast"] + 6, curve["forecast"] - 7, curve["forecast"] + 7
    prob["p_negative"], prob["p_spike"], prob["spike_threshold"], prob["conformal_margin"], prob["p_negative_cal"] = 0.0004, 0.3, 140.0, 1.0, 0.0117
    store.save_probabilistic(fid, prob)
    export_published(store, tmp_path / "published", days=90, now=pd.Timestamp("2026-10-12T12:00Z"), market_path=tmp_path / "none.csv")
    store.close()
    tomorrow = pd.read_csv(tmp_path / "published" / "tomorrow_probabilistic.csv")
    assert tomorrow["p_negative"].eq(0.0117).all() and tomorrow["p_negative_raw"].eq(0.0004).all() and tomorrow["calibrated"].eq(1).all()
