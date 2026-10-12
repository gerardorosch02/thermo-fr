"""Low-end reliability bins (computed, merged, stored) and the labels, summary line and calibration sentence that read them."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from thermo_fr.dashboard import probability_text as pt
from thermo_fr.forecast import probabilistic as pb


def frame(p, y, strict=True):
    return pd.DataFrame({"p_negative": p, "is_negative": y, "strict": strict})


def synthetic(seed=0, n=20000, rates=(0.02, 0.06, 0.12)):
    """Forecasts spread over the three low-end bins with known observed rates, plus some hours at 10% and above."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 0.01, n), rng.uniform(0.01, 0.05, n // 4), rng.uniform(0.05, 0.10, n // 8), rng.uniform(0.10, 0.9, n // 10)])
    y = np.concatenate([rng.uniform(size=n) < rates[0], rng.uniform(size=n // 4) < rates[1], rng.uniform(size=n // 8) < rates[2],
                        rng.uniform(size=n // 10) < 0.4]).astype(float)
    return p, y


def test_low_end_bins_count_hours_and_rates_per_bin():
    p, y = synthetic()
    rows = pb.low_end_bins(y, p)
    assert [r["bin"] for r in rows] == ["below 1%", "1% to 5%", "5% to 10%"]
    assert rows[0]["hours"] == 20000 and rows[1]["hours"] == 5000 and rows[2]["hours"] == 2500
    assert rows[0]["observed_rate"] == pytest.approx(0.02, abs=0.004) and rows[1]["observed_rate"] == pytest.approx(0.06, abs=0.01)
    assert rows[2]["observed_rate"] == pytest.approx(0.12, abs=0.02) and 0.049 < rows[1]["mean_forecast"] * 2 < 0.07
    assert sum(r["hours"] for r in rows) == int((p < 0.10).sum())  # hours at 10% and above are not in the low end


def test_sparse_bins_merge_with_their_neighbour_and_say_so():
    rows = [{"bin": "below 1%", "lower": 0.0, "upper": 0.01, "hours": 1000, "negative_hours": 20, "mean_forecast": 0.002, "observed_rate": 0.02, "merged_from": []},
            {"bin": "1% to 5%", "lower": 0.01, "upper": 0.05, "hours": 100, "negative_hours": 4, "mean_forecast": 0.03, "observed_rate": 0.04, "merged_from": []},
            {"bin": "5% to 10%", "lower": 0.05, "upper": 0.10, "hours": 50, "negative_hours": 3, "mean_forecast": 0.07, "observed_rate": 0.06, "merged_from": []}]
    merged = pb.merge_sparse_bins(rows, min_events=10)
    # 1 to 5% (4 events) joins 5 to 10% (3 events): 7 events, still sparse, so the pair then joins the bin below: one bin is left
    assert len(merged) == 1 and merged[0]["bin"] == "below 10%" and merged[0]["negative_hours"] == 27 and merged[0]["hours"] == 1150
    assert merged[0]["merged_from"] == ["below 1%", "1% to 5%", "5% to 10%"]
    rows[1]["negative_hours"], rows[2]["negative_hours"] = 15, 3  # only the top bin is sparse: it merges downward
    merged = pb.merge_sparse_bins(rows, min_events=10)
    assert [r["bin"] for r in merged] == ["below 1%", "1% to 10%"] and merged[1]["merged_from"] == ["1% to 5%", "5% to 10%"]
    assert merged[1]["observed_rate"] == pytest.approx((0.04 * 100 + 0.06 * 50) / 150, abs=1e-5)
    assert pb.merge_sparse_bins(rows[:1], min_events=10) == rows[:1]  # a single bin is never merged away


def test_low_end_table_fixes_the_bins_on_the_selection_window_and_scores_the_holdout_on_them():
    p, y = synthetic(seed=1)
    selection = frame(p, y)
    hp, hy = synthetic(seed=2, n=2000, rates=(0.025, 0.07, 0.10))
    table = pb.low_end_table(selection, frame(hp, hy))
    assert table["edges"] == [0.0, 0.01, 0.05, 0.10] and table["merge_note"] == "no bin merged"
    assert [r["bin"] for r in table["selection"]] == [r["bin"] for r in table["holdout"]] == ["below 1%", "1% to 5%", "5% to 10%"]
    assert table["holdout"][0]["observed_rate"] == pytest.approx(0.025, abs=0.012)
    rng = np.random.default_rng(5)
    sparse = frame(np.concatenate([rng.uniform(0, 0.01, 3000), rng.uniform(0.01, 0.05, 300), rng.uniform(0.05, 0.10, 400)]),
                   np.concatenate([(np.arange(3000) < 30), (np.arange(300) < 3), (np.arange(400) < 20)]).astype(float))
    merged = pb.low_end_table(sparse, None)
    assert "1% to 5% and 5% to 10% merged into 1% to 10%" in merged["merge_note"] and merged["edges"] == [0.0, 0.01, 0.10]
    assert [r["bin"] for r in merged["selection"]] == ["below 1%", "1% to 10%"] and merged["selection"][1]["merged_from"] == ["1% to 5%", "5% to 10%"]
    assert merged["selection"][1]["negative_hours"] == 23 and merged["selection"][1]["hours"] == 700
    assert all(r["hours"] == 0 for r in pb.low_end_table(frame(p, y, strict=False), None)["selection"])  # non-strict rows are excluded


BACKTEST = {"selection": {"events": {"negative": {"low_end_bins": [
    {"bin": "below 1%", "lower": 0.0, "upper": 0.01, "hours": 18000, "negative_hours": 360, "mean_forecast": 0.0012, "observed_rate": 0.02, "merged_from": []},
    {"bin": "1% to 5%", "lower": 0.01, "upper": 0.05, "hours": 1200, "negative_hours": 84, "mean_forecast": 0.024, "observed_rate": 0.07, "merged_from": []},
    {"bin": "5% to 10%", "lower": 0.05, "upper": 0.10, "hours": 300, "negative_hours": 42, "mean_forecast": 0.071, "observed_rate": 0.14, "merged_from": []}],
    "low_end_merge_note": "no bin merged"}}},
    "holdout": {"events": {"negative": {"low_end_bins": [
        {"bin": "below 1%", "lower": 0.0, "upper": 0.01, "hours": 2200, "negative_hours": 53, "mean_forecast": 0.0015, "observed_rate": 0.0241, "merged_from": []},
        {"bin": "1% to 5%", "lower": 0.01, "upper": 0.05, "hours": 90, "negative_hours": 8, "mean_forecast": 0.026, "observed_rate": 0.0889, "merged_from": []},
        {"bin": "5% to 10%", "lower": 0.05, "upper": 0.10, "hours": 20, "negative_hours": 2, "mean_forecast": 0.07, "observed_rate": 0.1, "merged_from": []}]}}}}


def hourly(values):
    return pd.DataFrame({"hour": range(len(values)), "p_negative": values})


def test_labels_below_ten_percent_come_from_the_selection_bins_and_above_from_the_model():
    bins = pt.low_end_bins(BACKTEST)
    assert [b["bin"] for b in bins] == ["below 1%", "1% to 5%", "5% to 10%"]
    assert pt.format_probability(0.0001, bins) == "about 2.0% historically" and pt.format_probability(0.0099, bins) == "about 2.0% historically"
    assert pt.format_probability(0.01, bins) == "about 7.0% historically" and pt.format_probability(0.049, bins) == "about 7.0% historically"
    assert pt.format_probability(0.05, bins) == "about 14% historically" and pt.format_probability(0.0999, bins) == "about 14% historically"
    assert pt.format_probability(0.10, bins) == "10%" and pt.format_probability(0.234, bins) == "23%" and pt.format_probability(1.0, bins) == "100%"
    assert pt.format_probability(float("nan"), bins) == "n/a" and pt.format_probability(None, bins) == "n/a"
    assert pt.format_probability(0.004) == "low" and pt.format_probability(0.034) == "3.4%"  # no table published
    for p in (0.0, 0.0001, 0.004):
        assert "0.0%" not in pt.format_probability(p, bins) and "0.01%" not in pt.format_probability(p, bins)
    assert pt.low_end_bins({}) == [] and pt.low_end_bins({"selection": {"events": {"negative": {}}}}) == []


def test_summary_line_uses_the_highest_hours_label():
    bins = pt.low_end_bins(BACKTEST)
    values = [0.0001] * 24
    values[16] = 0.0002
    s = pt.negative_summary(hourly(values), bins)
    assert s["max_hour"] == 16 and s["label"] == "about 2.0% historically" and s["hours_above_5pct"] == 0
    assert s["text"] == "Highest hour 16:00 Paris, about 2.0% historically."
    values[14], values[15], values[9] = 0.23, 0.08, 0.051
    s = pt.negative_summary(hourly(values), bins)
    assert s["text"] == "Highest hour 14:00 Paris, 23%; 3 hour(s) above 5%." and s["hours_above_5pct"] == 3
    mid = pt.negative_summary(hourly([0.0001] * 23 + [0.07]), bins)
    assert mid["text"] == "Highest hour 23:00 Paris, about 14% historically; 1 hour(s) above 5%."
    assert pt.negative_summary(hourly([0.0001] * 24))["text"] == "Highest hour 00:00 Paris, low."  # no table: plain "low"
    assert pt.negative_summary(pd.DataFrame(), bins)["text"] == "" and pt.negative_summary(hourly([float("nan")] * 3), bins)["text"] == ""


def test_axis_adapts_and_the_half_line_appears_only_when_reached():
    assert pt.probability_axis_top(0.0002) == 0.10 and pt.probability_axis_top(None) == 0.10 and pt.probability_axis_top(0.05) == 0.10
    assert pt.probability_axis_top(0.23) == pytest.approx(0.276) and pt.probability_axis_top(0.9) == 1.0
    assert not pt.show_half_line(0.10) and not pt.show_half_line(0.276) and pt.show_half_line(0.5) and pt.show_half_line(1.0)


def test_calibration_sentence_lists_the_bins_with_both_windows_side_by_side():
    sentence = pt.low_end_sentence(BACKTEST)
    assert sentence.startswith("Low-end reliability of the forecasts below 10%, selection window with the holdout alongside: below 1%: the model gave "
                               "0.1% on average, 2.0% of 18,000 hours went negative (holdout 2.4% of 2,200 hours); 1% to 5%: the model gave 2.4% on "
                               "average, 7.0% of 1,200 hours went negative (holdout 8.9% of 90 hours); 5% to 10%: the model gave 7.1% on average, "
                               "14% of 300 hours went negative (holdout 10% of 20 hours).")
    assert "about X% historically" in sentence and "full values" in sentence and "merged" not in sentence
    merged = json.loads(json.dumps(BACKTEST))
    merged["selection"]["events"]["negative"]["low_end_merge_note"] = "1% to 5% and 5% to 10% merged into 1% to 10%: fewer than 10 negative hours on the selection window"
    assert "1% to 5% and 5% to 10% merged into 1% to 10%" in pt.low_end_sentence(merged)
    assert pt.low_end_sentence({}) == "" and pt.low_end_sentence({"x": {"events": {}}}) == ""


def test_public_app_carries_the_same_display_functions():
    import streamlit_app as app

    bins = app.low_end_bins(BACKTEST)
    assert bins == pt.low_end_bins(BACKTEST)
    values = [0.0001] * 24
    values[7] = 0.17
    for f in (hourly(values), hourly([0.0] * 24), hourly([0.0001] * 23 + [0.07])):
        assert app.negative_summary(f, bins) == pt.negative_summary(f, bins)
    for p in (0.0, 0.004, 0.01, 0.034, 0.07, 0.17, 0.5, math.nan):
        assert app.format_probability(p, bins) == pt.format_probability(p, bins)
    assert app.probability_axis_top(0.17) == pt.probability_axis_top(0.17) and app.show_half_line(0.6) == pt.show_half_line(0.6)
    assert app.low_end_sentence(BACKTEST) == pt.low_end_sentence(BACKTEST)


def test_cli_writes_the_bins_into_the_published_file(tmp_path, capsys):
    from thermo_fr import cli

    p, y = synthetic(seed=3)
    frame(p, y).to_parquet(tmp_path / "sel.parquet")
    hp, hy = synthetic(seed=4, n=2000)
    frame(hp, hy).to_parquet(tmp_path / "hold.parquet")
    published = tmp_path / "probabilistic_backtest.json"
    published.write_text(json.dumps({"selection": {"label": "selection", "events": {"negative": {"brier": {"model": 0.01}}}},
                                     "holdout": {"label": "holdout", "events": {"negative": {}}}}))
    cli.main(["prob-low-end", "--selection", str(tmp_path / "sel.parquet"), "--holdout", str(tmp_path / "hold.parquet"), "--publish", str(published)])
    out = capsys.readouterr().out
    assert "below 1%" in out and "selection" in out and "holdout" in out
    written = json.loads(published.read_text())
    assert written["selection"]["events"]["negative"]["brier"] == {"model": 0.01}  # the existing aggregates are kept
    sel, hold = written["selection"]["events"]["negative"]["low_end_bins"], written["holdout"]["events"]["negative"]["low_end_bins"]
    assert [r["bin"] for r in sel] == [r["bin"] for r in hold] == ["below 1%", "1% to 5%", "5% to 10%"]
    assert written["selection"]["events"]["negative"]["low_end_merge_note"] == "no bin merged"
    assert pt.format_probability(0.0001, pt.low_end_bins(written)).startswith("about ")


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
