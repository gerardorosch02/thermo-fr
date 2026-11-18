"""The Research page: its sentences and charts are filled from published/research.json and never from typed figures."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
import streamlit_app as app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("build_research", ROOT / "scripts" / "build_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def research():
    return json.loads((ROOT / "published" / "research.json").read_text())


def test_published_file_is_aggregates_only(research):
    assert {"backtest", "live", "regime", "band", "ledger", "forward_test"} <= set(research)
    text = json.dumps(research)
    assert "timestamp_utc" not in text  # no hourly row of any kind
    assert "naive" not in text.lower() and "yesterday" not in text.lower()  # no comparison with the previous day's price
    assert len(research["live"]["daily"]) == research["live"]["days"]
    assert all(set(day) == {"delivery_day", "tree", "lear", "s1"} for day in research["live"]["daily"])


def test_backtest_sentence_carries_the_published_figures(research):
    backtest = research["backtest"]
    models = app.research_models(backtest["models"])
    text = app.research_backtest_text(backtest)
    assert f"{models['s1']['mae']:.2f}" in text and f"{models['tree']['mae']:.2f}" in text
    assert f"{backtest['days']} delivery days" in text
    assert f"{backtest['months_s1_better']} of {backtest['months']} months" in text


def test_live_sentence_says_undecided_only_when_the_interval_includes_zero(research):
    live = json.loads(json.dumps(research["live"]))
    live["paired_s1_tree"].update(mean=1.0, lo=-1.0, hi=3.0)
    assert app.R_LIVE_UNDECIDED.strip() in app.research_live_text(live)
    live["paired_s1_tree"].update(mean=2.0, lo=0.5, hi=3.0)
    assert app.R_LIVE_UNDECIDED.strip() not in app.research_live_text(live)


def test_charts_draw_one_point_per_published_row(research):
    backtest, live, regime = research["backtest"], research["live"], research["regime"]
    assert all(len(trace.x) == len(backtest["monthly"]) for trace in app.research_monthly_chart(backtest).data)
    assert len(app.research_difference_chart(backtest).data[0].y) == len(backtest["monthly"])
    assert all(len(trace.x) == live["days"] for trace in app.research_live_chart(live).data)
    shares = app.research_regime_chart(regime).data
    assert sum(shares[0].y) == pytest.approx(100, abs=0.1) and sum(shares[1].y) == pytest.approx(100, abs=0.1)


def test_builder_pairs_days_and_bins_prices():
    build = _builder()
    days = pd.date_range("2025-01-01", periods=30).strftime("%Y-%m-%d")
    frame = pd.DataFrame({"delivery_day": np.repeat(days, 24), "actual": 50.0})
    frame["good"], frame["bad"] = frame["actual"] + 1.0, frame["actual"] - 3.0
    pair = build.paired(frame, "good", "bad")
    assert pair["days"] == 30 and pair["mean"] == pytest.approx(-2.0) and pair["share_days_better"] == 1.0
    assert pair["lo"] == pytest.approx(-2.0) and pair["hi"] == pytest.approx(-2.0)
    bins = build.price_bins(pd.Series([10.0, 50.0, 199.9, 200.0, 400.0]))
    assert list(bins.astype(str)) == ["under 50", "50 to 100", "150 to 200", "200 to 250", "250 and over"]
