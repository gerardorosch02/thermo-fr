"""The forecast against EEX traded prices: window timing, product averages, P&L signs, the hand-entered store."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_store import curve_for
from thermo_fr import cli
from thermo_fr.forecast import market
from thermo_fr.forecast.market import (
    MarketDataError,
    MarketRow,
    NoForecastBeforeWindowError,
    add_row,
    evaluate,
    evaluate_day,
    forecast_before,
    load_market,
    product_average,
    public_summary,
    window_bounds,
)
from thermo_fr.forecast.store import Store

# fictional traded prices: real EEX figures stay in the local, git-ignored file and never in the repository
SEED = MarketRow("2026-10-10", "base", "11:15", "12:00", 60.10, 61.50, 59.20, 60.90, 60.60, "test desk")


def flat(day: str, level: float) -> pd.DataFrame:
    curve = curve_for(day)
    curve["forecast"] = level
    return curve


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "forecast.db")
    # 2026-10-10: one version before the window (10:00 Paris = 08:00Z), one after (12:30 Paris); auction 51.26 flat
    s.save_forecast(1, "2026-10-10", "honest", pd.Timestamp("2026-10-09T08:00Z"), "gbm:honest.txt", "scheduled", 0, True, flat("2026-10-10", 80.0))
    s.save_forecast(2, "2026-10-10", "honest", pd.Timestamp("2026-10-09T10:30Z"), "gbm", "manual", 50000, True, flat("2026-10-10", 40.0))
    c = curve_for("2026-10-10")
    s.save_actuals(pd.Series(51.26, index=c.index), pd.Series(["2026-10-10"] * len(c)))
    # 2026-10-11: forecast below the market, auction below the entry: a winning short
    s.save_forecast(3, "2026-10-11", "honest", pd.Timestamp("2026-10-10T07:00Z"), "gbm:honest.txt", "scheduled", 0, True, flat("2026-10-11", 30.0))
    c = curve_for("2026-10-11")
    s.save_actuals(pd.Series(45.0, index=c.index), pd.Series(["2026-10-11"] * len(c)))
    # 2026-10-12: only a version issued after the window
    s.save_forecast(4, "2026-10-12", "honest", pd.Timestamp("2026-10-11T10:00Z"), "gbm:honest.txt", "manual", 0, True, flat("2026-10-12", 60.0))
    yield s
    s.close()


def test_window_lies_on_the_day_before_delivery_in_paris_time():
    start, end = window_bounds("2026-10-10", "11:15", "12:00")
    assert start == pd.Timestamp("2026-10-09T09:15Z") and end == pd.Timestamp("2026-10-09T10:00Z")  # CEST
    start, _ = window_bounds("2026-12-10", "11:15", "12:00")
    assert start == pd.Timestamp("2026-12-09T10:15Z")  # CET


def test_only_forecasts_issued_before_the_window_are_used(store):
    start, _ = window_bounds("2026-10-10", "11:15", "12:00")
    meta, curve = forecast_before(store, "2026-10-10", start)
    assert meta["forecast_id"] == 1 and curve["forecast"].iloc[0] == 80.0  # the 12:30 version is ignored even though it is later and closer
    with pytest.raises(NoForecastBeforeWindowError, match="issued before the window"):
        forecast_before(store, "2026-10-12", window_bounds("2026-10-12", "11:15", "12:00")[0])
    with pytest.raises(NoForecastBeforeWindowError, match="no honest forecast stored"):
        forecast_before(store, "2026-10-13", window_bounds("2026-10-13", "11:15", "12:00")[0])
    # a version issued exactly at the window start does not count as before it
    with pytest.raises(NoForecastBeforeWindowError):
        forecast_before(store, "2026-10-10", pd.Timestamp("2026-10-09T08:00Z"))


def test_pnl_sign_convention_long_and_short(store):
    long = evaluate_day(store, SEED)
    assert long["direction"] == "long" and long["forecast"] == 80.0 and long["entry_vwap"] == 60.60 and long["auction"] == 51.26
    assert long["pnl_per_mwh"] == pytest.approx(51.26 - 60.60) and long["pnl_per_mwh_close"] == pytest.approx(51.26 - 60.90)
    assert long["model_error"] == pytest.approx(28.74) and long["market_error"] == pytest.approx(9.34) and long["model_beats_market"] == False  # noqa: E712
    assert long["issued_paris"] == "10:00" and long["model"] == "gbm:honest.txt"
    short = evaluate_day(store, MarketRow("2026-10-11", "base", "11:15", "12:00", 50.0, 51.0, 49.0, 50.0, 50.0, "test"))
    assert short["direction"] == "short" and short["pnl_per_mwh"] == pytest.approx(5.0)  # sold at 50, auction 45
    assert short["model_error"] == pytest.approx(15.0) and short["market_error"] == pytest.approx(5.0)
    flat_row = MarketRow("2026-10-11", "base", "11:15", "12:00", 30.0, 30.0, 30.0, 30.0, 30.0, "test")
    assert evaluate_day(store, flat_row)["direction"] == "none" and np.isnan(evaluate_day(store, flat_row)["pnl_per_mwh"])


def test_base_and_peak_averages_across_daylight_saving_days():
    autumn, spring = curve_for("2026-10-25"), curve_for("2026-03-29")
    assert len(autumn) == 25 and len(spring) == 23
    # curve_for puts forecast = 100 + local hour, so the base mean of the 25-hour day counts hour 2 twice
    assert product_average(autumn["forecast"], "base") == pytest.approx((sum(range(24)) + 2) / 25 + 100)
    assert product_average(spring["forecast"], "base") == pytest.approx((sum(range(24)) - 2) / 23 + 100)
    for curve in (autumn, spring):
        assert product_average(curve["forecast"], "peak") == pytest.approx(100 + np.mean(range(8, 20)))
        assert int(market.product_mask(curve.index, "peak").sum()) == 12
    assert np.isnan(product_average(pd.Series(dtype=float), "base"))
    with pytest.raises(MarketDataError):
        product_average(autumn["forecast"], "offpeak")


def test_evaluate_reports_days_totals_bands_and_skipped(store):
    rows = pd.DataFrame([SEED.__dict__, MarketRow("2026-10-11", "base", "11:15", "12:00", 50.0, 51.0, 49.0, 50.0, 50.0, "test").__dict__,
                         MarketRow("2026-10-12", "base", "11:15", "12:00", 50.0, 51.0, 49.0, 50.0, 50.0, "test").__dict__])
    result = evaluate(store, rows, bands=(0.0, 19.5))
    table, summary = result["table"], result["summary"]
    assert len(table) == 2 and summary["days"] == 2 and summary["scored_days"] == 2
    assert table["cumulative_pnl_per_mwh"].tolist() == pytest.approx([-9.34, -4.34])
    assert summary["hit_rate"] == 0.5 and summary["total_pnl_per_mwh"] == pytest.approx(-4.34) and summary["share_model_beats_market"] == 0.0
    assert summary["model_mae"] == pytest.approx((28.74 + 15.0) / 2, abs=0.01) and summary["market_mae"] == pytest.approx((9.34 + 5.0) / 2, abs=0.01)
    assert len(summary["skipped"]) == 1 and summary["skipped"][0]["delivery_date"] == "2026-10-12" and "before the window" in summary["skipped"][0]["reason"]
    bands = result["bands"].set_index("band_eur_mwh")
    # signals are +19.4 (day 1) and -20.0 (day 2): the 19.5 band keeps only the second trade
    assert bands.loc[0.0, "trades"] == 2 and bands.loc[19.5, "trades"] == 1 and bands.loc[19.5, "total_pnl_per_mwh"] == pytest.approx(5.0)
    assert "in sample" in result["bands_note"]


def test_public_summary_hides_prices_until_enough_days(store):
    result = evaluate(store, pd.DataFrame([SEED.__dict__]))
    hidden = public_summary(result)  # MIN_PUBLIC_DAYS is 20: the public app collects data first
    assert hidden["scored_days"] == 1 and hidden["note"] == "Versus the market: collecting data, 1 of 20 days." and "mean_pnl_per_mwh" not in hidden
    shown = public_summary(result, min_days=1)
    assert shown["mean_pnl_per_mwh"] == pytest.approx(-9.34) and shown["hit_rate"] == 0.0
    assert not {"entry_vwap", "entry_close", "open", "high", "low", "close", "vwap"} & set(shown)
    assert not any(str(v) in ("60.6", "60.9", "60.1", "61.5", "59.2") for v in shown.values())


def test_add_row_validates_and_refuses_duplicates(tmp_path):
    path = tmp_path / "market" / "eex_fr_da.csv"
    frame = add_row(SEED, path)
    assert path.exists() and len(frame) == 1 and list(frame.columns) == market.COLUMNS and frame.loc[0, "vwap"] == 60.60
    with pytest.raises(MarketDataError, match="already stored"):
        add_row(SEED, path)
    add_row(MarketRow("2026-10-10", "peak", "11:15", "12:00", 60.0, 62.0, 59.0, 61.0, 60.5, "test"), path)
    assert len(load_market(path)) == 2 and load_market(tmp_path / "missing.csv").empty
    with pytest.raises(MarketDataError, match="outside the low to high"):
        MarketRow("2026-10-10", "base", "11:15", "12:00", 65.0, 61.50, 59.20, 60.90, 60.60, "x").validate()
    with pytest.raises(MarketDataError, match="product must be"):
        MarketRow("2026-10-10", "offpeak", "11:15", "12:00", 50.0, 52.0, 49.0, 51.0, 51.0, "x").validate()
    with pytest.raises(MarketDataError, match="HH:MM"):
        MarketRow("2026-10-10", "base", "1115", "12:00", 50.0, 52.0, 49.0, 51.0, 51.0, "x").validate()
    with pytest.raises(MarketDataError, match="after window_start"):
        MarketRow("2026-10-10", "base", "12:00", "11:15", 50.0, 52.0, 49.0, 51.0, 51.0, "x").validate()
    with pytest.raises(MarketDataError, match="source"):
        MarketRow("2026-10-10", "base", "11:15", "12:00", 50.0, 52.0, 49.0, 51.0, 51.0, "").validate()


def test_market_file_is_git_ignored():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "check-ignore", "-q", str(market.DEFAULT_PATH)], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, f"{market.DEFAULT_PATH} is not ignored by git: {result.stderr}"
    assert "/data/" in (repo / ".gitignore").read_text(encoding="utf-8")


def test_cli_add_and_evaluate(store, tmp_path, capsys):
    path = tmp_path / "eex.csv"
    args = ["market", "add", "--path", str(path), "--date", "2026-10-10", "--product", "base", "--window", "11:15-12:00", "--open", "60.10",
            "--high", "61.50", "--low", "59.20", "--close", "60.90", "--vwap", "60.60", "--source", "test desk"]
    cli.main(args)
    assert "1 row" in capsys.readouterr().out and len(load_market(path)) == 1
    db = str(store.path)
    cli.main(["market", "evaluate", "--path", str(path), "--db", db])
    printed = capsys.readouterr().out
    assert "long" in printed and "-9.34" in printed and "hit rate" in printed.lower()
    with pytest.raises(SystemExit):
        cli.main(args)
