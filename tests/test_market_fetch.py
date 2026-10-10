"""Getting traded prices into the local file: the paste parser, trade dates, the collector contract, duplicates, the log, git-ignore.

Every price here is made up. The real collector module is git-ignored and is not exercised by the tests; a fake module written
into tmp_path stands in for it.
"""

import subprocess
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from thermo_fr import cli
from thermo_fr.forecast import market
from thermo_fr.forecast.market import (
    FetchOutcome,
    MarketDataError,
    MarketRow,
    add_row,
    delivery_days_to_collect,
    load_market,
    parse_paste,
    read_fetch_log,
    run_fetcher,
    trade_date_for,
    window_bounds,
)

PASTE = "FR DA Base EEX Trades 11:15-12:00 O: 70.10 H: 72.40 L: 69.30 C: 71.80 VWAP: 71.05"


def fake_plugin(tmp_path, body: str) -> Path:
    path = tmp_path / "fake_collector.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_paste_parser_reads_the_trader_line_and_variants():
    row = parse_paste(PASTE, "2026-10-13")
    assert row == MarketRow("2026-10-13", "base", "11:15", "12:00", 70.10, 72.40, 69.30, 71.80, 71.05, "EEX via trader", "pasted text")
    assert row.traded_on == "2026-10-12"
    peak = parse_paste("eex fr da PEAK 2026-10-14 trades 11:15 - 12:00 Open 80,5 High 81 Low 79,25 Close 80 vwap 80,2 (12 trades, 1440 MWh)")
    assert peak.product == "peak" and peak.delivery_date == "2026-10-14" and peak.open == 80.5 and peak.vwap == 80.2
    assert peak.trades == 12 and peak.volume_mwh == 1440.0
    assert parse_paste(PASTE + " 2026-10-20", "2026-10-13").delivery_date == "2026-10-13"  # the argument wins over a date in the text


def test_paste_parser_never_guesses():
    with pytest.raises(MarketDataError, match="neither 'base' nor 'peak'"):
        parse_paste("FR DA 11:15-12:00 O: 70 H: 72 L: 69 C: 71 VWAP: 71", "2026-10-13")
    with pytest.raises(MarketDataError, match="no vwap price"):
        parse_paste("FR DA Base 11:15-12:00 O: 70 H: 72 L: 69 C: 71", "2026-10-13")
    with pytest.raises(MarketDataError, match="no trading window"):
        parse_paste("FR DA Base O: 70 H: 72 L: 69 C: 71 VWAP: 71", "2026-10-13")
    with pytest.raises(MarketDataError, match="--date"):
        parse_paste(PASTE)
    with pytest.raises(MarketDataError, match="outside the low to high"):
        parse_paste("FR DA Base 11:15-12:00 O: 75 H: 72 L: 69 C: 71 VWAP: 71", "2026-10-13")


def test_trade_dates_and_the_days_to_collect():
    assert trade_date_for("2026-10-14") == "2026-10-13"  # Wednesday, traded Tuesday
    assert trade_date_for("2026-10-10") == "2026-10-09"  # Saturday, traded Friday
    assert trade_date_for("2026-10-11") == "2026-10-09"  # Sunday, traded Friday
    assert trade_date_for("2026-10-12") == "2026-10-09"  # Monday, traded Friday
    start, end = window_bounds("2026-10-12", "11:15", "12:00")  # Monday delivery: the window is Friday's
    assert start == pd.Timestamp("2026-10-09T09:15Z") and end == pd.Timestamp("2026-10-09T10:00Z")
    assert window_bounds("2026-10-12", "11:15", "12:00", trade_date="2026-10-11")[0] == pd.Timestamp("2026-10-11T09:15Z")
    assert delivery_days_to_collect("2026-10-09") == ["2026-10-10", "2026-10-11", "2026-10-12"]  # Friday
    assert delivery_days_to_collect("2026-10-08") == ["2026-10-09"]
    assert delivery_days_to_collect("2026-10-10") == ["2026-10-11"]  # a Saturday run, if any, collects only Sunday
    with pytest.raises(MarketDataError, match="before delivery_date"):
        MarketRow("2026-10-13", "base", "11:15", "12:00", 70, 72, 69, 71, 71, "x", trade_date="2026-10-13").validate()


def test_old_files_without_the_new_columns_are_read_and_migrated(tmp_path):
    path = tmp_path / "eex.csv"
    path.write_text("delivery_date,product,window_start,window_end,open,high,low,close,vwap,source,note\n"
                    "2026-10-12,base,11:15,12:00,70.1,72.4,69.3,71.8,71.05,test desk,\n", encoding="utf-8")
    frame = load_market(path)
    assert list(frame.columns) == market.COLUMNS and frame.loc[0, "trade_date"] == "2026-10-09" and pd.isna(frame.loc[0, "trades"])
    add_row(MarketRow("2026-10-13", "base", "11:15", "12:00", 70, 72, 69, 71, 71, "test", trades=5, volume_mwh=120.0), path)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(market.COLUMNS) and len(load_market(path)) == 2
    with pytest.raises(MarketDataError, match="already stored"):
        add_row(MarketRow("2026-10-12", "base", "11:15", "12:00", 1, 2, 1, 1, 1, "again"), path)


def test_run_fetcher_stores_logs_and_skips_what_exists(tmp_path):
    plugin = fake_plugin(tmp_path, '''
        from thermo_fr.forecast.market import FetchOutcome, MarketRow
        CALLS = []
        def fetch(delivery_dates, window, raw_dir, spacing_s, log):
            CALLS.append(list(delivery_dates))
            out = []
            for day in delivery_dates:
                out.append(FetchOutcome(day, "base", "stored", "", MarketRow(day, "base", window[0], window[1], 70.0, 72.0, 69.0, 71.0, 70.5,
                                                                            "fake site", "exact, from fake tape", trades=9, volume_mwh=216.0)))
                out.append(FetchOutcome(day, "peak", "skipped", "no peak day future for a weekend delivery"))
            return out
    ''')
    csv_path, log_path = tmp_path / "m.csv", tmp_path / "log.csv"
    outcomes = run_fetcher(["2026-10-10", "2026-10-11"], plugin=plugin, market_path=csv_path, log_path=log_path, raw_dir=tmp_path / "raw",
                           log=lambda *a: None, now=pd.Timestamp("2026-10-09T10:20Z"))
    assert [(o.delivery_date, o.product, o.status) for o in outcomes] == [
        ("2026-10-10", "base", "stored"), ("2026-10-10", "peak", "skipped"), ("2026-10-11", "base", "stored"), ("2026-10-11", "peak", "skipped")]
    stored = load_market(csv_path)
    assert len(stored) == 2 and stored["trade_date"].tolist() == ["2026-10-09", "2026-10-09"] and stored["trades"].tolist() == [9, 9]
    assert stored["note"].tolist() == ["exact, from fake tape"] * 2 and stored["source"].tolist() == ["fake site"] * 2
    log = read_fetch_log(log_path)
    assert list(log.columns) == market.LOG_COLUMNS and len(log) == 4 and set(log["status"]) == {"stored", "skipped"}
    # the retry: nothing is requested again for what is stored; the skipped peaks are asked again
    again = run_fetcher(["2026-10-10"], plugin=plugin, market_path=csv_path, log_path=log_path, raw_dir=tmp_path / "raw", log=lambda *a: None)
    assert [(o.product, o.status) for o in again] == [("base", "exists"), ("peak", "skipped")]
    assert len(load_market(csv_path)) == 2 and len(read_fetch_log(log_path)) == 6


def test_run_fetcher_refuses_rows_for_the_wrong_day_and_logs_errors(tmp_path):
    wrong_day = fake_plugin(tmp_path, '''
        from thermo_fr.forecast.market import FetchOutcome, MarketRow
        def fetch(delivery_dates, window, raw_dir, spacing_s, log):
            day = delivery_dates[0]
            return [FetchOutcome(day, "base", "stored", "", MarketRow("2026-10-20", "base", window[0], window[1], 70, 72, 69, 71, 70.5, "fake")),
                    FetchOutcome(day, "peak", "stored", "", MarketRow(day, "peak", window[0], window[1], 70, 72, 69, 71, 70.5, "fake",
                                                                       trade_date="2026-10-12"))]
    ''')
    csv_path, log_path = tmp_path / "m.csv", tmp_path / "log.csv"
    outcomes = run_fetcher(["2026-10-14"], plugin=wrong_day, market_path=csv_path, log_path=log_path, raw_dir=tmp_path / "raw", log=lambda *a: None)
    assert [o.status for o in outcomes] == ["error", "error"]
    assert "2026-10-20 when base 2026-10-14 was asked for" in outcomes[0].message and "not the trade date" in outcomes[1].message
    assert load_market(csv_path).empty and read_fetch_log(log_path)["status"].tolist() == ["error", "error"]

    blocked = fake_plugin(tmp_path, '''
        from thermo_fr.forecast.market import FetchBlocked
        def fetch(delivery_dates, window, raw_dir, spacing_s, log):
            raise FetchBlocked("HTTP 403 from the market data endpoint")
    ''')
    outcomes = run_fetcher(["2026-10-14"], plugin=blocked, market_path=csv_path, log_path=log_path, raw_dir=tmp_path / "raw", log=lambda *a: None)
    assert all(o.status == "error" and "blocked: HTTP 403" in o.message for o in outcomes) and len(outcomes) == 2
    silent = fake_plugin(tmp_path, '''
        def fetch(delivery_dates, window, raw_dir, spacing_s, log):
            return []
    ''')
    outcomes = run_fetcher(["2026-10-14"], plugin=silent, market_path=csv_path, log_path=log_path, raw_dir=tmp_path / "raw", log=lambda *a: None)
    assert all("reported nothing" in o.message for o in outcomes)
    missing = run_fetcher(["2026-10-14"], plugin=tmp_path / "nowhere.py", market_path=csv_path, log_path=log_path, raw_dir=tmp_path / "raw",
                          log=lambda *a: None)
    assert all(o.status == "error" and "no collector module" in o.message for o in missing)
    assert load_market(csv_path).empty


def test_cli_paste_and_fetch(tmp_path, capsys):
    csv_path = tmp_path / "m.csv"
    cli.main(["market", "paste", "--date", "2026-10-13", "--path", str(csv_path), "--text", PASTE])
    out = capsys.readouterr().out
    assert "Parsed and stored base row for 2026-10-13" in out and "on 2026-10-12" in out and len(load_market(csv_path)) == 1
    with pytest.raises(SystemExit, match="already stored"):
        cli.main(["market", "paste", "--date", "2026-10-13", "--path", str(csv_path), "--text", PASTE])
    with pytest.raises(SystemExit, match="no vwap"):
        cli.main(["market", "paste", "--date", "2026-10-14", "--path", str(csv_path), "--text", "FR DA Base 11:15-12:00 O: 70 H: 72 L: 69 C: 71"])
    plugin = fake_plugin(tmp_path, '''
        from thermo_fr.forecast.market import FetchOutcome, MarketRow
        def fetch(delivery_dates, window, raw_dir, spacing_s, log):
            return [FetchOutcome(d, p, "stored", "", MarketRow(d, p, window[0], window[1], 70, 72, 69, 71, 70.5, "fake", "exact, from fake tape"))
                    for d in delivery_dates for p in ("base", "peak")]
    ''')
    cli.main(["market", "fetch", "--date", "2026-10-14", "--plugin", str(plugin), "--path", str(csv_path), "--log", str(tmp_path / "log.csv"),
              "--raw-dir", str(tmp_path / "raw"), "--spacing", "0"])
    out = capsys.readouterr().out
    assert "2 stored" in out and len(load_market(csv_path)) == 3


def test_local_collector_and_raw_cache_are_git_ignored():
    repo = Path(__file__).resolve().parents[1]
    for path in (market.DEFAULT_PLUGIN, market.RAW_DIR / "anything.json", market.FETCH_LOG_PATH, market.DEFAULT_PATH):
        result = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=repo, capture_output=True, text=True)
        assert result.returncode == 0, f"{path} is not ignored by git"
    ignore = (repo / ".gitignore").read_text(encoding="utf-8")
    assert "/local/" in ignore and "/data/" in ignore
    tracked = subprocess.run(["git", "ls-files", "local", "data"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert tracked == "", f"files under local/ or data/ are tracked: {tracked}"
