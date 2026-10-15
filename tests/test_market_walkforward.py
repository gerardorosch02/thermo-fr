"""The aggregates-only walk-forward record against EEX traded prices, on fictional data."""

import json

import numpy as np
import pandas as pd
import pytest

from thermo_fr.forecast import walkforward_record as wr

DAYS = pd.date_range("2030-03-04", periods=6, freq="D")


def fictional_predictions(days=DAYS, level=100.0, offset=5.0):
    """Hourly rows for each Paris delivery day: actual = level + day index, forecast = actual + offset, D-1 = previous day's actual."""
    frames = []
    for i, day in enumerate(days):
        local = pd.date_range(day, periods=24, freq="h", tz="Europe/Paris")
        actual = level + i + np.zeros(24)
        frames.append(pd.DataFrame({"timestamp_utc": local.tz_convert("UTC"), "delivery_day": day.normalize(), "actual": actual,
                                    "gbm": actual + offset, "naive_day": actual - 1.0, "strict": True}))
    return pd.concat(frames).set_index("timestamp_utc")


def fictional_market(entries):
    rows = [{"delivery_date": f"{d:%Y-%m-%d}", "product": p, "vwap": v} for d, p, v in entries]
    return pd.DataFrame(rows)


def test_window_scoring_matches_market_rules():
    preds = fictional_predictions()
    # day 0: auction 100, forecast 105; entry 102 -> long, pnl -2 (loss); entry 98 on day 1 -> long, auction 101, pnl +3
    market = fictional_market([(DAYS[0], "base", 102.0), (DAYS[1], "base", 98.0), (DAYS[1], "peak", 110.0)])
    table = wr.window_table(preds, market)
    assert list(table["pnl"].round(6)) == [-2.0, 3.0, 9.0]  # peak: forecast 106 < 110 -> short, auction 101 -> +9
    rec = wr.build_record(preds, market, now="2030-03-10T12:00:00Z")
    assert rec["windows"] == 3 and rec["base_windows"] == 2 and rec["peak_windows"] == 1 and rec["days"] == 2
    assert rec["hits"] == 2 and rec["hit_rate"] == round(2 / 3, 3)
    assert rec["mean_pnl_per_mwh"] == round((-2 + 3 + 9) / 3, 2)
    assert rec["model_mae"] == 5.0
    assert rec["market_mae"] == round((2 + 3 + 9) / 3, 2)
    assert rec["base_only"]["windows"] == 2 and rec["base_only"]["hits"] == 1
    assert rec["period"] == {"first_delivery_day": "2030-03-04", "last_delivery_day": "2030-03-05"}
    assert rec["written_at_utc"] == "2030-03-10T12:00:00Z"


def test_binomial_matches_known_values():
    assert wr.binomial_two_sided(14, 42) == pytest.approx(0.0436, abs=1e-4)
    assert wr.binomial_two_sided(14, 38) == pytest.approx(0.1433, abs=1e-4)
    assert wr.binomial_two_sided(5, 10) == pytest.approx(1.0)


def test_bootstrap_is_reproducible_and_keeps_days_together():
    preds = fictional_predictions()
    entries = [(d, "base", 100.0 + i + (3 if i % 2 else -3)) for i, d in enumerate(DAYS)] + [(DAYS[2], "peak", 90.0)]
    table = wr.window_table(preds, fictional_market(entries))
    a, b = wr.block_bootstrap(table), wr.block_bootstrap(table)
    assert a == b
    assert a["days"] == 6
    lo, hi = a["mean_pnl_per_mwh"]["ci95"]
    assert lo <= table["pnl"].mean() <= hi


def test_repricing_share_and_persistence():
    preds = fictional_predictions(offset=0.0)
    # forecast = actual; D-1 = actual - 1; entries move 2x as far from D-1 as the forecast does: slope 0.5
    entries = [(d, "base", (100.0 + i - 1.0) + 2.0 * (1.0 + (i % 3))) for i, d in enumerate(DAYS)]
    rec = wr.build_record(preds, fictional_market(entries))
    assert 0 <= rec["persistence"]["model_calls_equal_persistence"] <= rec["windows"]
    # forecast - D1 is constant 1 while entry - D1 varies: slope 0
    assert rec["repricing_share"]["slope"] == pytest.approx(0.0, abs=1e-9)


def test_json_has_no_per_window_values_or_traded_prices(tmp_path):
    preds = fictional_predictions()
    vwaps = [123.456, 97.531, 111.222, 88.888, 104.321, 99.007]
    market = fictional_market([(d, "base", v) for d, v in zip(DAYS, vwaps)])
    pred_path, market_path, out = tmp_path / "p.csv", tmp_path / "m.csv", tmp_path / "out.json"
    preds.reset_index().to_csv(pred_path, index=False)
    market.to_csv(market_path, index=False)
    wr.run(pred_path, market_path, out, selection_path=pred_path)
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            # the only lists allowed are two-number confidence intervals
            assert path.endswith(".ci95") and len(node) == 2, f"list at {path}"
        else:
            assert not isinstance(node, float) or round(node, 3) not in {round(v, 3) for v in vwaps}, f"traded price at {path}"

    walk(data)
    for v in vwaps:
        assert str(v) not in text
    for key in ("windows", "hits", "hit_rate", "mean_pnl_per_mwh", "model_mae", "market_mae", "base_only", "bootstrap", "repricing_share",
                "persistence", "comparison", "source", "written_at_utc", "binomial_p_two_sided", "period_mean_price_eur_mwh"):
        assert key in data
    assert set(data["comparison"]) == {"selection", "holdout", "record_days"}


def test_cli_writes_the_file(tmp_path, monkeypatch):
    from thermo_fr import cli

    preds = fictional_predictions()
    market = fictional_market([(DAYS[0], "base", 102.0), (DAYS[1], "base", 98.0)])
    (tmp_path / "p.csv").write_text(preds.reset_index().to_csv(index=False))
    (tmp_path / "m.csv").write_text(market.to_csv(index=False))
    monkeypatch.chdir(tmp_path)
    cli.main(["market", "walkforward-record", "--predictions", "p.csv", "--path", "m.csv", "--selection", "", "--fallback-root", "",
              "--out", "out.json"])
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["windows"] == 2 and "selection" not in data["comparison"]
