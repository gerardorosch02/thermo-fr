"""Since 2026-10-12 the EEX traded price is the only benchmark of the level forecast: nothing published compares with the previous day's price."""

import json
from pathlib import Path

import pandas as pd

from test_publish import NOW, filled_store
from thermo_fr.forecast.publish import export_published, public_model_meta

FORBIDDEN = ("naive", "baseline", "same hour", "same-hour")


def test_published_files_and_status_carry_no_previous_day_comparison(tmp_path):
    out = tmp_path / "published"
    status = export_published(filled_store(tmp_path / "src.sqlite"), out, days=90, now=NOW, market_path=tmp_path / "no_market.csv")
    for name in ("forecasts.csv", "tomorrow.csv", "scores.csv"):
        columns = pd.read_csv(out / name).columns
        assert not {"naive_day", "naive_mae", "naive_rmse", "mae_below_baseline"} & set(columns), name
    text = json.dumps(status).lower()
    for word in FORBIDDEN:
        assert word not in text, word
    assert status["market"]["benchmark"].startswith("the only benchmark")
    assert "scored_against" not in status["live_record"]["default"] or "auction result" in status["live_record"]["default"]["scored_against"]


def test_model_metadata_for_status_drops_the_internal_comparison_figures():
    meta = {"feature_set": "honest_v2", "recent_check": {"mae": 32.97, "rmse": 44.13, "naive_mae": 38.57, "naive_rmse": 56.3},
            "frozen_holdout": {"mae": 25.81, "naive_mae": 29.37, "crash_2026_10_01_to_10_10_mae": 36.0}}
    public = public_model_meta(meta, ("feature_set", "recent_check", "frozen_holdout"))
    assert public["recent_check"] == {"mae": 32.97, "rmse": 44.13} and public["frozen_holdout"] == {"mae": 25.81, "crash_2026_10_01_to_10_10_mae": 36.0}
    assert public_model_meta(None, ("feature_set",)) == {}


def test_committed_model_metadata_and_public_app_source_carry_no_previous_day_comparison():
    root = Path(__file__).resolve().parents[1]
    for name in ("honest.json", "honest_v2.json"):
        path = root / "published" / "model" / name
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            for key in ("recent_check", "frozen_holdout"):
                assert not {"naive_mae", "naive_rmse"} & set(meta.get(key) or {}), (name, key)
    app = (root / "streamlit_app.py").read_text(encoding="utf-8").lower()
    assert "naive_day" not in app and "naive_mae" not in app and "baseline" not in app.replace("brier_base_rate", "")
    local = (root / "src" / "thermo_fr" / "dashboard" / "app.py").read_text(encoding="utf-8").lower()
    assert "naive_day" not in local and "naive_mae" not in local and "baseline" not in local
