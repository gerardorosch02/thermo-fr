import pandas as pd
import pytest

from thermo_fr.data.dataset import daily_frame
from thermo_fr.model import ThermoModel
from thermo_fr.report import out_of_sample
from thermo_fr.synthetic import make_synthetic


@pytest.fixture(scope="module")
def daily():
    return daily_frame(make_synthetic(threshold=15.0, gradient=2400.0, price_gradient=4.0))


def test_recovers_threshold_and_load_gradient(daily):
    fit = ThermoModel("load_mw").fit(daily).fit_
    assert abs(fit.threshold - 15.0) <= 0.5
    assert fit.gradient == pytest.approx(2400.0, rel=0.03)
    assert fit.r2 > 0.9


def test_recovers_price_gradient(daily):
    load = ThermoModel("load_mw").fit(daily)
    price = ThermoModel("price_eur_mwh").fit(daily, threshold=load.fit_.threshold)
    assert price.fit_.gradient == pytest.approx(4.0, rel=0.10)


def test_fixed_threshold_skips_search(daily):
    model = ThermoModel("load_mw").fit(daily, threshold=12.0)
    assert model.fit_.threshold == 12.0


def test_predict_requires_fit(daily):
    with pytest.raises(RuntimeError):
        ThermoModel().predict(daily)


def test_out_of_sample_error_is_small(daily):
    result = out_of_sample(daily)
    assert result["test_days"] > 300
    assert result["mape_pct"] < 5.0


def test_yearly_breakdown_recovers_gradient_each_year(daily, tmp_path):
    from thermo_fr.report import run, yearly_breakdown

    yearly = yearly_breakdown(daily, 15.0)
    assert list(yearly.index) == [2021, 2022, 2023, 2024, 2025]
    assert (yearly["load_gradient_mw_per_c"].sub(2400.0).abs() < 2400.0 * 0.06).all()
    assert (yearly["price_gradient_eur_mwh_per_c"].sub(4.0).abs() < 1.0).all()
    expected_pct = 100.0 * yearly["price_gradient_eur_mwh_per_c"] / yearly["mean_price_eur_mwh"]
    assert (yearly["price_gradient_pct_of_mean"].sub(expected_pct).abs() < 0.02).all()

    summary = run(daily, tmp_path)
    assert (tmp_path / "yearly.csv").exists() and (tmp_path / "yearly.md").exists()
    written = pd.read_csv(tmp_path / "yearly.csv", index_col="year")
    assert list(written.columns) == list(yearly.columns) and len(written) == 5
    text = (tmp_path / "yearly.md").read_text(encoding="utf-8")
    assert text.startswith("# Yearly breakdown, threshold fixed at") and "| 2025 |" in text
    ex = summary["price_excluding_peak_year"]
    assert ex["excluded_year"] in range(2021, 2026) and abs(ex["gradient_eur_mwh_per_c"] - 4.0) < 1.0
