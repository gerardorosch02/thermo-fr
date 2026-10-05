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
