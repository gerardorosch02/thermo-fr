"""The risk band: scores with known answers, no future error in any band, and the thesis tail code."""

import numpy as np
import pandas as pd

from thermo_fr.forecast import risk_band as rb
from thermo_fr.forecast.tail_model import SplicedDistribution, fit_gpd_mle, fit_spliced


def _errors(days=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-02-18", periods=days, freq="D")
    return pd.DataFrame(rng.standard_t(4, size=(days, 24)) * 10, index=index, columns=range(24))


def test_interval_score_known_values():
    assert rb.interval_score([5.0], [0.0], [10.0], 0.2)[0] == 10.0  # inside: the width
    assert rb.interval_score([12.0], [0.0], [10.0], 0.2)[0] == 10.0 + 10 * 2.0  # 2 above, weight 2 / 0.2
    assert rb.interval_score([-1.0], [0.0], [10.0], 0.05)[0] == 10.0 + 40 * 1.0


def test_baseline_reads_only_earlier_days():
    errors = _errors()
    base = rb.baseline_quantiles(errors)
    changed = errors.copy()
    changed.iloc[300:] = 1e6  # day 300 and every later day
    again = rb.baseline_quantiles(changed)
    for name in rb.NAMES:
        pd.testing.assert_frame_equal(base[name].iloc[:301], again[name].iloc[:301])
    assert base["q10"].iloc[:rb.MIN_DAYS].isna().all().all() and base["q10"].iloc[rb.MIN_DAYS].notna().all()


def test_baseline_is_the_empirical_quantile_of_the_previous_days():
    errors = _errors()
    base = rb.baseline_quantiles(errors)
    expected = np.quantile(errors.iloc[250 - rb.B_DAYS:250, 5].to_numpy(), 0.9)
    assert abs(base["q90"].iloc[250, 5] - expected) < 1e-12


def test_tail_band_reads_only_earlier_days():
    errors = _errors(days=260)
    tables, _ = rb.tail_quantiles(errors)
    changed = errors.copy()
    changed.iloc[200:] = 1e6
    again, _ = rb.tail_quantiles(changed)
    for name in rb.NAMES:
        pd.testing.assert_frame_equal(tables[name].iloc[:201], again[name].iloc[:201])
    assert tables["q10"].iloc[200].notna().all()


def test_tail_band_orders_its_quantiles_and_covers_a_stationary_sample():
    errors = _errors(days=500, seed=3)
    tables, params = rb.tail_quantiles(errors)
    row = 480
    values = [tables[name].iloc[row, 0] for name in rb.NAMES]
    assert values == sorted(values)
    inside = ((errors >= tables["q10"]) & (errors <= tables["q90"])).iloc[200:].to_numpy().mean()
    assert 0.76 < inside < 0.84 and len(params) > 0


def test_spliced_distribution_round_trip_and_tail_masses():
    d = SplicedDistribution(mu=0.0, sigma=1.0, u_upper=1.3, xi_upper=0.2, beta_upper=0.7, p_upper=0.1, u_lower=1.3, xi_lower=0.1, beta_lower=0.8, p_lower=0.1)
    q = np.array([0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999])
    assert np.allclose(d.cdf(d.ppf(q)), q, atol=1e-10)
    assert abs(d.cdf(1.3) - 0.9) < 1e-12 and abs(d.cdf(-1.3) - 0.1) < 1e-12
    assert abs(d.tail_masses["bulk"] - 0.8) < 1e-12


def test_gpd_fit_recovers_known_parameters():
    from scipy import stats

    y = stats.genpareto.rvs(0.2, loc=0, scale=2.0, size=20000, random_state=1)
    xi, beta, ll = fit_gpd_mle(y)
    assert abs(xi - 0.2) < 0.03 and abs(beta - 2.0) < 0.08 and np.isfinite(ll)


def test_fit_spliced_puts_the_splice_points_at_the_sample_deciles():
    z = np.random.default_rng(2).standard_t(5, size=5000)
    d = fit_spliced(z)
    assert abs(d.u_upper - np.quantile(z, 0.9)) < 1e-12 and abs(-d.u_lower - np.quantile(z, 0.1)) < 1e-12
    assert np.allclose(d.ppf(np.array([0.1, 0.9])), [np.quantile(z, 0.1), np.quantile(z, 0.9)])
