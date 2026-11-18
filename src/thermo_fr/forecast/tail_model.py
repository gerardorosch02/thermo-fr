"""Normal body with generalised Pareto tails, and the maximum-likelihood tail fit.

Copied on 2026-11-18 from the author's MSc thesis code (de-power-caps, src/spliced_distribution.py and src/evt_model.py)
for the risk band (docs/prereg/risk_band_prereg.md). The low-level generalised Pareto helpers, the SplicedDistribution class
(density, distribution and quantile functions) and fit_gpd_mle are the thesis code line for line. Left behind: the
regime-switching fit, sampling, return levels, the saved-model loaders and the thesis configuration module, whose two
constants used here are written out below.

Design (thesis, Design A, free tail masses). For splice points a = -u_lower and b = u_upper, with tail masses
p_lower = P(X < a) and p_upper = P(X > b):

    F(x) = p_lower * GPD_SF(a - x; xi_l, beta_l)                     x < a
    F(x) = p_lower + p_bulk * [Phi(x) - Phi(a)] / [Phi(b) - Phi(a)]  a <= x <= b
    F(x) = (1 - p_upper) + p_upper * GPD_CDF(x - b; xi_u, beta_u)    x > b

The distribution function is continuous at both splice points; the density generally is not.

References: Coles (2001), An Introduction to Statistical Modeling of Extreme Values, sections 4.3 and 5. Scarrott and
MacDonald (2012), REVSTAT 10(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import stats as sp_stats

SPLICED_XI_ZERO_TOL: float = 1e-8  # thesis config: below this |xi| the exponential limit is used
CALM_LOWER_XI_FIXED: float = 0.0  # thesis config; referred to in a docstring only

logger = logging.getLogger("thermo_fr.tail_model")

ArrayLike = Union[float, np.ndarray]


def _gpd_log_pdf(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """Log-PDF of GPD(xi, beta) at non-negative excesses y.

    Uses the exponential limit when |xi| < SPLICED_XI_ZERO_TOL.
    Returns -inf outside the support (y < 0, or y > -beta/xi when xi < 0).

    Parameters
    ----------
    y : array-like
        Non-negative excesses above the threshold.
    xi : float
        GPD shape parameter.
    beta : float
        GPD scale parameter (> 0).

    Returns
    -------
    np.ndarray of log-densities, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, -np.inf)
    valid = y >= 0.0
    if xi < 0:
        valid &= y < -beta / xi
    if not valid.any():
        return out
    yv = y[valid]
    if abs(xi) < SPLICED_XI_ZERO_TOL:
        out[valid] = -np.log(beta) - yv / beta
    else:
        arg = 1.0 + xi * yv / beta
        out[valid] = -np.log(beta) - (1.0 + 1.0 / xi) * np.log(arg)
    return out


def _gpd_log_sf(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """Log-survival of GPD(xi, beta) at non-negative excesses y.

    Parameters
    ----------
    y : array-like
        Non-negative excesses.
    xi : float
        GPD shape parameter.
    beta : float
        GPD scale parameter (> 0).

    Returns
    -------
    np.ndarray of log-survival values.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)          # log SF = 0 at y=0 (SF=1)
    if abs(xi) < SPLICED_XI_ZERO_TOL:
        out = -y / beta
    else:
        arg = 1.0 + xi * y / beta
        if xi < 0:
            bounded = y < -beta / xi
            out = np.where(bounded, (-1.0 / xi) * np.log(np.maximum(arg, 1e-300)), -np.inf)
        else:
            out = (-1.0 / xi) * np.log(arg)
    return out


def _gpd_sf(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """Survival function of GPD(xi, beta)."""
    return np.exp(_gpd_log_sf(y, xi, beta))


def _gpd_cdf(y: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """CDF of GPD(xi, beta)."""
    return 1.0 - _gpd_sf(y, xi, beta)


def _gpd_ppf(p: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """Quantile function of GPD(xi, beta) for p in [0, 1).

    Parameters
    ----------
    p : array-like
        Probabilities in [0, 1).
    xi, beta : float
        GPD parameters.

    Returns
    -------
    np.ndarray of excesses y such that CDF(y) = p.
    """
    p = np.asarray(p, dtype=float)
    if abs(xi) < SPLICED_XI_ZERO_TOL:
        return -beta * np.log1p(-p)
    return (beta / xi) * (np.power(1.0 - p, -xi) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Spliced distribution — Design A
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplicedDistribution:
    """Gaussian-bulk / GPD-tail spliced distribution (Design A: free tail-mass).

    The distribution is a 3-component mixture:

      Lower tail  (x < -u_lower):  p_lower * reflected_GPD(xi_lower, beta_lower)
      Bulk        (-u_lower <= x <= u_upper):  p_bulk * TruncGaussian(mu, sigma)
      Upper tail  (x > u_upper):  p_upper * GPD(xi_upper, beta_upper)

    where p_bulk = 1 - p_upper - p_lower.  The CDF is continuous at both
    splice points; the PDF is generally discontinuous there.

    Parameters
    ----------
    mu : float
        Gaussian bulk mean.
    sigma : float
        Gaussian bulk standard deviation (> 0).
    u_upper : float
        Upper splice threshold magnitude (positive).  Upper GPD tail: x > u_upper.
    xi_upper : float
        GPD shape for the upper tail.
    beta_upper : float
        GPD scale for the upper tail (> 0).
    p_upper : float
        Probability mass assigned to the upper tail, P(X > u_upper).
        Must be in (0, 1).  Free of bulk geometry under Design A.
    u_lower : float
        Lower splice threshold magnitude (positive).  Lower GPD tail: x < -u_lower.
    xi_lower : float
        GPD shape for the lower tail.  Fixed at CALM_LOWER_XI_FIXED for the
        calm regime when fix_xi_lower=True.
    beta_lower : float
        GPD scale for the lower tail (> 0).
    p_lower : float
        Probability mass assigned to the lower tail, P(X < -u_lower).
        Must be in (0, 1).  Free of bulk geometry under Design A.
    fix_xi_lower : bool
        If True (default), xi_lower is a held-fixed modelling assumption from
        Step 2 and must not be updated by any optimiser.

    Attributes (derived, not constructor parameters)
    ------------------------------------------------
    _p_bulk : float
        Probability mass in the bulk: 1 - p_upper - p_lower.
    _bulk_gauss_prob : float
        Gaussian CDF mass in the bulk interval [Phi(u_upper) - Phi(-u_lower)].
        Used to normalise the truncated-Gaussian component.
    """

    mu: float
    sigma: float
    u_upper: float
    xi_upper: float
    beta_upper: float
    p_upper: float
    u_lower: float
    xi_lower: float
    beta_lower: float
    p_lower: float
    fix_xi_lower: bool = True

    _p_bulk: float = field(init=False, repr=False)
    _bulk_gauss_prob: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.beta_upper <= 0:
            raise ValueError(f"beta_upper must be > 0, got {self.beta_upper}")
        if self.beta_lower <= 0:
            raise ValueError(f"beta_lower must be > 0, got {self.beta_lower}")
        if not (0.0 < self.p_upper < 1.0):
            raise ValueError(f"p_upper must be in (0, 1), got {self.p_upper}")
        if not (0.0 < self.p_lower < 1.0):
            raise ValueError(f"p_lower must be in (0, 1), got {self.p_lower}")
        if self.p_upper + self.p_lower >= 1.0:
            raise ValueError(
                f"p_upper + p_lower must be < 1, "
                f"got {self.p_upper} + {self.p_lower} = {self.p_upper + self.p_lower}"
            )
        self._recompute_normalisation()

    def _recompute_normalisation(self) -> None:
        """Compute derived quantities from current parameters.

        Called in __post_init__ and by the EM M-step after parameter updates.
        Under Design A, p_upper and p_lower are free parameters; this method
        only derives the implied bulk mass and the Gaussian CDF mass in the
        bulk interval.
        """
        b = self.u_upper
        a = -self.u_lower
        self._p_bulk = 1.0 - self.p_upper - self.p_lower
        self._bulk_gauss_prob = (
            sp_stats.norm.cdf(b, loc=self.mu, scale=self.sigma)
            - sp_stats.norm.cdf(a, loc=self.mu, scale=self.sigma)
        )

    # ── probability mass bookkeeping ─────────────────────────────────────────

    @property
    def tail_masses(self) -> dict[str, float]:
        """Return probability masses in each region."""
        return {
            "lower_tail": self.p_lower,
            "bulk": self._p_bulk,
            "upper_tail": self.p_upper,
        }

    # ── core density / distribution methods ─────────────────────────────────

    def logpdf(self, x: ArrayLike) -> np.ndarray:
        """Log-density, computed directly in log-space for tail stability.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the log-density.

        Returns
        -------
        np.ndarray, same shape as x.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        b = self.u_upper
        a = -self.u_lower

        # Upper tail: x > b
        mask_up = x > b
        if mask_up.any():
            y = x[mask_up] - b
            out[mask_up] = (
                np.log(self.p_upper)
                + _gpd_log_pdf(y, self.xi_upper, self.beta_upper)
            )

        # Bulk: a <= x <= b
        # f(x) = p_bulk * phi(x; mu, sigma) / bulk_gauss_prob
        mask_bulk = (x >= a) & (x <= b)
        if mask_bulk.any():
            out[mask_bulk] = (
                np.log(self._p_bulk)
                + sp_stats.norm.logpdf(x[mask_bulk], loc=self.mu, scale=self.sigma)
                - np.log(self._bulk_gauss_prob)
            )

        # Lower tail: x < a
        mask_dn = x < a
        if mask_dn.any():
            y = a - x[mask_dn]
            out[mask_dn] = (
                np.log(self.p_lower)
                + _gpd_log_pdf(y, self.xi_lower, self.beta_lower)
            )

        return out

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """Probability density function."""
        return np.exp(self.logpdf(x))

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """Cumulative distribution function.

        F(x) =
          p_lower * GPD_SF(-u_lower - x; xi_l, beta_l)       x < -u_lower
          p_lower + p_bulk * [Phi(x)-Phi(a)] / bulk_gauss_prob   -u_lower <= x <= u_upper
          (1 - p_upper) + p_upper * GPD_CDF(x - u_upper)     x > u_upper

        CDF is continuous at both splice points by construction.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        b = self.u_upper
        a = -self.u_lower

        # Upper tail
        mask_up = x > b
        if mask_up.any():
            y = x[mask_up] - b
            out[mask_up] = (
                1.0 - self.p_upper
                + self.p_upper * _gpd_cdf(y, self.xi_upper, self.beta_upper)
            )

        # Bulk
        mask_bulk = (x >= a) & (x <= b)
        if mask_bulk.any():
            out[mask_bulk] = (
                self.p_lower
                + self._p_bulk * (
                    sp_stats.norm.cdf(x[mask_bulk], loc=self.mu, scale=self.sigma)
                    - sp_stats.norm.cdf(a, loc=self.mu, scale=self.sigma)
                ) / self._bulk_gauss_prob
            )

        # Lower tail
        mask_dn = x < a
        if mask_dn.any():
            y = a - x[mask_dn]
            out[mask_dn] = self.p_lower * _gpd_sf(y, self.xi_lower, self.beta_lower)

        return np.clip(out, 0.0, 1.0)

    def sf(self, x: ArrayLike) -> np.ndarray:
        """Survival function, computed directly for upper-tail stability."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        b = self.u_upper

        # Upper tail: SF(x) = p_upper * GPD_SF(x - b)
        mask_up = x > b
        if mask_up.any():
            y = x[mask_up] - b
            out[mask_up] = self.p_upper * _gpd_sf(y, self.xi_upper, self.beta_upper)

        # Bulk and lower tail: SF = 1 - CDF (no underflow risk here)
        mask_rest = ~mask_up
        if mask_rest.any():
            out[mask_rest] = 1.0 - self.cdf(x[mask_rest])

        return np.clip(out, 0.0, 1.0)

    def log_sf(self, x: ArrayLike) -> np.ndarray:
        """Log-survival function, computed in log-space for upper-tail stability."""
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        b = self.u_upper

        mask_up = x > b
        if mask_up.any():
            y = x[mask_up] - b
            out[mask_up] = (
                np.log(self.p_upper)
                + _gpd_log_sf(y, self.xi_upper, self.beta_upper)
            )

        mask_rest = ~mask_up
        if mask_rest.any():
            sf_vals = self.sf(x[mask_rest])
            out[mask_rest] = np.where(sf_vals > 0, np.log(sf_vals), -np.inf)

        return out

    def ppf(self, q: ArrayLike) -> np.ndarray:
        """Quantile function (inverse CDF).

        Parameters
        ----------
        q : array-like
            Probabilities in (0, 1).

        Returns
        -------
        np.ndarray of quantiles.
        """
        q = np.asarray(q, dtype=float)
        out = np.empty_like(q)
        a = -self.u_lower

        # Lower tail: q <= p_lower
        # F(x) = p_lower * GPD_SF(a - x) = q
        # GPD_SF(y) = q / p_lower  =>  GPD_CDF(y) = 1 - q/p_lower
        # y = GPD_PPF(1 - q/p_lower)  =>  x = a - y
        mask_dn = q <= self.p_lower
        if mask_dn.any():
            ratio = q[mask_dn] / self.p_lower
            ratio = np.clip(ratio, 0.0, 1.0)
            y = _gpd_ppf(1.0 - ratio, self.xi_lower, self.beta_lower)
            out[mask_dn] = a - y

        # Bulk: p_lower < q <= p_lower + p_bulk
        # F(x) = p_lower + p_bulk * [Phi(x) - Phi(a)] / bulk_gauss_prob = q
        # Phi(x) = Phi(a) + (q - p_lower) * bulk_gauss_prob / p_bulk
        mask_bulk = (q > self.p_lower) & (q <= self.p_lower + self._p_bulk)
        if mask_bulk.any():
            phi_a = sp_stats.norm.cdf(a, loc=self.mu, scale=self.sigma)
            target_cdf = (
                phi_a
                + (q[mask_bulk] - self.p_lower)
                * self._bulk_gauss_prob
                / self._p_bulk
            )
            target_cdf = np.clip(target_cdf, 1e-15, 1.0 - 1e-15)
            out[mask_bulk] = sp_stats.norm.ppf(target_cdf, loc=self.mu, scale=self.sigma)

        # Upper tail: q > p_lower + p_bulk  (i.e. q > 1 - p_upper)
        # (1 - p_upper) + p_upper * GPD_CDF(x - u_upper) = q
        # GPD_CDF(y) = (q - (1 - p_upper)) / p_upper
        mask_up = q > self.p_lower + self._p_bulk
        if mask_up.any():
            p_gpd = (q[mask_up] - self.p_lower - self._p_bulk) / self.p_upper
            p_gpd = np.clip(p_gpd, 0.0, 1.0 - 1e-15)
            out[mask_up] = self.u_upper + _gpd_ppf(p_gpd, self.xi_upper, self.beta_upper)

        return out



def fit_gpd_mle(y: np.ndarray) -> tuple[float, float, float]:
    """Fit GPD to exceedances y (all > 0) by maximum likelihood (thesis evt_model._fit_gpd_mle).

    Returns (xi, beta, log likelihood); the exponential approximation with log likelihood minus infinity if the fit fails.
    """
    try:
        xi, _, beta = sp_stats.genpareto.fit(y, floc=0)
        ll = float(sp_stats.genpareto.logpdf(y, xi, loc=0, scale=beta).sum())
        return float(xi), float(beta), ll
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPD MLE failed (%s); returning exponential approximation", exc)
        return 0.0, float(y.mean()), -np.inf


def fit_spliced(z: np.ndarray, tail_mass: float = 0.10) -> SplicedDistribution:
    """A SplicedDistribution for the sample z: splice points at the empirical `tail_mass` and 1 - `tail_mass` quantiles, tail
    masses equal to `tail_mass`, a generalised Pareto fit to the exceedances beyond each, body mean and standard deviation from
    the values between. Written for the risk band; the thesis fitted the same object inside an EM step."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    a, b = np.quantile(z, [tail_mass, 1.0 - tail_mass])
    upper, lower = z[z > b] - b, a - z[z < a]
    xi_u, beta_u, _ = fit_gpd_mle(upper[upper > 0])
    xi_l, beta_l, _ = fit_gpd_mle(lower[lower > 0])
    body = z[(z >= a) & (z <= b)]
    return SplicedDistribution(mu=float(body.mean()), sigma=float(body.std()), u_upper=float(b), xi_upper=xi_u, beta_upper=beta_u,
                               p_upper=tail_mass, u_lower=float(-a), xi_lower=xi_l, beta_lower=beta_l, p_lower=tail_mass,
                               fix_xi_lower=False)
