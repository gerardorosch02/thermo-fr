"""Variance-stabilising transform shared by the LEAR model and the asinh option of the gradient boosting walk-forward.

z = asinh((y - m) / s), with m the median and s the median absolute deviation x 1.4826 of the values the transform is fitted
on (always training rows only); the inverse is y = m + s sinh(z).
"""

import numpy as np

MAD_SCALE = 1.4826


def robust_scale(values) -> tuple[float, float]:
    """Median and MAD x 1.4826 of the finite values; a zero or undefined MAD becomes 1."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    m = float(np.median(v))
    s = float(np.median(np.abs(v - m)) * MAD_SCALE)
    return m, (s if s > 0 else 1.0)


def asinh_forward(x, m: float, s: float):
    return np.arcsinh((np.asarray(x, dtype=float) - m) / s)


def asinh_inverse(z, m: float, s: float):
    return m + s * np.sinh(np.asarray(z, dtype=float))
