"""Variance-stabilising transform shared by the LEAR model and the asinh option of the gradient boosting walk-forward.

z = asinh((y - m) / s), with m the median and s the median absolute deviation x 1.4826 of the values the transform is fitted
on (always training rows only); the inverse is y = m + s sinh(z).

The slope variant (pre-registration docs/prereg/lear_recipe_prereg.md, candidate C; Uniejewski, arXiv 2511.13603, equation 2)
applies, to the scaled value x = (y - m) / s, z = sign(x) [asinh(|x| + a) - asinh(a)] with a = sqrt(1 / c^2 - 1). c is the
slope of the transform at the origin: c = 1 gives a = 0 and the plain asinh above, a smaller c a flatter transform.
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


def slope_offset(c: float) -> float:
    """a = sqrt(1 / c^2 - 1) for a slope c in (0, 1]."""
    c = float(c)
    if not 0.0 < c <= 1.0:
        raise ValueError(f"the asinh slope must be in (0, 1], not {c!r}")
    return float(np.sqrt(1.0 / c**2 - 1.0))


def asinh_slope_forward(x, m: float, s: float, c: float = 1.0):
    """sign(u) [asinh(|u| + a) - asinh(a)] with u = (x - m) / s; c = 1 is asinh_forward itself (the same code path)."""
    if c == 1.0:
        return asinh_forward(x, m, s)
    a = slope_offset(c)
    u = (np.asarray(x, dtype=float) - m) / s
    return np.sign(u) * (np.arcsinh(np.abs(u) + a) - np.arcsinh(a))


def asinh_slope_inverse(z, m: float, s: float, c: float = 1.0):
    """The inverse of asinh_slope_forward: m + s sign(z) [sinh(|z| + asinh(a)) - a]; c = 1 is asinh_inverse itself."""
    if c == 1.0:
        return asinh_inverse(z, m, s)
    a = slope_offset(c)
    z = np.asarray(z, dtype=float)
    return m + s * np.sign(z) * (np.sinh(np.abs(z) + np.arcsinh(a)) - a)
