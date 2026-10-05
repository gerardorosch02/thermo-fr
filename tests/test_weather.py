import numpy as np
import pandas as pd

from thermo_fr.data.weather import weighted_temperature


def test_weighted_average():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    frames = {"A": pd.Series([10.0, 10.0], index=idx), "B": pd.Series([0.0, 0.0], index=idx)}
    out = weighted_temperature(frames, {"A": 3.0, "B": 1.0})
    assert np.allclose(out, [7.5, 7.5])


def test_missing_city_is_renormalised_not_zeroed():
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    frames = {"A": pd.Series([10.0], index=idx), "B": pd.Series([np.nan], index=idx)}
    out = weighted_temperature(frames, {"A": 3.0, "B": 1.0})
    assert np.isclose(out.iloc[0], 10.0)
