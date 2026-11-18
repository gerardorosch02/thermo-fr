"""The price-series measurement: the corrected inputs change two columns and nothing else."""

import numpy as np
import pandas as pd

from thermo_fr.forecast import price_seq as ps
from thermo_fr.forecast.inputs import INPUT_COLUMNS


def _sequences(index):
    half = len(index) // 2
    table = pd.DataFrame(index=index, columns=[c for pair in ps.COLUMNS.values() for c in pair], dtype=float)
    table.iloc[:half, [0, 2]] = 10.0  # the hourly series, first half
    table.iloc[half:, [1, 3]] = 20.0  # the quarter-hourly series, second half
    return table


def test_coupled_series_takes_the_hourly_series_then_the_quarter_hourly_one():
    index = pd.date_range("2025-09-29", periods=96, freq="h", tz="UTC")
    coupled = ps.coupled_series(_sequences(index))
    assert (coupled.iloc[:48] == 10.0).all().all() and (coupled.iloc[48:] == 20.0).all().all()


def test_corrected_inputs_change_only_the_two_price_columns_where_the_raw_files_reach():
    index = pd.date_range("2025-09-29", periods=120, freq="h", tz="UTC")
    base = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    corrected = ps.corrected_inputs(base, _sequences(index[:96]))
    changed = [c for c in base.columns if not np.array_equal(base[c].to_numpy(), corrected[c].to_numpy())]
    assert sorted(changed) == sorted(ps.COLUMNS)
    assert (corrected.loc[index[96:], list(ps.COLUMNS)] == 1.0).all().all()  # beyond the raw files the stored value stays
    assert (corrected.loc[index[:48], "price_de_lu_eur_mwh"] == 10.0).all() and (corrected.loc[index[48:96], "price_es_eur_mwh"] == 20.0).all()
