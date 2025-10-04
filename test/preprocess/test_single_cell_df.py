import pandas as pd
import numpy as np

from tdm.utils import microns
from tdm.preprocess.single_cell_df import (
    check_single_cell_df,
    X_COL,
    Y_COL,
    DIVISION_COL,
    CELL_TYPE_COL,
    IMG_ID_COL,
    SUBJECT_ID_COL,
    remap_dict_values,
    restrict_df_to_required_columns,
)


def test_check_single_cell_df():
    """
    Test that check_single_cell_df fails if any column of the 6 is missing.
    """

    single_cell_df = pd.DataFrame(
        {
            X_COL: np.random.uniform(low=0, high=microns(500), size=100),
            Y_COL: np.random.uniform(low=0, high=microns(500), size=100),
            DIVISION_COL: np.random.choice([True, False], size=100),
            CELL_TYPE_COL: np.random.choice(["A", "B", "C"], size=100),
            IMG_ID_COL: np.arange(100),
            SUBJECT_ID_COL: np.random.choice(["a", "b", "c"], size=100),
        }
    )

    # stub passes:
    assert check_single_cell_df(single_cell_df, verbose=False)

    # check_single_cell_df fails if any column of the 6 is missing.
    for col in single_cell_df.columns:
        assert not check_single_cell_df(single_cell_df.drop(columns=col), verbose=False)

    # test that check_single_cell_df fails if x,y are not float:
    df_int = single_cell_df.copy()
    df_int[X_COL] = df_int[X_COL].astype(int)
    assert not check_single_cell_df(df_int, verbose=False)

    df_int = single_cell_df.copy()
    df_int[Y_COL] = df_int[Y_COL].astype(int)
    assert not check_single_cell_df(df_int, verbose=False)

    # test that check_single_cell_df fails if division is not bool:
    df_int = single_cell_df.copy()
    df_int[DIVISION_COL] = df_int[DIVISION_COL].astype(int)
    assert not check_single_cell_df(df_int, verbose=False)

    # test that check_single_cell_df fails if cell_type has nan values:
    df_nan = single_cell_df.copy()
    df_nan.loc[0, CELL_TYPE_COL] = np.nan
    assert not check_single_cell_df(df_nan, verbose=False)


def test_remap_dict_values():
    """
    Test that remap_dict_values remaps the values of a dictionary.
    """
    d = {"a": 1, "b": 2, "c": 3}
    val_map = {1: 10, 2: 20}

    assert remap_dict_values(d, val_map) == {"a": 10, "b": 20, "c": 3}


def test_restrict_df_to_required_columns():
    """
    Test that restrict_df_to_required_columns returns the correct columns.
    """
    single_cell_df = pd.DataFrame(
        {
            X_COL: np.random.uniform(low=0, high=microns(500), size=100),
            Y_COL: np.random.uniform(low=0, high=microns(500), size=100),
            DIVISION_COL: np.random.choice([True, False], size=100),
            CELL_TYPE_COL: np.random.choice(["A", "B", "C"], size=100),
            IMG_ID_COL: np.arange(100),
            SUBJECT_ID_COL: np.random.choice(["a", "b", "c"], size=100),
            "extra": np.random.choice(["a", "b", "c"], size=100),
        }
    )

    # works if all columns are present:
    assert restrict_df_to_required_columns(single_cell_df).shape[1] == 6

    # warns if a column is missing:
    single_cell_df.drop(columns=[X_COL], inplace=True)
    assert restrict_df_to_required_columns(single_cell_df, warn_missing_columns=True).shape[1] == 5
