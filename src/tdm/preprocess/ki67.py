import pandas as pd
import numpy as np
from tdm.preprocess.single_cell_df import CELL_TYPE_COL, KI67_COL


def _transform_ki67_series(x: np.ndarray, typical_noise=0.5, drop_values_below_noise: bool = False) -> np.ndarray:
    """Selects values greater than typical_noise, subtracts typical_noise and divides by the standard deviation of
    the shifted selected values.

    Args:
        x (np.ndarray): ki67 values
        typical_noise (float, optional): typical magnitude of noise. Defaults to 0.5.

    Returns:
        np.ndarray: transformed ki67 values
    """
    x = x - typical_noise
    std = x[x > 0].std()  # standard deviation of values above noise
    x = x / std

    if drop_values_below_noise:
        x = x[x > 0]
    else:
        x[x < 0] = 0

    return x


def transform_ki67(
    single_cell_df: pd.DataFrame,
    typical_noise: float = 0.5,
    ki67_col: str = KI67_COL,
    cell_type_col: str = CELL_TYPE_COL,
):
    """Return a single-cell dataframe with standardized Ki67 values above noise, the transformed distributions should
    have similar shapes.

    Args:
        single_cell_df (pd.DataFrame): dataframe with row per cell, columns for cell type and Ki67 values.
        typical_noise (float, optional): magnitude of typical noise in the data. See example plot for finding the typical noise
        in :ref:`tutorial 01<tutorials>`.
        ki67_col (str): name of the column with Ki67 values.
        cell_type_col (str): name of the column with cell types.

    Note:
        The transformed values should have similar distributions accross different cell types.
        To plot the transformed values:

        .. code-block:: python

            from tdm.preprocess.ki67 import transform_ki67, plot_marker_distributions

            transformed_ki67_single_cell_df = transform_ki67(single_cell_df)
            plot_marker_distributions(transformed_ki67_single_cell_df, ki67_col)


    Returns:
        _type_: _description_
    """

    transformed_df = (
        single_cell_df.groupby(cell_type_col)
        .apply(lambda g: _transform_ki67_series(g[ki67_col], typical_noise=typical_noise, drop_values_below_noise=True))
        .reset_index()
        .drop(columns="level_1")
    )

    return transformed_df


def _compute_ki67_division_cutoff_series(
    single_cell_df: pd.Series, ki67_col: str, typical_noise: float, ki67_threshold: float
):
    """Computes the ki67 cutoff for division.

    Args:
        single_cell_df (pd.Series): single-cell dataframe for one cell type.
        ki67_col (str): column with the non-transformed ki67 values
        typical_noise (float): magnitude of typical noise in the data. See example plot for finding the typical noise
        in :ref:`tutorial 01<tutorials>`.
        ki67_threshold (float): _description_

    Returns:
        _type_: _description_
    """
    x = single_cell_df[ki67_col]
    std = np.std(x[x > typical_noise] - typical_noise)
    return (ki67_threshold * std) + typical_noise


def _compute_ki67_division_cutoffs(
    single_cell_df: pd.DataFrame,
    typical_noise: float,
    ki67_col: str,
    ki67_threshold: float = 0.5,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    def func(df):
        return _compute_ki67_division_cutoff_series(
            df, ki67_col=ki67_col, typical_noise=typical_noise, ki67_threshold=ki67_threshold
        )

    return single_cell_df.groupby(cell_type_col).apply(func, include_groups=False)


def is_dividing(single_cell_df: pd.DataFrame, typical_noise: float, ki67_threshold: float, ki67_col: str) -> pd.Series:
    r"""Compute a binary division label for each cell.

    Computes a cutoff for each cell-type:

    .. math::
        \\text{cutoff} = K \cdot \sigma + N

    Where :math:`K` is the ki67 threshold, :math:`N` is the typical noise level and :math:`\sigma` is the standard
    deviation of ki67 values above typical noise, after subtracting the noise level.

    Args:
        single_cell_df (pd.DataFrame): dataframe with row per cell, columns for cell type and Ki67 values.
        typical_noise (float): magnitude of typical noise in the data. See example plot for finding the typical noise
        in :ref:`tutorial 01<tutorials>`.
        ki67_threshold (float): fraction of standard deviation above noise to use as the cutoff for division.
        ki67_col (str): name of the column with Ki67 values.

    Returns:
        pd.Series: boolean series with True for dividing cells.
    """

    ki67_threshold_per_type = _compute_ki67_division_cutoffs(
        single_cell_df=single_cell_df, typical_noise=typical_noise, ki67_threshold=ki67_threshold, ki67_col=ki67_col
    )

    return single_cell_df[ki67_col] > single_cell_df.cell_type.map(ki67_threshold_per_type)
