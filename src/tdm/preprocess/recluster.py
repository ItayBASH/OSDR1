"""
Utils for reclustering cell-types.

In some cases a cluster was defined based on a dominating phenotype, such as hypoxia
proliferation etc. We want to reassign the cells to our canonical cell types.
"""

import pandas as pd
import numpy as np
from numba import jit


def recluster(
    single_cell_df: pd.DataFrame,
    gene_signature_protein_columns: list[str],
    cell_type_column: str,
    cell_type_to_recluster: str,
    ignore_types: list[str] | None = None,
) -> pd.DataFrame:
    """Recluster a cell-type by maximal correlation with the mean gene expressions within each cluster.

    Note:

        In some cases a cluster results from a dominating phenotype, such as hypoxia or
        proliferation. We want to reassign the cells to our "canonical" cell types.

        To ignore the specific phenotype the gene_signature_protein_columns shouldn't
        include the corresponding proteins (e.g Ki67 for proliferation).

    Args:
        single_cell_df (pd.DataFrame): dataframe with row per cell and columns including cell type and protein levels
        gene_signature_protein_columns (list[str]): proteins / genes used for clustering.
        cell_type_column (str): column with the cell types
        cell_type_to_recluster (str): cell type to reassign to other types in the cell_type_column.
        ignore_types (list[str] | None, optional): cell_types to ignore. Typically, this includes other types we will recluster. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    single_cell_df = single_cell_df.copy()

    # fetch cell-type +  protein columns:
    protein_df = single_cell_df.loc[:, [cell_type_column] + gene_signature_protein_columns]

    # use mean expressions as reference gene signatures for each cell type:
    if ignore_types is None:
        ignore_types = []
    ignore_types = [cell_type_to_recluster] + ignore_types

    reference_protein_df = protein_df[~np.isin(protein_df[cell_type_column], ignore_types)]
    reference_protein_df = reference_protein_df.groupby(cell_type_column).agg("mean")
    cell_types = reference_protein_df.index
    reference_protein_df = reference_protein_df.reset_index().drop(columns=cell_type_column)

    # fetch matrix of cells from the ki67 cluster
    recluster_type_protein_df = protein_df[protein_df[cell_type_column] == cell_type_to_recluster]
    recluster_type_protein_df = recluster_type_protein_df.drop(columns=cell_type_column)
    recluster_type_cell_idxs = recluster_type_protein_df.index
    recluster_type_protein_df = recluster_type_protein_df.reset_index().drop(columns="index")

    # # initialize a matrix to hold the correlations:
    # # row for each cell in the recluster type, column for each type in the reference df:
    # correlations = np.zeros((recluster_type_protein_df.shape[0], reference_protein_df.shape[0]))

    # # compute correlation between each recluster cell and each reference gene signature
    # for idx1, row1 in recluster_type_protein_df.iterrows():
    #     for idx2, row2 in reference_protein_df.iterrows():
    #         correlation = np.corrcoef(row1, row2)[0, 1]
    #         correlations[idx1, idx2] = correlation

    correlations = fast_correlations(recluster_type_protein_df.to_numpy(), reference_protein_df.to_numpy())

    # assign each cell to the type with maximal correlation:
    reassigned_types = cell_types[np.argmax(correlations, axis=1)]

    # write into single_cell_df:
    single_cell_df.loc[recluster_type_cell_idxs, cell_type_column] = reassigned_types

    return single_cell_df


@jit
def fast_correlations(mat1: np.ndarray, mat2: np.ndarray):  # pragma: no cover
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]

    correlations = np.zeros((n1, n2))

    # compute correlation between each recluster cell and each reference gene signature
    for i1 in range(n1):
        for i2 in range(n2):
            correlation = np.corrcoef(mat1[i1], mat2[i2])[0, 1]
            correlations[i1, i2] = correlation

    return correlations
