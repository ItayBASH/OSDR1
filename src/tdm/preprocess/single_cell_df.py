"""
Validations of the main single-cell dataframe provided to the Analysis object.
"""

import pandas as pd
import warnings
from anndata import AnnData
from pandas.api.types import is_float_dtype

from tdm.utils import cprint, verbosity


X_COL = "x"
Y_COL = "y"
DIVISION_COL = "division"
CELL_TYPE_COL = "cell_type"
IMG_ID_COL = "img_id"
SUBJECT_ID_COL = "subject_id"

REQUIRED_COLUMNS = [X_COL, Y_COL, DIVISION_COL, CELL_TYPE_COL, IMG_ID_COL, SUBJECT_ID_COL]

KI67_COL = "ki67"


@verbosity
def check_single_cell_df(single_cell_df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Checks that ``single_cell_df`` is preprocessed correctly and provides hints in case it isn't.

    Note:
        The ``single_cell_df`` must have:
            - ``x (float)`` and ``y (float)`` columns with spatial coordinates in standard units (e.g 1 micron = 1e-6)
            - a ``cell_type (str)`` column.
            - a ``division (bool)`` column.
            - an ``img_id (int)`` column with an identifier of the tissue section
            - a ``subject_id (int | str)`` column with an identifier of the patient (or mouse etc.)
    """
    cprint("Validating single cell dataframe...\n", color="blue")

    valid = True

    """
    'x' column:
    """

    if X_COL not in single_cell_df.columns:
        cprint(
            f"[ERROR] Missing {X_COL} column in the dataframe."
            """
            To rename the current x,y columns try:
            >>> from tdm.preprocess.single_cell_df import X_COL, Y_COL
            >>> df = df.rename(columns={"current x column name": X_COL, "current y column name": Y_COL})
            """,
            color="red",
        )
        valid = False
    else:
        cprint(f"[SUCCESS] Found {X_COL} column", color="green")

        x = single_cell_df[X_COL]

        # check correct dtype:
        if not is_float_dtype(x):
            cprint(f"[ERROR] {X_COL} column is not of type float.", color="red")
            valid = False

        # check correct units:
        if max(x) - min(x) > 5e-2:
            cprint(
                f"""
                Warning: the difference between maximal and minimal {X_COL} values exceeds 5e-2 (5cm).

                Did you provide positions in standard units (e.g 1 micron = 1e-6)?

                To convert integer numbers of microns to standard units run:
                >>> from tdm.utils import microns
                >>> df[x] = microns(df[x]) # equivalent to df[x]*1e-6
                """,
                color="yellow",
            )

        # minimal x is within 10 microns of zero:
        if not abs(min(x)) < 1e-5:
            cprint(
                """
                Warning: the minimal x position is not close to 0.

                The spatial positions within the tissue should range from (0,0) to (x max, y max)
                """,
                color="yellow",
            )

    """
    'y' column:
    """

    if Y_COL not in single_cell_df.columns:
        cprint(
            f"[ERROR] Missing {Y_COL} column in the dataframe."
            """
            To rename the current x,y columns try:
            >>> from tdm.preprocess.single_cell_df import X_COL, Y_COL
            >>> df = df.rename(columns={"current x column name": X_COL, "current y column name": Y_COL})
            """,
            color="red",
        )
        valid = False
    else:
        cprint(f"[SUCCESS] Found {Y_COL} column", color="green")

        y = single_cell_df[Y_COL]

        # check correct dtype:
        if not is_float_dtype(y):
            cprint(f"[ERROR] {Y_COL} column is not of type float.", color="red")
            valid = False

        # check correct units:
        if max(y) - min(y) > 5e-2:
            cprint(
                f"""
                Warning: the difference between maximal and minimal {Y_COL} values exceeds 5e-2 (5cm).

                Did you provide positions in standard units (e.g 1 micron = 1e-6)?

                To convert integer numbers of microns to standard units run:
                >>> from tdm.utils import microns
                >>> df[y] = microns(df[y]) # equivalent to df[y]*1e-6
                """,
                color="yellow",
            )

        # minimal y is within 10 microns of zero:
        if not abs(min(y)) < 1e-5:
            cprint(
                """
                Warning: the minimal y position is not close to 0.

                The spatial positions within the tissue should range from (0,0) to (x max, y max)
                """,
                color="yellow",
            )

    """
    'cell_type' column:
    """
    if CELL_TYPE_COL not in single_cell_df.columns:
        cprint(f"[ERROR] Missing {CELL_TYPE_COL} column in the dataframe.", color="red")
        valid = False
    else:

        cprint(f"[SUCCESS] Found {CELL_TYPE_COL} column.", color="green", new_line=False)

        # there shouldn't be more than 100 types:
        types = cell_types(single_cell_df)
        num_types = len(types)
        if num_types < 100:
            num_types_color = "green"
            cprint(f"Number of cell types: {num_types}", color=num_types_color)
        else:
            num_types_color = "yellow"
            cprint(
                f"Number of cell types: {num_types}. \nThis is a large number of cell-types, do the types look ok?",
                color=num_types_color,
            )

        # there shouldn't be any nan cell types (can cause problems in RestrictedNeighbors):
        if single_cell_df[CELL_TYPE_COL].isna().sum() > 0:
            cprint("[ERROR] Found nan values in the cell_type column.", color="red")
            valid = False

        cprint(f"\tCell types: {cell_types(single_cell_df)}", color=num_types_color)

    """
    'division' column:
    """

    if DIVISION_COL not in single_cell_df.columns:
        cprint(f"[ERROR] Missing {DIVISION_COL} column in the dataframe.", color="red")
        valid = False
    else:

        division = single_cell_df[DIVISION_COL]

        # check correct dtype:
        if not division.dtype == bool:
            cprint(f"[ERROR] {DIVISION_COL} column is not of type bool.", color="red")
            valid = False

        else:

            avg_divisions = division.mean()
            cprint(
                f"[SUCCESS] Found {DIVISION_COL} column. Fraction of dividing cells: {avg_divisions:.3f}",
                color="green",
            )

    """
    'img_num' column:
    """
    if IMG_ID_COL not in single_cell_df.columns:
        cprint(f"[ERROR] Missing {IMG_ID_COL} column in the dataframe.", color="red")
        valid = False
    else:
        img_num = single_cell_df[IMG_ID_COL]

        cprint(
            f"[SUCCESS] Found {IMG_ID_COL} column. Number of images found: {len(img_num.unique())}",
            color="green",
        )

    """
    'subject_id' column:
    """
    if SUBJECT_ID_COL not in single_cell_df.columns:
        cprint(f"[ERROR] Missing {SUBJECT_ID_COL} column in the dataframe.", color="red")
        valid = False
    else:
        subject_id = single_cell_df[SUBJECT_ID_COL]

        cprint(
            f"[SUCCESS] Found {SUBJECT_ID_COL} column. Number of subjects found: {len(subject_id.unique())}",
            color="green",
        )

    if valid:
        cprint("\n[SUCCESS] Validation complete!", color="green")
    else:
        cprint("\n[FAIL] Validation complete!", color="red")

    return valid


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    types = {
        CELL_TYPE_COL: str,
        X_COL: float,
        Y_COL: float,
        DIVISION_COL: bool,
        IMG_ID_COL: int,
    }
    return df.astype({k: v for k, v in types.items() if k in df.columns})


def restrict_df_to_required_columns(df: pd.DataFrame, warn_missing_columns: bool = True) -> pd.DataFrame:
    """
    Restrict the dataframe to the required columns.

    Note:
        If any required columns are missing, a warning is issued and the available columns are returned.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        if warn_missing_columns:
            warnings.warn(f"Warning: Missing required columns: {missing_cols}", stacklevel=2)
        available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
        return df[available_cols]
    return df[REQUIRED_COLUMNS]


def cell_types(single_cell_df: pd.DataFrame) -> list[str]:
    """Return all cell types present in the single_cell_df.

    Args:
        single_cell_df (pd.DataFrame): the single cell dataframe,
        see :func:`~tdm.preprocess.single_cell_df.check_single_cell_df` for more details.

    Returns:
        list[str]: list of cell types.
    """
    return list(single_cell_df[CELL_TYPE_COL].unique())


def n_cells_per_type(single_cell_df: pd.DataFrame) -> pd.DataFrame:
    """Return the number of cells per cell type in the single_cell_df.

    Args:
        single_cell_df (pd.DataFrame): the single_cell_df,
        see: :func:`~tdm.preprocess.single_cell_df.check_single_cell_df`

    Returns:
        pd.DataFrame: a dataframe with columns 'cell_type' and '#', where '#' is the number of cells of that type.
    """
    return single_cell_df.groupby(CELL_TYPE_COL).size().reset_index(name="#")


def remap_dict_values(d: dict, val_map: dict):
    """
    Remap values from d to new values given in val_map.

    Note:
        only maps values in d that appear in val_map.

    Tip:
        useful for reducing the granularity of a cell-type definitions dictionary.

    Example:
        >>> val_map = {'CD4': 'T', 'CD8':'T'}
        >>> d = {'CD4 T-cell Marker': 'CD4', 'CD8 T-cell Marker': 'CD8'}
        >>> remap_dict_values(d, val_map)
        {'CD4 T-cell Marker': 'T', 'CD8 T-cell Marker': 'T'}
    """
    return {k: val_map[v] if v in val_map else v for k, v in d.items()}


def add_and_rename_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    """
    Add and rename columns in the dataframe.

    Args:
        df (pd.DataFrame): the dataframe to add and rename columns in.
        rename_dict (dict): a dictionary with the original column names as keys and the new column names as values.

    Returns:
        pd.DataFrame: the dataframe with the added and renamed columns.
    """
    # create columns with tdm names:
    # x,y, ki67, cell_type, img_num, subject_id
    for original_col, renamed_col in rename_dict.items():
        df[renamed_col] = df[original_col]

    return df


def single_cell_df_from_adata(adata: AnnData) -> pd.DataFrame:
    """
    Create a single cell dataframe from an AnnData object.

    Args:
        adata (AnnData): the AnnData object to create the single cell dataframe from.

    Note:
        The adata object is expected to be the output of the MCMICRO pipeline followed by
        SCIMAP phenotyping workflow. This means it should contain:
            * spatial coordinates as ``X_centroid`` and ``Y_centroid``
            * marker counts in ``adata.raw.X``
            * phenotype annotations in ``adata.obs['phenotype']``

    Note:
        This function doesn't define division events or add a ``division`` column.
        See tdm.preprocess.ki67 for defining division events based on Ki67 expression.

    Note:
        The ``adata`` should have a ``raw`` attribute with the original counts.

    Returns:
        pd.DataFrame: the single cell dataframe.
    """
    # assert adata has the required columns:
    assert "X_centroid" in adata.obs.columns, "Missing X_centroid column in adata.obs"
    assert "Y_centroid" in adata.obs.columns, "Missing Y_centroid column in adata.obs"
    assert "phenotype" in adata.obs.columns, "Missing phenotype column in adata.obs"
    assert hasattr(adata, "raw"), "Missing raw attribute in adata"

    # fetch marker counts from adata:
    protein_counts = pd.DataFrame(data=adata.raw.X, index=adata.obs.index, columns=adata.raw.var.index)
    protein_counts = protein_counts.rename(columns={"KI67": KI67_COL})

    # fetch x,y positions and phenotype:
    scdf = adata.obs.copy()

    # rename columns to match required columns:
    scdf = scdf.rename(
        columns={
            "X_centroid": X_COL,
            "Y_centroid": Y_COL,
            "imageid": IMG_ID_COL,
            "phenotype": CELL_TYPE_COL,
        }
    )

    # assume single subject
    scdf[SUBJECT_ID_COL] = 1
    scdf[IMG_ID_COL] = 1  # TODO: support string image type and fetch from adata

    # filter out some irrelevant columns:
    scdf = restrict_df_to_required_columns(scdf, warn_missing_columns=False)

    # join marker counts:
    scdf = scdf.join(protein_counts)
    scdf = set_dtypes(scdf)

    return scdf
