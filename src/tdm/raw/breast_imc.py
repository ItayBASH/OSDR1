"""
IMC breast-cancer dataset by
Fischer et. al., Cell Reports Medicine 2023
https://www.nature.com/articles/s41588-022-01041-y

Not automatically testing these functions because data requires some preprocessing in R.
"""

import os
from typing import Literal
import pandas as pd
import numpy as np

from tdm.utils import microns
from tdm.cell_types import FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL, T_CELL, B_CELL
from tdm.paths import DATA_DIR
from tdm.cache import persistent_cache
from tdm.preprocess.ki67 import is_dividing
from tdm.preprocess.single_cell_df import (
    X_COL,
    Y_COL,
    CELL_TYPE_COL,
    KI67_COL,
    IMG_ID_COL,
    SUBJECT_ID_COL,
    DIVISION_COL,
    set_dtypes,
)

from functools import cache

# Paths:
BREAST_IMC_DATA_DIR = DATA_DIR / "breast_imc"
SINGLE_CELL_TABLE_PATH = BREAST_IMC_DATA_DIR / "breast_imc_processed_single_cells.csv"

# CLINICAL_TABLE_NAME = "breast_imc_clinical_data_ut8.csv"
CLINICAL_TABLE_NAME = "Patient_metadata_DOWNLOADED_1_12_2024_AFTER_UTF.csv"
CLINICAL_TABLE_PATH = BREAST_IMC_DATA_DIR / CLINICAL_TABLE_NAME

# Constants from paper, used for validations
N_PATIENTS_WITH_PRIMARY_TUMORS = 771
N_PATIENTS_WITH_MATCHED_LYMPH_METS = 271

# used to map columns to standard tdm names
RENAME_COLUMNS = {
    "AreaShape_Center_X": X_COL,
    "AreaShape_Center_Y": Y_COL,
    "Er168_Ki-67": KI67_COL,
    "cell_type": CELL_TYPE_COL,
    "ImageNumber": IMG_ID_COL,
    "PID": SUBJECT_ID_COL,
}

# see plots in tutorial 1
TYPICAL_KI67_NOISE = 0.5

CELL_TYPE_DEFINITIONS = {
    "CD4": T_CELL,
    "CD8": T_CELL,
    "F": FIBROBLAST,
    "Tu": TUMOR,
    "M": MACROPHAGE,
    "En": ENDOTHELIAL,
    "B": B_CELL,
}


# Why two caches:
# 1. persistent cache saves the csv as a pickle: ~20 seconds -> ~1.5 seconds
# 2. in-memory cache reduces time of subsequent reads: ~1.5 seconds -> ~0 seconds


@persistent_cache(ignore=["download"])
def _read_single_cell_df(download: bool = True):  # pragma: no cover
    # read the table, or download it if it doesn't exist:
    if os.path.exists(SINGLE_CELL_TABLE_PATH):
        df = pd.read_csv(SINGLE_CELL_TABLE_PATH, index_col=0)
    else:
        if download:
            print(
                "Download is currently unsupported as it is required to preprocess data in R using the original authors code."
            )
            raise NotImplementedError()
        else:
            raise UserWarning("Set download=True to download the breast-cancer IMC dataset before the first use.")

    return df


@cache
def read_single_cell_df(  # pragma: no cover
    tissue_type: Literal["primary tumor", "lymph node mts", "both"] = "primary tumor",
    ki67_threshold: float = 0.5,
    download: bool = True,
) -> pd.DataFrame:
    """
    Reads the SingleCells.csv dataframe, persistent_cache for fast reading.

    Note:
        There were some additional types in the data, including 'normal breast tissue' (~30K cells from 37 patients)
    """
    # read the table, or download it if it doesn't exist:
    df = _read_single_cell_df(download)

    # select samples from primary tumor site and/or metastatic lymph nodes.
    if tissue_type == "both":
        tissue_types = ["primary tumor", "lymph node mts"]
    else:
        tissue_types = [tissue_type]

    df = df.loc[np.isin(df["TissueType"], tissue_types), :]

    # create columns with tdm names:
    # x,y, ki67, cell_type, img_num, subject_id
    for original_col, renamed_col in RENAME_COLUMNS.items():
        df[renamed_col] = df[original_col]

    # add binary division event column:
    df[DIVISION_COL] = is_dividing(
        single_cell_df=df, typical_noise=TYPICAL_KI67_NOISE, ki67_threshold=ki67_threshold, ki67_col=KI67_COL
    )

    # transform x,y values to standard units:
    df[X_COL] = microns(df[X_COL])
    df[Y_COL] = microns(df[Y_COL])

    # map cell-types:
    df[CELL_TYPE_COL] = df[CELL_TYPE_COL].map(CELL_TYPE_DEFINITIONS)
    df = df[~df[CELL_TYPE_COL].isna()]

    # exclude outlier samples:
    df = df[~np.isin(df[IMG_ID_COL], [221, 568])]

    # set standard dtypes:
    df = set_dtypes(df)

    return df


def read_clinical_df():  # pragma: no cover
    """"""
    return pd.read_csv(CLINICAL_TABLE_PATH)


def all_image_numbers():  # pragma: no cover
    sc_df = read_single_cell_df()
    return np.sort(sc_df[IMG_ID_COL].unique())


def read_image_single_cell_df(img_num):  # pragma: no cover
    sc_df = read_single_cell_df()
    return sc_df[sc_df[IMG_ID_COL] == img_num]
