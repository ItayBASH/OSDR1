"""
IMC breast-cancer dataset by
Wang et. al, Nature - September 2023
https://www.nature.com/articles/s41586-023-06498-3

Not automatically testing these functions because data requires a license.
"""

import os
from typing import Literal
import pandas as pd
import numpy as np
from frozendict import frozendict

from tdm.utils import microns, subtract_from_list
from tdm.cell_types import FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL, T_CELL, B_CELL, CD4_T_CELL, CD8_T_CELL
from tdm.paths import DATA_DIR
from tdm.cache import persistent_cache

from tdm.preprocess.ki67 import is_dividing
from tdm.preprocess.recluster import recluster
from tdm.preprocess.single_cell_df import (
    X_COL,
    Y_COL,
    CELL_TYPE_COL,
    KI67_COL,
    IMG_ID_COL,
    SUBJECT_ID_COL,
    DIVISION_COL,
    set_dtypes,
    remap_dict_values,
)

from functools import cache

# # Paths:
TRIPLE_NEGATIVE_IMC_DATA_DIR = DATA_DIR / "triple_negative_imc"
SINGLE_CELL_TABLE_PATH = TRIPLE_NEGATIVE_IMC_DATA_DIR / "cells.csv"
CLINICAL_TABLE_PATH = TRIPLE_NEGATIVE_IMC_DATA_DIR / "clinical.csv"

"""
Patient numbers from paper, used for validations:
"""

# timepoint 1 - baseline:
N_BASELINE_CHEMO_IMMUNO = 124
N_BASELINE_CHEMO = 119

# timepoint 2 - on treatment:
N_ON_TREATMENT_CHEMO_IMMUNO = 106
N_ON_TREATMENT_CHEMO = 101

# timepoint 3 - post treatment:
N_POST_CHEMO_IMMUNO = 106
N_POST_CHEMO = 104


"""
Number of responders from paper, used for validations:
"""

# timepoint 1 - baseline:
N_CR_BASELINE_CHEMO_IMMUNO = 57
N_CR_BASELINE_CHEMO = 53

# timepoint 2 - on treatment:
N_CR_ON_TREATMENT_CHEMO_IMMUNO = 49
N_CR_ON_TREATMENT_CHEMO = 47

# timepoint 3 - post treatment:
N_CR_POST_CHEMO_IMMUNO = 57
N_CR_POST_CHEMO = 54

"""
Days between time points
"""
DAYS_PER_CYCLE = 3 * 7
N_DAYS_BASELINE_TO_EARLY = 1 * DAYS_PER_CYCLE  # 1st day of second cycle
N_DAYS_EARLY_TO_POST = 7 * DAYS_PER_CYCLE  # time between cycle 2 to 8
N_DAYS_BASELINE_TO_POST = 8 * DAYS_PER_CYCLE  # 1st day of second cycle

"""
Phases and arms:
"""

# phases:
BIOPSY_PHASE_COLUMN = "BiopsyPhase"
BASELINE = "Baseline"
ON_TREATMENT = "On-treatment"
POST_TREATMENT = "Post-treatment"

# arms:
ARM_COLUMN = "Arm"
CHEMO = "C"
CHEMO_IMMUNO = "C&I"

# response:
RESPONSE_COLUMN = "pCR"
NO_RESPONSE = "RD"
RESPONSE = "pCR"

ALL_PROTEINS = [
    "H3",
    "CD163",
    "CD20",
    "PD-L1 (SP142)",
    "CD56",
    "Helios",
    "CD8",
    "OX40",
    "CD11c",
    "CD3",
    "GATA3",
    "SMA",
    "TOX",
    "T-bet",
    "PD-1",
    "IDO",
    "AR",
    "FOXP3",
    "PD-L1 (73-10)",
    "ICOS",
    "Ki67",
    "CD4",
    "CK5/14",
    "TCF1",
    "PDGFRB",
    "CD31",
    "GZMB",
    "PDPN",
    "HLA-ABC",
    "c-PARP",
    "panCK",
    "CD79a",
    "DNA1",
    "CK8/18",
    "DNA2",
    "Carboplatin",
    "Vimentin",
    "Calponin",
    "Caveolin-1",
    "CD15",
    "MPO",
    "HLA-DR",
    "CD68",
    "pH2AX",
    "CD45",
    "CA9",
]


EXCLUDE_FROM_GENE_SIGNATURE = [
    "CA9",
    "PD-L1 (SP142)",
    "PD-L1 (73-10)",
    "Ki67",
    # exclude apoptosis proteins for same reason?
]


GENE_SIGNATURE_PROTEINS = subtract_from_list(ALL_PROTEINS, EXCLUDE_FROM_GENE_SIGNATURE)

ORIGINAL_CELL_TYPE_COLUMN = "Label"

# used to map columns to standard tdm names
RENAME_COLUMNS = {
    "Location_Center_X": X_COL,
    "Location_Center_Y": Y_COL,
    "Ki67": KI67_COL,
    ORIGINAL_CELL_TYPE_COLUMN: CELL_TYPE_COL,
    "ImageNumber": IMG_ID_COL,
    "PatientID": SUBJECT_ID_COL,
}

# see plots in tutorial 1
TYPICAL_KI67_NOISE = 0.0002

# maps Label column as defined in the SingleCells.csv table to cell_type values used in the analysis:
CELL_TYPE_DEFINITIONS_WITH_SUBTYPES = {
    "Fibroblasts": FIBROBLAST,
    "Myofibroblasts": FIBROBLAST,
    "PDPN^+Stromal": FIBROBLAST,
    # CD4 T-cells
    "CD4^+TCF1^+T": CD4_T_CELL,
    "CD4^+PD1^+T": CD4_T_CELL,
    # CD8 T-cell
    "CD8^+T": CD8_T_CELL,
    "CD8^+TCF1^+T": CD8_T_CELL,
    "CD8^+PD1^+T_{Ex}": CD8_T_CELL,
    "CD8^+GZMB^+T": CD8_T_CELL,
    # B-cells:
    "CD20^+B": B_CELL,
    # "CD79a^+Plasma", # -> B-cells?
    # Macs:
    "M2 Mac": MACROPHAGE,
    # TODO - where are the M1 macs?
    "Endothelial": ENDOTHELIAL,
    # "DCs": DENDRITIC,  # Dendritic cells
    # EPITHELIAL CELLS
    "CK8/18^{med}": TUMOR,
    "CK^{hi}GATA3^{+}": TUMOR,
    "CK^{lo}GATA3^{+}": TUMOR,
    "panCK^{med}": TUMOR,
    "AR^{+}LAR": TUMOR,
    "TCF1^{+}": TUMOR,  # was an epithelial cell in the paper
    "MHC I&II^{hi}": TUMOR,  # was an epithelial cell in the paper
    "Helios^{+}": TUMOR,  # Treg marker
    "PD-L1^{+}GZMB^{+}": TUMOR,  # was an epithelial cell in the paper
    "CA9^{+}Hypoxia": TUMOR,  # was an epithelial cell in the paper
    "pH2AX^{+}DSB": TUMOR,  # double-strand breaks - dying cells, was an epithelial cell in the paper
    "Apoptosis": TUMOR,  # was epithelial in the paper
    "PD-L1^{+}IDO^{+}": TUMOR,  # was epithelial in the paper
    "Vimentin^{+}EMT": TUMOR,  # was epithelial in the paper
    "Basal": TUMOR,  # was epithelial in the paper
    #
    # re-assign clusters:
    # "CA9^+", # hypoxia - re-assign cluster?
    # "PD-L1^+IDO^+APCs", # Macs or Dendritic cells - reassign
    # "PD-L1^+APCs", # Macs or Dendritic cells - reassign
    #
    # Unassigned non-epithelial cells:
    # "CD15^{+}", # granulocytes
    # "Treg", # Tregs
    # "Neutrophils", # Neutrophils
    # "CD56^+NK", # NK cells
    # "CD56^{+}NE", # Neuro-endocrine cells?
}

CELL_TYPE_DEFINITIONS_CD4_CD8 = frozendict(CELL_TYPE_DEFINITIONS_WITH_SUBTYPES.copy())
CELL_TYPE_DEFINITIONS = frozendict(
    remap_dict_values(CELL_TYPE_DEFINITIONS_CD4_CD8, {CD4_T_CELL: T_CELL, CD8_T_CELL: T_CELL})  # type: ignore
)

CELL_TYPES_TO_RECLUSTER = ["CA9^+", "PD-L1^+IDO^+APCs", "PD-L1^+APCs"]

# # Why two caches:
# # 1. persistent cache saves the csv as a pickle: ~20 seconds -> ~1.5 seconds
# # 2. in-memory cache reduces time of subsequent reads: ~1.5 seconds -> ~0 seconds


@persistent_cache(ignore=["download"])
def _read_single_cell_df(download: bool = True):  # pragma: no cover
    # read the table, or download it if it doesn't exist:
    if os.path.exists(SINGLE_CELL_TABLE_PATH):
        df = pd.read_csv(SINGLE_CELL_TABLE_PATH, index_col=0)
    else:
        if download:
            print(
                f"""
            File not found for the triple-negative breast cancer dataset by Wang et. al (Nature 2023).

            We can't download it automatically because it requires a license.

            But you can follow these steps:

            1. Get a license by following instructions at:
            https://zenodo.org/records/7990870

            2. Download cells.csv and place the file at:
            {SINGLE_CELL_TABLE_PATH}

            3. Download clinical.csv and place the file at:
            {CLINICAL_TABLE_PATH}

            Make sure to name the files EXACTLY.
            """
            )
            raise NotImplementedError()
        else:
            raise UserWarning(f"{SINGLE_CELL_TABLE_PATH} was not found. Set download=True.")

    return df


@cache
def read_single_cell_df(  # pragma: no cover
    cell_type_definitions: frozendict = CELL_TYPE_DEFINITIONS,
    biopsy_phase: Literal["Baseline", "On-treatment", "Post-treatment", "all"] = "Baseline",
    treatment_arm: Literal["C", "C&I", "all"] = "all",
    response: Literal["RD", "pCR", "all"] = "all",
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
    df = df.reset_index()

    # add columns for response (pCR) and trial arm:
    clinical_df = read_clinical_df()
    df = df.merge(clinical_df, how="left", on="PatientID")

    # recluster hypoxia and pd-1+ cell-types:
    for cell_type_to_recluster in CELL_TYPES_TO_RECLUSTER:
        df = recluster(
            single_cell_df=df,
            gene_signature_protein_columns=GENE_SIGNATURE_PROTEINS,
            cell_type_column=ORIGINAL_CELL_TYPE_COLUMN,
            ignore_types=CELL_TYPES_TO_RECLUSTER,
            cell_type_to_recluster=cell_type_to_recluster,
        )

    # create columns with tdm-compatible names:
    # x,y, ki67, cell_type, img_num, subject_id
    for original_col, renamed_col in RENAME_COLUMNS.items():
        df[renamed_col] = df[original_col]

    # add binary division event column:
    df[DIVISION_COL] = is_dividing(
        single_cell_df=df, typical_noise=TYPICAL_KI67_NOISE, ki67_threshold=ki67_threshold, ki67_col=KI67_COL
    )

    # select tissues by phase:
    if biopsy_phase != "all":
        df = df[df[BIOPSY_PHASE_COLUMN] == biopsy_phase]

    # select tissues by treatment group:
    if treatment_arm != "all":
        df = df[df[ARM_COLUMN] == treatment_arm]

    # select tissues by response:
    if response != "all":
        df = df[df[RESPONSE_COLUMN] == response]

    # transform x,y values to standard units:
    df[X_COL] = microns(df[X_COL])
    df[Y_COL] = microns(df[Y_COL])

    # map cell-types:
    df[CELL_TYPE_COL] = df[ORIGINAL_CELL_TYPE_COLUMN].map(cell_type_definitions)
    df = df[~df[CELL_TYPE_COL].isna()]

    # set standard dtypes:
    df = set_dtypes(df)

    # exclude outlier samples:
    df = df[~np.isin(df[IMG_ID_COL], [1167, 1090, 1725, 64, 801, 720])]

    return df


# def all_image_numbers():
#     sc_df = read_single_cell_df()
#     return np.sort(sc_df[IMG_ID_COL].unique())


# def read_image_single_cell_df(img_num):
#     sc_df = read_single_cell_df()
#     return sc_df[sc_df[IMG_ID_COL] == img_num]


@persistent_cache
def read_clinical_df():  # pragma: no cover
    clinical_df = pd.read_csv(TRIPLE_NEGATIVE_IMC_DATA_DIR / "clinical.csv")
    clinical_df = clinical_df.groupby("PatientID").first().drop(columns=["BiopsyPhase"]).reset_index()
    return clinical_df
