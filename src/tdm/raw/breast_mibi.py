"""
IMC breast-cancer dataset by
The mibi file name is an early mistake in the codebase, it should be IMC.
Danenberg et. al., Nature Genetics 2022
https://www.nature.com/articles/s41588-022-01041-y
"""

import os
import shutil
import pandas as pd
import numpy as np
from frozendict import frozendict
from tdm.utils import microns
from tdm.paths import DATA_DIR
from tdm.cell_types import FIBROBLAST, MACROPHAGE, T_CELL, TUMOR, ENDOTHELIAL, B_CELL, CD4_T_CELL, CD8_T_CELL
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
)

# non-persistent cache:
from functools import cache

# persistent cache:
from tdm.cache import persistent_cache
from tdm.download import download_file, unzip_file


"""
Constants:
"""

# used to map columns to standard tdm names
RENAME_COLUMNS = {
    "Location_Center_X": X_COL,
    "Location_Center_Y": Y_COL,
    "Ki-67": KI67_COL,
    "cellPhenotype": CELL_TYPE_COL,
    "ImageNumber": IMG_ID_COL,
    "metabric_id": SUBJECT_ID_COL,
}

# see plots in tutorial 1
TYPICAL_KI67_NOISE = 0.5


# See notebook 11 for ki67 cutoff analysis:
KI67_THRESHOLD_MAP = {
    0.0: {"Tu": 0.5, "T": 0.5, "En": 0.5, "F": 0.5, "M": 0.5, "B": 0.5},
    0.1: {
        "Tu": 0.7013736312498289,
        "T": 0.574690093471027,
        "En": 0.6093185091748804,
        "F": 0.6621323619426547,
        "M": 0.5757646514578609,
        "B": 0.6505044823776146,
    },
    0.2: {
        "Tu": 0.9027472624996578,
        "T": 0.6493801869420539,
        "En": 0.7186370183497608,
        "F": 0.8242647238853092,
        "M": 0.6515293029157219,
        "B": 0.8010089647552292,
    },
    0.3: {
        "Tu": 1.1041208937494866,
        "T": 0.724070280413081,
        "En": 0.8279555275246413,
        "F": 0.9863970858279639,
        "M": 0.7272939543735828,
        "B": 0.9515134471328439,
    },
    0.4: {
        "Tu": 1.3054945249993155,
        "T": 0.7987603738841079,
        "En": 0.9372740366995216,
        "F": 1.1485294477706185,
        "M": 0.8030586058314437,
        "B": 1.1020179295104584,
    },
    0.5: {
        "Tu": 1.5068681562491442,
        "T": 0.8734504673551349,
        "En": 1.046592545874402,
        "F": 1.310661809713273,
        "M": 0.8788232572893045,
        "B": 1.252522411888073,
    },
    0.6: {
        "Tu": 1.7082417874989733,
        "T": 0.9481405608261619,
        "En": 1.1559110550492826,
        "F": 1.4727941716559279,
        "M": 0.9545879087471656,
        "B": 1.4030268942656878,
    },
    0.7: {
        "Tu": 1.909615418748802,
        "T": 1.0228306542971888,
        "En": 1.2652295642241629,
        "F": 1.6349265335985823,
        "M": 1.0303525602050265,
        "B": 1.5535313766433023,
    },
    0.8: {
        "Tu": 2.110989049998631,
        "T": 1.0975207477682158,
        "En": 1.3745480733990432,
        "F": 1.797058895541237,
        "M": 1.1061172116628875,
        "B": 1.7040358590209168,
    },
    0.9: {
        "Tu": 2.3123626812484597,
        "T": 1.172210841239243,
        "En": 1.4838665825739237,
        "F": 1.9591912574838914,
        "M": 1.1818818631207482,
        "B": 1.8545403413985315,
    },
    1.0: {
        "Tu": 2.5137363124982883,
        "T": 1.2469009347102697,
        "En": 1.593185091748804,
        "F": 2.121323619426546,
        "M": 1.257646514578609,
        "B": 2.005044823776146,
    },
}

# after reassigning ki67 cluster threshold:
KI67_THRESHOLD_MAP = {
    0.0: {"B": 0.5, "En": 0.5, "F": 0.5, "M": 0.5, "T": 0.5, "Tu": 0.5},
    0.1: {
        "B": 0.6649737918907466,
        "En": 0.609531643014187,
        "F": 0.6658766174806148,
        "M": 0.5978567775064421,
        "T": 0.6302745912564897,
        "Tu": 0.7015586127487251,
    },
    0.2: {
        "B": 0.829947583781493,
        "En": 0.719063286028374,
        "F": 0.8317532349612295,
        "M": 0.6957135550128841,
        "T": 0.7605491825129793,
        "Tu": 0.9031172254974502,
    },
    0.3: {
        "B": 0.9949213756722396,
        "En": 0.8285949290425609,
        "F": 0.9976298524418443,
        "M": 0.7935703325193262,
        "T": 0.8908237737694689,
        "Tu": 1.1046758382461754,
    },
    0.4: {
        "B": 1.159895167562986,
        "En": 0.938126572056748,
        "F": 1.163506469922459,
        "M": 0.8914271100257682,
        "T": 1.0210983650259586,
        "Tu": 1.3062344509949004,
    },
    0.5: {
        "B": 1.3248689594537324,
        "En": 1.047658215070935,
        "F": 1.3293830874030736,
        "M": 0.9892838875322103,
        "T": 1.1513729562824482,
        "Tu": 1.5077930637436257,
    },
    0.6: {
        "B": 1.4898427513444792,
        "En": 1.1571898580851219,
        "F": 1.4952597048836886,
        "M": 1.0871406650386524,
        "T": 1.2816475475389377,
        "Tu": 1.7093516764923509,
    },
    0.7: {
        "B": 1.6548165432352255,
        "En": 1.266721501099309,
        "F": 1.6611363223643034,
        "M": 1.1849974425450944,
        "T": 1.4119221387954273,
        "Tu": 1.910910289241076,
    },
    0.8: {
        "B": 1.819790335125972,
        "En": 1.376253144113496,
        "F": 1.8270129398449182,
        "M": 1.2828542200515365,
        "T": 1.542196730051917,
        "Tu": 2.112468901989801,
    },
    0.9: {
        "B": 1.9847641270167184,
        "En": 1.4857847871276828,
        "F": 1.9928895573255327,
        "M": 1.3807109975579785,
        "T": 1.6724713213084066,
        "Tu": 2.3140275147385263,
    },
    1.0: {
        "B": 2.149737918907465,
        "En": 1.5953164301418699,
        "F": 2.1587661748061473,
        "M": 1.4785677750644206,
        "T": 1.8027459125648961,
        "Tu": 2.5155861274872513,
    },
}

# use T-cell threshold for CD4 and CD8 T-cells
for threshold in KI67_THRESHOLD_MAP.keys():
    KI67_THRESHOLD_MAP[threshold][CD4_T_CELL] = KI67_THRESHOLD_MAP[threshold][T_CELL]
    KI67_THRESHOLD_MAP[threshold][CD8_T_CELL] = KI67_THRESHOLD_MAP[threshold][T_CELL]


NBRHOOD_SIZES = list(np.arange(50, 160, 10))


# maps CellPhenotype column as defined in the SingleCells.csv table to cell_type values used in the analysis:
CELL_TYPE_DEFINITIONS = frozendict(
    {
        # fibroblasts:
        "Fibroblasts": FIBROBLAST,
        "Fibroblasts FSP1^{+}": FIBROBLAST,
        "Myofibroblasts PDPN^{+}": FIBROBLAST,
        "Myofibroblasts": FIBROBLAST,
        # macrophages:
        "Macrophages": MACROPHAGE,
        "Macrophages & granulocytes": MACROPHAGE,
        # 'MHC I & II^{hi}': MACROPHAGE, # Leeat said these might be macs
        # tumor:
        "CK^{med}ER^{lo}": TUMOR,
        "CK^{+} CXCL12^{+}": TUMOR,
        "CK8-18^{hi}ER^{lo}": TUMOR,
        "CK8-18^{+} ER^{hi}": TUMOR,
        "CK^{lo}ER^{lo}": TUMOR,
        "CK8-18^{hi}CXCL12^{hi}": TUMOR,
        "CK^{lo}ER^{med}": TUMOR,
        # Endothelial:
        "Endothelial": ENDOTHELIAL,
        # cd4 T-cells:
        "CD4^{+} T cells & APCs": T_CELL,  # CD4_T_CELL,
        "CD4^{+} T cells": T_CELL,  # CD4_T_CELL,
        # cd8 T-cells:
        "CD8^{+} T cells": T_CELL,  # CD8_T_CELL,
        # B-cells:
        "B cells": B_CELL,
    }
)

# Fibroblast subtypes:
CELL_TYPE_DEFINITIONS_ONLY_MYOFIBROBLASTS = dict(CELL_TYPE_DEFINITIONS.copy())
CELL_TYPE_DEFINITIONS_ONLY_MYOFIBROBLASTS.pop("Fibroblasts")
CELL_TYPE_DEFINITIONS_ONLY_MYOFIBROBLASTS.pop("Fibroblasts FSP1^{+}")
CELL_TYPE_DEFINITIONS_ONLY_MYOFIBROBLASTS = frozendict(CELL_TYPE_DEFINITIONS_ONLY_MYOFIBROBLASTS)

CELL_TYPE_DEFINITIONS_NON_MYOFIBROBLASTS = dict(CELL_TYPE_DEFINITIONS.copy())
CELL_TYPE_DEFINITIONS_NON_MYOFIBROBLASTS.pop("Myofibroblasts")
CELL_TYPE_DEFINITIONS_NON_MYOFIBROBLASTS.pop("Myofibroblasts PDPN^{+}")
CELL_TYPE_DEFINITIONS_NON_MYOFIBROBLASTS = frozendict(CELL_TYPE_DEFINITIONS_NON_MYOFIBROBLASTS)

# T-cell subtypes:
CELL_TYPE_DEFINITIONS_CD4_T = dict(CELL_TYPE_DEFINITIONS.copy())
CELL_TYPE_DEFINITIONS_CD4_T.pop("CD8^{+} T cells")
CELL_TYPE_DEFINITIONS_CD4_T = frozendict(CELL_TYPE_DEFINITIONS_CD4_T)

CELL_TYPE_DEFINITIONS_CD8_T = dict(CELL_TYPE_DEFINITIONS.copy())
CELL_TYPE_DEFINITIONS_CD8_T.pop("CD4^{+} T cells & APCs")
CELL_TYPE_DEFINITIONS_CD8_T.pop("CD4^{+} T cells")
CELL_TYPE_DEFINITIONS_CD8_T = frozendict(CELL_TYPE_DEFINITIONS_CD8_T)

CELL_TYPE_DEFINITIONS_CD4_CD8_B = dict(CELL_TYPE_DEFINITIONS.copy())
CELL_TYPE_DEFINITIONS_CD4_CD8_B["CD8^{+} T cells"] = CD8_T_CELL
CELL_TYPE_DEFINITIONS_CD4_CD8_B["CD4^{+} T cells & APCs"] = CD4_T_CELL
CELL_TYPE_DEFINITIONS_CD4_CD8_B["CD4^{+} T cells"] = CD4_T_CELL
CELL_TYPE_DEFINITIONS_CD4_CD8_B = frozendict(CELL_TYPE_DEFINITIONS_CD4_CD8_B)


"""
Downloading the MIBI data:
"""

BREAST_MIBI_DATA_DIR = DATA_DIR / "breast_mibi"

BREAST_MIBI_DOWNLOAD_URL = "https://zenodo.org/record/7324285/files/MBTMEStrPublicProcessedDataCode.zip"
BREAST_MIBI_ZIP_PATH = BREAST_MIBI_DATA_DIR / "breast_mibi.zip"
BREAST_UNZIPPED_DIR = BREAST_MIBI_DATA_DIR / "MBTMEIMCPublic"

SINGLE_CELL_TABLE_NAME = "SingleCells.csv"
SINGLE_CELL_TABLE_PATH = BREAST_MIBI_DATA_DIR / SINGLE_CELL_TABLE_NAME
SINGLE_CELL_TABLE_PATH_IN_UNZIPPED_DIR = BREAST_UNZIPPED_DIR / SINGLE_CELL_TABLE_NAME

CLINICAL_TABLE_NAME = "brca_metabric_clinical_data.tsv"
CLINICAL_TABLE_PATH = BREAST_MIBI_DATA_DIR / CLINICAL_TABLE_NAME


# test through read_single_cell_df()
def download_breast_mibi_data():  # pragma: no cover
    # clear previous attempts in case something failed:
    if os.path.exists(BREAST_MIBI_ZIP_PATH):
        os.remove(BREAST_MIBI_ZIP_PATH)

    if os.path.exists(BREAST_UNZIPPED_DIR):
        shutil.rmtree(BREAST_UNZIPPED_DIR)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    if not os.path.exists(BREAST_MIBI_DATA_DIR):
        os.mkdir(BREAST_MIBI_DATA_DIR)

    download_file(BREAST_MIBI_DOWNLOAD_URL, BREAST_MIBI_ZIP_PATH)
    unzip_file(BREAST_MIBI_ZIP_PATH, BREAST_MIBI_DATA_DIR)
    shutil.move(SINGLE_CELL_TABLE_PATH_IN_UNZIPPED_DIR, SINGLE_CELL_TABLE_PATH)

    # clear redundant files:
    if os.path.exists(BREAST_UNZIPPED_DIR):
        shutil.rmtree(BREAST_UNZIPPED_DIR)

    if os.path.exists(BREAST_MIBI_ZIP_PATH):
        os.remove(BREAST_MIBI_ZIP_PATH)


"""
Functions for reading the main data tables for the analysis:
"""

MISSING_IMAGES = [110, 518, 571]


def all_image_numbers():
    return np.setdiff1d(np.arange(1, 798), MISSING_IMAGES)


def read_clinical_df():
    """
    https://www.cbioportal.org/study/summary?id=brca_metabric
    """
    if not os.path.exists(CLINICAL_TABLE_PATH):

        msg = f"""
            Clinical data table not found at:
            {CLINICAL_TABLE_PATH}.

            Please download the clinical data table from:

            https://www.cbioportal.org/study/summary?id=brca_metabric"

            Then save it as 'brca_metabric_clinical_data.tsv' in the directory:

            {BREAST_MIBI_DATA_DIR}
            """

        print(msg)
        raise FileNotFoundError(msg)

    return pd.read_csv(CLINICAL_TABLE_PATH, sep="\t")


# Why two caches:
# 1. persistent cache saves the csv as a pickle: ~20 seconds -> ~1.5 seconds
# 2. in-memory cache reduces time of subsequent reads: ~1.5 seconds -> ~0 seconds


@persistent_cache
def _read_single_cell_df() -> pd.DataFrame:  # pragma: no cover
    """Downloads and reads the single_cell_df.

    Returns:
        pd.DataFrame: _description_
    """
    if not os.path.exists(SINGLE_CELL_TABLE_PATH):
        print(
            """
        Downloading the breast-cancer IMC dataset by Danenberg et. al., Nature Genetics 2022:

        Link to paper:
        https://www.nature.com/articles/s41588-022-01041-y

        Link to dataset:
        https://zenodo.org/record/7324285/


        Downloading this large dataset takes a few minutes (duration depends on internet connection)
        """
        )
        download_breast_mibi_data()

    df = pd.read_csv(SINGLE_CELL_TABLE_PATH)

    return df


@cache
def read_single_cell_df(
    ki67_threshold: float = 0.5,
    cell_type_definitions: frozendict = CELL_TYPE_DEFINITIONS,
    _reassign_ki67_cluster: bool = True,
) -> pd.DataFrame:

    df = _read_single_cell_df()

    if _reassign_ki67_cluster:
        df = reassign_ki67_cluster(df)

    # create columns with tdm names:
    # x,y, ki67, cell_type, img_num, subject_id
    for original_col, renamed_col in RENAME_COLUMNS.items():
        df[renamed_col] = df[original_col]

    # map cell-types:
    df[CELL_TYPE_COL] = df[CELL_TYPE_COL].map(cell_type_definitions)
    df = df[~df[CELL_TYPE_COL].isna()].copy()

    # add binary division event column:
    df.loc[:, DIVISION_COL] = is_dividing(
        single_cell_df=df, typical_noise=TYPICAL_KI67_NOISE, ki67_threshold=ki67_threshold, ki67_col=KI67_COL
    )

    # transform x,y values to standard units:
    df.loc[:, X_COL] = microns(df[X_COL])
    df.loc[:, Y_COL] = microns(df[Y_COL])

    # set standard dtypes:
    df = set_dtypes(df)

    # exclude outlier patient with disproportionate division number:
    df = df[df.subject_id != "MB-0340"]  # FM
    df = df[~np.isin(df.subject_id, ["MB-0222", "MB-0620"])]  # TB

    df = df.reset_index()

    return df


def reassign_ki67_cluster(single_cell_df: pd.DataFrame) -> pd.DataFrame:
    """Reassign each cell from the  Ki67+ cluster to one of the cell types.

    Args:
        single_cell_df (pd.DataFrame): the single-cell dataframe from Danenberg et. al 2022

    Returns:
        pd.DataFrame: the single-cell dataframe with Ki67+ cells reassigned (in-place change)
    """

    ALL_PROTEINS_EXCEPT_KI67 = [
        "Histone H3",
        "SMA",
        "CK5",
        "CD38",
        "HLA-DR",
        "CK8-18",
        "CD15",
        "FSP1",
        "CD163",
        "ICOS",
        "OX40",
        "CD68",
        "HER2 (3B5)",
        "CD3",
        "Podoplanin",
        "CD11c",
        "PD-1",
        "GITR",
        "CD16",
        "HER2 (D8F12)",
        "CD45RA",
        "B2M",
        "CD45RO",
        "FOXP3",
        "CD20",
        "ER",
        "CD8",
        "CD57",
        "PDGFRB",
        "Caveolin-1",
        "CD4",
        "CD31-vWF",
        "CXCL12",
        "HLA-ABC",
        "panCK",
    ]  # no 'Ki-67'

    return recluster(
        single_cell_df=single_cell_df,
        gene_signature_protein_columns=ALL_PROTEINS_EXCEPT_KI67,
        cell_type_column="cellPhenotype",
        cell_type_to_recluster="Ki67^{+}",
    )


"""
all_types = [
    'CK^{med}ER^{lo}',
    'ER^{hi}CXCL12^{+}',
    'CD4^{+} T cells & APCs',
    'CD4^{+} T cells',
    'Endothelial',
    'Fibroblasts',
    'Myofibroblasts PDPN^{+}',
    'CD8^{+} T cells',
    'CK8-18^{hi}CXCL12^{hi}',
    'Myofibroblasts',
    'CK^{lo}ER^{lo}',
    'Macrophages', # M
    'CK^{+} CXCL12^{+}',
    'CK8-18^{hi}ER^{lo}',
    'CK8-18^{+} ER^{hi}',
    'CD15^{+}',
    'MHC I & II^{hi}',
    'T_{Reg} & T_{Ex}',
    'CD57^{+}',
    'Ep Ki67^{+}',
    'CK^{lo}ER^{med}',
    'Macrophages & granulocytes', # M
    'CD38^{+} lymphocytes',
    'Ki67^{+}',
    'HER2^{+}',
    'B cells',
    'Basal',
    'Fibroblasts FSP1^{+}',
    'Granulocytes',
    'MHC I^{hi}CD57^{+}',
    'Ep CD57^{+}',
    'MHC^{hi}CD15^{+}'
]
"""
