from tdm.raw.breast_mibi import read_single_cell_df as read_breast_mibi
from tdm.raw.breast_imc import read_single_cell_df as read_breast_imc
from tdm.raw.triple_negative_imc import read_single_cell_df as read_triple_negative_imc


BREAST_MIBI = "breast_mibi"
BREAST_IMC = "breast_imc"
TRIPLE_NEGATIVE_IMC = "triple_negative_imc"


def read_single_cell_df(dataset: str):
    """Read a single_cell_df by dataset name.

    Args:
        dataset (str): Name of dataset to read. Must be one of: "breast_mibi", "breast_imc", "triple_negative_imc"

    Returns:
        _type_: _description_
    """
    assert dataset in [
        BREAST_MIBI,
        BREAST_IMC,
        TRIPLE_NEGATIVE_IMC,
    ], f"dataset must be one of: {[BREAST_MIBI, BREAST_IMC, TRIPLE_NEGATIVE_IMC]}"
    if dataset == BREAST_MIBI:
        return read_breast_mibi()
    elif dataset == BREAST_IMC:
        return read_breast_imc()
    elif dataset == TRIPLE_NEGATIVE_IMC:
        return read_triple_negative_imc()
