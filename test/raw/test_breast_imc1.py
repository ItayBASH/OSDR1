from tdm.raw.breast_mibi import read_single_cell_df
from tdm.preprocess.single_cell_df import check_single_cell_df


def test_breast_imc1_preprocessing():
    assert check_single_cell_df(read_single_cell_df(), verbose=False)
