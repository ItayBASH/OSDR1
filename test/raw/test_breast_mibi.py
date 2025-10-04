from tdm.raw.breast_mibi import read_single_cell_df
from tdm.preprocess.single_cell_df import check_single_cell_df
from tdm.raw.breast_mibi import all_image_numbers
import numpy as np


def test_read_single_cell_df():
    df = read_single_cell_df()
    assert check_single_cell_df(df, verbose=False)


def test_all_image_numbers():
    img_nums = all_image_numbers()
    assert isinstance(img_nums, np.ndarray)
    assert len(img_nums) == 794  # 797 - 3 missing images
