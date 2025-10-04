# import numpy as np
# from tdm.cell_types import CELL_TYPES_ARRAY
# from tdm.tissue.breast_mibi import BreastMIBI
import pytest


@pytest.mark.skip("Needs a re-implementation post refactor.")
def test_tissue():
    """
    What does this test?
    """
    raise NotImplementedError()


# TISSUE_LIST = [BreastMIBI]


# def test_tissues():
#     for t in TISSUE_LIST:
#         tissue = t()
#         cell_df = tissue.cell_df()

#         # must have core columns:
#         assert "x" in cell_df.columns
#         assert "y" in cell_df.columns
#         assert "cell_type" in cell_df.columns

#         # shouldn't have any redundant columns:
#         assert set(cell_df.columns).issubset(["x", "y", "cell_type", "division", "death"])

#         # values should make sense:

#         # x
#         assert np.all(~np.isnan(cell_df.x))
#         assert np.all(cell_df.x >= 0)
#         assert np.all(cell_df.x <= 0.02)  # less than 2cm

#         # y
#         assert np.all(~np.isnan(cell_df.y))
#         assert np.all(cell_df.y >= 0)
#         assert np.all(cell_df.y <= 0.02)  # less than 2cm

#         # cell_type
#         assert np.all(np.isin(cell_df.cell_type, CELL_TYPES_ARRAY))
