"""
Construct a tissue from cells provided as tuples or as a dataframe.
"""

from tdm.preprocess.single_cell_df import SUBJECT_ID_COL, IMG_ID_COL
from tdm.tissue import Tissue
from typing import Sequence
import pandas as pd


class StubTissue(Tissue):
    """
    Construct a tissue from cells provided as tuples or as a dataframe.
    """

    def __init__(
        self,
        cell_types: list[str],
        cell_df: pd.DataFrame | None = None,
        cell_type_xy_tuples: Sequence[tuple[str, float, float, int]] | None = None,
        tissue_dimensions: tuple[float, float] | None = None,
        subject_id: int = 1,
        img_id: int = 1,
    ) -> None:
        """
        Parameters:
            cell_types (list[str]):
                list of cell types

            cell_df (pd.DataFrame, optional):
                dataframe with columns: cell_type, x, y, division

            cell_type_xy_tuples (list[tuple[str, float,float, Literal[0] | Literal[1] ]], optional):
                a list of tuples that contain:
                    - cell_type (str)
                    - x (float)
                    - y (float)
                    - division (0 or 1)

            tissue_dimensions (tuple[float,flaot], optional):
                x,y dimensions of the tissue. If not provided, inferred from data.

        """
        # set cell_df:
        if cell_df is None:
            cell_df = pd.DataFrame(cell_type_xy_tuples, columns=["cell_type", "x", "y", "division"])
            cell_df[SUBJECT_ID_COL] = subject_id
            cell_df[IMG_ID_COL] = img_id

        # use provided dimensions or infer from data:
        tissue_dimensions = tissue_dimensions or (
            cell_df.x.max(),
            cell_df.y.max(),
        )

        super().__init__(single_cell_df=cell_df, tissue_dimensions=tissue_dimensions, cell_types=cell_types)

    def set_cell_df(self, new_cell_df: pd.DataFrame) -> None:
        self._cell_df = new_cell_df
