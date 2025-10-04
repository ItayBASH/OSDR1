"""
Base class for all tissues.

A tissue encapsulates all knowledge related to one real-life tissue.
"""

import pandas as pd
import numpy as np
import copy
import warnings
from typing import Literal
from tdm.preprocess.single_cell_df import X_COL, Y_COL, IMG_ID_COL, restrict_df_to_required_columns


class Tissue:
    """A tissue represents one real-life tissue section."""

    def __init__(
        self,
        single_cell_df: pd.DataFrame,
        cell_types: list[str],
        tissue_dimensions: tuple[float, float] | None = None,
    ):
        """Initialize the tissue from a subset of the single_cell_df corresponding with a single img_id (see :ref:`Preprocess`).

        Args:
            single_cell_df (pd.DataFrame): a subset of the single_cell_df corresponding with a single img_id (see: :ref:`Preprocess`)
            cell_types (list[str]): list of supported cell types.
            tissue_dimensions (tuple[float, float] | None, optional): maximal x,y limits of the tissue section (assumes coordinates start at 0,0). Defaults to the maximal x,y values in the data.
        """

        self._cell_df = restrict_df_to_required_columns(single_cell_df)
        self._cell_types = cell_types
        self._tissue_dimensions = tissue_dimensions or self._init_tissue_dimensions(self._cell_df)

        self._tissue_id = self._init_tissue_id(single_cell_df)
        self._subject_id = self._init_subject_id(single_cell_df)

    @property
    def img_id(self):
        """
        A unique identifier of the tissue.
        """
        return self._tissue_id

    @property
    def subject_id(self):
        """
        A unique identifier of the patient.
        """
        return self._subject_id

    def cell_df(self) -> pd.DataFrame:
        """
        Returns the subset of the single_cell_df containing information this tissue (see: :ref:`Preprocess`)

        Returns:
            pd.DataFrame: The DataFrame containing cell information.
        """
        return self._cell_df

    def tissue_dimensions(self) -> tuple[float, float]:
        """
        Returns:
            (x_max, y_max): maximal x,y limits of the tissue section (assumes coordinates start at 0,0).
        """
        return self._tissue_dimensions

    def cell_types(self) -> list[str]:
        """
        List of cell types supported by this tissue.

        Note:
            A tissue supporting a list of cell_types might not have an instance of each one of them.
            The reason is that we want to be clear which cell types are included in this analysis.
        """
        return self._cell_types

    def __add__(self, other: "Tissue") -> "Tissue":
        """
        Add two Tissue objects together by creating a new Tissue object with cells from both tissues.

        Parameters:
            other (Tissue): The Tissue object to be added.

        Returns:
            Tissue: A new Tissue object that is the result of the addition.
        """
        self_copy = copy.deepcopy(self)

        # pandas will take into account empty dataframe dtype inference, which is fine.
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)

            self_copy._cell_df = pd.concat([self.cell_df(), other.cell_df()])  # concat returns a copy

        return self_copy

    def present_cell_types(self) -> list[str]:
        """List of cell types with at least one cell in the tissue.

        Returns:
            list[str]: List of cell types.
        """
        return list(self.cell_df().cell_type.unique())

    def n_cells(self, cell_type: str | None = None, neighborhood_size: float | None = None) -> float:
        """Number of cells in the tissue.

        Note:
            Returns a float type only if result is scaled to a specific neighborhood size
            (i.e neighborhood_size is not None)

        Args:
            cell_type (str | None, optional): return only the number of this cell type. Defaults to None.
            neighborhood_size (float | None, optional): scale the number of cells to match a neighborhood with the same density. Defaults to None.

        Returns:
            float: number of cells
        """
        if cell_type is None:
            n = self.cell_df().shape[0]
        else:
            n = (self.cell_df().cell_type == cell_type).sum()

        if neighborhood_size is None:
            return n
        else:
            w, h = self.tissue_dimensions()
            c = correction_factor(
                tissue_width=w, tissue_height=h, nbrhood_size=neighborhood_size, mode="to_neighborhood"
            )
            return n * c

    def n_cells_df(self, neighborhood_size: float | None = None):
        ns = {}
        for cell_type in self.cell_types():
            ns[cell_type] = self.n_cells(cell_type=cell_type, neighborhood_size=neighborhood_size)

        return pd.DataFrame(ns, index=[0], columns=self.cell_types())

    def n_cells_list(self, neighborhood_size: float | None = None, cell_types: list[str] | None = None):
        if cell_types is None:
            cell_types = self.cell_types()

        ns = []
        for cell_type in cell_types:
            ns.append(self.n_cells(cell_type=cell_type, neighborhood_size=neighborhood_size))

        return ns

    def _init_tissue_dimensions(self, cell_df) -> tuple[float, float]:
        """
        Initializes the dimensions of the tissue based on the maximum X and Y values in the cell dataframe.

        Args:
            cell_df (pandas.DataFrame): The dataframe containing the cell data.

        Returns:
            tuple[float, float]: A tuple containing the maximum X and Y values in the cell dataframe.
        """
        return cell_df[X_COL].max(), cell_df[Y_COL].max()

    def _init_tissue_id(self, single_cell_df: pd.DataFrame) -> int:
        """
        Initializes the tissue ID based on the given single cell DataFrame.

        Args:
            single_cell_df (pd.DataFrame): The DataFrame containing single cell data.

        Returns:
            int: The initialized tissue ID.

        Raises:
            ValueError: If the number of unique image IDs is not equal to 1.
        """
        img_ids = single_cell_df[IMG_ID_COL].unique()

        if len(img_ids) != 1:
            raise ValueError(f"Expected a single tissue id, got {img_ids}")

        return img_ids.item()

    def _init_subject_id(self, single_cell_df: pd.DataFrame) -> str:
        """
        Initializes the subject ID based on the given single cell DataFrame.

        Args:
            single_cell_df (pd.DataFrame): The DataFrame containing single cell data.

        Returns:
            str: The initialized subject ID.

        Raises:
            ValueError: If the number of unique subject IDs is not equal to 1.
        """
        subject_ids = single_cell_df["subject_id"].unique()

        if len(subject_ids) != 1:
            raise ValueError(f"Expected a single subject ID, got {subject_ids}")

        return subject_ids.item()


def tissue_area(tissue_width, tissue_height):
    return tissue_width * tissue_height


def neighborhood_area(nbrhood_size):
    return np.pi * nbrhood_size**2


def correction_factor(
    tissue_width: float,
    tissue_height: float,
    nbrhood_size: float,
    mode: Literal["to_tissue"] | Literal["to_neighborhood"] = "to_tissue",
) -> float:
    """Computes the ratio of tissue and neighborhood areas.

    This method is useful for transforming cell numbers from a tissue to neigborhood or the other way around.

    Args:
        tissue_width (float): tissue width, e.g microns(1000) (See: tdm.utils.microns)
        tissue_height (float): tissue height, e.g microns(1000) (See: tdm.utils.microns)
        nbrhood_size (float): neiborhood radius, e.g microns(150) (See: tdm.utils.microns)
        mode (Literal[&quot;to_tissue&quot;] | Literal[&quot;to_neighborhood&quot;], optional): _description_.
        Defaults to "to_tissue".

    Raises:
        ValueError: user must provide one of the supported modes.

    Returns:
        float: area ratio

    Examples:
        >>> w,h,r = tissue_width, tissue_height, neighborhood_size
        >>> n_cells_per_neighborhood = 10
        >>> n_cells_per_tissue = n_cells_per_neighborhood * correction_factor(w,h,r, mode='to_tissue')
        >>> n_cells_per_neighborhood = n_cells_per_tissue * correction_factor(w,h,r, mode='to_neighborhood')
    """
    if mode == "to_tissue":
        return tissue_area(tissue_width, tissue_height) / neighborhood_area(nbrhood_size)
    elif mode == "to_neighborhood":
        return neighborhood_area(nbrhood_size) / tissue_area(tissue_width, tissue_height)
    else:
        raise ValueError(f"Invalid argument mode = {mode}")
