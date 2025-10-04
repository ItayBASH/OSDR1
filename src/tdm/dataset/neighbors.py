"""
NeighborsDataset implements the Dataset API and constructs a matrix X whose features depend
only on the number of neighbors of each cell type.
"""

from pandas.core.api import DataFrame as DataFrame
from tdm.dataset.dataset import Dataset
from tdm.tissue import Tissue

import math
import pandas as pd
import numpy as np
from numba import jit

from tdm.utils import microns

_default_neighborhood_size = microns(80)


class NeighborsDataset(Dataset):
    """Counts the number of neighbors of each type for each cell in the tissue."""

    def __init__(
        self,
        tissue: Tissue,
        neighborhood_size: float = _default_neighborhood_size,
        exclude_cells_near_edge: bool = True,
    ) -> None:
        """Counts the number of neighbors of each type for each cell in the tissue.

        Args:
            tissue (Tissue): instance of Tissue.
            neighborhood_size (float, optional): neighborhood radius in microns. Default: 80 microns.
            exclude_cells_near_edge (bool, optional): remove cells whose neighborhood exceeds tissue limits. Defaults to True.

        Warning:
            :class:`~tdm.dataset.NeighborsDataset` excludes cells whose neighborhood exceeds tissue limits!

        Note:
            If :class:`~tdm.tissue.Tissue` is a small sections of a large tissue, then cells near the edges have neighbors we do not see.
            :class:`~tdm.dataset.NeighborsDataset` excludes these cells. :class:`~tdm.dataset.ExtrapolateNeighborsDataset` includes these cells but corrects for the bias due to the partially observed neighborhood.
            The latter increases sample size.

            It makes sense to keep cells near the edge without correction if the edge of the image is indeed the edge of the tissue.
            This could be the case for complete tissue sections.
        """
        self.tissue = tissue
        self.neighborhood_size = neighborhood_size
        self.exclude_cells_near_edge = exclude_cells_near_edge
        super().__init__()

    def _init_dataset_dict(self) -> dict[str, tuple[DataFrame, DataFrame]]:
        return {c: self.construct_dataset(self.tissue, c) for c in self.tissue.cell_types()}

    def construct_dataset(self, tissue: Tissue, cell_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parameters:
            tissue: instance of type Tissue
            cell_type: str from the list CELL_TYPES_ARRAY, as defined in tdm.tissue.cell_types

        Returns:
            features,observations (tuple):
                features: dataframe with shape (n_cells, n_features)
                observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        features = self.construct_features(tissue, cell_type)
        obs = self.construct_observations(tissue, cell_type)

        if self.exclude_cells_near_edge:
            is_valid = self.valid_cells(tissue, cell_type)
            return features[is_valid], obs[is_valid]
        else:
            return features, obs

    def construct_features(self, tissue: Tissue, cell_type: str) -> pd.DataFrame:
        """
        Computes n_neighbors of each type for each cell in the dataset.
        """
        first_type_xys = self.get_xys(cell_type)

        features = {}
        for _type in tissue.cell_types():
            second_type_xys = self.get_xys(_type)
            features[_type] = n_neighbors(first_type_xys, second_type_xys, self.neighborhood_size)

        return pd.DataFrame(features)

    def n_neighbors(
        self,
        cell_xy: np.ndarray,
        cell_type_xys: np.ndarray,
        neighborhood_size: float = _default_neighborhood_size,
    ) -> int:
        """
        Parameters:
            cell_xy: np.array of shape (2,)
            cell_type_xys: np.array of shape (n_cells, 2) containing all xys of a specific cell type
                           (e.g all macrophage xys).
            neighborhood_size: cells within this distance are considered neighors. default: 100 microns.

        Returns:
            (int) number of neighbors
        """
        all_distances = np.linalg.norm(cell_xy - cell_type_xys, ord=2, axis=1)  # L2 norm
        return np.sum(all_distances < neighborhood_size).astype(int)

    def construct_observations(self, tissue: Tissue, cell_type: str) -> pd.DataFrame:
        """
        Fetches division & death columns for this cell type.
        """
        cell_df = tissue.cell_df()
        return cell_df.loc[
            cell_df.cell_type == cell_type,
            cell_df.columns.intersection(["division", "death"]),
        ]

    def get_xys(self, cell_type: str | None) -> np.ndarray:
        """Get the xy coords for a cell type (str) or all cells (None).

        Note:
            Caches previous calls in a dictionary. Avoiding the lru_cache decorator so class
            can be pickled.

        Returns:
            numpy array of shape (n_cells,2) with x,y locations of all cells of type cell_type.
        """
        # poor-man's cache to avoid lru_cache decorator, which can't be pickled
        if not hasattr(self, "cell_type_to_xys"):
            self._cell_type_to_xys: dict[str | None, np.ndarray] = {}

        if cell_type not in self._cell_type_to_xys.keys():
            self._cell_type_to_xys[cell_type] = self._get_xys(cell_type)

        return self._cell_type_to_xys[cell_type]

    def _get_xys(self, cell_type: str | None) -> np.ndarray:
        """
        Returns:
            numpy array of shape (n_cells,2) with x,y locations of all cells of type cell_type.
        """
        cell_df = self.tissue.cell_df()

        if cell_type is None:
            return cell_df[["x", "y"]].to_numpy()
        else:
            return cell_df[["x", "y"]].to_numpy()[cell_df.cell_type == cell_type]

    def valid_cells(self, tissue: Tissue, cell_type: str):
        """
        Returns a subset of xys with cells that have a distance of at least NEIGHBORHOOD_SIZE
        to each edge of the slide.

        1. Cells with less than neighborhood_size to each edge of the slide are
           excluded from the dataset so we don't get an under-estimate of their n_neighbors.

        2. Cells near the edge are of-course taken into account as neighbors of other cells.
        """
        xys = self.get_xys(cell_type)

        x_min, y_min = 0, 0
        x_max, y_max = tissue.tissue_dimensions()

        # compute x,y masks:
        ds_x, ds_y = xys[:, 0], xys[:, 1]
        x_is_in_bounds = (ds_x <= x_max - self.neighborhood_size) & (ds_x >= x_min + self.neighborhood_size)
        y_is_in_bounds = (ds_y <= y_max - self.neighborhood_size) & (ds_y >= y_min + self.neighborhood_size)

        is_valid = x_is_in_bounds & y_is_in_bounds

        return is_valid

    def get_n_neighbors_by_xy(self, xy: tuple[float, float]):
        """
        Computes the number of neighbors of each cell type, for a hypothetical cell placed in position
        xy within the tissue.

        xy: position in microns
        tissue: Tissue object
        """
        features = {}
        for cell_type in self.cell_types():
            type_xys = self.get_xys(cell_type)
            features[cell_type] = self.n_neighbors(np.array(xy), type_xys, self.neighborhood_size)

        return features


@jit(nopython=True)
def n_neighbors(first_type_xys, second_type_xys, neighborhood_size):  # pragma: no cover
    """
    Computes of the number of neighbors of the second type, for each cell of the first type.

    Note:
        This compiled loop is faster than scipy.spatial.kdtree.
    """
    ns = []
    for x1, y1 in first_type_xys:
        n = 0
        for x2, y2 in second_type_xys:
            if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < neighborhood_size:
                n += 1
        ns.append(n)

    return ns
