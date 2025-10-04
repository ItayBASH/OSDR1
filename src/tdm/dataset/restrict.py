import pandas as pd
import numpy as np
from tdm.dataset import Dataset, NeighborsDataset


class RestrictedNeighborsDataset(NeighborsDataset):
    """
    Filters out cells according to lists of allowed cell types and cell types
    that must be neighboring each cell.
    """

    def __init__(
        self,
        nds: Dataset,
        allowed_neighbor_types: list[str] | None = None,
        required_neighbor_types: list[str] | None = None,
        keep_types: list[str] | None = None,
    ) -> None:
        """
        Parameters:

            nds (NeighborsDataset):
                instance of NeighborsDataset

            allowed_neighbor_types (list[str], optional):
                A list of cell types. Cells that are or have neighbors outside this list are excluded
                from the restricted dataset. Default behavior: allow all types.

            required_neighbor_types (list[str], optional):
                A list of cell types. Cells that do not have at least one neighbors of each type
                in this list are excluded from the restricted dataset. Note that a cell is by definition
                a neighbor of itself. Default behavior: don't require any type.

            keep_types (list[str], optional):
                A list of cell types. Drops all columns for cell types outside this
                list. If not provided, uses allowed_types because all columns outside
                allowed_neighbor_types will have only zeros.
        """
        self.nds = nds
        self.allowed_types = allowed_neighbor_types
        self.must_types = required_neighbor_types
        self.keep_types = keep_types or allowed_neighbor_types

        self.dataset_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        if self.keep_types is not None:
            for cell_type in self.keep_types:
                features, obs = nds.fetch(cell_type)
                include_mask = self.get_include_mask(features)

                self.dataset_dict[cell_type] = features.loc[include_mask, self.keep_types].reset_index(
                    drop=True
                ), obs.loc[include_mask].reset_index(drop=True)
        else:
            raise UserWarning("keep_types was None, no cell types included!")

    def get_include_mask(self, features) -> np.ndarray:
        """
        Returns boolean a mask that includes cells that have:
            1. zero neighbors of unallowed types.
            2. at least one neighbor of each must type.
        """
        # start by including all cells:
        mask = np.repeat(True, repeats=features.shape[0])

        # include only cells that have 0 neighbors of unallowed type:
        if self.allowed_types is not None:
            all_types = self.nds.cell_types()
            unallowed_types = [t for t in all_types if t not in self.allowed_types]
            for t in unallowed_types:
                mask = mask & (features[t] == 0)

        # include only cells that have at least 1 neighbor of each must-type:
        if self.must_types is not None:
            for t in self.must_types:
                mask = mask & (features[t] > 0)

        return mask
