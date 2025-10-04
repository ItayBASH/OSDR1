"""
Base class for all datasets.

A Dataset maps cell types to features & observations.
"""

from typing import Sequence, Literal
import pandas as pd
import numpy as np


class Dataset:
    """
    Base class for all datasets.

    A dataset maps cell types to features & labels.

    Note:
        A dataset is typically constructed based on one of the following sources:
            - Tissue:
                Used for direct computations on tissue cells, such as counting neighbors (see: NeighborsDataset)
            - Dataset:
                Typically used for transforming features (see: PolynomialDataset)
            - A list of Datasets:
                Used for combining datasets (see: ConcatDataset)
    """

    def __init__(self) -> None:
        """
        Initializes the Dataset with a dictionary mapping cell type to features and obs.

        dataset_dict:
            - key:
                - cell_type (str)
            - value:
                - features: dataframe with shape (n_cells, n_features)
                - observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        self.dataset_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = self._init_dataset_dict()

    def _init_dataset_dict(self) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Returns a dataset dictionary mapping cell type to features and obs:
            - key:
                cell_type (str)
            - value:
                - features: dataframe with shape (n_cells, n_features)
                - observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        raise NotImplementedError

    def fetch(self, cell_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the features and observations associated with a cell type.

        Parameters:
            cell_type: a str from tdm.tissue.cell_types.CELL_TYPES_ARRAY

        Returns:
            features, observations (tuple):

                - features: dataframe with shape (n_cells, n_features)
                - observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        return self.dataset_dict[cell_type]

    def set_dataset(self, cell_type: str, features: pd.DataFrame, obs: pd.DataFrame):
        """Manually write the features and obs for a cell type.

        Args:
            cell_type (str): string identifier of a cell type.
            features (pd.DataFrame): dataframe with shape (n_cells, n_features)
            obs (pd.DataFrame): dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        self.dataset_dict[cell_type] = features, obs

    def cell_types(self) -> list[str]:
        """
        Returns the cell types in the dataset.

        See: tdm.tissue.cell_types.CELL_TYPES_ARRAY for possible values.
        """
        return list(self.dataset_dict.keys())

    def fetch_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns features and observations from all cell types, concatenated.

        Returns:
            features, observations (tuple):

                - features: dataframe with shape (n_cells, n_features)
                - observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        features_and_obs = [self.fetch(c) for c in self.cell_types()]

        features = pd.concat([f for f, o in features_and_obs])
        obs = pd.concat([o for f, o in features_and_obs])

        return features, obs

    def n_cells(self, cell_type: str | None = None) -> int:
        """
        Returns the number of cells of cell_type in the dataset, or all cell types if cell_type = None
        """
        if cell_type is None:
            return sum([self.dataset_dict[c][0].shape[0] for c in self.dataset_dict.keys()])
        else:
            return self.dataset_dict[cell_type][0].shape[0]

    def n_obs(self, cell_type: str, obs: Literal["division", "death"]) -> int:
        """
        Returns the number of division or death events
        """
        return self.fetch(cell_type)[1][obs].sum()

    def construct_features_from_counts(
        self, cell_counts: dict[str, float | Sequence[float]], target_cell: str, **kwargs
    ) -> pd.DataFrame:
        """
        Constructs features compatible with construct_polynomial_features
        Input is in raw values!
        """
        raise NotImplementedError

    def n_features(self) -> int:
        """
        Returns the number of features in the dataset.

        warning:
            Fails if there are different numbers of features for different cell types
        """
        n_features = [self.fetch(c)[0].shape[1] for c in self.cell_types()]
        assert len(np.unique(n_features) == 1)
        return n_features[0]
