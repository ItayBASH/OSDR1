"""
Generic ConcatDataset class.
"""

from typing import Sequence

from pandas.core.api import DataFrame as DataFrame
from tdm.dataset import Dataset
import numpy as np
import pandas as pd


class ConcatDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        """
        Initializes a dictionary that maps cell_type (str) to an (features, observations) tuple:
            features: dataframe with shape (n_cells, n_features)
            observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death

        Parameters:
            datasets: a Sequence of Dataset instances.
        """
        self.dataset_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.datasets = datasets

        super().__init__()

    def _init_dataset_dict(self) -> dict[str, tuple[DataFrame, DataFrame]]:
        return {c: self.construct_dataset(c) for c in self._cell_types_in_list_of_datasets(self.datasets)}

    def construct_dataset(self, cell_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concatenates features and observations corresponding with cell_type from all datasets.
        """
        features = []
        obs = []
        for ds in self.datasets:
            f, o = ds.fetch(cell_type)
            features.append(f)
            obs.append(o)

        return pd.concat(features, ignore_index=True), pd.concat(obs, ignore_index=True)

    def _cell_types_in_list_of_datasets(self, datasets: Sequence[Dataset]) -> list[str]:
        # all datasets should have the same cell types:
        types = np.array([ds.cell_types() for ds in datasets])
        assert np.all(types[:, :] == types[0, :])

        return list(types[0, :])
