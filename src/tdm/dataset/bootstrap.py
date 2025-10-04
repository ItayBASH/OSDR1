from typing import Callable, Sequence
import pandas as pd
import numpy as np
from tdm.dataset.dataset import Dataset
from tdm.dataset.neighbors import NeighborsDataset
from tdm.dataset.concat import ConcatDataset


class BootstrapDataset(Dataset):
    def __init__(
        self,
        ds: Dataset,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        A wrapper for ds that resamples a dataset at the cellular level.
        """
        self.ds = ds
        self.n_samples = n_samples
        np.random.seed(seed)

        self.dataset_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        # n cells for each type:
        d = self.n_samples_per_type()
        for cell_type in ds.cell_types():
            self.dataset_dict[cell_type] = self.sample_features_and_obs(cell_type, d[cell_type])

    def n_samples_per_type(self) -> dict[str, int]:
        """
        Returns a dictionary that maps cell_type to n_samples.

        The number of samples according to a multinomial distribution with n_samples trials and
        probabilities equal to the fraction of cells of each type in the dataset.
        """
        counts = np.array([self.ds.n_cells(c) for c in self.ds.cell_types()])
        probs = counts / counts.sum()
        total_samples = self.n_samples or counts.sum()
        return dict(zip(self.ds.cell_types(), np.random.multinomial(total_samples, probs)))

    def sample_features_and_obs(self, cell_type: str, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        features, obs = self.ds.fetch(cell_type)
        bootstrap_idxs = np.random.choice(a=np.array(features.shape[0], dtype=int), size=n_samples, replace=True)
        return features.iloc[bootstrap_idxs].reset_index(drop=True), obs.iloc[bootstrap_idxs].reset_index(drop=True)

    def construct_features_from_counts(
        self, cell_counts: dict[str, float | Sequence[float]], target_cell: str, **kwargs
    ) -> pd.DataFrame:
        """
        Overrides base class function and passes to self.ds
        """
        return self.ds.construct_features_from_counts(cell_counts, target_cell, **kwargs)

    def __getattr__(self, name):
        """
        Passes calls to the wrapped Dataset.
        """
        return getattr(self.ds, name)


class PatientLevelBootstrap:
    def __init__(self, neighbors_ds_to_features_ds: Callable) -> None:
        self.dataset_func = neighbors_ds_to_features_ds

    def __call__(
        self,
        ndss: list[NeighborsDataset],
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """
        Resamples a list of neighbors datasets.
        """
        n_samples = n_samples or len(ndss)

        # resample datasets:
        np.random.seed(seed)
        random_idxs = np.random.choice(np.arange(len(ndss), dtype=int), n_samples, replace=True)
        resampled_ndss = [ndss[i] for i in random_idxs]

        # initialize the dataset:
        return self.dataset_func(ConcatDataset(resampled_ndss))
