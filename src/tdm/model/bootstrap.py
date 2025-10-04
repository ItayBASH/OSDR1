from typing import Literal, Type
from tdm.model.model import Model
from tdm.dataset.dataset import Dataset
from tdm.dataset.bootstrap import BootstrapDataset
import tqdm

import pandas as pd


class BootstrapModel:
    """
    Fits models to resampled subsets of cells from the provided dataset.
    """

    def __init__(
        self,
        model: Type[Model],
        ds: Dataset,
        n_bootstrap_tries: int,
        bs_initial_seed: int,
        disable_tqdm: bool = True,
        **model_kwargs,
    ) -> None:
        """
        Parameters:
            model (Model):
                class used in each fit, e.g LogisticRegressionModel

            ds (Dataset):
                dataset to resample.

            n_bootstrap_tries (int):
                number of times a dataset is resampled and model is fit.

            bs_initial_seed (int):
                in try number k BootstrapModel receives the seed bs_initial_seed + k

            disable_tqdm (bool):
                show progress bar during model fits.

        Returns:
            None
        """
        self.bs_models = []

        for i in tqdm.tqdm(range(n_bootstrap_tries), disable=disable_tqdm):
            bsds = BootstrapDataset(ds, seed=bs_initial_seed + i)
            bs_model = model(bsds, **model_kwargs)
            self.bs_models.append(bs_model)

    def predict(
        self,
        cell_type: str,
        obs: Literal["death"] | Literal["division"] | Literal["division_minus_death"],
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predicts division of cell_type using each of the bootstrapped models.

        Parameters:
            cell_type (str):
                name of cell whose division rate is predicted.

            obs (str): the observation type predicted.
                - 'division': computes the probability for division.
                - 'death': computes the probability for death.
                - 'division_minus_death': computes the probability for division minus death

            features (pd.DataFrame):
                features dataframe, as produced by the fitted dataset, e.g from ds.fetch()

        Returns:
            a dataframe with column 'p' for the probability for each cell, and column 'idx' with the index associated
            with the resampled model evaluated.
        """
        preds = []

        for idx, m in enumerate(self.bs_models):
            p = m.predict(cell_type, obs, features)
            preds.append(pd.DataFrame({"p": p, "idx": idx}))

        return pd.concat(preds)
