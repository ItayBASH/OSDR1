import pandas as pd
from tdm.dataset.dataset import Dataset
from tdm.dataset.polynomial import PolynomialDataset
from tdm.model import Model


class ApplyModelDataset(PolynomialDataset):
    """
    Applies a model to all cells of a dataset and replaces the
    observation (currently: division) with the model's predictions.
    """

    def __init__(self, nds: Dataset, model: Model) -> None:
        """
        Parameters:
            ds (Dataset):
                a dataset of cell counts (e.g NeighborsDataset)
        """
        self.nds = nds
        self.model = model

        self.dataset_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        for cell_type in nds.cell_types():
            self.dataset_dict[cell_type] = self.counts_and_obs_from_model(cell_type)

    def counts_and_obs_from_model(self, cell_type: str):
        cell_counts, _ = self.nds.fetch(cell_type)

        # sample observations:
        obs = self.model.sample_observations(cell_type, cell_counts, obs="division")

        return cell_counts, pd.DataFrame(obs, columns=["division"])

    def __getattr__(self, name):
        """
        Passes calls to the wrapped Dataset so that ApplyModelDataset can be used as a PolynomialDataset.
        """
        return getattr(self.nds, name)
