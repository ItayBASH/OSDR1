from typing import Sequence
from tdm.utils import log2_1p, inv_log2_1p
from tdm.dataset import NeighborsDataset
from tdm.model.model import Model
import numpy as np
import pandas as pd

from tdm.dataset.utils import max_density_per_cell_type


class MaximalDensityEnforcer:
    """
    Base class for maximal density enforcers, defines the api implemented by subclasses.
    A density enforcer corrects for infeasible positive fluxes at maximal density.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, cell_counts: pd.DataFrame, cell_type: str) -> np.ndarray:
        """
        Computes the correction flux for each cell of type cell_type,
        based on the density provided in cell_counts.

        Parameters:
            cell_counts (pd.DataFrame):
                A dataframe with one column per cell_type, values are raw cell counts.

            cell_type (str):
                A cell type identifier (e.g 'F' for fibroblasts)

        Returns:
            np.ndarray:
                An array of correction fluxes, one for each cell of type cell_type.
        """
        return np.zeros(cell_counts.shape[0])


class CellTypeSpecificDensityEnforcer(MaximalDensityEnforcer):
    """
    Density correction for a 2 cell system, possibly with other cells at fixed values.

    For each cell type, the corrected flux function is of the form:
        c * (density ** pow)

    Where the constant c is specific to each cell_type and is constructed such that
    we introduce the minimal correction needed to ensure a non-positive flux at max density.
    That is:
        c = (maximal positive flux at max density) / (max density ** pow)
    """

    def __init__(
        self,
        model: Model,
        nds: NeighborsDataset,
        cell_types: list[str],
        power: int = 2,
    ) -> None:
        """
        Initializes a maximal density enforcer model that corrects for positive fluxes.

        Parameters:
            model (Model):
                The model that is being corrected.

            max_density_per_cell_type (dict[str, float]):
                Maximal number of cell per neighborhood for each cell type.

            varied_cell_types (list[str]):
                The 2 cell types we are correcting.

            fixed_cell_counts (dict[str, float], optional):
                Constant values of other cells.

            power (int, optional):
                The power to which the density is raised, controls the steepness of the correction.
                Large power results in minimal influence on most of the phase-portrait with a large
                effect at the edges.
        """
        # calculations assume there is no current max-density enforcement:
        model.reset_maximal_density_enforcement()
        if not model.truncate_division_rate:
            raise UserWarning(
                "Fitting a maximal density enforcer without truncating division rate can produce excessive corrections."
            )

        # save init args:
        self.cell_types = cell_types
        self.power = power

        # use 95th quantile is more robust than max
        self.max_density_per_cell_type: dict[str, float] = max_density_per_cell_type(nds, quantile=0.95)

        # compute constants:
        self.c_per_type = self._compute_c_per_type(model)

    def __call__(self, cell_counts: pd.DataFrame, cell_type: str) -> np.ndarray:
        density = cell_counts[cell_type].values
        c = self.c_per_type[cell_type]
        correction = c * np.power(density, self.power)

        # truncate correction at -0.5
        return np.where(correction < -0.5, -0.5, correction)

    def _compute_c_per_type(self, model: Model) -> dict[str, float]:
        """
        Computes the constant c for each cell type.
        """
        return {t: self._compute_c(model, t) for t in self.cell_types}

    def _compute_c(self, model: Model, cell_type: str):
        max_density_a = self.max_density_per_cell_type[cell_type]
        max_flux_at_max_density = self._max_flux_at_max_density(model, cell_type)
        return -1 * max_flux_at_max_density / (max_density_a**self.power)

    def _max_flux_at_max_density(self, model: Model, cell_type: str):
        fluxes_at_max_density = self._fluxes_at_max_density(model, cell_type)
        return max(0, fluxes_at_max_density.max())

    def _fluxes_at_max_density(self, model: Model, cell_type: str, n_samples: int = 10000):
        """
        Computes the rate of division minus death at region of max density of cell_type.

        Note:
            Values of other cell types are sampled at random between minimal and maximal values, on a logscale.
            This is done for sample efficiency. Sampling even 10 samples per dimension explodes very quickly.
        """

        cell_counts: dict[str, float | Sequence[float]] = {}

        # fix density of target cell to max
        cell_counts[cell_type] = self.max_density_per_cell_type[cell_type]

        # sample n_samples of random points for other cell types:
        for c in self.cell_types:
            max_c = log2_1p(self.max_density_per_cell_type[c])
            cell_counts[c] = inv_log2_1p(np.random.uniform(low=0, high=max_c, size=n_samples))

        # vals -> features:
        features = model._construct_features_from_counts(cell_counts=cell_counts, target_cell=cell_type)

        # compute maximal overflow to correct:
        fluxes_at_max_density = model.predict(cell_type, "division_minus_death", features=features)

        return fluxes_at_max_density
