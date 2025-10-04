"""
Base class for all models.

A Model fits two models per cell_type, one for division and one for death.
"""

from typing import Literal, Callable, Any, Mapping
import pandas as pd
import numpy as np

from tdm.utils import dict_to_dataframe
from tdm.dataset.dataset import Dataset
from tdm.model.constant import ConstantProbabilityModel
from abc import ABC, abstractmethod

STABLE = "stable"
UNSTABLE = "unstable"


class Model(ABC):
    """
    Base class for all models.

    A Model fits two models per cell_type, one for division and one for death.
    """

    def __init__(
        self,
        dataset: Dataset,
        fit_cell_types: list[str] | None = None,
        death_estimation: Literal["mean"] = "mean",
        truncate_division_rate: bool = True,
        **kwargs,
    ) -> None:
        """Base class for models.

        Args:
            dataset (Dataset): :class:`~tdm.dataset.Dataset` used to fit the model.
            fit_cell_types (list[str] | None, optional): fit a model to this subset. Defaults to all cell types.
            death_estimation (Literal[&quot;mean&quot;, &quot;fit&quot;], optional): method for death rate estimation. Defaults to "mean".
            truncate_division_rate (bool, optional): truncate division rates at the maximal rate for cells in the data.
                                                     Prevents extreme values in extrapolated regions. Defaults to False.
        """
        self.ds = dataset
        self._cell_types = fit_cell_types or self.ds.cell_types()

        self.death_estimation = death_estimation

        # ensure non-positive flux at max density
        self.maximal_density_enforcer = None

        # _debug mode applies only enforcer component of dynamics
        self._debug_maximal_density_enforcer = False

        # fit models:
        self.models: dict[str, dict[str, Any]] = {}
        for cell_type in self.cell_types():

            features, obs = dataset.fetch(cell_type)

            self.models[cell_type] = {}

            # division:
            self.models[cell_type]["division"] = self.fit(features, obs.division, cell_type=cell_type)

            # death:
            self.models[cell_type]["death"] = self._estimate_death(features, obs, cell_type=cell_type)

        # truncate division rate at the maximal value:
        self.truncate_division_rate = truncate_division_rate
        if self.truncate_division_rate:
            self._max_observed_fluxes: dict[str, float] = self._init_max_observed_fluxes(dataset)

    @abstractmethod
    def fit(self, features: pd.DataFrame, obs: pd.Series, cell_type: str) -> object:
        """
        Fits a single model to X=features, y=obs
        """
        raise NotImplementedError

    def cell_types(self):
        """
        The cell types the model has been fitted to.
        """
        return self._cell_types

    def _estimate_death(self, features: pd.DataFrame, obs: pd.DataFrame, cell_type: str, method: str = "mean"):
        if method == "mean":
            return ConstantProbabilityModel(p=obs.division.mean())
        else:
            raise ValueError(f"{method} is not a valid death estimation method")

    def predict(
        self,
        cell_type: str,
        obs: Literal["death", "division", "division_minus_death"],
        features: pd.DataFrame,
    ) -> np.ndarray:
        """
        Uses the model fit on cell_type data to predict death / division rates.

        predict() implements shared logic for all Model classes.
        Classes inheriting Model should implement the _predict function
        """
        if self._debug_maximal_density_enforcer:
            return np.zeros(features.shape[0])

        if obs == "death":
            return self._predict(cell_type=cell_type, obs=obs, features=features)

        elif obs == "division":
            p_div = self._predict(cell_type=cell_type, obs=obs, features=features)
            return self._truncate_if(p_div, cell_type)

        elif obs == "division_minus_death":
            p_death = self._predict(cell_type=cell_type, obs="death", features=features)
            p_div = self._predict(cell_type=cell_type, obs="division", features=features)
            p_div = self._truncate_if(division_rates=p_div, cell_type=cell_type)
            return p_div - p_death

        else:
            raise ValueError(f"Invalid argument: predict(obs={obs})")

    @abstractmethod
    def _predict(
        self,
        cell_type: str,
        obs: Literal["death", "division"],
        features: pd.DataFrame,
    ) -> np.ndarray:
        pass

    def delta_cells(
        self,
        cell_counts: Mapping[str, float | np.ndarray],
        return_order: list[str],
        mode: Literal["cells", "rates"] = "cells",
    ) -> list[np.ndarray] | np.ndarray:
        """
        Uses division and death rate to compute the absolute number of cells gained / lost.

        Note:
        This method works with ACTUAL CELL NUMBERS!
        Do not perform any transformation to cell vals. The dataset object used to fit the model
        is responsible for performing the transformation.

        Parameters:
            cell_counts (dict):
                maps a cell type to an array or single integer value of non-transformed cell counts.
                Note: cell_counts must contain a value for every cell type the model was fit to.

            return_order (list[str]):
                determines the order of cells for returning results as a tuple
        """

        # cell_counts should contain exactly the cell types used during model fit:
        # return order should be a subset of these types.
        if isinstance(cell_counts, dict):
            assert set(self.ds.cell_types()) == set(cell_counts.keys()) >= set(return_order)
        elif isinstance(cell_counts, pd.DataFrame):
            assert set(self.ds.cell_types()) == set(cell_counts.columns) >= set(return_order)

        # perform once here for performance
        cell_counts = dict_to_dataframe(cell_counts, columns=self.ds.cell_types())

        delta_cells = []
        for cell_type in return_order:
            delta_cells.append(
                self._delta_cells(
                    cell_counts=cell_counts,
                    target_cell=cell_type,
                    mode=mode,
                )
            )

        # when computing a single point return the result as a single numpy array:
        return np.squeeze(delta_cells)

    def get_delta_cells_func(
        self,
        predicted_cell_types: list[str],
        fixed_cell_counts: dict[str, float] | None = None,
    ) -> Callable:
        """Return the dynamics function f, such that dx/dt = f(x). Useful for numerical ode solvers.

        Args:
            predicted_cell_types (list[str]): return order of predictions.
            fixed_cell_counts (dict[str, float] | None, optional): provide these in the case of 2D dynamics with fixed cell counts for other cells. Defaults to None.

        Returns:
            Callable: the dynamics function
        """
        if fixed_cell_counts is None:
            fixed_cell_counts = {}

        def f(x):
            # merge dictionaries of predicted and fixed counts:
            predicted_cell_counts = {c: x[i] for i, c in enumerate(predicted_cell_types)}
            cell_counts = {**predicted_cell_counts, **fixed_cell_counts}

            # to list, ordered like the dataset used for fit
            cell_counts = np.array([cell_counts[c] for c in self.ds.cell_types()]).reshape(1, -1)

            # to dataframe - faster than dict-to-dataframe
            cell_counts = pd.DataFrame(cell_counts, index=[0], columns=self.ds.cell_types())

            return np.array(self.delta_cells(cell_counts=cell_counts, return_order=predicted_cell_types))

        return f

    def _delta_cells(
        self,
        cell_counts: pd.DataFrame,
        target_cell: str,
        mode: Literal["cells", "rates"],
    ) -> np.ndarray:
        """
        Computes the number or rate of change of cells using features based on
        the absolute numbers of cells provided in cell_counts.
        """
        features = self._construct_features_from_counts(cell_counts, target_cell=target_cell)

        division_minus_death_rates = self.predict(target_cell, "division_minus_death", features)

        # term for ensuring a non-positive flux at maximal density:
        maximal_density_correction = self._get_maximal_density_correction(cell_counts, target_cell)

        rate = division_minus_death_rates + maximal_density_correction

        if mode == "rates":
            return rate
        elif mode == "cells":
            # return number of cells lost or gained:
            n_cells = cell_counts[target_cell]
            return n_cells * rate
        else:
            ValueError(f"Invalid argument for _delta_cells( mode = {mode})")

    def _construct_features_from_counts(self, cell_counts: pd.DataFrame, target_cell: str):
        return self.ds.construct_features_from_counts(cell_counts, target_cell=target_cell)

    def sample_observations(
        self,
        cell_type: str,
        cell_counts: dict | pd.DataFrame,
        obs: Literal["death", "division", "both"],
    ) -> np.ndarray:
        """
        Sample death or division according to modeled probabilities. Useful for Monte Carlo methods.
        """
        cell_counts = dict_to_dataframe(cell_counts)

        # compute division / death probabilities per cell:
        features = self.ds.construct_features_from_counts(cell_counts=cell_counts, target_cell=cell_type)
        p_div = self.predict(cell_type, "division", features)
        p_death = self.predict(cell_type, "death", features)

        # Note: only corrects if a maximal density enforcer was initialized
        abs_maximal_density_correction = np.abs(
            self._get_maximal_density_correction(cell_counts=cell_counts, target_cell=cell_type)
        )

        # sample observations:
        n = p_div.shape[0]
        u = np.random.uniform(size=n, low=0, high=1)

        sampled_obs = np.zeros(n, dtype=np.int8)
        if obs == "division":
            sampled_obs[u < p_div] = 1
        elif obs == "death":
            sampled_obs[u > 1 - p_death] = -1
        elif obs == "both":
            sampled_obs[u < p_div] = 1
            sampled_obs[u > 1 - (p_death + abs_maximal_density_correction)] = -1

        return sampled_obs

    def parameters(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns parameter values.
        """
        raise NotImplementedError

    def parameter_stds(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns standard deviation of parameters.
        """
        raise NotImplementedError

    def parameter_pvalues(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns parameter pvalues.
        """
        raise NotImplementedError

    def parameter_names(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns the parameter names associated with the death / division model for cells of type cell_type.
        """
        raise NotImplementedError

    def parameters_df(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "name": self.parameter_names(cell_type, obs),
                "val": self.parameters(cell_type, obs),
                "std": self.parameter_stds(cell_type, obs),
                "pval": self.parameter_pvalues(cell_type, obs),
            }
        )

    def death_prob(self, cell_type: str) -> float:
        try:
            return self.models[cell_type]["death"].p
        except AttributeError:
            raise AttributeError("Try fitting the model with `set_death_rate_to_mean_division_rate = True`")

    def set_death_prob(self, cell_type: str, val: float):
        try:
            self.models[cell_type]["death"].p = val
        except AttributeError:
            raise AttributeError("Try fitting the model with `set_death_rate_to_mean_division_rate = True`")

    def set_maximal_density_enforcement(self, enforcer):
        """
        Initializes a maximal density enforcer model that corrects for positive fluxes
        at maximal density.
        """
        self.maximal_density_enforcer = enforcer

    def reset_maximal_density_enforcement(self):
        self.maximal_density_enforcer = None

    def _set_debug_maximal_density_enforcer(self, debug=True):
        self._debug_maximal_density_enforcer = debug

    def _get_maximal_density_correction(self, cell_counts: dict | pd.DataFrame, target_cell: str) -> np.ndarray:
        """
        Computes the signed (negative) rate (fraction / dt) of cells lost via the maximal density correction.

        Parameters:
            cell_counts (dict or dataframe):
                - dict: maps a cell type to an array or single integer value of non-transformed cell counts.
                - dataframe: one column per cell type

                Note: cell_counts must contain a value for every cell type the model was fit to.

            target_cell (str):
                the cell type for which the maximal density correction is computed.

        Returns:
            np.ndarray:
                the signed rate (fraction / dt) of cells lost via the maximal density correction.
        """
        cell_counts = dict_to_dataframe(cell_counts)

        if self.maximal_density_enforcer is None:
            return np.zeros(cell_counts.shape[0])

        return self.maximal_density_enforcer(cell_counts, target_cell)

    def _init_max_observed_fluxes(self, dataset: Dataset):
        d = {}
        for t in dataset.cell_types():
            features = dataset.fetch(t)[0]
            rates = self._predict(t, obs="division", features=features)  # intentionally using the "raw" _predict
            d[t] = np.max(rates)
        return d

    def _truncate_if(self, division_rates: np.ndarray, cell_type: str) -> np.ndarray:
        if self.truncate_division_rate:
            _max = self._max_observed_fluxes[cell_type]
            return np.where(division_rates > _max, _max, division_rates)
        else:
            return division_rates
