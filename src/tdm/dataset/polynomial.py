"""
Implements a wrapper for NeighborsDataset which constructs polynomial features from neighbor counts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tdm.log import logger
from tdm.dataset.dataset import Dataset
from tdm.utils import log2_1p, dict_to_dataframe
from itertools import combinations_with_replacement


class PolynomialDataset(Dataset):
    """
    Performs feature transformations and constructs polynomial features from a dataset.
    """

    def __init__(
        self,
        ds: Dataset,
        degree: int = 2,
        feature_cell_types: list[str] | None = None,
        include_bias: bool = True,
        scale_features: bool = False,
        log_transform: bool = False,
    ) -> None:
        """Performs feature transformations and constructs polynomial features from a dataset.

        Args:
            ds (Dataset): dataset whose features we transform
            degree (int, optional): order of polynomial interactions to construct. Defaults to 2.
            feature_cell_types (list[str] | None, optional): selects a subset of cell types for features. Defaults to all cell types present.
            include_bias (bool, optional): add a bias term to the features. Defaults to True.
            scale_features (bool, optional): standardize features (subtract mean and divide by standard deviation). Defaults to False.
            log_transform (bool, optional): log2(1+x) transformation of features. Defaults to False.

        Examples:
            >>> # nds is a NeighborsDataset instance
            >>> pds = PolynomialDataset(nds, degree=2)

        """
        self.degree = degree
        self.include_bias = include_bias
        self.pf = MyPolynomialFeatures(degree=self.degree, include_bias=self.include_bias)

        self.scale_features = scale_features
        self.log_transform = log_transform

        self.dataset_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.scaler_dict: dict[str, StandardScaler] = {}  # saved for scaling new features

        # default use all cells in the neighbors dataset as features:
        self.feature_cell_types = feature_cell_types or ds.cell_types()
        self.feature_names = self.pf.feature_names_out(self.feature_cell_types)

        for cell_type in ds.cell_types():
            neighbor_features, observations = ds.fetch(cell_type)

            # don't create features for missing cells:
            if neighbor_features.shape[0] == 0:
                continue

            polynomial_features, scaler = self.construct_polynomial_features(
                neighbor_features,
                scale_features=self.scale_features,
                fit_scaler=True,
                target_cell=cell_type,
            )

            self.dataset_dict[cell_type] = polynomial_features, observations
            self.scaler_dict[cell_type] = scaler

    def construct_polynomial_features(
        self,
        neighbor_features: pd.DataFrame,
        scale_features: bool,
        fit_scaler: bool = False,
        target_cell: str | None = None,
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """
        Parameters:
            neighbor_features: output of fetch() from a NeighborsDataset
            scale_features: True performs standard scaling on the resulting polynomial features
            fit_scaler: True fits a new standard scaler to the resulting polynomial features
            target_cell: cell we're modelling, used to fetch the correct StandardScaler

        Returns:
            tuple:
                (a polynomial features dataframe, a standard scaler)
        """
        x = neighbor_features
        # x = x.loc[:, cell_subset]  # TODO this is the slowest line here!!!! wow

        # make sure using same feature order:
        if hasattr(self, "feature_names_in_"):
            assert np.all(self.feature_names_in_ == x.columns)

        if self.log_transform:
            x = log2_1p(x)

        # x = x.values  # to numpy
        x = np.array(x)
        x = self.pf.transform(x)

        # OLD:
        # # construct polynomial features:
        # if hasattr(self.pf, "n_output_features_"):
        #     x = x.values
        #     x = self.pf.transform(x)
        # else:
        #     # on init - fit_transform twice:
        #     # once, to get feature names in and out:
        #     self.pf.fit_transform(x)  # intentionally not overwriting x!
        #     self.feature_names_in_ = self.pf.feature_names_in_
        #     self.feature_names_out_ = self.pf.get_feature_names_out()

        #     # again, to fit to a numpy array so that future transforms don't perform a slow validation.
        #     x = x.values
        #     x = self.pf.fit_transform(x)

        # scale after creating polynomial features:
        scaler = None
        if scale_features:
            if fit_scaler:
                scaler = StandardScaler().fit(x)
            else:
                if target_cell is not None:
                    scaler = self.scaler_dict[target_cell]
                else:
                    raise ValueError(
                        "Please provide cell_type. \
                         Required to scale features according to the scale of the original dataset."
                    )

            x = scaler.transform(x)

            if self.include_bias:
                x[:, 0] = 1  # reset the centered intercept

        # re-cast to a readable dataframe:
        x = pd.DataFrame(x, columns=self.feature_names, copy=False)  # don't have to copy, already a copy

        return x, scaler

    def construct_features_from_counts(
        self, cell_counts: dict | pd.DataFrame, target_cell: str, **kwargs
    ) -> pd.DataFrame:
        """
        Constructs features compatible with construct_polynomial_features
        Cell vals is in raw counts (i.e "64" cells, not the log-value: 6)!
        """
        neighbor_features = dict_to_dataframe(cell_counts, columns=self.cell_types())

        if (neighbor_features.shape[0] > 1) and (neighbor_features.values.max() < 10):
            logger.debug("expected raw cell counts. max was = %s", neighbor_features.values.max())

        if not set(neighbor_features.columns) == set(self.feature_cell_types):
            raise ValueError(
                f"Cell types provided do not match the original dataset: \nExpected: {set(self.feature_cell_types)}\nGot: {neighbor_features.columns}"
            )

        x, _ = self.construct_polynomial_features(
            neighbor_features,
            scale_features=self.scale_features,
            fit_scaler=False,
            target_cell=target_cell,
        )

        return x

    def intercept_idx(self):
        if not self.include_bias:
            raise LookupError("Can't fetch intercept index because model has no intercept term!")
        else:
            return 0


class MyPolynomialFeatures:
    """A minimalist numpy-only polynomial feature transformer."""

    def __init__(self, degree, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit_transform(self, X: np.ndarray):
        return self._transform(X)

    def transform(self, X: np.ndarray):
        return self._transform(X)

    def feature_names_out(self, feature_names_in: list[str]):
        all_combinations = self._all_combinations(len(feature_names_in))
        feature_names = []

        for comb in all_combinations:
            # Count occurrences of each feature index in the combination
            counts = {i: comb.count(i) for i in comb}
            # Generate the name based on counts, using exponents when needed
            parts = [
                f"{feature_names_in[i]}^{count}" if count > 1 else feature_names_in[i] for i, count in counts.items()
            ]
            # Join the parts to form the final feature name
            feature_names.append(" ".join(parts))

        if not self.include_bias:
            # Exclude the first entry which corresponds to the bias term
            feature_names = feature_names[1:]

        # intercept term
        if self.include_bias:
            feature_names[0] = "1"

        return feature_names

    def _all_combinations(self, n_features_in: int):
        return [
            comb
            for d in range(self.degree + 1)  # Loop over all degrees from 0 to the given degree
            for comb in combinations_with_replacement(range(n_features_in), d)
        ]

    def _transform(self, X: np.ndarray):

        n_samples, n_features_in = X.shape
        all_combinations = self._all_combinations(n_features_in)

        # Create polynomial features by multiplying elements according to combinations
        X_poly = np.empty((n_samples, len(all_combinations)))
        for i, comb in enumerate(all_combinations):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)

        if not self.include_bias:
            X_poly = X_poly[:, 1:]

        return X_poly
