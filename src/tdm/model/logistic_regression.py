"""
Logistic Regression Model
"""

from tdm.log import logger
from tdm.model.model import Model
from tdm.dataset.dataset import Dataset
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper, L1BinaryResultsWrapper
from typing import Literal
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import warnings
from typing import Union

"""
Helper functions:
"""


class LogisticRegressionModel(Model):
    """A :class:`~tdm.model.Model` that models probabilities using logistic regression."""

    def __init__(
        self,
        dataset: Dataset,
        regularization_alpha: float | Literal["score"] = 0,
        score_kwargs: dict | None = None,
        cv_kwargs: dict | None = None,
        maxiter: int = 100,
        **kwargs,
    ) -> None:
        """Initialize and fit the model

        Args:
            dataset (Dataset): _description_
            regularization_alpha (float | Literal[&quot;score&quot;, &quot;significance&quot;], optional): _description_. Defaults to 0.
            score_kwargs (dict | None, optional): passed to self._fit_score_optimized_model(...)
            maxiter (int, optional): _description_. Defaults to 100.

        Note:
            See :class:`~tdm.model.Model` base class for all available model kwargs.
        """
        self.regularization_alpha = regularization_alpha
        self.score_kwargs = score_kwargs or {}  # passed to self._fit_score_optimized_model(...)
        self.cv_kwargs = cv_kwargs or {}  # passed to self._fit_score_optimized_model(...)
        self.best_alphas: dict[str, float] = {}  # save best alphas from parameter search
        self.maxiter = maxiter
        super().__init__(dataset, **kwargs)

    def fit(
        self,
        features: pd.DataFrame,
        obs: pd.Series,
        cell_type: str,
        regularization_alpha: float | Literal["score", "cv"] | dict[str, float] = 0,
        score_kwargs: dict | None = None,
        cv_kwargs: dict | None = None,
        **kwargs,
    ) -> Union[BinaryResultsWrapper, L1BinaryResultsWrapper]:
        """
        Fits a single model to X=features, y=obs

        Note: By default, Logit doesn't fit an intercept term.
        """

        if not np.all(features.iloc[:, 0] == 1):
            logger.debug("Fitting a logistic regression model with no intercept!")

        # Override arguments prodived in __init__ if provided:
        alpha = regularization_alpha or self.regularization_alpha  # 0 or anything -> anything
        score_kwargs = score_kwargs or self.score_kwargs
        cv_kwargs = cv_kwargs or self.cv_kwargs

        if isinstance(alpha, dict):
            return self._fit_regularized_model(features=features, obs=obs, alpha=alpha[cell_type])
        elif not isinstance(alpha, str):
            return self._fit_regularized_model(features=features, obs=obs, alpha=alpha)
        elif alpha == "score":
            return self._fit_score_optimized_model(cell_type=cell_type, features=features, obs=obs, **score_kwargs)
        elif alpha == "cv":
            return self._fit_cv_optimized_model(cell_type=cell_type, features=features, obs=obs, **cv_kwargs)
        else:
            raise ValueError(f"Invalid value for alpha: {alpha}")

    def _predict(
        self,
        cell_type: str,
        obs: Literal["death"] | Literal["division"],
        features: pd.DataFrame,
    ) -> np.ndarray:
        """
        Uses the model fit on cell_type data to predict death / division probabilities.
        """
        if obs in ["death", "division"]:
            return np.array(self.models[cell_type][obs].predict(features))
        else:
            raise ValueError(f"Invalid argument: _predict(obs={obs})")

    def _fit_regularized_model(self, features: pd.DataFrame, obs: pd.DataFrame, alpha: float):
        """Fits a single regularized model with L1 penalty alpha.

        Args:
            features (pd.DataFrame): Feature matrix.
            obs (pd.DataFrame): Observations (dependent variable).
            alpha (float): L1 penalty.

        Returns:
            model: statsmodels result.
        """

        # some alpha values don't converge, during optimization this floods the terminal
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m = sm.Logit(exog=features, endog=obs)

            if alpha > 0:
                alpha_array = self._no_intercept_alpha(
                    alpha, n_features=features.shape[1]
                )  # Doesn't regularize intercept
                result = m.fit_regularized(alpha=alpha_array, maxiter=100, disp=0)  # type: ignore
            else:
                result = m.fit(maxiter=100, disp=0)

            return result

    def _fit_score_optimized_model(
        self,
        cell_type: str,
        features: pd.DataFrame,
        obs: pd.DataFrame,
        metric: Literal["bic", "aic"] = "bic",
        alpha_logspace_start: float = -8,
        alpha_logspace_stop: float = 4,
        k_fits: int = 20,
    ):
        """Fits models to a range of regularization values and returns the optimal model according to the metric.

        Args:
            features (pd.DataFrame): _description_
            obs (pd.DataFrame): _description_
            metric (Literal[&quot;bic&quot;, &quot;aic&quot;], optional): _description_. Defaults to "bic".
            alpha_logspace_start (float, optional): _description_. Defaults to -4.
            alpha_logspace_stop (float, optional): _description_. Defaults to 4.
            k_fits (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        alphas = np.logspace(alpha_logspace_start, alpha_logspace_stop, num=k_fits)
        ms = [self._fit_regularized_model(features=features, obs=obs, alpha=alpha) for alpha in alphas]

        if metric == "bic":
            scores = [m.bic for m in ms]
        elif metric == "aic":
            scores = [m.aic for m in ms]

        opt_idx = np.argmin(scores)

        # result is likely suboptimal if obtained at edge of parameter range:
        if opt_idx == 0:
            logger.debug(
                f"[LogisticRegression] The minimal alpha was optimal, try decreasing alpha_logspace_start. Current value {alpha_logspace_start}"
            )

        if opt_idx == k_fits - 1:
            logger.debug(
                f"[LogisticRegression] The maximal alpha was optimal, try increasing alpha_logspace_stop. Current value {alpha_logspace_stop}"
            )

        # save best alpha for quicker fit later:
        self.best_alphas[cell_type] = alphas[opt_idx].item()

        return ms[opt_idx]

    def _fit_cv_optimized_model(
        self,
        cell_type: str,
        features: pd.DataFrame,
        obs: pd.Series,
        alpha_logspace_start: float = -8,
        alpha_logspace_stop: float = 4,
        k_fits: int = 20,
        k_folds: int = 3,
    ) -> float:
        """Finds the optimal alpha using k-fold cross-validation.

        Args:
            features (pd.DataFrame): Input features.
            obs (pd.Series): Target values.
            alpha_logspace_start (float, optional): Start of log-spaced alpha values. Defaults to -8.
            alpha_logspace_stop (float, optional): End of log-spaced alpha values. Defaults to 4.
            k_fits (int, optional): Number of alpha values to test. Defaults to 20.
            k_folds (int, optional): Number of folds for cross-validation. Defaults to 3.

        Returns:
            float: The best alpha value found.
        """
        alphas = np.logspace(alpha_logspace_start, alpha_logspace_stop, num=k_fits)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_alpha = None
        best_loss = np.inf

        for alpha in alphas:
            scores = []
            for train_idx, val_idx in kf.split(features):
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = obs.iloc[train_idx], obs.iloc[val_idx]

                model = self._fit_regularized_model(features=X_train, obs=y_train, alpha=alpha)
                predictions = model.predict(X_val)

                loss = log_loss(y_val, predictions)
                scores.append(loss)

            avg_score = np.mean(scores)
            if avg_score < best_loss:
                best_loss = avg_score
                best_alpha = alpha

        # best loss is always less than inf at least once
        best_alpha = float(best_alpha)  # type: ignore

        # save best alpha for quicker fit later:
        self.best_alphas[cell_type] = best_alpha

        return self._fit_regularized_model(features=features, obs=obs, alpha=best_alpha)

    def parameters(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns the parameters associated with the death / division model for cells of type cell_type.
        """
        return np.array(self.models[cell_type][obs].params.values)  # type: ignore

    def parameter_stds(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns the parameters associated with the death / division model for cells of type cell_type.
        """
        return np.array(self.models[cell_type][obs].bse.values)  # type: ignore

    def parameter_pvalues(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns the parameters associated with the death / division model for cells of type cell_type.
        """
        return np.array(self.models[cell_type][obs].pvalues.values)  # type: ignore

    def set_parameters(
        self,
        cell_type: str,
        obs: Literal["death"] | Literal["division"],
        new_params: list[float] | np.ndarray,
    ) -> None:
        """
        Replaces the parameters associated with the death / division model for cells of type cell_type with values.
        """
        original_params = self.models[cell_type][obs]._results.params

        assert len(original_params) == len(new_params)

        self.models[cell_type][obs]._results.params = np.array(new_params, dtype=np.float64)

    def parameter_names(self, cell_type: str, obs: Literal["death"] | Literal["division"]) -> np.ndarray:
        """
        Returns the parameter names associated with the death / division model for cells of type cell_type.
        """
        return np.array(self.models[cell_type][obs].model.exog_names)  # type: ignore

    def set_parameter(
        self, cell_type: str, parameter_name: str, value: float, obs: Literal["division"] = "division"
    ) -> None:
        """
        Sets the value of a parameter in the model.
        """
        parameter_idx = np.argwhere(self.parameter_names(cell_type, obs) == parameter_name)
        self.models[cell_type][obs]._results.params[parameter_idx] = value

    def get_parameter(self, cell_type: str, parameter_name: str, obs: Literal["division"] = "division") -> None:
        """
        Gets the value of a parameter in the model.
        """
        parameter_idx = np.argwhere(self.parameter_names(cell_type, obs) == parameter_name)
        return self.models[cell_type][obs]._results.params[parameter_idx].item()

    """
    Private functions:
    """

    def _no_intercept_alpha(self, alpha: float, n_features: int, intercept_idx: int = 0):
        """
        Returns a vector with n repeats of alpha and a zero at the index of the intercept.

        statsmodels fit_regularized alpha regularizes intercept unless given a vector with regularization per feature.
        """
        a = np.repeat(alpha, n_features)
        a[intercept_idx] = 0.0  # don't regularize intercept
        return a
