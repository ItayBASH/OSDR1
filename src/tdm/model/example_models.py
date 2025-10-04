"""
A module containing models with predefined parameters, according to
the number, type and location of fixed points.
"""

import numpy as np
from sympy import symbols, solve
from tdm.cell_types import FIBROBLAST, MACROPHAGE
from tdm.cache import persistent_cache


"""
Getter functions for models and funcs:
"""


def get_models():
    return [
        model_with_one_stable_fixed_point(),
        model_with_two_stable_fixed_points(),
        model_with_three_stable_fixed_points(),
        model_with_two_stable_fixed_points_on_axes(),
    ]


# def get_metric_funcs():
#     return [metric_func_1, metric_func_2, metric_func_3, metric_func_4]


"""
Code for computing logistic regression coefficients:
"""


def logit(p: float):
    return np.log(p / (1 - p))


def coefficients_for_one_line_nullcline():
    """
    Parameters for a logistic regression model for a cell whose division rate decreases
    with the number of neighbors of its own type.
    """
    # death rate at steady state
    D = 0.02
    D_logit = logit(D)

    # maximal death rate, at position = 0
    D_max = 0.05
    D_max_logit = logit(D_max)

    # maximal death rate, at position = 0
    D_low = 0.0025
    D_low_logit = logit(D_low)

    # number of cells (log) at steady state
    n_stable = 4
    n_max = 2
    n_low = 7

    beta0, beta1, beta2 = symbols("beta0 beta1 beta2")

    params = solve(
        [
            beta0 + (beta1 * n_stable) + (beta2 * n_stable**2) - D_logit,
            beta0 + (beta1 * n_max) + (beta2 * n_max**2) - D_max_logit,
            beta0 + (beta1 * n_low) + (beta2 * n_low**2) - D_low_logit,
        ],
        [beta0, beta1, beta2],
    )

    beta0, beta1, beta2 = (
        float(params[beta0]),
        float(params[beta1]),
        float(params[beta2]),
    )

    return beta0, beta1, beta2, D, D_max


def coefficients_for_two_line_nullcline():
    """
    Parameters for a logistic regression model for a cell whose division rate
    increases and then decreases with the number of neighbors of its own type.
    """
    # death rate at steady state
    D = 0.02
    D_logit = np.log(D / (1 - D))

    # maximal death rate, at position = 2 (similar to the decrease in fibroblast proliferation in the data)
    D_max = 0.05
    D_max_logit = np.log(D_max / (1 - D_max))

    # number of cells (log) at stable and unstable steady states:
    n_stable = 4
    n_unstable = 2
    n_max = 3

    beta0, beta1, beta2 = symbols("beta0 beta1 beta2")

    params = solve(
        [
            # Division equals death at n = Nstable
            beta0 + (beta1 * n_stable) + (beta2 * (n_stable**2)) - D_logit,
            # Division equals death at n = Nunstable
            beta0 + (beta1 * n_unstable) + (beta2 * (n_unstable**2)) - D_logit,
            # Maximum death rate is D_max, at n = 1
            beta0 + (beta1 * n_max) + (beta2 * (n_max**2)) - D_max_logit,
        ],
        [beta0, beta1, beta2],
    )

    beta0, beta1, beta2 = (
        float(params[beta0]),
        float(params[beta1]),
        float(params[beta2]),
    )

    return beta0, beta1, beta2, D, D_max


def coefficients_for_parabola_nullcline():
    """
    Parameters for a logistic regression model for a cell whose division rate increases
    and then decreases with the number of neighbors of its own type, but its division rate
    also decreases as a function of the other type, resulting in a parabola nullcline
    """

    # death rate at steady state
    D = 0.02
    D_logit = np.log(D / (1 - D))

    # maximal death rate, at position = 2 (similar to the decrease in fibroblast proliferation in the data)
    D_max = 0.06
    D_max_logit = np.log(D_max / (1 - D_max))

    # number of cells (log) at stable and unstable steady states:
    n_stable = 6
    n_unstable = 2
    n_max = 4.5
    m_max = 2

    beta0, beta1, beta2, beta3 = symbols("beta0 beta1 beta2 beta3")  # beta3 is the coefficient of the second cell type

    params = solve(
        [
            # Division equals death at n = Nstable
            beta0 + (beta1 * n_stable) + (beta2 * (n_stable**2)) - D_logit,
            # Division equals death at n = Nunstable
            beta0 + (beta1 * n_unstable) + (beta2 * (n_unstable**2)) - D_logit,
            # Maximum death rate is D_max, at n_max, m = 0
            beta0 + (beta1 * n_max) + (beta2 * (n_max**2)) - D_max_logit,
            # Maximum death rate is D_max, at n = 1
            beta0 + (beta1 * n_max) + (beta2 * (n_max**2)) + (beta3 * m_max) - D_logit,
        ],
        [beta0, beta1, beta2, beta3],
    )

    beta0, beta1, beta2, beta3 = [float(params[b]) for b in [beta0, beta1, beta2, beta3]]

    return beta0, beta1, beta2, beta3, D, D_max


"""
Code for initializing models based on pre-computed coefficients:
"""


@persistent_cache
def get_some_fitted_model(return_dataset=False):  # pragma: no cover
    """
    Get a logistic regression model fitted to a dataset.
    In this module, the model parameters are later replaced with the pre-computed ones.
    """

    from tdm.raw.breast_mibi import read_single_cell_df
    from tdm.analysis import Analysis
    from tdm.cell_types import TUMOR, ENDOTHELIAL

    ana = Analysis(
        single_cell_df=read_single_cell_df(),  # smaller df for testing
        neighborhood_mode="exclude",
        allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL],
        cell_types_to_model=[FIBROBLAST, MACROPHAGE],
        polynomial_dataset_kwargs={
            "degree": 2,
            "log_transform": True,  # important - transforms raw counts to log!
            "scale_features": False,
        },
        enforce_max_density=False,
        verbose=False,
    )

    m = ana.model
    pds = ana.pds

    if return_dataset:
        return m, pds
    else:
        return m


def model_with_one_stable_fixed_point():
    """
    Returns a model with one stable fixed point at (4,4)
    """
    m = get_some_fitted_model()

    # beta0, beta1, D, D_max = coefficients_for_one_line_nullcline()
    beta0, beta1, beta2, D, D_max = coefficients_for_one_line_nullcline()

    # m.set_parameters(FIBROBLAST, "division", [beta0, beta1, 0, 0, 0, 0])
    m.set_parameters(FIBROBLAST, "division", [beta0, beta1, 0, beta2, 0, 0])
    m.models[FIBROBLAST]["death"].p = D

    m.set_parameters(MACROPHAGE, "division", [beta0, 0, beta1, 0, 0, beta2])
    m.models[MACROPHAGE]["death"].p = D

    return m


def model_with_two_stable_fixed_points():
    """
    Returns a model with stable fixed points at (4,4) and (0,4)
    """
    m = get_some_fitted_model()

    beta0, beta1, beta2, D, _ = coefficients_for_two_line_nullcline()
    m.set_parameters(FIBROBLAST, "division", [beta0, beta1, 0, beta2, 0, 0])
    m.models[FIBROBLAST]["death"].p = D

    beta0, beta1, beta2, D, _ = coefficients_for_one_line_nullcline()
    m.set_parameters(MACROPHAGE, "division", [beta0, 0, beta1, 0, 0, beta2])
    m.models[MACROPHAGE]["death"].p = D

    return m


def model_with_two_stable_fixed_points_on_axes():
    """
    Returns a model with stable fixed points at (6,0) and (0,6)
    """
    m = get_some_fitted_model()

    beta0, beta1, beta2, beta3, D, D_max = coefficients_for_parabola_nullcline()

    m.set_parameters(FIBROBLAST, "division", [beta0, beta1, beta3, beta2, 0, 0])
    m.models[FIBROBLAST]["death"].p = D

    m.set_parameters(MACROPHAGE, "division", [beta0, beta3, beta1, 0, 0, beta2])
    m.models[MACROPHAGE]["death"].p = D

    return m


def model_with_three_stable_fixed_points():
    """
    Returns a model with stable fixed points at (4,4), (4,0) and (0,4)
    """
    m = get_some_fitted_model()

    beta0, beta1, beta2, D, D_max = coefficients_for_two_line_nullcline()

    m.set_parameters(FIBROBLAST, "division", [beta0, beta1, 0, beta2, 0, 0])
    m.models[FIBROBLAST]["death"].p = D

    m.set_parameters(MACROPHAGE, "division", [beta0, 0, beta1, 0, 0, beta2])
    m.models[MACROPHAGE]["death"].p = D

    return m


"""
Functions for identifying the example models:
"""


def exists_fp_near_xy(fps: list[tuple[float, float]], x, y, tol=1.5):
    for ux, uy in fps:
        if abs(ux - x) < tol and abs(uy - y) < tol:
            return True
    return False


def get_valid_stable_points(stable_fps: list[tuple[float, float]], unstable_fps: list[tuple[float, float]]):
    """
    Computes the number of valid stable points.

    A stable point is considered semi-stable if it is on an axis and there
    is an unstable fixed point within ~1 cell from it. Such a point effectively has
    no real basin of attraction.
    """
    valid_fps = []
    for x, y in stable_fps:
        # on-axis: need to consider discretization error
        if np.allclose(x, 0) or np.allclose(y, 0):
            if not exists_fp_near_xy(unstable_fps, x, y, tol=1.5):
                valid_fps.append((x, y))
            else:
                continue
        # off-axis:
        else:
            valid_fps.append((x, y))

    return valid_fps
