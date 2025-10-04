"""
Code for computing deterministic trajectories based on fitted models.
"""

import numpy as np
from scipy.integrate import odeint

from tdm.analysis import Analysis
from tdm.model import Model


def compute_trajectory(ana: Analysis, initial_cell_counts: list[float], odeint_timepoints: np.ndarray) -> np.ndarray:
    f = _construct_f_for_odeint(ana.cell_types, ana.model)
    # setting atol increases speed by reducing function calls
    return odeint(f, initial_cell_counts, odeint_timepoints, atol=1e-3)


def _construct_f_for_odeint(cell_types: list[str], model: Model):
    _f = model.get_delta_cells_func(predicted_cell_types=cell_types, fixed_cell_counts=None)

    def f(x, t):
        cell_counts = np.array(x)
        delta_cells = _f(cell_counts)
        return list(delta_cells)

    return f
