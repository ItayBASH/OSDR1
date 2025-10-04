"""
Utilities for plotting phase-portraits based on analyses fit to 3 cells.
"""

import numpy as np
from tdm.utils import log2_1p
from tdm.analysis import Analysis


def get_cell_c_typical_vals(ana: Analysis, low_q: float = 0.2, high_q: float = 0.8) -> np.ndarray:
    """Returns a list of consecutive integers between the low and high quantiles of cell c density (log).

    Args:
        rnds (NeighborsDataset): _description_
        low_q (float, optional): _description_. Defaults to 0.2.
        high_q (float, optional): _description_. Defaults to 0.8.

    Returns:
        _type_: _description_
    """
    rnds = ana.rnds
    cell_a = ana.cell_a
    cell_b = ana.cell_b
    cell_c = ana.cell_c

    # compute the low and high quantiles of cell c in neighborhoods of cells of type a:
    ac_vals = rnds.fetch(cell_a)[0][cell_c]
    low_a, high_a = ac_vals.quantile(low_q), ac_vals.quantile(high_q)

    # compute the low and high quantiles of cell c in neighborhoods of cells of type b:
    bc_vals = rnds.fetch(cell_b)[0][cell_c]
    low_b, high_b = bc_vals.quantile(low_q), bc_vals.quantile(high_q)

    # combine quantiles "conservatively":
    low = max(low_a, low_b)
    high = min(high_a, high_b)

    low, high = log2_1p(low), log2_1p(high)

    return np.arange(*np.round([low, high], 1))
