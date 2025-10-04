"""
Utils for bootstrapping analyses
"""

from concurrent.futures import ThreadPoolExecutor
from tdm.analysis import Analysis
from copy import deepcopy
import numpy as np
from typing import Literal


# def n_boostraps(ana: Analysis, n: int, mode: Literal["tissue", "cell"]):
#     if mode == "tissue":
#         f = tissue_level_bootstrap
#     elif mode == "cell":
#         f = cell_level_bootstrap

#     return [f(ana) for _ in range(n)]


def _run_bootstrap(f_ana):
    f, ana = f_ana
    return f(ana)


def n_boostraps(ana: Analysis, n: int, mode: Literal["tissue", "cell"]):
    if mode == "tissue":
        f = tissue_level_bootstrap
    elif mode == "cell":
        f = cell_level_bootstrap

    # parallelize:
    with ThreadPoolExecutor() as executor:
        bootstrap_anas = list(executor.map(_run_bootstrap, [(f, ana) for _ in range(n)]))

    return bootstrap_anas


def tissue_level_bootstrap(ana: Analysis):
    # copy:
    new_ana = deepcopy(ana)

    # resample ndss with replacement:
    new_ana.ndss = np.random.choice(a=ana.ndss, size=len(ana.ndss), replace=True)  # type: ignore
    new_ana.tissues = [nds.tissue for nds in new_ana.ndss]  # overwrite tissues

    # rerun analysis starting with the resampled datasets:
    new_ana.run(start_phase=3, verbose=False)

    return new_ana


def cell_level_bootstrap(ana: Analysis):
    # copy:
    new_ana = deepcopy(ana)

    # resample ndss with replacement:
    for cell_type in new_ana.rnds.cell_types():

        # original cells and observations:
        features, obs = new_ana.rnds.fetch(cell_type)

        # resample cells:
        n_cells = len(features)
        sampled_indices = np.random.choice(a=n_cells, size=n_cells, replace=True)
        sampled_features = features.iloc[sampled_indices].reset_index(drop=True)
        sampled_obs = obs.iloc[sampled_indices].reset_index(drop=True)

        # assign resampled cells:
        new_ana.rnds.set_dataset(cell_type=cell_type, features=sampled_features, obs=sampled_obs)

    # rerun analysis starting with the resampled datasets:
    new_ana.run(start_phase=4, verbose=False)

    return new_ana
