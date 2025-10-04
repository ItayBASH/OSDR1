"""
Model calibration plot.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from copy import deepcopy

from tdm.analysis import Analysis
from tdm.model.model import Model
from tdm.dataset.dataset import Dataset
from tdm.cell_types import cell_type_to_full_name


def plot_calibration(ana: Analysis, n_cells_per_bin: int = 2000, max_p: dict | float | None = None):
    """Plots model calibration for all cell types.

    Args:
        ana (Analysis): an Analysis object.
        n_cells_per_bin (int, optional): number of cells used in each point. A small number introduces
        greater error in estimating the true probability. Defaults to 2000.
        max_p (float | None, optional): sets the y-lim of the true probability axis. Defaults to None.
    """

    n_types = len(ana.cell_types)

    fig, axs = plt.subplots(ncols=n_types, figsize=(2 * n_types, 2))

    for i, cell_type in enumerate(ana.cell_types):
        ax = axs[i]

        if isinstance(max_p, dict):
            _p = max_p[cell_type]
        else:
            _p = max_p

        _plot_model_calibration(
            cell_type=cell_type,
            model=ana.model,
            dataset=ana.pds,
            n_cells_per_bin=n_cells_per_bin,
            ax=ax,
            max_p=_p,
        )

    fig.tight_layout()


def plot_calibration_against_permuted_divisions(ana: Analysis, n_cells_per_bin: int = 2000):
    """Plots model calibration for all cell types.
    Plots calibration on real data and a version with permuted divisions side by side.

    Args:
        ana (Analysis): an Analysis object.
        n_cells_per_bin (int, optional): number of cells used in each point. A small number introduces
        greater error in estimating the true probability. Defaults to 2000.
        max_p (float | None, optional): sets the y-lim of the true probability axis. Defaults to None.
    """

    # original model and dataset:
    pds = ana.pds
    m = ana.model

    # permuted model and dataset:
    pds_permuted = deepcopy(pds)
    for cell_type in pds.cell_types():
        counts, div = pds_permuted.fetch(cell_type)
        div_permuted = div.iloc[np.random.permutation(len(div))].reset_index(drop=True)
        pds_permuted.dataset_dict[cell_type] = counts, div_permuted
    m_permuted = ana._fit_model(pds_permuted)

    # plot calibration plots for all types:
    for cell_type in m.cell_types():

        fig, axs = plt.subplots(figsize=(6, 3), ncols=2)

        ax = axs[0]
        min_p, max_p = _plot_model_calibration(
            cell_type,
            m,
            pds,
            n_cells_per_bin=n_cells_per_bin,
            ax=ax,
            max_p=None,
            plot_min_max_lines=True,
        )

        ax = axs[1]
        _plot_model_calibration(
            cell_type,
            m_permuted,
            pds_permuted,
            n_cells_per_bin=n_cells_per_bin,
            ax=ax,
            min_p=min_p,
            max_p=max_p,
            plot_min_max_lines=True,
        )
        ax.set_title("Calibration over permuted divisions")

        fig.tight_layout()


def _plot_model_calibration(
    cell_type: str,
    model: Model,
    dataset: Dataset,
    n_cells_per_bin: int = 2000,
    ax=None,
    min_p: float | None = None,
    max_p: float | None = None,
    obs_type="division",
    plot_min_max_lines: bool = False,
) -> tuple[float, float]:
    """ """
    features, obs = dataset.fetch(cell_type)
    probs = model.predict(cell_type, obs=obs_type, features=features)
    n_bins = int(len(probs) / n_cells_per_bin)
    prob_true, prob_pred = calibration_curve(
        obs[obs_type],
        probs,
        n_bins=n_bins,
        strategy="quantile",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(y=prob_true, x=prob_pred, color="#53AC69", edgecolor="#252526", ax=ax)

    min_p = min_p or prob_true.min()
    max_p = max_p or prob_true.max()

    sns.lineplot(
        x=np.linspace(0, max_p, 10),
        y=np.linspace(0, max_p, 10),
        color="black",
        linestyle="--",
        ax=ax,
    )

    if plot_min_max_lines:
        ax.axhline(y=prob_true.min(), color="red", linestyle="--")
        ax.axhline(y=prob_true.max(), color="red", linestyle="--")

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("true probability")
    ax.set_title(f"{cell_type_to_full_name(cell_type)} model")

    eps = 0.01
    ax.set_xlim(min_p - eps, max_p + eps)
    ax.set_ylim(min_p - eps, max_p + eps)
    sns.despine(ax=ax)

    return min_p, max_p
