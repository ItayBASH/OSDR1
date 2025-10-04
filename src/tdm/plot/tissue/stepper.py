from tdm.utils import log2_1p
from tdm.dataset import NeighborsDataset
from tdm.plot.tissue.spatial import plot_tissue
from tdm.cell_types import CELL_TYPE_TO_COLOR, CELL_TYPE_TO_FULL_NAME
from tdm.simulate.tissue_step import TissueStep
from tdm.simulate.generate_distribution import n_cells_per_neighborhood

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_steps(stepper, steps: list[int] | None = None, figsize=(4, 4)):

    steps = steps or [0]

    fig, axs = plt.subplots(ncols=len(steps), figsize=figsize, sharex=True, sharey=True)

    for i, t in enumerate(steps):
        ax = axs[i]

        # if there aren't enough tissues, plot empty axes
        if len(stepper.tissues) <= t:
            continue
        else:

            plot_tissue(stepper.tissues[t], ax=ax)

            if i < len(steps) - 1:
                ax.legend().remove()

        ax.set_title(f"{t} steps")
        sns.despine(ax=ax)

    fig.tight_layout()


def plot_cell_distribution_over_time(
    stepper: TissueStep,
    cell_types: list[str],
    steps: list[int] | None = None,
    figsize=(4, 4),
    bw_adjust: float = 2.0,
    axvline_x_vals: list[float] | None = None,
):
    """ """
    steps = steps or [0]

    fig, axs = plt.subplots(nrows=len(steps), figsize=figsize, sharex=True, sharey=False)

    for ax_i, t in enumerate(steps):

        # each timestep has an axis:
        ax = axs[ax_i]
        ax.set_title(f"{t} steps")
        sns.despine(ax=ax)
        ax.set_xlim(left=0, right=7)

        if len(stepper.tissues) <= t:
            continue
        else:
            tissue = stepper.tissues[t]

        nds = NeighborsDataset(tissue)

        for c in cell_types:

            # take number of neighbors for cells of this type.
            n_cells = nds.fetch(c)[0][c]
            sns.kdeplot(log2_1p(n_cells), bw_adjust=bw_adjust, fill=True, color=CELL_TYPE_TO_COLOR[c], ax=ax, label=c)

        if ax_i == 0:
            ax.legend(frameon=False, loc="upper right")

        if axvline_x_vals is not None:
            for x, c in zip(axvline_x_vals, cell_types):
                ax.axvline(x, linestyle="--", linewidth=1, color=CELL_TYPE_TO_COLOR[c])

    fig.tight_layout()


def plot_cells_over_time(
    stepper: TissueStep,
    cell_types: list[str],
    ahline_y_vals: list[float] | None = None,
    ax: plt.Axes | None = None,
    ylim: tuple[float, float] = (0, 8),
    log_cells: bool = True,
    n_steps: tuple[int] | int | None = None,
    legend_loc: str = "upper left",
):
    """
    Plots the number of cells of each cell_types using the simulated tissues
    produced by stepper.
    """
    ahline_y_vals = ahline_y_vals or []

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    for cell_type in cell_types:
        # get total number of cells of this type in each tissue:
        n_cells = np.array([(t.cell_df().cell_type == cell_type).sum() for t in stepper.tissues])

        # convert to number of cells per neighborhood:
        n_cells_per_nbrhood = n_cells_per_neighborhood(
            n_cells_per_tissue=n_cells,
            tissue_width=stepper.tissue_width,
            tissue_height=stepper.tissue_height,
            neighborhood_size=stepper.neighborhood_size,
            round_to_int=False,
        )

        # log:
        vals = log2_1p(n_cells_per_nbrhood)
        if not log_cells:
            vals = n_cells_per_nbrhood

        x = np.arange(len(vals))

        if n_steps is not None:
            if isinstance(n_steps, int):
                vals = vals[:n_steps]
                x = np.arange(len(vals))
            else:
                l, r = n_steps  # type: ignore
                vals = vals[l:r]
                x = np.arange(l, r)

        # plot:
        sns.lineplot(
            x=x,
            y=vals,
            label=CELL_TYPE_TO_FULL_NAME[cell_type],
            color=CELL_TYPE_TO_COLOR[cell_type],
            ax=ax,
        )

    ax.set_xlim(x[0], x[-1])

    if log_cells:
        ax.set_ylim(*ylim)
        ax.set_ylabel("# cells (log2)")
    else:
        ax.set_ylim(ylim[0], 2 ** ylim[1])
        ax.set_ylabel("# cells")

    for y_val in ahline_y_vals:
        ax.axhline(y_val, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("t (steps)")
    ax.legend(loc=legend_loc, fontsize="large", frameon=False)

    sns.despine(ax=ax)
