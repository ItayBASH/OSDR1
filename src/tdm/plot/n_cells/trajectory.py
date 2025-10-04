from tdm.utils import log2_1p
from tdm.analysis import Analysis
from tdm.simulate.deterministic import compute_trajectory
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trajectory(
    ana: Analysis,
    initial_cell_counts: list[float],
    odeint_timepoints: np.ndarray,
    title: str | None = None,
    logscale: bool = False,
    ax: plt.Axes | None = None,
):
    """Plot a deterministic 3D trajectory.

    Args:
        ana (Analysis): Analysis object.
        initial_cell_counts (list[float]): start state for the trajectory.
        odeint_timepoints (np.ndarray): timepoints for the scipy.integrate.odeint function
        title (str | None, optional): alternative title for plot. Defaults to None.

    Example:
    >>> # ana is an analysis fit to 3 cell types
    >>> plot_nD_trajectory(ana, initial_cell_counts=[40,4,1], odeint_timepoints=np.linspace(0,5000, 10000))
    """

    sol = compute_trajectory(ana=ana, initial_cell_counts=initial_cell_counts, odeint_timepoints=odeint_timepoints)

    if logscale:
        sol = log2_1p(sol)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    for i in range(sol.shape[1]):
        label = f"{ana.cell_types[i]}"
        sns.lineplot(x=odeint_timepoints, y=sol[:, i], ax=ax, label=label)

    # Set title
    if title is None:
        ax.set_title("n-dimensional trajectory")
    else:
        ax.set_title(title)

    # Add a legend
    ax.legend()
    sns.despine(ax=ax)

    if logscale:
        ax.set_ylabel("cell counts (log)")
    else:
        ax.set_ylabel("cell counts")

    ax.set_xlabel("time (steps)")

    fig.tight_layout()

    # Show the plot
    plt.show()


def plot_n_trajectories(
    anas: list[Analysis],
    initial_cell_counts: list[float] | list[list[float]],
    odeint_timepoints: np.ndarray,
    title: str | None = None,
    logscale: bool = False,
    axhlines1: list[float] | None = None,
    axhlines2: list[float] | None = None,
    ylim: float = 10,
    ana_colors: list[str] | None = None,
    plot: bool = True,
):
    """Plot a deterministic nD trajectory, with one dimension per figure.

    Args:
        ana (list[Analysis]): Analysis objects.
        initial_cell_counts (list[float]): start state for the trajectory.
        odeint_timepoints (np.ndarray): timepoints for the scipy.integrate.odeint function
        title (str | None, optional): alternative title for plot. Defaults to None.

    Example:
    >>> # ana is an analysis fit to 3 cell types
    >>> plot_nD_trajectory(ana, initial_cell_counts=[40,4,1], odeint_timepoints=np.linspace(0,5000, 10000))
    """

    # different initial state per ana:
    if isinstance(initial_cell_counts[0], list):
        sols = [
            compute_trajectory(ana=ana, initial_cell_counts=initial_cell_counts[i], odeint_timepoints=odeint_timepoints)  # type: ignore
            for i, ana in enumerate(anas)
        ]
    else:
        sols = [
            compute_trajectory(ana=ana, initial_cell_counts=initial_cell_counts, odeint_timepoints=odeint_timepoints)  # type: ignore
            for ana in anas
        ]

    if logscale:
        sols = [log2_1p(sol) for sol in sols]

    if not plot:
        return sols

    cell_types = anas[0].cell_types
    n_types = len(cell_types)
    fig, axs = plt.subplots(figsize=(6, 1 * n_types), nrows=n_types, sharey=True, sharex=True)

    # one axis per cell-type:
    for i, ax in enumerate(axs):
        # ylim = 0
        for ana_i, sol in enumerate(sols):
            y = sol[:, i]

            if ana_colors is None:
                sns.lineplot(x=odeint_timepoints, y=y, ax=ax, color="#3B76BC")
            else:
                sns.lineplot(x=odeint_timepoints, y=y, ax=ax, color=ana_colors[ana_i])

            # ylim = max(ylim, max(y))

        sns.despine(ax=ax)
        ax.set_xlabel("time (steps)")
        if logscale:
            ax.set_ylabel("cells (log)")
        else:
            ax.set_ylabel("cells")

        ax.set_ylim(bottom=0, top=ylim)
        ax.set_title(f"{cell_types[i]}")

    # add one axvline to each axis:
    if axhlines1 is not None:
        for i, ax in enumerate(axs):
            ax.axhline(axhlines1[i], color="green", linestyle="--", alpha=0.5)

    # add one axvline to each axis:
    if axhlines2 is not None:
        for i, ax in enumerate(axs):
            ax.axhline(axhlines2[i], color="red", linestyle="--", alpha=0.5)

    # Set title
    if title is None:
        fig.suptitle("n-dimensional trajectory")
    else:
        fig.suptitle(title)

    fig.tight_layout()

    return sols
