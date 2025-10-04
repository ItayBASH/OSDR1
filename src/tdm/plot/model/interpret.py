from typing import Literal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tdm.cell_types import CELL_TYPE_TO_FULL_NAME
from tdm.analysis import Analysis
from tdm.style import RESPONSE_COLOR, NO_RESPONSE_COLOR

LOW_PROLIFERATION_COLOR = "#3F3C3C"  # black
HIGH_PROLIFERATION_COLOR = "#CA493C"  # red


def plot_high_and_low_proliferation_states(
    ana: Analysis,
    plot_background_distribution: bool = False,
    k_states: int = 100,
    xlim: float | None = None,
    ylim: float | None = None,
):
    """Plot one figure per cell-type with the distributions of the k most and least proliferative states.

    Args:
        ana (Analysis): _description_
        plot_background_distribution (bool, optional): _description_. Defaults to False.
    """

    # compute top and bottom dividing states over the distribution of states of that type:
    for i, cell_a in enumerate(ana.cell_types):

        # compute division rate at all states:
        cell_counts = ana.rnds.fetch(cell_type=cell_a)[0]
        division_rates = ana.model.delta_cells(cell_counts=cell_counts, mode="rates", return_order=ana.cell_types)[i]
        sorted_idxs = np.argsort(division_rates)
        sorted_counts = cell_counts.iloc[sorted_idxs]

        # add small uniform noise for zero-variance error:
        all_states = sorted_counts + np.random.uniform(low=-0.5, high=0.5, size=sorted_counts.shape)

        # fetch extremes:
        lowest_dividing_states = all_states.iloc[:100]
        highest_dividing_states = all_states.iloc[-100:]

        nrows = len(ana.cell_types)
        fig, axs = plt.subplots(nrows=nrows, figsize=(7, 1 * nrows))
        fig.suptitle(cell_a)

        kde_kwargs = {
            "bw_adjust": 2,
            "fill": True,
            "alpha": 0.5,
        }

        for i, cell_b in enumerate(ana.cell_types):

            ax = axs[i]
            ax.set_title(cell_b)
            sns.kdeplot(
                lowest_dividing_states[cell_b],
                ax=ax,
                **kde_kwargs,
                color=LOW_PROLIFERATION_COLOR,
                label="low-proliferation states",
            )
            sns.kdeplot(
                highest_dividing_states[cell_b],
                ax=ax,
                **kde_kwargs,
                color=HIGH_PROLIFERATION_COLOR,
                label="high-proliferation states",
            )

            if plot_background_distribution:
                sns.kdeplot(all_states[cell_b], ax=ax, **kde_kwargs, color="black")

            # aesthetics:
            ax.set_xlim(left=-1, right=xlim)
            ax.set_xlabel(f"number of {cell_b} cells")
            ax.set_ylabel("probability")
            sns.despine(ax=ax)
            ax.set_title("")

            if i == 0:
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.2, 0.8))

            ax.set_ylim(0, ylim)

        fig.tight_layout()
        plt.show()


def plot_states_space_regions_with_different_sign(
    cell_type: str,
    reference_ana: Analysis,
    ana1: Analysis,
    ana2: Analysis,
    mode: Literal["1gt2", "2gt1"],
    eps: float = 0,
    color1: str = RESPONSE_COLOR,
    color2: str = NO_RESPONSE_COLOR,
    return_states: bool = False,
):
    features = reference_ana.pds.fetch(cell_type)[0]
    cell_types = reference_ana.cell_types

    p1 = ana1.model.predict(cell_type=cell_type, obs="division_minus_death", features=features)
    p2 = ana2.model.predict(cell_type=cell_type, obs="division_minus_death", features=features)

    if mode == "1gt2":
        mask = (p1 > eps) & (p2 < -eps)
    elif mode == "2gt1":
        mask = (p2 > eps) & (p1 < -eps)
    else:
        raise ValueError("mode must be either '1gt2' or '2gt1'")

    print(f"n states:{mask.sum()}")

    max_diff_states = features.loc[mask, :]

    print(
        "response proliferation rate at max diff states: ",
        p1[mask].mean().round(3),
        "mean division rate: ",
        # p_res.mean().round(2),
        ana1.rnds.fetch("Tu")[1].mean().round(2).item(),
    )
    print(
        "no-response proliferation rate at max diff states:",
        p2[mask].mean().round(3),
        "mean division rate: ",
        # p_no.mean().round(2),
        ana2.rnds.fetch("Tu")[1].mean().round(2).item(),
    )

    fig, axs = plt.subplots(figsize=(12, 1.5), ncols=len(cell_types))

    for i, c in enumerate(cell_types):

        ax = axs[i]

        kde_kwargs = {
            "bw_adjust": 2.0,
            "clip_on": True,
            "alpha": 0.5,
            "fill": True,
            "linewidth": 1.5,
        }
        sns.kdeplot(
            x=max_diff_states[c],
            label="different sign",
            **kde_kwargs,
            color=NO_RESPONSE_COLOR,
            ax=ax,
        )
        sns.kdeplot(x=features[c], label="all states", **kde_kwargs, ax=ax, color=RESPONSE_COLOR)

        if i == len(cell_types) - 1:
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 1.0), loc="upper left")
        if i == 0:
            ax.set_ylabel("probability")
        else:
            ax.set_ylabel("")

        ax.set_title(CELL_TYPE_TO_FULL_NAME[c])
        sns.despine(ax=ax)
        ax.set_xlim(left=0, right=features[c].quantile(0.995))
        ax.set_xlabel("cell density")

    fig.tight_layout()
    plt.show()

    if return_states:
        return max_diff_states


def plot_states_space_regions_with_maximal_difference(
    cell_type: str,
    reference_ana: Analysis,
    ana1: Analysis,
    ana2: Analysis,
    mode: Literal["1gt2", "2gt1"],
    color1: str = RESPONSE_COLOR,
    color2: str = NO_RESPONSE_COLOR,
):

    features = reference_ana.pds.fetch(cell_type)[0]
    cell_types = reference_ana.cell_types

    p1 = ana1.model.predict(cell_type=cell_type, obs="division", features=features)
    p2 = ana2.model.predict(cell_type=cell_type, obs="division", features=features)

    max_diff_idxs = np.argsort(p2 - p1)[-100:]
    # max_diff_idxs = np.argsort(p_no - p_res)[:100]
    max_diff_states = features.iloc[max_diff_idxs]

    print(
        "response proliferation rate at max diff states: ",
        p1[max_diff_idxs].mean().round(2),
        "mean division rate: ",
        # p_res.mean().round(2),
        ana1.rnds.fetch("Tu")[1].mean().round(2).item(),
    )
    print(
        "no-response proliferation rate at max diff states:",
        p2[max_diff_idxs].mean().round(2),
        "mean division rate: ",
        # p_no.mean().round(2),
        ana2.rnds.fetch("Tu")[1].mean().round(2).item(),
    )

    fig, axs = plt.subplots(figsize=(12, 1.5), ncols=len(cell_types))

    for i, c in enumerate(cell_types):

        ax = axs[i]

        kde_kwargs = {
            "bw_adjust": 2.0,
            "clip_on": True,
            "alpha": 0.5,
            "fill": True,
            "linewidth": 1.5,
        }
        sns.kdeplot(
            x=max_diff_states[c],
            label="maximal difference between groups",
            **kde_kwargs,
            color=NO_RESPONSE_COLOR,
            ax=ax,
        )
        sns.kdeplot(x=features[c], label="all states", **kde_kwargs, ax=ax, color=RESPONSE_COLOR)

        if i == len(cell_types) - 1:
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 1.0), loc="upper left")
        if i == 0:
            ax.set_ylabel("probability")
        else:
            ax.set_ylabel("")

        ax.set_title(CELL_TYPE_TO_FULL_NAME[c])
        sns.despine(ax=ax)
        ax.set_xlim(left=0)
        ax.set_xlabel("cell density")
        # plt.show()

    # fig.suptitle("States with maximal difference in tumor proliferation")
    fig.tight_layout()
    plt.show()
