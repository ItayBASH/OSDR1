"""
Code for plots related to the temporal and response predictions
over the triple negative dataset.
"""

from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from functools import cache
from tdm.cache import persistent_cache
from tdm.style import RESPONSE_COLOR, NO_RESPONSE_COLOR
from tdm.analysis import Analysis
from tdm.cell_types import CELL_TYPE_TO_FULL_NAME
from tdm.preprocess.single_cell_df import SUBJECT_ID_COL
from tdm.raw.triple_negative_imc import BIOPSY_PHASE_COLUMN, read_single_cell_df

from tdm.analysis.bootstrap import tissue_level_bootstrap
from tdm.plot.n_cells.trajectory import compute_trajectory


CELL_TYPES = ["Tu", "F", "M", "T", "B", "En"]


"""
Code for initial exploration, number of patients etc.
"""


def patient_ids_with_paired_biopsies(
    arm: str = "all",
    phases: Literal["baseline_to_early", "early_to_post"] = "baseline_to_early",
    response: Literal["RD", "pCR", "all"] = "all",
) -> np.ndarray:
    """Get the ids for patients that have biopsies from two consecutive phases.

    Args:
        arm (str, optional): restrict patient ids to one arm. Defaults to all.
        phases (Literal[&quot;baseline_to_early&quot;, &quot;early_to_post&quot;], optional): the two consecutive phases. Defaults to "baseline_to_early".

    Returns:
        np.ndarray: the patient ids
    """
    scdf = read_single_cell_df(biopsy_phase="all", treatment_arm=arm, response=response)

    subject_phase = (
        scdf.groupby(["subject_id", BIOPSY_PHASE_COLUMN])
        .first()
        .reset_index()
        .loc[:, [SUBJECT_ID_COL, BIOPSY_PHASE_COLUMN]]
    )
    unmelted_subject_phase = pd.get_dummies(subject_phase, columns=["BiopsyPhase"])
    unmelted_subject_phase = unmelted_subject_phase.groupby(SUBJECT_ID_COL).max()

    if phases == "baseline_to_early":
        mask = unmelted_subject_phase["BiopsyPhase_Baseline"] & unmelted_subject_phase["BiopsyPhase_On-treatment"]
    else:
        mask = unmelted_subject_phase["BiopsyPhase_On-treatment"] & unmelted_subject_phase["BiopsyPhase_Post-treatment"]

    return unmelted_subject_phase.loc[mask].index.values


"""
Code for temporal predictions:
"""


@cache
@persistent_cache
def get_analysis(
    biopsy_phase: str = "all",
    treatment_arm: str = "all",
    response: str = "all",
    degree: int = 1,
    ki67_threshold: float = 0.5,
):
    """Single function for loading analyses.

    Args:
        biopsy_phase (str, optional): _description_. Defaults to "all".
        treatment_arm (str, optional): _description_. Defaults to "all".
        response (str, optional): _description_. Defaults to "all".

    Returns:
        _type_: _description_
    """
    scdf = read_single_cell_df(
        biopsy_phase=biopsy_phase, treatment_arm=treatment_arm, response=response, ki67_threshold=0.5
    )
    return Analysis(
        single_cell_df=scdf,
        neighborhood_mode="extrapolate",
        cell_types_to_model=CELL_TYPES,
        supported_cell_types=CELL_TYPES,  # sets the order in the tissue objects
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        polynomial_dataset_kwargs={
            "degree": degree,
        },
        enforce_max_density=True,
        max_density_enforcer_power=8,
    )


# two levels of caching - persistence and in-memory cache
# 20 seconds on first call, 0 on second.
@cache
@persistent_cache
def get_bootstrap_analysis(
    idx: int = 0, biopsy_phase: str = "all", treatment_arm: str = "all", response: str = "all", degree: int = 1
):
    ana = get_analysis(biopsy_phase=biopsy_phase, treatment_arm=treatment_arm, response=response, degree=degree)
    np.random.seed(42 + idx)
    return tissue_level_bootstrap(ana=ana)


def assign_death_rates(from_ana: Analysis, to_ana: Analysis, alpha: float = 1):

    for cell_type in to_ana.cell_types:

        d_new = from_ana.model.death_prob(cell_type)
        d_old = to_ana.model.death_prob(cell_type)

        combined_d = (alpha * d_new) + ((1 - alpha) * d_old)

        to_ana.model.set_death_prob(cell_type, combined_d)

    to_ana.set_maximal_density_enforcer(to_ana.model, max_density_enforcer_power=8)


def compute_n_trajectories(ana, initial_states, timepoints):
    sols = []
    for i in range(len(initial_states)):
        trajectory = compute_trajectory(
            ana=ana, initial_cell_counts=initial_states.iloc[i], odeint_timepoints=timepoints
        )
        trajectory = pd.DataFrame(trajectory, columns=ana.cell_types)
        trajectory["tissue"] = i
        trajectory["t"] = timepoints
        sols.append(trajectory)

    return pd.concat(sols)


def melt_sols(sols):
    sols_df = sols.melt(id_vars=["tissue", "t"], var_name="cell_type", value_name="n")
    sols_df["days"] = sols_df["t"] * 0.25  # every step is a quarter day
    return sols_df


def plot_cell_over_time(sols_df, cell_type, ax, label, color=None, linestyle="-"):
    sns.lineplot(
        sols_df[sols_df.cell_type == cell_type], x="days", y="n", ax=ax, label=label, color=color, linestyle=linestyle
    )


def cells_over_time_dfs(initial_states, timepoints, response_ana, no_response_ana):

    response_sols = compute_n_trajectories(ana=response_ana, initial_states=initial_states, timepoints=timepoints)
    no_response_sols = compute_n_trajectories(ana=no_response_ana, initial_states=initial_states, timepoints=timepoints)

    response_sols_df = melt_sols(response_sols)
    no_response_sols_df = melt_sols(no_response_sols)

    return response_sols_df, no_response_sols_df


def plot_all_cell_types_over_time(
    response_sols_df,
    no_response_sols_df,
    ylim: float = 60,
    xlim: float | None = None,
    cell_types: list[str] = CELL_TYPES,
    share_ylim: bool = True,
):

    fig, axes = plt.subplots(ncols=len(cell_types), figsize=(len(cell_types) * 1.8, 2), sharex=True, sharey=share_ylim)

    # Flatten the axes for easy iteration
    axes = axes.flatten()

    for i, cell_type in enumerate(cell_types):
        ax = axes[i]

        # Plot data for each cell type
        plot_cell_over_time(
            response_sols_df, cell_type=cell_type, ax=ax, label="response dynamics", color=RESPONSE_COLOR
        )
        plot_cell_over_time(
            no_response_sols_df, cell_type=cell_type, ax=ax, label="no-response dynamics", color=NO_RESPONSE_COLOR
        )

        # for patents:
        # plot_cell_over_time(
        #     response_sols_df, cell_type=cell_type, ax=ax, label="response dynamics", color="black", linestyle="-"
        # )
        # plot_cell_over_time(
        #     no_response_sols_df, cell_type=cell_type, ax=ax, label="no-response dynamics", color="black", linestyle="--"
        # )

        # Adjust appearance
        sns.despine(ax=ax)
        ax.set_title(f"{CELL_TYPE_TO_FULL_NAME[cell_type]}")

        ax.legend().remove()

        # # add one legend only:
        # if i == 1:
        #     ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
        # else:
        #     # remove:
        #     ax.legend().remove()

        ax.set_xlim(0, xlim)
        if share_ylim:
            ax.set_ylim(0, ylim)
        ax.set_ylabel("# cells")

    plt.tight_layout()
    plt.show()


def plot_states_space_regions_with_maximal_difference(cell_type, reference_ana, response_ana, no_response_ana):

    features = reference_ana.pds.fetch(cell_type)[0]

    # remove from features all rows with values over 99th percentile per column:
    features = features[(features.iloc[:, 1:] < features.iloc[:, 1:].quantile(0.99)).all(axis=1)]

    p_res = response_ana.model.predict(cell_type=cell_type, obs="division", features=features)
    p_no = no_response_ana.model.predict(cell_type=cell_type, obs="division", features=features)

    max_diff_idxs = np.argsort(p_no - p_res)[-100:]
    # max_diff_idxs = np.argsort(p_no - p_res)[:100]
    max_diff_states = features.iloc[max_diff_idxs]

    print(
        "response proliferation rate at max diff states: ",
        p_res[max_diff_idxs].mean().round(2),
        "mean division rate: ",
        # p_res.mean().round(2),
        response_ana.rnds.fetch("Tu")[1].mean().round(2).item(),
    )
    print(
        "no-response proliferation rate at max diff states:",
        p_no[max_diff_idxs].mean().round(2),
        "mean division rate: ",
        # p_no.mean().round(2),
        no_response_ana.rnds.fetch("Tu")[1].mean().round(2).item(),
    )

    fig, axs = plt.subplots(figsize=(12, 1.5), ncols=len(CELL_TYPES))

    for i, c in enumerate(CELL_TYPES):

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

        if i == len(CELL_TYPES) - 1:
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
