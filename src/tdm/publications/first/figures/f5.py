from tdm.raw.triple_negative_imc import (
    ON_TREATMENT,
    CHEMO,
    CHEMO_IMMUNO,
    RESPONSE,
    NO_RESPONSE,
)

from tdm.publications.first.triple_negative_plots import (
    plot_all_cell_types_over_time,
    cells_over_time_dfs,
    get_analysis,
    RESPONSE_COLOR,
    NO_RESPONSE_COLOR,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


def fig_5b():
    """
    Figure 5B: Dynamics of triple-negative breast cancer response to chemotherapy and chemotherapy + immunotherapy.
    """

    def get_response_dfs(treatment_arm):
        degree = 1
        biopsy_phase = ON_TREATMENT

        on_treatment_ana = get_analysis(
            biopsy_phase=biopsy_phase,
            treatment_arm=treatment_arm,
            response="all",
            degree=degree,
        )

        # responder analyses:
        response_on_treatment_ana = get_analysis(
            biopsy_phase=biopsy_phase,
            treatment_arm=treatment_arm,
            response=RESPONSE,
            degree=degree,
        )

        # non-responder analyses:
        no_response_on_treatment_ana = get_analysis(
            biopsy_phase=biopsy_phase,
            treatment_arm=treatment_arm,
            response=NO_RESPONSE,
            degree=degree,
        )

        # ALL states from early on-treatment:
        initial_states = on_treatment_ana.tissue_states()
        timepoints = np.linspace(0, 1000, 100)

        response_sols_df, no_response_sols_df = cells_over_time_dfs(
            initial_states=initial_states,
            timepoints=timepoints,
            response_ana=response_on_treatment_ana,
            no_response_ana=no_response_on_treatment_ana,
        )

        return response_sols_df, no_response_sols_df

    print(
        "Note: This function is computationally intensive and may take several minutes to complete (>10 minutes on some systems)"
    )

    # chemo:
    response_sols_df, no_response_sols_df = get_response_dfs(CHEMO)
    plot_all_cell_types_over_time(response_sols_df, no_response_sols_df, ylim=60)

    # chemo and immuno:
    response_sols_df, no_response_sols_df = get_response_dfs(CHEMO_IMMUNO)
    plot_all_cell_types_over_time(response_sols_df, no_response_sols_df, ylim=60)


def fig_5c():
    """
    Comparison of tumor division rate between high T-cell neighborhoods and all neighborhoods.
    """
    q = 0.9

    treatment_arm = CHEMO_IMMUNO
    degree = 1
    biopsy_phase = ON_TREATMENT

    # responder analyses:
    response_on_treatment_ana = get_analysis(
        biopsy_phase=biopsy_phase,
        treatment_arm=treatment_arm,
        response=RESPONSE,
        degree=degree,
    )

    # non-responder analyses:
    no_response_on_treatment_ana = get_analysis(
        biopsy_phase=biopsy_phase,
        treatment_arm=treatment_arm,
        response=NO_RESPONSE,
        degree=degree,
    )

    # all neighborhoods:
    _, all_nbrhoods_response_obs = response_on_treatment_ana.rnds.fetch("Tu")
    _, all_nbrhoods_no_response_obs = no_response_on_treatment_ana.rnds.fetch("Tu")

    response_counts, response_obs = response_on_treatment_ana.rnds.fetch("Tu")
    mask = response_counts["T"] > response_counts["T"].quantile(q)
    response_counts = response_counts[mask]
    response_obs = response_obs[mask].loc[:, ["division"]]

    no_response_counts, no_response_obs = no_response_on_treatment_ana.rnds.fetch("Tu")
    mask = no_response_counts["T"] > no_response_counts["T"].quantile(q)
    no_response_counts = no_response_counts[mask]
    no_response_obs = no_response_obs[mask].loc[:, ["division"]]

    high_t_response_obs = response_obs.copy()
    high_t_no_response_obs = no_response_obs.copy()

    all_nbrhoods_response_obs = all_nbrhoods_response_obs.copy()
    all_nbrhoods_no_response_obs = all_nbrhoods_no_response_obs.copy()
    high_t_response_obs = high_t_response_obs.copy()
    high_t_no_response_obs = high_t_no_response_obs.copy()

    # statistical tests:
    all_res_vs_no = ttest_ind(
        all_nbrhoods_response_obs.astype(float), all_nbrhoods_no_response_obs.astype(float)
    ).pvalue.item()
    all_res_vs_high_t_res = ttest_ind(
        all_nbrhoods_response_obs.astype(float), high_t_response_obs.astype(float)
    ).pvalue.item()
    all_no_vs_high_t_no = ttest_ind(
        all_nbrhoods_no_response_obs.astype(float), high_t_no_response_obs.astype(float)
    ).pvalue.item()
    high_t_res_vs_no = ttest_ind(high_t_response_obs.astype(float), high_t_no_response_obs.astype(float)).pvalue.item()

    print("t-test p-values:")
    print(f"All neighborhoods - response vs no: {all_res_vs_no:.3e}")
    print(f"High T-cell neighborhoods - response vs no: {high_t_res_vs_no:.3e}")
    print(f"Response - all vs high T-cell: {all_res_vs_high_t_res:.3e}")
    print(f"No Response - all vs high T-cell: {all_no_vs_high_t_no:.3e}")

    # dataframes:
    all_nbrhoods_response_obs["group"] = "response"
    all_nbrhoods_no_response_obs["group"] = "no response"
    all_df = pd.concat([all_nbrhoods_response_obs, all_nbrhoods_no_response_obs])

    high_t_response_obs["group"] = "response"
    high_t_no_response_obs["group"] = "no response"
    high_t_df = pd.concat([high_t_response_obs, high_t_no_response_obs])

    # barplot:

    fig, axs = plt.subplots(figsize=(6, 4), ncols=2, sharey=True)

    ax = axs[0]
    sns.barplot(
        all_df,
        x="group",
        y="division",
        hue="group",
        palette=[RESPONSE_COLOR, NO_RESPONSE_COLOR],
        err_kws={"color": "black"},
        ax=ax,
    )
    ax.set_title("All Neighborhoods")

    ax = axs[1]
    sns.barplot(
        high_t_df,
        x="group",
        y="division",
        hue="group",
        palette=[RESPONSE_COLOR, NO_RESPONSE_COLOR],
        err_kws={"color": "black"},
        ax=ax,
    )
    ax.set_title("High T-cell Neighborhoods")

    for ax in axs:
        sns.despine(ax=ax)
        ax.set_xlabel("")

    axs[0].set_ylabel("Ki67+ cancer cells (fraction)")

    plt.show()
