from typing import Literal
from tdm.publications.first.analyses import fm_analysis, fm_tu_analysis
from tdm.analysis import Analysis
from tdm.raw.breast_mibi import read_single_cell_df, read_clinical_df
from tdm.raw.get_raw import BREAST_MIBI
from tdm.cell_types import (
    FIBROBLAST,
    MACROPHAGE,
    ENDOTHELIAL,
    TUMOR,
    T_CELL,
    B_CELL,
    FIBROBLAST_COLOR,
    MACROPHAGE_COLOR,
)
from tdm.plot.two_cells.phase_portrait import plot_phase_portrait, plot_growth_rate
from tdm.utils import log2_1p
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from tdm.style import set_style
from tdm.plot.model.calibration import plot_calibration

set_style()


def fig_3a():
    """
    Neighbor cell counts are strong predictors of division probability
    """

    MAX_P_DICT = {"T": 0.06, "B": 0.028, "F": 0.085, "M": 0.042, "Tu": 0.12, "En": 0.04}

    all_types = [TUMOR, T_CELL, B_CELL, ENDOTHELIAL, MACROPHAGE, FIBROBLAST]

    ana = Analysis(
        single_cell_df=read_single_cell_df(),
        cell_types_to_model=all_types,
        polynomial_dataset_kwargs={
            "degree": 2,
            "log_transform": True,
        },
        neighborhood_mode="extrapolate",
    )

    plot_calibration(ana, n_cells_per_bin=4000, max_p=MAX_P_DICT)
    plt.show()


def fig_3e():
    """
    Phase-portrait plot.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_phase_portrait(ana=fm_analysis(), plot_nullclines=True, ax=ax)
    plt.show()


def fig_3def(dataset: Literal["mibi", "imc"] = "mibi"):
    """
    Growth rate plot.
    """

    print("\nRunning.. \nPlease note: fig_3def() may take a few minutes to complete.")

    fma = fm_analysis()
    plot_growth_rate(
        ana=fma,
        kde_bw=0.35,
        cell_a_rbf_gamma=1.2,
        cell_b_rbf_gamma=0.2,
        include_titles=False,
        plot_nullclines=True,
        ylim=(0, 5.5),
    )
    plt.show()


def fig_3g():
    """
    Plot FM against tumor-density.
    """
    fmtu = fm_tu_analysis(BREAST_MIBI)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4))

    titles = ["No Tumor Neighbors", "With Tumor Neighbors (Mean)"]

    tus = [0, 64]  # no tumor cells, mean tumor cells
    for i in range(len(tus)):
        ax = axs[i]
        tu = tus[i]

        plot_phase_portrait(fmtu, fixed_cell_counts={"Tu": tu}, plot_nullclines=False, ax=ax)
        ax.set_title(titles[i])
        ax.set_ylim(0, 4)

    plt.show()


def fig_3h():
    """
    Perform Cox regression analysis on survival data and plot survival curves based on macrophage density.
    """
    fma1 = fm_analysis(BREAST_MIBI)
    clin_df1 = read_clinical_df()

    clin_df1 = clin_df1.rename(columns={"Overall Survival (Months)": "duration", "Patient ID": "subject_id"})

    clin_df1 = clin_df1.loc[:, ["subject_id", "Overall Survival Status", "duration"]]
    states1 = fma1.tissue_states()
    states1 = log2_1p(states1)

    states1["subject_id"] = [t.subject_id for t in fma1.tissues]
    states1 = states1.groupby("subject_id").mean().reset_index()
    df1 = states1.merge(clin_df1, on="subject_id", how="left").dropna()

    df1["event"] = df1["Overall Survival Status"].map(lambda x: 1 if "DECEASED" in x else 0)
    df1 = df1.drop(columns="Overall Survival Status")

    # Plot survival curves
    M_cutoff = 1.5
    group1 = df1[df1["M"] > M_cutoff]
    group2 = df1[df1["M"] <= M_cutoff]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(5, 5))

    # Fit and plot for group 1 (M > cutoff)
    kmf.fit(group1["duration"], event_observed=group1["event"], label="hot-fibrosis")
    kmf.plot_survival_function(color=MACROPHAGE_COLOR)
    print(f"Median survival time for hot-fibrosis: {kmf.median_survival_time_}")

    # Fit and plot for group 2 (M <= cutoff)
    kmf.fit(group2["duration"], event_observed=group2["event"], label="cold-fibrosis")
    kmf.plot_survival_function(color=FIBROBLAST_COLOR)
    print(f"Median survival time for cold-fibrosis: {kmf.median_survival_time_}")

    # Perform log-rank test
    results = logrank_test(
        group1["duration"], group2["duration"], event_observed_A=group1["event"], event_observed_B=group2["event"]
    )
    p_value = results.p_value

    # Display p-value and title
    plt.text(0.1, 0.1, f"p-value: {p_value:.4f}", transform=plt.gca().transAxes, fontsize=12)
    plt.title(f"Danenberg et. al (n = {len(df1)})")
    plt.xlim(0, 12 * 10)
    plt.ylim(0.3, 1.05)
    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.legend(frameon=False)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.show()
