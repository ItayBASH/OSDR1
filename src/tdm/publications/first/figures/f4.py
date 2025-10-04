from tdm.utils import microns
from tdm.publications.first.analyses import tb_analysis, get_3d_tb_ana
from tdm.plot.two_cells.phase_portrait import plot_growth_rate, plot_trajectory
from tdm.simulate.generate_distribution import simulate_one_tissue, get_tissue0
from tdm.plot.tissue.stepper import plot_cells_over_time
from tdm.plot.three_cells.trajectory import plot_3D_trajectory
from tdm.plot.model.interpret import plot_high_and_low_proliferation_states
import numpy as np
import matplotlib.pyplot as plt


def fig_4abc():

    print("\nRunning.. \nPlease note: fig_4abc() may take a few minutes to complete.")

    ana = tb_analysis()
    fig = plot_growth_rate(ana, plot_nullclines=False, include_titles=False)

    # add one immune-flare trajectory over the phase-portrait:
    ax = fig.axes[0]
    odeint_timepoints = np.linspace(0, 3500, 10000)

    # 80 microns:
    arrow_timepoint_idxs = [0, 1250, 1900, 2800, 5900]

    state0 = (4.0, 0.001)  # T,B

    plot_trajectory(
        ana, state0=state0, odeint_timepoints=odeint_timepoints, arrow_timepoint_idxs=arrow_timepoint_idxs, ax=ax
    )

    plt.show()


def fig_4d():
    """A single adaptive immune flare."""
    np.random.seed(42)

    tba = tb_analysis()

    n_steps = 2000
    tissue_width = tissue_height = microns(2000)
    state0 = (4.5, 1.0)

    # perform cell-population simulation:
    stepper = simulate_one_tissue(
        cell_a=tba.cell_a,
        cell_b=tba.cell_b,
        model=tba.model,
        neighborhood_size=tba.neighborhood_size,
        n_steps=n_steps,
        tissue_width=tissue_width,
        tissue_height=tissue_height,
        state0=state0,
        return_stepper=True,
        verbose=True,
        diffusion_coef=0,
    )

    # plot cells over time:
    fig, ax = plt.subplots(figsize=(5, 2))

    # _correction_factor = correction_factor(tissue_width, tissue_height, nbrhood_size, mode="to_neighborhood")
    plot_cells_over_time(
        stepper=stepper,
        cell_types=["T", "B"],
        log_cells=False,
        legend_loc="upper right",
        ax=ax,
        ylim=(0, 6.2),
        n_steps=1500,
    )

    plt.show()


def fig_4e():
    """Refractory period plot."""

    np.random.seed(42)

    tba = tb_analysis()

    nbrhood_size = tba.neighborhood_size
    tissue_width = tissue_height = microns(2000)
    state0 = (4.5, 0.5)

    stepper = simulate_one_tissue(
        cell_a=tba.cell_a,
        cell_b=tba.cell_b,
        model=tba.model,
        neighborhood_size=tba.neighborhood_size,
        n_steps=1000,
        tissue_width=tissue_width,
        tissue_height=tissue_height,
        state0=state0,
        return_stepper=True,
        verbose=True,
        diffusion_coef=0,
    )

    # ineffective pulse @ 1000
    last_tissue = stepper.tissues[-1]
    T_influx_tissue = get_tissue0(
        cell_a=tba.cell_a,
        cell_b=tba.cell_b,
        start_a=4.5,
        start_b=0,
        nbrhood_size=nbrhood_size,
        tissue_width=tissue_width,
        tissue_height=tissue_height,
    )
    stepper.tissues[-1] = last_tissue + T_influx_tissue
    stepper.step_n_times(1500, verbose=True)

    # ineffective pulse @ 2500
    last_tissue = stepper.tissues[-1]
    T_influx_tissue = get_tissue0(
        cell_a=tba.cell_a,
        cell_b=tba.cell_b,
        start_a=4.5,
        start_b=0,
        nbrhood_size=nbrhood_size,
        tissue_width=tissue_width,
        tissue_height=tissue_height,
    )
    stepper.tissues[-1] = last_tissue + T_influx_tissue
    stepper.step_n_times(1000, verbose=True)

    fig, ax = plt.subplots(figsize=(5, 2))
    plot_cells_over_time(
        stepper=stepper,
        cell_types=["T", "B"],
        log_cells=False,
        legend_loc="upper right",
        ax=ax,
        ylim=(0, 6.2),
    )
    ax.get_legend().remove()  # type: ignore

    plt.show()


def fig_4f():
    """Plot 3D trajectory of T and B cell dynamics."""
    ana = get_3d_tb_ana()

    # Plot trajectory with initial conditions and time points
    plot_3D_trajectory(
        ana, initial_cell_counts=[40, 4, 1], odeint_timepoints=np.linspace(0, 5000, 10000), logspace=False
    )

    plt.show()


def fig_4g():
    """Plot high and low proliferation states for T and B cells."""
    ana = get_3d_tb_ana()
    return plot_high_and_low_proliferation_states(ana, ylim=0.12, xlim=120)


"""
OLD CODE BELOW
"""

# def fig_4d():
#     tba = TBAnalysis()
#     m, rnds, pds = tba.fit_model(return_datasets=True)
#     fig, ax = plt.subplots(figsize=(4, 4))
#     visualize_model(m, T_CELL, B_CELL, ax, None, xlim=tba.xlim, ylim=tba.ylim, _plot_nullclines=False)
#     return fig


# def fig_4e():
#     """A single adaptive immune flare."""
#     np.random.seed(42)

#     tba = TBAnalysis()
#     m = tba.fit_model()

#     nbrhood_size = microns(tba.neighborhood_size)
#     n_steps = 2000
#     tissue_width = tissue_height = microns(2000)
#     state0 = (4.5, 1.0)
#     _correction_factor = correction_factor(tissue_width, tissue_height, nbrhood_size, mode="to_neighborhood")

#     stepper = simulate_one_tissue(
#         tba.cell_a,
#         tba.cell_b,
#         model=m,
#         nbrhood_size=nbrhood_size,
#         n_steps=n_steps,
#         tissue_width=tissue_width,
#         tissue_height=tissue_height,
#         state0=state0,
#         return_stepper=True,
#         verbose=True,
#     )

#     fig, ax = plt.subplots(figsize=(5, 2))
#     plot_cells_over_time(
#         stepper=stepper,
#         cell_types=["T", "B"],
#         correction_factor=_correction_factor,
#         log_cells=False,
#         legend_loc="upper right",
#         ax=ax,
#         ylim=(0, 6.2),
#         n_steps=1500,
#     )

#     return fig
#     # ax.legend_.remove()


# def fig_4f():
#     """Refractory period plot."""

#     np.random.seed(42)

#     tba = TBAnalysis()
#     m = tba.fit_model()

#     nbrhood_size = microns(tba.neighborhood_size)
#     tissue_width = tissue_height = microns(2000)
#     state0 = (4.5, 0.5)
#     _correction_factor = correction_factor(tissue_width, tissue_height, nbrhood_size, mode="to_neighborhood")

#     stepper = simulate_one_tissue(
#         tba.cell_a,
#         tba.cell_b,
#         model=m,
#         nbrhood_size=nbrhood_size,
#         n_steps=1000,
#         tissue_width=tissue_width,
#         tissue_height=tissue_height,
#         state0=state0,
#         return_stepper=True,
#         verbose=True,
#     )

#     # ineffective pulse @ 1000
#     last_tissue = stepper.tissues[-1]
#     T_influx_tissue = get_tissue0(tba.cell_a, tba.cell_b, 4.5, 0, nbrhood_size, tissue_width, tissue_height)
#     updated_cell_df = pd.concat([last_tissue.cell_df(), T_influx_tissue.cell_df()])
#     last_tissue.set_cell_df(updated_cell_df)
#     stepper.step_n_times(1500, verbose=True)

#     # ineffective pulse @ 1500
#     last_tissue = stepper.tissues[-1]
#     T_influx_tissue = get_tissue0(tba.cell_a, tba.cell_b, 4.5, 0, nbrhood_size, tissue_width, tissue_height)
#     updated_cell_df = pd.concat([last_tissue.cell_df(), T_influx_tissue.cell_df()])
#     last_tissue.set_cell_df(updated_cell_df)
#     stepper.step_n_times(1000, verbose=True)

#     fig, ax = plt.subplots(figsize=(5, 2))
#     plot_cells_over_time(
#         stepper=stepper,
#         cell_types=["T", "B"],
#         correction_factor=_correction_factor,
#         log_cells=False,
#         legend_loc="upper right",
#         ax=ax,
#         ylim=(0, 6.2),
#     )
#     ax.legend_.remove()

#     return fig


# def sup_4b(n=30):
#     """Cell-level level bootstrap.

#     Args:
#         n (int, optional): Number of samples. Defaults to 30.
#     """
#     tba = TBAnalysis()
#     return two_cell_fixed_point_bootstrap(tba, n_tries=n)


# def sup_4c(n=30):
#     """Sample level bootstrap.

#     Args:
#         n (int, optional): Number of samples. Defaults to 30.
#     """
#     tba = TBAnalysis()
#     return patient_level_two_cell_fixed_point_bootstrap(tba, n_tries=n)


# def sup_4d():
#     tba = TBAnalysis()
#     return plot_death_prob_perturbations(tba)


# def sup_4e():
#     tba = TBAnalysis()
#     plot_n_divisions_per_threshold(tba, mode="count")


# def sup_4f():
#     from tdm.raw.breast_mibi import NBRHOOD_SIZES, KI67_THRESHOLD_MAP

#     nbrhood_sizes = NBRHOOD_SIZES[1:10]
#     ki67_thresholds = list(KI67_THRESHOLD_MAP.keys())[2:7]
#     tba = TBAnalysis()
#     visualize_model_over_parameter_grid(tba, ki67_thresholds, nbrhood_sizes)


# def sup_4g():
#     from tdm.eval.two_cell_dynamics import plot_dynamics_against_third_cell
#     from tdm.analysis.TB import TBAnalysis

#     tba = TBAnalysis()
#     plot_dynamics_against_third_cell(tba, "Tu")
#     plot_dynamics_against_third_cell(tba, "F")
#     plot_dynamics_against_third_cell(tba, "M")
#     plot_dynamics_against_third_cell(tba, "En")
