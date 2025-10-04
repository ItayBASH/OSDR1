import matplotlib.pyplot as plt
import numpy as np
from tdm.dataset import PolynomialDataset, BootstrapDataset, ApplyModelDataset, Dataset
from tdm.cell_types import FIBROBLAST, MACROPHAGE
from tdm.model.example_models import get_models
from tdm.utils import microns
from tdm.simulate.generate_distribution import generate_cell_distribution_from_ground_truth_model
from tdm.plot.two_cells.phase_portrait import (
    _plot_phase_portrait,
    _plot_nullclines,
    plot_fixed_points,
    _plot_fixed_points,
)
from tdm.model import Model
from tdm.model.logistic_regression import LogisticRegressionModel
from tdm.numerical.phase_portrait_analysis import get_classified_fixed_points_inbounds


def plot_model_reconstruction(
    nds: Dataset,
    cell_a: str,
    cell_b: str,
    ax: plt.Axes,
    ground_truth_model: Model,
    model: type[Model] = LogisticRegressionModel,
    xlim: tuple[float, float] = (0, 8),
    ylim: tuple[float, float] = (0, 8),
    plot_first_phase_portrait: bool = True,
    n_repeats: int = 10,
    sample_size: int = 10000,
):
    """
    Plots the distribution of fixed points for n_tries bootstrap fits to samples of size sample_size.
    """
    for try_i in range(n_repeats):

        # resample 10K cells:
        bds = BootstrapDataset(
            nds,
            seed=42 + try_i,
            n_samples=sample_size,
        )

        # apply true model to resampled cells:
        bds = ApplyModelDataset(bds, ground_truth_model)

        # transform counts to features:
        pds = PolynomialDataset(
            bds, degree=2, log_transform=True
        )  # Ugly code warning: This matches the dataset used for the ground truth model, see model.example_models.py

        # fit model:
        bootstrap_model = model(pds, death_estimation="mean")

        # classify fixed points:
        fps = get_classified_fixed_points_inbounds(bootstrap_model, cell_a, cell_b, xlim, ylim)

        # plot fixed points for this repeat:
        _plot_fixed_points(fps, ax)

        # visualize phase portrait for one repeat:
        if plot_first_phase_portrait and try_i == 0:
            _plot_phase_portrait(
                model=bootstrap_model,
                cell_a=cell_a,
                cell_b=cell_b,
                xlim=xlim,
                ylim=ylim,
                ax=ax,
                streamplot_density=0.5,
                streamplot_linewidth=0.5,
            )


def fig_2d(n=10):
    """
    OSDR accurately reconstructs neighborhood dynamics from simulations of known dynamical circuits.
    """

    np.random.seed(42)

    neighborhood_size = microns(80)
    tissue_width = tissue_height = microns(500)  # 0.5mm by 0.5mm

    cell_a = FIBROBLAST
    cell_b = MACROPHAGE

    models = get_models()
    n_models = len(models)

    print(
        "Note: simulations are computationally intensive and may take several minutes to run (>15 minutes on some systems)"
    )
    model_datasets = []
    for ground_truth_model in models:
        nds, tissues = generate_cell_distribution_from_ground_truth_model(
            cell_a=cell_a,
            cell_b=cell_b,
            model=ground_truth_model,
            neighborhood_size=neighborhood_size,
            tissue_width=tissue_width,
            tissue_height=tissue_height,
            n_steps=35,  # 35 steps produces variance similar to empirical data
            n_initial_conditions=300,  # 300 random initial conditions is default
        )

        model_datasets.append((nds, tissues))

    for i in range(n_models):
        ground_truth_model = models[i]
        nds, tissues = model_datasets[i]

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(4, 2), sharex=True, sharey=True)

        # plot true model:
        ax = axs[0]
        _plot_phase_portrait(
            model=ground_truth_model,
            cell_a=FIBROBLAST,
            cell_b=MACROPHAGE,
            ax=ax,
            streamplot_density=0.5,
            xlim=(0, 8),
            ylim=(0, 8),
        )
        _plot_nullclines(
            model=ground_truth_model,
            cell_a=FIBROBLAST,
            cell_b=MACROPHAGE,
            ax=ax,
            xlim=(0, 8),
            ylim=(0, 8),
            fixed_cell_counts=None,
            linewidth=2,
        )
        plot_fixed_points(
            model=ground_truth_model,
            cell_a=FIBROBLAST,
            cell_b=MACROPHAGE,
            ax=ax,
            xlim=(0, 8),
            ylim=(0, 8),
            fixed_cell_counts=None,
        )

        ax.set_title("Known")
        ax.set_xlabel("")
        ax.set_ylabel("Cell B density")
        ax.set_xlabel("Cell A density")

        # plot estimates
        ax = axs[1]
        plot_model_reconstruction(
            nds=nds,
            cell_a=FIBROBLAST,
            cell_b=MACROPHAGE,
            ax=ax,
            ground_truth_model=ground_truth_model,
            model=LogisticRegressionModel,
            xlim=(0, 8),
            ylim=(0, 8),
            n_repeats=10,
        )
        ax.set_title("Inferred")
        ax.set_xlabel("Cell A density")
        ax.set_ylabel("")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)

        plt.show()


# MAX_P_DICT = {"T": 0.06, "B": 0.028, "F": 0.085, "M": 0.042, "Tu": 0.12, "En": 0.04}


# def get_fig2_model_and_pds():
#     nds = get_neighbors_datasets(ki67_thresholds=0.5, mode="extrapolate")
#     pds = PolynomialDataset(nds, degree=2)
#     m = LogisticRegressionModel(pds)

#     return m, pds


# def fig_2d():
#     m, pds = get_fig2_model_and_pds()

#     for c in m.cell_types():
#         fig, ax = plt.subplots(figsize=(2, 2))
#         plot_calibration(c, m, pds, 4000, ax, max_p=MAX_P_DICT[c])

#     return fig


# def sup_2a():
#     m, pds = get_fig2_model_and_pds()

#     F_summary = m.models["F"]["division"].summary()

#     pvals = pd.DataFrame(
#         {
#             "cell_type": m.cell_types(),
#             "LLR p-value": [m.models[c]["division"].llr_pvalue for c in m.cell_types()],
#         }
#     )

#     return F_summary, pvals


# def sup_2b():
#     ki67_threshold = 0.5
#     degree = 2

#     nds = get_neighbors_datasets(ki67_thresholds=ki67_threshold, mode="extrapolate")
#     pds = PolynomialDataset(nds, degree=degree)
#     m = LogisticRegressionModel(pds)

#     # load again so don't overwrite previous ones:
#     random_nds = get_neighbors_datasets(ki67_thresholds=ki67_threshold, mode="extrapolate")

#     for t in nds.cell_types():
#         counts, div = random_nds.dataset_dict[t]
#         random_div = div.iloc[np.random.permutation(len(div))].reset_index(drop=True)
#         random_nds.dataset_dict[t] = counts, random_div

#     random_pds = PolynomialDataset(random_nds, degree=degree)
#     random_m = LogisticRegressionModel(random_pds)

#     for c in m.cell_types():
#         fig, axs = plt.subplots(figsize=(6, 3), ncols=2)

#         ax = axs[0]
#         plot_calibration(
#             c,
#             m,
#             pds,
#             n_cells_per_bin=4000,
#             ax=ax,
#             max_p=MAX_P_DICT[c],
#             plot_min_max_lines=True,
#         )

#         ax = axs[1]
#         plot_calibration(
#             c,
#             random_m,
#             random_pds,
#             n_cells_per_bin=4000,
#             ax=ax,
#             max_p=MAX_P_DICT[c],
#             plot_min_max_lines=True,
#         )
#         ax.set_title("Calibration over permuted divisions")

#         fig.tight_layout()

#     return fig
