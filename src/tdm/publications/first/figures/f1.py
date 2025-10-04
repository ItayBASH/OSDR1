import matplotlib.pyplot as plt
import seaborn as sns
from tdm.cell_types import FIBROBLAST, MACROPHAGE
from tdm.utils import microns
from tdm.simulate.tissue_step import TissueStep
from tdm.simulate.generate_distribution import get_tissue0
from tdm.publications.first.analyses import fm_analysis
from tdm.plot.two_cells.phase_portrait import plot_phase_portrait
from tdm.plot.tissue.stepper import plot_cells_over_time


# TODO run these in a notebook..


def fig_1a():
    fma = fm_analysis()
    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    plot_phase_portrait(fma, plot_nullclines=False, ax=ax)
    sns.despine(ax=ax)


def fig_1e():
    # fetch the model from the fibroblast-macrophage analysis:
    fma = fm_analysis()

    # tissue params:
    neighborhood_size = microns(fma.neighborhood_size)
    width, height = microns(500), microns(500)
    tissue0 = get_tissue0(FIBROBLAST, MACROPHAGE, 1, 1, neighborhood_size, width, height)

    # simulate 200 steps:
    stepper = TissueStep(tissue=tissue0, ana=fma, division_offset=1.0, diffusion_coeff=0)
    stepper.step_n_times(200)

    # plot:
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_cells_over_time(stepper=stepper, cell_types=fma.cell_types)
