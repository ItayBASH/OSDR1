import matplotlib.pyplot as plt

from tdm.publications.first.analyses import fm_analysis, tb_analysis
from tdm.raw.get_raw import BREAST_MIBI, BREAST_IMC, TRIPLE_NEGATIVE_IMC
from tdm.plot.two_cells import plot_phase_portrait
from typing import Literal


def map_dataset_name_to_full_title(dataset):

    if dataset == BREAST_MIBI:
        return "Danenberg, 2022 (n=693)"
    if dataset == BREAST_IMC:
        return "Fischer, 2023 (n=771)"
    if dataset == TRIPLE_NEGATIVE_IMC:
        return "Wang, 2023 (n=243)"


def plot_three_phase_portraits(cell_pair: Literal["fm", "tb"] = "fm"):
    if cell_pair == "fm":
        ana_func = fm_analysis
    elif cell_pair == "tb":
        ana_func = tb_analysis
    else:
        raise ValueError(f"Unknown cell pair: {cell_pair}")

    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))

    for i, dataset in enumerate([BREAST_MIBI, BREAST_IMC, TRIPLE_NEGATIVE_IMC]):
        ana = ana_func(dataset)

        ax = axs[i]
        plot_phase_portrait(ana, ax=ax, plot_nullclines=False)

        ax.set_title(map_dataset_name_to_full_title(dataset))
