from tdm.tissue import Tissue
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy

# do i want this dependency?
from tdm.cell_types import (
    CELL_TYPES_ARRAY,
    CELL_TYPE_COLORS_ARRAY,
    CELL_TYPE_TO_FULL_NAME,
    CELL_FULL_NAME_TO_COLOR,
)


def plot_tissue(
    tissue: Tissue,
    all_on_one_plot: bool = True,
    cell_subset: list[str] | None = None,
    ax: plt.Axes | None = None,
    plot_effective_capture_area=False,
    neighborhood_size: float | None = None,
    circle_xy: tuple | None = None,
    circle_r: float | None = None,
):
    """ """
    cell_df = tissue.cell_df().copy()

    if cell_subset is not None:
        cell_df = cell_df.loc[cell_df.cell_type.isin(cell_subset), :]

    # from units of meters (e.g 1 micron = 1e-6) to milli-meters:
    cell_df.loc[:, "x [mm]"] = cell_df["x"] / 1e-3
    cell_df.loc[:, "y [mm]"] = cell_df["y"] / 1e-3

    if all_on_one_plot:
        if ax is None:
            w, h = tissue.tissue_dimensions()
            width = 4
            height = width / w * h
            fig, ax = plt.subplots(figsize=(width, height))

        ax.set_title("Spatial distribution of cells")

        sns.scatterplot(
            cell_df,
            y="y [mm]",
            x="x [mm]",
            edgecolor="#252526",
            s=17,
            linewidth=0.5,
            hue=cell_df.cell_type.map(CELL_TYPE_TO_FULL_NAME),
            palette=CELL_FULL_NAME_TO_COLOR,
            ax=ax,
        )

        # legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False).set_title(
            ""
        )  # remove "cell_type" from legend
        sns.despine(ax=ax)

        w, h = tissue.tissue_dimensions()
        w_mm, h_mm = w / 1e-3, h / 1e-3

        ax.set_xlim(0, w_mm)
        ax.set_ylim(0, h_mm)

    else:
        g = sns.FacetGrid(
            cell_df,
            col="cell_type",
            col_order=CELL_TYPES_ARRAY,
            col_wrap=3,
            sharex=True,
            sharey=True,
            height=5,
            aspect=1,
            hue="cell_type",
            hue_order=CELL_TYPES_ARRAY,
            hue_kws={"color": CELL_TYPE_COLORS_ARRAY},
        )

        g.map_dataframe(
            sns.scatterplot,
            y="y [mm]",
            x="x [mm]",
            edgecolor="#252526",
            s=17,
            linewidth=0.5,
        )

    if plot_effective_capture_area:
        if neighborhood_size is None:
            raise UserWarning("Must provide neighborhood_size if plot_effective_capture_area=True")
        else:
            ns = neighborhood_size / 1e-3
            x_max, y_max = tissue.tissue_dimensions()
            x_max, y_max = x_max / 1e-3, y_max / 1e-3

            x_min, y_min = ns, ns
            x_max, y_max = x_max - ns, y_max - ns

            if y_min > y_max:
                print("y_min > y_max! Capture Area plot misleading! Aborting..")
                return g

            if x_min > x_max:
                print("x_min > x_max! Capture Area plot misleading! Aborting..")
                return g

        if not all_on_one_plot:
            for _ax in g.axes.flat:
                _ax.vlines([x_min, x_max], ymin=y_min, ymax=y_max, color="red")
                _ax.hlines([y_min, y_max], xmin=x_min, xmax=x_max, color="red")
        else:
            ax.vlines([x_min, x_max], ymin=y_min, ymax=y_max, color="red")  # type: ignore
            ax.hlines([y_min, y_max], xmin=x_min, xmax=x_max, color="red")  # type: ignore

    # add circle:
    if (circle_xy is not None) and (circle_r is not None):

        circle_x_mm = circle_xy[0] / 1e-3
        circle_y_mm = circle_xy[1] / 1e-3
        circle_r_mm = circle_r / 1e-3

        circle = patches.Circle(
            (circle_x_mm, circle_y_mm),
            circle_r_mm,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
            linestyle="--",
        )

        if not all_on_one_plot:
            for _ax in g.axes.flat:
                _ax.add_patch(deepcopy(circle))
        else:
            ax.add_patch(circle)  # type: ignore
