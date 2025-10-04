import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tdm.preprocess.single_cell_df import CELL_TYPE_COL, DIVISION_COL, IMG_ID_COL


def plot_marker_distributions(
    single_cell_df: pd.DataFrame,
    marker: str | list[str],
    include_cell_types: list[str] | None = None,
    cell_type_col: str = CELL_TYPE_COL,
    ax: plt.Axes | None = None,
    axvline: float | None = None,
    xlim: float = 5,
    bw: float = 1.0,
    highlight_cell_types: list[str] | None = None,
):
    """Plot the distribution of a marker over multiple cell types.

    Args:
        ax (_type_): a matplotlib axis
        single_cell_df (pd.DataFrame): dataframe with row per cell, columns for cell type and marker
        marker (str): name of the column with marker values
        marker_cell_type (str): cell type that the marker is associated with
        cell_type_col (str): name of the column with cell types
        include_cells (list[str]): list of cell types to include in the plot
        axvline (float, optional): x value to draw a dotted red line at. Defaults to 0.5.
        xlim (float, optional): maximal x value to plot. Defaults to 5.
        bw (float, optional): kdeplot bw. Defaults to 0.2.
    """
    plot_when_done = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_when_done = True

    if include_cell_types:
        mask = np.isin(single_cell_df[cell_type_col], include_cell_types)
    else:
        mask = np.repeat(True, single_cell_df.shape[0])

    if highlight_cell_types is None:
        sns.kdeplot(
            single_cell_df[mask], x=marker, hue=cell_type_col, common_norm=False, bw_method=bw, ax=ax, legend=True
        )
    else:
        highlight_mask = np.isin(single_cell_df[cell_type_col], highlight_cell_types)

        # Get all unique hue values
        unique_hues = single_cell_df[mask & ~highlight_mask][cell_type_col].unique()
        palette = {hue: "lightgray" for hue in unique_hues}

        sns.kdeplot(
            single_cell_df[mask & ~highlight_mask],
            x=marker,
            hue=cell_type_col,
            common_norm=False,
            bw_method=bw,
            ax=ax,
            legend=True,
            palette=palette,
        )

        # Get all unique hue values
        unique_hues = single_cell_df[mask & highlight_mask][cell_type_col].unique()
        palette = {hue: "r" for hue in unique_hues}
        sns.kdeplot(
            single_cell_df[mask & highlight_mask],
            x=marker,
            hue=cell_type_col,
            common_norm=False,
            bw_method=bw,
            ax=ax,
            legend=True,
            palette=palette,
        )

    ax.set_xlim(0, xlim)
    ax.set_title(f"Marker = {marker}")

    if axvline is not None:
        ax.axvline(axvline, color="r", linestyle="--")

    sns.despine(ax=ax)

    if plot_when_done:
        plt.show()


def plot_fraction_of_dividing_cells(
    single_cell_df: pd.DataFrame,
    cell_type_col: str = CELL_TYPE_COL,
    division_col: str = DIVISION_COL,
):
    """Display the fraction of dividing cells of each type.

    Args:
        single_cell_df (pd.DataFrame): dataframe with row per cell, and columns for cell type and division.
        cell_type_col (str, optional): name of the cell-type column. Defaults to CELL_TYPE_COL.
        division_col (str, optional): name of the division column. Defaults to DIVISION_COL.
    """

    fraction_dividing_df = (
        single_cell_df.groupby(cell_type_col)
        .agg(
            fraction_dividing=(division_col, lambda x: x.mean()),
            n_dividing=(division_col, lambda x: x.sum()),
            n_cells=(cell_type_col, lambda x: len(x)),
        )
        .sort_values("fraction_dividing", ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(fraction_dividing_df, y="cell_type", x="fraction_dividing", ax=ax)

    # Add annotations on each bar
    for i, row in fraction_dividing_df.iterrows():
        ax.annotate(
            f"{int(row['n_dividing'])}/{int(row['n_cells'])}",  # Text to display
            xy=(row["fraction_dividing"] + 0.001, i),  # Position at (x, y)
            xytext=(3, 0),  # Slightly right from the bar end to avoid overlap
            textcoords="offset points",  # Offset in points
            va="center",  # Vertically center
            ha="left",  # Left align text
        )

    fig.suptitle("Fraction of dividing cells")
    ax.set_ylabel("cell type")
    ax.set_xlabel("fraction")

    ax.set_xticks(np.arange(0, max(fraction_dividing_df["fraction_dividing"]), 0.01))
    sns.despine(ax=ax)
    plt.show()


def plot_divisions_per_image(single_cell_df: pd.DataFrame):
    """Display the fraction of dividing cells from each image. Plots a pie-chart per cell-type.

    Args:
        single_cell_df (pd.DataFrame): _description_
    """

    _df = (
        single_cell_df.groupby([CELL_TYPE_COL, IMG_ID_COL])
        .agg(n_divisions=("division", lambda x: x.sum()))
        .reset_index()
        .groupby("cell_type")
    )

    for _, cell_type_df in iter(_df):

        # Sort images by fraction of dividing cells:
        top_dividing_tissues = cell_type_df.sort_values("n_divisions", ascending=False)
        top_dividing_tissues["n_divisions"] = (
            top_dividing_tissues["n_divisions"] / top_dividing_tissues["n_divisions"].sum()
        )

        # Pie chart parameters:
        values = top_dividing_tissues["n_divisions"]
        labels = top_dividing_tissues[IMG_ID_COL].astype(int)
        labels = np.where(values > 0.02, labels, "")

        # Plot pie:
        plt.figure(figsize=(4, 4))
        plt.pie(values, labels=labels)  # type: ignore

        # Add title:
        cell_type = top_dividing_tissues["cell_type"].to_numpy()[0]
        plt.title(f"Fraction of divisions from each image - cell type = {cell_type}")
        plt.show()
