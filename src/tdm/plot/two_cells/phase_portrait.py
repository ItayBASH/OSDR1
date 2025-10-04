from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
from tdm.utils import log2_1p, inv_log2_1p
from tdm.cell_types import CELL_TYPE_TO_COLOR, CELL_TYPE_TO_FULL_NAME, FIBROBLAST_COLOR
from tdm.analysis import Analysis
from tdm.dataset import NeighborsDataset
from tdm.model import Model
from tdm.style import STABILITIY_TYPE_TO_MARKER_STYLE_KWARGS, STABLE, SEMI_STABLE, UNSTABLE
from tdm.plot.utils import log_density_axis
from tdm.numerical.phase_portrait_analysis import (
    get_nullclines,
    compute_change_in_each_direction,
    get_classified_fixed_points_inbounds,
)

from tdm.simulate.deterministic import compute_trajectory


def plot_bla():

    pass


def plot_phase_portrait(
    ana: Analysis,
    plot_nullclines: bool = True,
    cell_a: str | None = None,
    cell_b: str | None = None,
    fixed_cell_counts: dict[str, float] | None = None,
    phase_portrait_style: Literal["stream", "quiver"] = "stream",
    mode: Literal["cells", "rates"] = "cells",
    streamplot_density: float = 0.6,
    nullcline_width: float = 1.5,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    step: float = 0.35,
    ax: plt.Axes | None = None,
    nds_for_kdeplot: NeighborsDataset | None = None,
    streamplot_color: str = "black",
    add_fixed_points: bool = True,
):
    """Plot a 2D phase-portrait with nullclines and fixed-points.

    Args:
        ana (Analysis): _description_
        plot_nullclines (bool, optional): _description_. Defaults to True.
        fixed_cell_counts (dict[str, float] | None, optional): _description_. Defaults to None.
        phase_portrait_style (Literal[&quot;stream&quot;, &quot;quiver&quot;], optional): _description_.
            Defaults to "stream".
        mode (Literal[&quot;cells&quot;, &quot;rates&quot;], optional): _description_. Defaults to "cells".
        streamplot_density (float, optional): _description_. Defaults to 0.6.
        nullcline_width (float, optional): _description_. Defaults to 1.5.
        xlim (tuple[float, float] | None, optional): _description_. Defaults to None.
        ylim (tuple[float, float] | None, optional): _description_. Defaults to None.
        ax (plt.Axes | None, optional): _description_. Defaults to None.
        nds_for_kdeplot (NeighborsDataset | None, optional): _description_. Defaults to None.

    Examples:
        >>> from tdm.analysis import Analysis
        >>> from tdm.plot.two_cells.phase_portrait import plot_phase_portrait
        >>> ana = Analysis.load("fm.pkl")
        >>> plot_phase_portrait(ana)
    """
    model = ana.model
    cell_a = cell_a or ana.cell_a
    cell_b = cell_b or ana.cell_b
    xlim = xlim or ana.xlim
    ylim = ylim or ana.ylim

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    _plot_phase_portrait(
        model=model,
        cell_a=cell_a,
        cell_b=cell_b,
        xlim=xlim,
        ylim=ylim,
        ax=ax,
        fixed_cell_counts=fixed_cell_counts,
        style=phase_portrait_style,
        mode=mode,
        streamplot_density=streamplot_density,
        streamplot_color=streamplot_color,
        step=step,
    )

    if plot_nullclines:
        _plot_nullclines(
            model=model,
            cell_a=cell_a,
            cell_b=cell_b,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            fixed_cell_counts=fixed_cell_counts,
            linewidth=nullcline_width,
        )

    if nds_for_kdeplot is not None:
        plot_two_cell_density(nds=nds_for_kdeplot, cell_a=cell_a, cell_b=cell_b, ax=ax)

    if add_fixed_points:
        plot_fixed_points(
            model=model, cell_a=cell_a, cell_b=cell_b, ax=ax, xlim=xlim, ylim=ylim, fixed_cell_counts=fixed_cell_counts
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Phase Portrait")


def plot_growth_rate(
    ana: Analysis,
    kde_bw: float = 0.3,
    cell_a_rbf_gamma: float = 1.2,  # value for F in FM analysis
    cell_b_rbf_gamma: float = 0.3,  # value for M in FM analysis
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fixed_cell_counts: dict[str, float] | None = None,
    include_titles: bool = True,
    plot_nullclines: bool = True,
    streamplot_density: float = 0.6,
    _plot_contours: bool = True,
) -> Figure:
    """Display the growth rate of both cell-types over the phase-plane.
        Plots 3 axes: growth rate for each cell-type and a phase-portrait.

    Hint:
        **How to interpret these plots?**

        The growth rate is computed by taking a smoothed mean of the binary cell-division events over the phase-plane.
        We then subtract the mean division rate so that areas with high proliferation are positive (plotted red)
        and areas with low proliferation are negative (plotted blue).
        See `paper <https://www.biorxiv.org/content/10.1101/2024.04.22.590503v1>`_ for more on why the mean division
        rate is a good approximation of the death rate.

        For ``cell_a`` (x axis), regions with positive growth (red) correspond with flow to the right on the
        phase-portrait, and regions with negative growth (blue) correspond with flow to the left.
        For ``cell_b`` (y axis), positive and negative growth correspond with flow up or down respectively.

        Thus, this plot connects the measured division events directly with the modeled dynamics.


    Args:
        ana (Analysis): _description_
        kde_bw (float, optional): _description_. Defaults to 0.3.
        cell_a_rbf_gamma (float, optional): _description_. Defaults to 1.2.
        ylim (tuple[float, float] | None, optional): _description_. Defaults to None.
        fixed_cell_counts (dict[str, float] | None, optional): _description_. Defaults to None.
        include_titles (bool, optional): _description_. Defaults to True.
        plot_nullclines (bool, optional): _description_. Defaults to True.
        streamplot_density (float, optional): _description_. Defaults to 0.6.
        _plot_contours (bool, optional): _description_. Defaults to True.

    Returns:
        Figure: _description_
    """

    model = ana.model
    nds = ana.rnds
    cell_a = ana.cell_a
    cell_b = ana.cell_b
    xlim = xlim or ana.xlim
    ylim = ylim or ana.ylim

    fixed_cell_counts = fixed_cell_counts or {}
    fig, axs = plt.subplots(ncols=3, figsize=(10, 5), sharex=False, sharey=False)
    ax0, ax1, ax2 = axs

    nbrs_a, obs_a = nds.fetch(cell_a)
    nbrs_b, obs_b = nds.fetch(cell_b)
    nbrs_a, nbrs_b = log2_1p(nbrs_a), log2_1p(nbrs_b)

    for ax in axs:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # ax1 - cell a kde plot:
    _plot_growth_rate_axis(
        nbrs_a,
        obs_a,
        cell_a,
        cell_b,
        kde_bw=kde_bw,
        rbf_gamma=cell_a_rbf_gamma,  # 0.1 for restricted dataset, 1.2 for fibroblast macrophage analysis
        ax=ax1,
        _plot_contours=_plot_contours,
    )
    ax1.set_title(f"{CELL_TYPE_TO_FULL_NAME[cell_a]} growth Rate")

    # ax2 - cell b kde plot:
    _plot_growth_rate_axis(
        nbrs_b,
        obs_b,
        cell_a,
        cell_b,
        kde_bw=kde_bw,
        rbf_gamma=cell_b_rbf_gamma,  # 0.8 for restricted dataset
        ax=ax2,
        _plot_contours=_plot_contours,
    )
    ax2.set_title(f"{CELL_TYPE_TO_FULL_NAME[cell_b]} growth Rate")

    # ax0 - two-cell Phase Portrait
    plot_phase_portrait(
        ana,
        streamplot_density=streamplot_density,
        ax=ax0,
        fixed_cell_counts=fixed_cell_counts,
        plot_nullclines=plot_nullclines,
        ylim=ylim,
    )

    # plot nullclines over all axes:
    if plot_nullclines:
        _plot_nullclines(
            model,
            cell_a,
            cell_b,
            ax0,
            xlim,
            ylim,
            fixed_cell_counts,
        )
        _plot_nullclines(
            model,
            cell_a,
            cell_b,
            ax1,
            xlim,
            ylim,
            fixed_cell_counts,
            plot_b=False,
        )
        _plot_nullclines(
            model,
            cell_a,
            cell_b,
            ax2,
            xlim,
            ylim,
            fixed_cell_counts,
            plot_a=False,
        )

    plot_fixed_points(
        model,
        cell_a,
        cell_b,
        ax0,
        xlim,
        ylim,
        fixed_cell_counts,
    )

    # plot_fixed_points(model, cell_a, cell_b, ax0)
    _add_fixed_point_legend(ax0)

    if not include_titles:
        for ax in axs:
            ax.set_title("")

    fig.tight_layout()

    return fig


def plot_trajectory(
    ana: Analysis,
    state0: tuple[float, float],
    odeint_timepoints: np.ndarray,
    arrow_timepoint_idxs: list[int],
    color: str = "#3B76BC",
    ax: plt.Axes | None = None,
    zorder: int = 100,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    initial_cell_counts = inv_log2_1p(np.array(state0))
    sol = compute_trajectory(ana, initial_cell_counts=initial_cell_counts, odeint_timepoints=odeint_timepoints)

    # we log-transform for plotting:
    sol = log2_1p(sol)
    x_vals = sol[:, 0]
    y_vals = sol[:, 1]

    # plot trajectory line:
    ax.plot(x_vals, y_vals, color=color, linewidth=4, zorder=zorder)

    # add arrows to accentuate direction of trajectory:
    for i in arrow_timepoint_idxs:
        dx = x_vals[i + 1] - x_vals[i]
        dy = y_vals[i + 1] - y_vals[i]

        ax.arrow(
            x_vals[i],
            y_vals[i],
            dx,
            dy,
            shape="full",
            lw=1.0,
            length_includes_head=False,
            head_width=0.5,
            facecolor="black",
            zorder=zorder,
            edgecolor=None,
            clip_on=False,
        )


def plot_tissues_over_phase_portrait(
    ana: Analysis,
    ax: plt.Axes | None = None,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    plot_phase_portrait(
        ana,
        ax=ax,
        streamplot_color="black",
        mode="rates",
        plot_nullclines=False,
    )

    sns.scatterplot(
        log2_1p(ana.tissue_states()),
        x=ana.cell_a,
        y=ana.cell_b,
        c=FIBROBLAST_COLOR,
        edgecolor="black",
        zorder=-10,
        ax=ax,
    )

    sns.despine(ax=ax)
    plt.show()


"""
Internal plotting functions:
"""


def _plot_phase_portrait(
    model: Model,
    cell_a: str,
    cell_b: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: plt.Axes,
    fixed_cell_counts: dict[str, float] | None = None,
    style: Literal["stream", "quiver"] = "stream",
    mode: Literal["cells", "rates"] = "cells",
    max_density: int | dict | None = None,
    streamplot_density: float = 0.6,
    streamplot_linewidth: float = 1.0,
    streamplot_color: str = "black",
    quiver_arrow_scale: float | None = None,
    step: float = 0.35,
):
    """_summary_

    Args:
        model (Model): _description_
        cell_a (str): _description_
        cell_b (str): _description_
        xlim (tuple[float, float]): _description_
        ylim (tuple[float, float]): _description_
        ax (plt): _description_
        fixed_cell_counts (dict[str, float] | None, optional): _description_. Defaults to None.
        style (Literal[&quot;stream&quot;, &quot;quiver&quot;], optional): _description_. Defaults to "stream".
        mode (Literal[&quot;cells&quot;, &quot;rates&quot;], optional): _description_. Defaults to "cells".
        max_density (int | dict | None, optional): maximal density of cells (untransformed count). Defaults to None.
        streamplot_density (float, optional): _description_. Defaults to 0.6.
        streamplot_linewidth (float, optional): _description_. Defaults to 1.0.
        streamplot_color (str, optional): _description_. Defaults to "black".
        quiver_arrow_scale (float | None, optional): _description_. Defaults to None.
        step (float, optional): _description_. Defaults to 0.35.

    Raises:
        ValueError: _description_
    """
    # x,y,u,v is typical streamplot naming for meshgrid locations x,y and
    # rates of change in x and y directions u and v respectively
    x, y, u, v = compute_change_in_each_direction(
        model, cell_a, cell_b, xlim, ylim, fixed_cell_counts, step=step, mode=mode
    )

    if style == "stream":
        ax.streamplot(
            x,
            y,
            u,
            v,
            broken_streamlines=True,
            color=streamplot_color,
            density=streamplot_density,
            linewidth=streamplot_linewidth,
        )
    elif style == "quiver":
        # plot arrows beginning within max density:
        alpha = None
        if max_density is not None:
            abs_x = inv_log2_1p(x).ravel()
            abs_y = inv_log2_1p(y).ravel()

            if np.isscalar(max_density):
                alpha = abs_x + abs_y <= max_density

            elif isinstance(max_density, dict):
                alpha = (abs_x <= max_density[cell_a]) & (abs_y <= max_density[cell_b])

        ax.quiver(
            x,
            y,
            u,
            v,
            color=streamplot_color,
            angles="uv",
            alpha=alpha,
            scale=quiver_arrow_scale,
        )
    else:
        raise ValueError(f"{style} is not a valid phase portrait mode")

    ax.set_title("Phase-Portrait")
    ax.set_xlabel(log_density_axis(cell_a))
    ax.set_ylabel(log_density_axis(cell_b))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    sns.despine(ax=ax)


def _plot_nullclines(
    model: Model,
    cell_a: str,
    cell_b: str,
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    fixed_cell_counts: dict[str, float] | None = None,
    plot_a: bool = True,
    plot_b: bool = True,
    linewidth: float = 1.5,
):
    a_mesh, b_mesh, dadt, dbdt = compute_change_in_each_direction(
        model, cell_a, cell_b, xlim, ylim, fixed_cell_counts, step=0.01
    )

    a_nullclines, b_nullclines = get_nullclines(a_mesh, b_mesh, dadt, dbdt)

    if plot_a:
        for g in a_nullclines.geoms:
            x, y = g.xy
            ax.plot(x, y, color=CELL_TYPE_TO_COLOR[cell_a], linewidth=linewidth)

    if plot_b:
        for g in b_nullclines.geoms:
            x, y = g.xy
            ax.plot(x, y, color=CELL_TYPE_TO_COLOR[cell_b], linewidth=linewidth)


def plot_two_cell_density(
    nds: NeighborsDataset,
    cell_a: str,
    cell_b: str,
    ax: plt.Axes | None = None,
    kde_bw: float = 0.3,
    fraction_outlined: float = 0.95,
):
    np.random.seed(42)
    nbrs_a, nbrs_b = nds.fetch(cell_a)[0], nds.fetch(cell_b)[0]
    nbrs = log2_1p(pd.concat([nbrs_a, nbrs_b]).sample(1000))

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    sns.kdeplot(
        x=nbrs[cell_a],
        y=nbrs[cell_b],
        bw_method=kde_bw,
        ax=ax,
        levels=[1 - fraction_outlined, 1.0],
        fill=True,
        cbar=False,
        color="#F0F0F0",
        alpha=0.5,
        zorder=-100,
    )


def plot_fixed_points(
    model: Model,
    cell_a: str,
    cell_b: str,
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    fixed_cell_counts: dict[str, float] | None = None,
):
    fixed_points_dict = get_classified_fixed_points_inbounds(model, cell_a, cell_b, xlim, ylim, fixed_cell_counts)
    _plot_fixed_points(fixed_points_dict, ax)


def _plot_fixed_points(
    fixed_points: dict,
    ax: plt.Axes,
):
    for stability_type in ["stable", "semi-stable", "unstable"]:
        ps = fixed_points[stability_type]
        if len(ps) > 0:
            x, y = zip(*ps)
            ax.plot(
                x,
                y,
                **STABILITIY_TYPE_TO_MARKER_STYLE_KWARGS[stability_type],  # type: ignore
                linestyle="",
                clip_on=False,
                zorder=100,
            )


def _plot_growth_rate_axis(
    nbrs,
    obs,
    cell_a,
    cell_b,
    kde_bw,
    rbf_gamma,
    ax,
    return_densest_point=False,
    obs_type="division",
    _plot_contours: bool = True,
):
    ax.set_xlabel(log_density_axis(cell_a))
    ax.set_ylabel(log_density_axis(cell_b))

    """
    Plot empirical growth density
    """

    # to keep runtimes reasonable - subsample to 30K cells:
    max_cells = int(30e3)
    if nbrs.shape[0] > max_cells:
        nbrs = nbrs.sample(n=max_cells, random_state=42, replace=False)
        obs = obs.iloc[nbrs.index]

        nbrs.reset_index(drop=True, inplace=True)
        obs.reset_index(drop=True, inplace=True)

    weights = rbf_kernel(nbrs, gamma=rbf_gamma)
    division_density = weights @ obs[obs_type] / np.sum(weights, axis=1)
    division_density_minus_death = division_density - obs[obs_type].mean()

    # set colormap - red "hot" = division > death, blue "cold" = death > division
    vcenter = 0
    vmin, vmax = np.round([division_density_minus_death.min(), division_density_minus_death.max()], 3)
    eps = 1e-6
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin - eps, vmax=vmax + eps)
    colormap = plt.get_cmap("RdBu_r")

    sns.scatterplot(
        x=nbrs[cell_a],
        y=nbrs[cell_b],
        c=division_density_minus_death,
        cmap=colormap,
        norm=normalize,
        ax=ax,
    )

    # add colorbar:
    sm = cm.ScalarMappable(norm=normalize, cmap=colormap)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.8)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(np.round([vmin, vcenter, vmax], 3))  # type: ignore
    cbar.ax.set_title("fraction (Ki67 > threshold) - mean")
    cbar.ax.tick_params(rotation=45)

    """
    Plot empirical cell distribution
    """
    if _plot_contours:
        sns.kdeplot(
            x=nbrs[cell_a],
            y=nbrs[cell_b],
            bw_method=kde_bw,
            ax=ax,
            levels=5,
            fill=False,
            cbar=False,
            color="black",
            linestyles="-",
            linewidths=1,
        )

    # remove right and top limits
    sns.despine(ax=ax)


def _add_fixed_point_legend(ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.8)
    cax.set_xlim(-0.2, 1.5)

    # stable point:
    cax.plot(
        [0.0],
        [1],
        **STABILITIY_TYPE_TO_MARKER_STYLE_KWARGS[STABLE],
        linestyle="",
        clip_on=False,
        zorder=100,
    )

    # semi-stable
    cax.plot(
        [0.5],
        [1],
        **STABILITIY_TYPE_TO_MARKER_STYLE_KWARGS[SEMI_STABLE],
        linestyle="",
        clip_on=False,
        zorder=100,
    )

    # unstable:
    cax.plot(
        [1.0],
        [1],
        **STABILITIY_TYPE_TO_MARKER_STYLE_KWARGS[UNSTABLE],
        linestyle="",
        clip_on=False,
        zorder=100,
    )

    annotations = [
        (STABLE, (0.0, 1), (25, -5)),
        (SEMI_STABLE, (0.5, 1), (35, -5)),
        (UNSTABLE, (1, 1), (30, -5)),
    ]

    ax = cax

    for label, point, offset in annotations:
        ax.annotate(
            label,
            xy=point,
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    # Remove the ticks and tick labels for both axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove the axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove the grid lines
    ax.grid(False)

    # Remove the frame around the plot
    ax.set_frame_on(False)
