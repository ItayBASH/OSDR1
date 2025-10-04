from tdm.utils import log2_1p
from tdm.analysis import Analysis
from tdm.simulate.deterministic import compute_trajectory
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3D_trajectory(
    ana: Analysis,
    initial_cell_counts: list[float],
    odeint_timepoints: np.ndarray,
    title: str | None = None,
    plot_projections: bool = True,
    logspace: bool = True,
):
    """Plot a deterministic 3D trajectory.

    Args:
        ana (Analysis): Analysis object.
        initial_cell_counts (list[float]): start state for the trajectory.
        odeint_timepoints (np.ndarray): timepoints for the scipy.integrate.odeint function
        title (str | None, optional): alternative title for plot. Defaults to None.
        plot_projections (bool, optional): show gray trajectorys on each of planes: x=0, y=0, z=0. Defaults to True.
        logspace (bool, optional): plot in log-space. Defaults to True.

    Example:
    >>> # ana is an analysis fit to 3 cell types
    >>> plot_3D_trajectory(ana, initial_cell_counts=[40,4,1], odeint_timepoints=np.linspace(0,5000, 10000))
    """

    sol = compute_trajectory(ana=ana, initial_cell_counts=initial_cell_counts, odeint_timepoints=odeint_timepoints)

    # plot in log-space:
    if logspace:
        sol = log2_1p(sol)
    xs = sol[:, 0]
    ys = sol[:, 1]
    zs = sol[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d", facecolor="white")

    # Plot the trajectory
    ax.plot(xs, ys, zs, label="Trajectory", color="#3B76BC", linewidth=3)

    # Plot the 2D projections
    if plot_projections:
        ax.plot(xs, ys, zs * 0, "k", alpha=0.25, linewidth=3)  # Projection on xy-plane
        ax.plot(xs, ys * 0, zs, "k", alpha=0.25, linewidth=3)  # Projection on xz-plane
        ax.plot(xs * 0, ys, zs, "k", alpha=0.25, linewidth=3)  # Projection on yz-plane

    # axis limits:
    # ax.set_xlim(ana.xlim)
    # ax.set_ylim(ana.ylim)
    # ax.set_zlim(ana.zlim)

    # Reverse the Y-axis
    ax.invert_yaxis()

    # Set labels
    ax.set_xlabel(ana.cell_types[0])
    ax.set_ylabel(ana.cell_types[1])
    ax.set_zlabel(ana.cell_types[2])

    # Set title
    if title is None:
        x, y, z = ana.cell_types
        ax.set_title(f"3D trajectory: x={x}, y={y}, z={z}")
    else:
        ax.set_title(title)

    ax.set_box_aspect(aspect=None, zoom=0.85)

    # Add a green dot at the start state
    ax.scatter(xs[0], ys[0], zs[0], color="#96C75A", s=100, label="Start State")

    # Add a red dot at the final state
    ax.scatter(xs[-1], ys[-1], zs[-1], color="#EC6238", s=100, label="Final State")

    # Add a legend
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.2, 0.7))

    fig.tight_layout()

    # Show the plot
    plt.show()
