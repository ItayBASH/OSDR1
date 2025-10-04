"""
Numerical tools related to 2d dynamics (nullclines, fixed-points, stability etc.)
"""

import numpy as np
import numdifftools as nd
from typing import Mapping, Literal, Callable
from shapely.geometry import MultiLineString, Point, LineString
from contourpy import contour_generator


from tdm.utils import inv_log2_1p, log2_1p
from tdm.model import Model


def compute_change_in_each_direction(
    model: Model,
    cell_a: str,
    cell_b: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    fixed_cell_counts: Mapping[str, float] | None = None,
    step: float = 0.01,
    mode: Literal["cells", "rates"] = "cells",
):
    """
    Computes rates of change in the number of cells A,B (dA/dt and dB/dt)
    over a grid of values within the range xlim and ylim.

    Returns:
        (A, B, dAdt, dBdt): a tuple of meshgrids of shape (xlim / step) x (ylim / step)
    """
    # uniform values (in log-scale) of a and b
    A = np.arange(*xlim, step=step)
    B = np.arange(*ylim, step=step)
    A, B = np.meshgrid(A, B)

    # construct cell_counts for the model using the uniform values:
    cell_counts: dict[str, float | np.ndarray] = {}
    cell_counts[cell_a] = A.ravel()
    cell_counts[cell_b] = B.ravel()
    cell_counts = {k: inv_log2_1p(v) for k, v in cell_counts.items()}  # construct features expects raw cell counts
    if fixed_cell_counts is not None:
        cell_counts.update(fixed_cell_counts)

    # difference, in absolute number of cells:
    dAdt, dBdt = model.delta_cells(cell_counts, return_order=[cell_a, cell_b], mode=mode)

    # transform change in number of cells to log-scale for plot:
    if mode == "cells":
        dAdt = log2_1p(cell_counts[cell_a] + dAdt) - log2_1p(cell_counts[cell_a])
        dBdt = log2_1p(cell_counts[cell_b] + dBdt) - log2_1p(cell_counts[cell_b])

    # reshape into meshgrid:
    dAdt = dAdt.reshape(A.shape)
    dBdt = dBdt.reshape(B.shape)

    return A, B, dAdt, dBdt


def get_nullclines(
    a_meshgrid: np.ndarray, b_meshgrid: np.ndarray, dadt: np.ndarray, dbdt: np.ndarray
) -> tuple[MultiLineString, MultiLineString]:

    # contourpy output:
    a_ncs = get_nullclines_1_axis(a_meshgrid, b_meshgrid, dadt)
    b_ncs = get_nullclines_1_axis(a_meshgrid, b_meshgrid, dbdt)

    # contourpy sometimes misses the axes-nullclines.
    # manually add them:

    # add y=0 nullcline to a_ncs
    x_eq_zero_nc = LineString([(0, -10), (0, 10)])
    a_ncs = MultiLineString([*a_ncs.geoms, x_eq_zero_nc])

    # and x=0 nullclines to b_ncs
    y_eq_zero_nc = LineString([(-10, 0), (10, 0)])
    b_ncs = MultiLineString([*b_ncs.geoms, y_eq_zero_nc])

    return a_ncs, b_ncs


def get_nullclines_1_axis(a_meshgrid: np.ndarray, b_meshgrid: np.ndarray, dxdt: np.ndarray):
    return MultiLineString(contour_generator(a_meshgrid, b_meshgrid, dxdt).lines(0.0))


def get_classified_fixed_points_inbounds(
    model: Model,
    cell_a: str,
    cell_b: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    fixed_cell_counts: dict[str, float] | None = None,
):
    points = get_fixed_point_xys(model, cell_a, cell_b, fixed_cell_counts, return_as="tuples", xlim=xlim, ylim=ylim)

    points = remove_points_out_of_bounds(points, xlim, ylim)

    points_dict = classify_fixed_points(model, cell_a, cell_b, points, fixed_cell_counts)

    return points_dict


def remove_points_out_of_bounds(points, xlim, ylim):
    return [p for p in points if is_inbound(*p, *xlim, *ylim)]


def is_inbound(x, y, xmin, xmax, ymin, ymax, eps=1e-4):
    return (x >= xmin - eps) & (x <= xmax + eps) & (y >= ymin - eps) & (y <= ymax + eps)


def get_fixed_point_xys(
    model: Model,
    cell_a: str,
    cell_b: str,
    fixed_cell_counts: dict[str, float] | None = None,
    return_as: Literal["tuples", "lists"] = "tuples",
    xlim: tuple[float, float] = (0, 8),
    ylim: tuple[float, float] = (0, 8),
):
    # take small delta below each axis for discovering axes fixed points:
    eps = 1e-2
    xlim_eps = (-eps, xlim[1])
    ylim_eps = (-eps, xlim[1])

    a_mesh, b_mesh, dadt, dbdt = compute_change_in_each_direction(
        model=model,
        cell_a=cell_a,
        cell_b=cell_b,
        xlim=xlim_eps,
        ylim=ylim_eps,
        fixed_cell_counts=fixed_cell_counts,
        step=0.01,
    )

    # Two MultiLineString objects:
    a_nullclines, b_nullclines = get_nullclines(a_mesh, b_mesh, dadt, dbdt)

    # MultiPoint or Point object:
    fixed_points = a_nullclines.intersection(b_nullclines)

    # extract simple float tuples from shapely objects:
    if isinstance(fixed_points, Point):
        fixed_points = [(fixed_points.x, fixed_points.y)]
    else:  # Multipoint
        fixed_points = [(pt.x, pt.y) for pt in fixed_points.geoms]

    if return_as == "tuples":
        return fixed_points
    elif return_as == "lists":
        return zip(*fixed_points)


def classify_fixed_points(
    model: Model,
    cell_a: str,
    cell_b: str,
    fixed_points: list[tuple[float, float]],
    fixed_cell_counts: dict[str, float] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Classifies fixed points into:
        - Stable:
            all jacobian eigenvalues have negative real-values
        - Semi-stable:
            point is on one axis and dynamics contract along the other
        - Unstable:
            point has non-negative eigenvalues and isn't on an axis

    Parameters:
        model (Model):
            the model whose dynamics are analyzed

        fixed_points (list[tuple[float,float]]):
            list of x,y tuples corresponding with positions on the phase-portrait (!) for cell_a and cell_b respectively
    """
    # can't init a mutable in the function signature.
    cell_counts = fixed_cell_counts or {}

    # define function for numerical differentiation:
    def f(xy):
        cell_counts.update({cell_a: inv_log2_1p(xy[0]), cell_b: inv_log2_1p(xy[1])})
        delta_cells = model.delta_cells(cell_counts, return_order=[cell_a, cell_b])
        return np.array(delta_cells)

    d: dict[str, list[tuple[float, float]]] = {"stable": [], "semi-stable": [], "unstable": []}

    for p in fixed_points:
        d[classify_one_fixed_point(p, f)].append(p)

    return d


def classify_one_fixed_point(p: tuple[float, float], f: Callable):
    """
    Classifies the point p according to the 2-D dynamics function f
    """
    if not on_x_axis(p) and not on_y_axis(p):
        jac = nd.Jacobian(f, step=0.1)(p)
        if is_negative_definite(jac):
            return "stable"
        else:
            return "unstable"

    # p is on an axis:
    else:
        # use forward method to avoid negative values for cell number
        jac = nd.Jacobian(f, method="forward", step=0.1)(p)

        # stable regardless of position:
        if is_negative_definite(jac):
            return "stable"

        # We consider the (0,0) point semi-stable for non-stable dynamics
        if on_x_axis(p) and on_y_axis(p):
            return "semi-stable"

        elif on_x_axis(p):
            if (jac @ [1, 0])[0] < 0:  # contracts along x-axis
                return "semi-stable"
            else:
                return "unstable"

        elif on_y_axis(p):
            if (jac @ [0, 1])[1] < 0:  # contracts along y-axis
                return "semi-stable"
            else:
                return "unstable"
        else:
            raise ValueError(f"What happened here? {p}")


def is_negative_definite(M):
    return np.all(np.linalg.eigvals(M).real < 0)


def on_x_axis(p, eps=1e-6):
    return abs(p[1]) < eps


def on_y_axis(p, eps=1e-6):
    return abs(p[0]) < eps


"""
Analysing the classified fixed-points dictionary provided by get_classified_fixed_points_inbounds

TODO write tests using example models
"""


def has_stable_point_off_axis(fps: dict[str, list[tuple[float, float]]], eps: float = 1e-2):
    """"""

    stable_fps = fps["stable"]

    # has at least one fixed point off-axis:
    off_axes_stable_fps = [(x, y) for x, y in stable_fps if (x > eps and y > eps)]
    if len(off_axes_stable_fps) >= 1:
        return True

    return False


def has_stable_point_on_y_axis(fps: dict[str, list[tuple[float, float]]], eps: float = 1e-2, y_gt: float = 0.1):
    """"""

    stable_fps = fps["stable"]

    # has at least one stable fixed point y-axis:
    y_axis_stable_fps = [(x, y) for x, y in stable_fps if (np.allclose(x, 0, atol=eps) and y > y_gt)]
    if len(y_axis_stable_fps) >= 1:
        return True

    return False


def has_semi_stable_point_on_y_axis(fps: dict[str, list[tuple[float, float]]], eps: float = 1e-2, y_gt: float = 0.1):
    """"""

    semi_stable_fps = fps["semi-stable"]

    # has at least one semi-stable fixed point on y-axis:
    y_axis_semi_stable_fps = [(x, y) for x, y in semi_stable_fps if (np.allclose(x, 0, atol=eps) and y > y_gt)]
    if len(y_axis_semi_stable_fps) >= 1:
        return True

    return False


def has_stable_point_on_x_axis(fps: dict[str, list[tuple[float, float]]], eps: float = 1e-2, x_gt: float = 0.1):
    """"""

    stable_fps = fps["stable"]

    # has at least one stable fixed point on x-axis:
    x_axis_stable_fps = [(x, y) for x, y in stable_fps if (x > x_gt and np.allclose(y, 0, atol=eps))]
    if len(x_axis_stable_fps) >= 1:
        return True

    return False


def has_semi_stable_point_on_x_axis(fps: dict[str, list[tuple[float, float]]], eps: float = 1e-2, x_gt: float = 0.1):
    """"""

    semi_stable_fps = fps["semi-stable"]

    # has at least one semi-stable fixed point on x axis:
    x_axis_semi_stable_fps = [(x, y) for x, y in semi_stable_fps if (x > x_gt and np.allclose(y, 0, atol=eps))]
    if len(x_axis_semi_stable_fps) >= 1:
        return True

    return False
