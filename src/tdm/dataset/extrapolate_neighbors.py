from tdm.tissue import Tissue
from tdm.dataset.neighbors import NeighborsDataset, _default_neighborhood_size
from numba import jit
import math
import numpy as np
import pandas as pd


class ExtrapolateNeighborsDataset(NeighborsDataset):
    """A NeighoborsDataset that includes cells near the edge of the tissue, correcting the bias due to partially observed neighborhoods.

    We can't observe the entire neighborhood of cells near tissue edges. For example,
    we can observe only half the neighborhood of a cell near the middle of an edge.
    To correct for this bias we multiply this cell's counts by 2.

    Generally:

    .. code-block:: python

        corrected_counts = counts / fraction_of_neighborhood_observed
    """

    def __init__(self, tissue: Tissue, neighborhood_size: float = _default_neighborhood_size) -> None:
        """Initializes the ExtrapolateNeighborsDataset.

        Args:
            tissue (Tissue): The tissue instance.
            neighborhood_size (float, optional): The neighborhood radius. Defaults to 80 microns.
        """
        self.tissue = tissue
        self.neighborhood_size = neighborhood_size
        super().__init__(tissue, neighborhood_size)

    def construct_dataset(self, tissue: Tissue, cell_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parameters:
            tissue: instance of type Tissue
            cell_type: str from the list CELL_TYPES_ARRAY, as defined in tdm.tissue.cell_types

        Returns:
            features,observations (tuple):
                features: dataframe with shape (n_cells, n_features)
                observations: dataframe with shape (n_cells, 2) holding observations. columns: division, death
        """
        features = self.construct_features(tissue, cell_type)
        obs = self.construct_observations(tissue, cell_type)

        # correct counts:
        xys = self.get_xys(cell_type)
        if len(xys) == 0:  # no cell of this type
            return features, obs

        tissue_dimensions = tissue.tissue_dimensions()

        cell_fractions = compute_neighborhood_fractions_within_tissue(xys, self.neighborhood_size, *tissue_dimensions)

        # We only "inflate" the biased part, i.e the neighbors excluding the cell-itself.
        # For example, a cell at the corner (1/4 observed) with 0 neighbors except itself will
        # have 1 neighbor after correction, not 4
        features[cell_type] = features[cell_type] - 1
        features = features.divide(pd.Series(cell_fractions), axis=0)
        features[cell_type] = features[cell_type] + 1

        return features, obs


"""
Optimized computation of neighborhood fracion in tissue.

Note:
    These functions are not considered in codecov because it doesn't support numba.
    But, they are tested as part of the ExtrapolateNeighborsDataset tests.
"""


@jit(nopython=True)
def neighborhood_fraction_in_tissue(  # pragma: no cover
    neighborhood_size: float,
    right: float,
    top: float,
    cell_x: float,
    cell_y: float,
    left: float = 0,
    bottom: float = 0,
) -> float:
    nbrhood_area = circle_area(neighborhood_size)
    nbrhood_area_inbounds = fast_circle_rectangle_intersection_area(
        neighborhood_size, cell_x, cell_y, right, top, left, bottom
    )

    return nbrhood_area_inbounds / nbrhood_area


@jit(nopython=True)
def compute_neighborhood_fractions_within_tissue(
    xys, neighborhood_size, tissue_width, tissue_height
):  # pragma: no cover
    fractions = []
    for x, y in xys:
        fractions.append(neighborhood_fraction_in_tissue(neighborhood_size, tissue_width, tissue_height, x, y))
    return fractions


@jit(nopython=True)
def circle_area(r):  # pragma: no cover
    return math.pi * (r**2)


@jit(nopython=True)
def distance(x1, y1, x2, y2):  # pragma: no cover
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


@jit(nopython=True)
def fast_circle_rectangle_intersection_area(
    r: float,
    x: float,
    y: float,
    right: float,
    top: float,
    left: float = 0,
    bottom: float = 0,
):  # pragma: no cover
    """
    Fast shapely alternative for computing neighborhood intersection with tissue boundaries.

    Assumes circle center is within the rectangle ("cell in a tissue").
    Computes the area analyticlally, separating into cases by number of rectangle sides closer than r.
    """
    dists = (
        top - y,
        x - left,
        y - bottom,
        right - x,
    )  # counter clockwise, starting from top
    is_under_r_bools = [d < r for d in dists]
    case = sum(is_under_r_bools)

    # case 0 - (circle completely inside rectangle):
    if case == 0:
        return circle_area(r)

    # case 1 - one side is closer than r:
    elif case == 1:
        d = min(dists)  # necessarily the minimal distance:

        # angle between line perpendicular to side and radius to side's intersection with circle.
        theta = math.acos(d / r)

        # area from triangle enclosed by radii and closest side:
        a1 = d * math.sin(theta) * r

        # area from remaining circle:
        a2 = (1 - theta / math.pi) * circle_area(r)

        return a1 + a2

    # case 2 - two sides are closer than r:
    elif case == 2:
        idx1, idx2 = [i for i, _bool in enumerate(is_under_r_bools) if _bool]
        d1, d2 = dists[idx1], dists[idx2]

        # case 2a - the two sides are opposite:
        if (idx2 - idx1) == 2:  # (0, 2) or (1,3)
            # two computations like case 1:
            theta1 = math.acos(d1 / r)
            a1 = d1 * math.sin(theta1) * r

            theta2 = math.acos(d2 / r)
            a2 = d2 * math.sin(theta2) * r

            # area from remaining circle:
            a3 = (1 - (theta1 + theta2) / math.pi) * circle_area(r)

            return a1 + a2 + a3

        else:  # adjacent sides - (0,1), (1,2), (2,3), (3,4)
            # top triangle minus protruding right small triangle:
            theta1 = math.acos(d1 / r)
            leg = math.sin(theta1) * r
            sub_leg = leg - d2
            small_triangle_area = (leg * d1 / 2) * (sub_leg / leg) ** 2
            a1 = d1 * leg - small_triangle_area

            # half of right triangle:
            theta2 = math.acos(d2 / r)
            a2 = d2 * math.sin(theta2) * r / 2

            # small section missing:
            theta_small = (math.pi / 2) - theta1
            a3 = math.tan(theta_small) * d2 * d2 / 2

            # remaining circle:
            a4 = ((2 * math.pi - (2 * theta1 + theta_small + theta2)) / (math.pi * 2)) * circle_area(r)

            return a1 + a2 + a3 + a4

    # case 3 - three sides are closer than r:
    elif case == 3:
        # middle side is d3
        idx1, idx3, idx2 = np.roll([0, 1, 2, 3], 3 - np.argmax(~np.array(is_under_r_bools)))[:3]
        d1, d2, d3 = dists[idx1], dists[idx2], dists[idx3]

        # compute top and bottom halves separately, taking care of 2 cases for each:

        angle_covered = float(0)
        area = float(0)

        # top half:
        theta1 = math.acos(d1 / r)
        leg1 = math.sin(theta1) * r

        if d3 < leg1:
            # half of top triangle:
            angle_covered += theta1
            area += leg1 * d1 / 2

            # top rectangle
            angle_covered += math.pi / 2
            area += d1 * d3
        else:
            # complete top triangle:
            angle_covered += 2 * theta1
            area += leg1 * d1

            # half of side triangle:
            theta3 = math.acos(d3 / r)
            leg3 = math.sin(theta3) * r

            angle_covered += theta3
            area += d3 * leg3 / 2

        # bottom half:
        theta2 = math.acos(d2 / r)
        leg2 = math.sin(theta2) * r

        if d3 < leg2:
            # half of bottom triangle:
            angle_covered += theta2
            area += leg2 * d2 / 2

            # bottom rectangle
            angle_covered += math.pi / 2
            area += d2 * d3
        else:
            # complete bottom triangle:
            angle_covered += 2 * theta2
            area += leg2 * d2

            # half of side triangle:
            theta3 = math.acos(d3 / r)
            leg3 = math.sin(theta3) * r

            angle_covered += theta3
            area += d3 * leg3 / 2

        # remaining circle:
        area += (1 - (angle_covered / (2 * math.pi))) * circle_area(r)

        return area

    # case 4 - rectangle completely inside circle:
    elif case == 4:
        return (right - left) * (top - bottom)
    else:
        raise UserWarning("Failed to compute circle intersection with rectangle.")
