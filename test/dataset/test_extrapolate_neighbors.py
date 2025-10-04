from tdm.tissue import StubTissue
from tdm.dataset import ExtrapolateNeighborsDataset
from tdm.dataset.extrapolate_neighbors import neighborhood_fraction_in_tissue
import numpy as np
import pytest
from shapely.geometry import Point, Polygon


x_max, y_max = 0.0005, 0.0005


@pytest.mark.parametrize(
    "tups, neighbor_type, expected_count",
    [
        # cell at corner with one other neighbor should have 5 neighbors after correction:
        ([("F", x_max, y_max, 0), ("F", x_max, y_max, 0)], "F", 5),
        # cell at corner with no other neighbor should have 1 neighbor of it's kind after correction:
        ([("F", x_max, y_max, 0)], "F", 1),
        # cell at corner with no other neighbor should have 0 neighbors of other kinds after correction:
        ([("F", x_max, y_max, 0)], "M", 0),
    ],
)
def test_extrapolate_neighbors_dataset(tups, neighbor_type, expected_count):
    tissue = StubTissue(
        cell_types=["F", "M"],
        cell_type_xy_tuples=tups,
        tissue_dimensions=(x_max, y_max),
    )
    ends = ExtrapolateNeighborsDataset(tissue, 50 * 1e-6)
    assert np.allclose(ends.fetch("F")[0][neighbor_type][0], expected_count)


@pytest.mark.parametrize(
    "r, x, y, right, top, bottom, left",
    [
        (1, 2, 2, 4, 4, 0, 0),  # case 0
        (2, 2.5, 2.5, 5, 3, 0, 0),  # case 1
        (2, 2.5, 2.5, 5, 3, 0, 2),  # case 2a
        (2, 2.5, 2.5, 3, 5, 2, 0),
        (2, 2.5, 2.5, 3, 2.5, 0, 0),  # case 2b
        (2, 2.5, 2.5, 3.5, 3.5, 0, 0),
        (2, 2.5, 2.5, 5, 5, 2, 2),
        (3, 4.0, 4.0, 5.0, 5.0, 0, 3),  # case 3 - 3 sides in circle
        (3, 4.0, 4.0, 6.99, 5.0, 0, 3),
        (10, 2, 2, 4, 4, 0, 0),  # case 4 - rectangle totally inside circle
    ],
)
def test_neighborhood_intersection_with_tissue(r, x, y, right, top, bottom, left):
    shapely_frac = shapely_neighborhood_fraction_in_tissue(r, right, top, left, bottom, x, y)
    my_frac = neighborhood_fraction_in_tissue(r, right, top, x, y, left, bottom)

    assert np.allclose(shapely_frac, my_frac, rtol=0.01)


def shapely_neighborhood_fraction_in_tissue(neighborhood_size: float, right, top, left, bottom, cell_x, cell_y):
    """
    Returns the fraction out of the cell's neighborhood that's contained
    in the tissue boundaries.

    Slow shapely function, only used for validating my code.
    """
    # circle:
    circle_center = cell_x, cell_y
    r = neighborhood_size
    circle = Point(circle_center).buffer(r)

    # rectangle:
    corners = [
        (left, bottom),
        (right, bottom),
        (right, top),
        (left, top),
        (left, bottom),
    ]
    rectangle = Polygon(corners)

    return circle.intersection(rectangle).area / circle.area
