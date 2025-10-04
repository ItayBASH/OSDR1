from tdm.dataset import NeighborsDataset
from tdm.tissue import StubTissue
from tdm.cell_types import MACROPHAGE, FIBROBLAST
from tdm.utils import microns
from tdm.simulate.generate_distribution import get_tissue0

import numpy as np

tissue_width = tissue_height = microns(500)
neighborhood_size = microns(100)


def test_filter_cells_near_edges():
    # example with 4 macrophages out of bounds
    tissue = StubTissue(
        cell_types=["F", "M"],
        cell_type_xy_tuples=[
            # some cells inbounds:
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            # four macs out of bounds:
            ("M", microns(450), microns(450), 0),
            ("M", microns(450), microns(450), 0),
            ("M", microns(450), microns(450), 0),
            ("M", microns(450), microns(450), 0),
        ],
        tissue_dimensions=(tissue_width, tissue_height),
    )

    # Fetch number of macs from complete tissue:
    cell_df = tissue.cell_df()
    cell_df = cell_df.loc[cell_df.cell_type == MACROPHAGE, :]

    # Exclude:
    ds = NeighborsDataset(tissue, neighborhood_size=neighborhood_size, exclude_cells_near_edge=True)
    X, y = ds.construct_dataset(tissue, MACROPHAGE)
    assert cell_df.shape[0] == X.shape[0] + 4
    assert cell_df.shape[0] == y.shape[0] + 4

    # Don't Exclude:
    ds = NeighborsDataset(tissue, neighborhood_size=neighborhood_size, exclude_cells_near_edge=False)
    X, y = ds.construct_dataset(tissue, MACROPHAGE)
    assert cell_df.shape[0] == X.shape[0]
    assert cell_df.shape[0] == y.shape[0]


def test_n_neighbors_func():
    """
    tissue with more fibroblasts than macrophages
    """
    tissue = get_tissue0("F", "M", 8, 1, tissue_width, tissue_height, neighborhood_size)
    ds = NeighborsDataset(tissue, neighborhood_size=neighborhood_size)

    features, obs = ds.fetch(MACROPHAGE)
    assert np.all(features.F > features.M)

    features, obs = ds.fetch(FIBROBLAST)
    assert np.all(features.F > features.M)

    """
    tissue with more MACs than FBs:
    """
    tissue = get_tissue0("F", "M", 1, 8, neighborhood_size, tissue_width, tissue_height)
    ds = NeighborsDataset(tissue, neighborhood_size=neighborhood_size)

    features, obs = ds.fetch(MACROPHAGE)
    assert np.all(features.F < features.M)

    features, obs = ds.fetch(FIBROBLAST)
    assert np.all(features.F < features.M)

    """
    tissue with no MACs
    """
    tissue = get_tissue0("F", "M", 8, 0, neighborhood_size, tissue_width, tissue_height)
    ds = NeighborsDataset(tissue, neighborhood_size=neighborhood_size)

    features, obs = ds.fetch(MACROPHAGE)
    assert features.shape[0] == 0

    features, obs = ds.fetch(FIBROBLAST)
    assert np.all(features.M == 0)
    assert np.all(features.F > 0)
