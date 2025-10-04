"""
Test the ConcatDataset class.
"""

from tdm.tissue import StubTissue
from tdm.dataset import NeighborsDataset, ConcatDataset
from tdm.utils import microns


def test_concat_dataset():
    tissue = StubTissue(
        cell_types=["F", "M"],
        cell_type_xy_tuples=[
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("F", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
            ("M", microns(250), microns(250), 0),
        ],
        tissue_dimensions=(microns(500), microns(500)),
    )
    ds = NeighborsDataset(tissue)

    cds = ConcatDataset([ds, ds])

    for cell_type in ds.cell_types():
        assert cds.fetch(cell_type)[0].shape[0] == 2 * ds.fetch(cell_type)[0].shape[0]
        assert cds.fetch(cell_type)[0].shape[1] == ds.fetch(cell_type)[0].shape[1]

        assert cds.fetch(cell_type)[1].shape[0] == 2 * ds.fetch(cell_type)[1].shape[0]
        assert cds.fetch(cell_type)[1].shape[1] == ds.fetch(cell_type)[1].shape[1]


test_concat_dataset()
