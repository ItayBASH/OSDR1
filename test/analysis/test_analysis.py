from tdm.raw.breast_mibi import read_single_cell_df
from tdm.analysis import Analysis
from tdm.cell_types import FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL
from typing import Literal
from tdm.tissue import Tissue
from tdm.dataset import ConcatDataset, NeighborsDataset, RestrictedNeighborsDataset, PolynomialDataset


def fmtu_analysis_generator(
    neighborhood_mode: Literal["exclude", "extrapolate"] = "extrapolate", nds_class_kwargs: dict | None = None
):
    return Analysis(
        single_cell_df=read_single_cell_df().head(1000),  # smaller df for testing
        neighborhood_mode=neighborhood_mode,
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL],
        cell_types_to_model=[FIBROBLAST, MACROPHAGE, TUMOR],
        polynomial_dataset_kwargs={
            "degree": 2,
            "log_transform": False,
            "scale_features": False,
        },
        nds_class_kwargs=nds_class_kwargs,
        verbose=False,
        end_phase=4,  # skip model fit
    )


def test_neighbors_dataset_modes():

    ana1 = fmtu_analysis_generator(neighborhood_mode="extrapolate")
    ana2 = fmtu_analysis_generator(neighborhood_mode="exclude", nds_class_kwargs={"exclude_cells_near_edge": False})
    ana3 = fmtu_analysis_generator(neighborhood_mode="exclude", nds_class_kwargs={"exclude_cells_near_edge": True})

    assert ana1.nds.n_cells("F") == ana2.nds.n_cells("F")
    assert ana1.nds.n_cells("F") > ana3.nds.n_cells("F")


def test_analysis_properties():
    ana = fmtu_analysis_generator()

    assert ana.cell_a == FIBROBLAST
    assert ana.cell_b == MACROPHAGE
    assert ana.cell_c == TUMOR
    assert (
        isinstance(ana.xlim, tuple) and isinstance(ana.xlim[0], (int, float)) and isinstance(ana.xlim[1], (int, float))
    )
    assert (
        isinstance(ana.ylim, tuple) and isinstance(ana.ylim[0], (int, float)) and isinstance(ana.ylim[1], (int, float))
    )
    assert (
        isinstance(ana.zlim, tuple) and isinstance(ana.zlim[0], (int, float)) and isinstance(ana.zlim[1], (int, float))
    )
    assert isinstance(ana.tissues, list)
    assert isinstance(ana.tissues[0], Tissue)
    assert isinstance(ana.ndss, list)
    assert isinstance(ana.ndss[0], NeighborsDataset)
    assert isinstance(ana.nds, ConcatDataset)
    assert isinstance(ana.rnds, RestrictedNeighborsDataset)
    assert isinstance(ana.pds, PolynomialDataset)

    # model property should be None since end_phase=4 skips model fit
    assert not hasattr(ana, "_model")
