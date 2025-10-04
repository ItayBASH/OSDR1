from tdm.raw import read_single_cell_df
from tdm.analysis import Analysis
from tdm.cache import persistent_cache
from tdm.cell_types import FIBROBLAST, MACROPHAGE, T_CELL, B_CELL, TUMOR, ENDOTHELIAL
from tdm.raw.breast_mibi import CELL_TYPE_DEFINITIONS_CD4_CD8_B, CD4_T_CELL, CD8_T_CELL
from tdm.raw.breast_mibi import read_single_cell_df as read_single_cell_df_mibi


@persistent_cache
def fm_analysis(dataset: str = "breast_mibi"):
    return _fm_analysis(read_single_cell_df(dataset=dataset))


def _fm_analysis(single_cell_df):
    return Analysis(
        single_cell_df=single_cell_df,
        neighborhood_mode="extrapolate",
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL],
        cell_types_to_model=[FIBROBLAST, MACROPHAGE],
        polynomial_dataset_kwargs={
            "degree": 2,
            "log_transform": False,
            "scale_features": False,
        },
        enforce_max_density=True,  # original
        max_density_enforcer_power=8,
        verbose=False,
    )


@persistent_cache
def fm_tu_analysis(dataset: str = "breast_mibi", enforce_max_density: bool = True):
    return Analysis(
        single_cell_df=read_single_cell_df(dataset=dataset),
        neighborhood_mode="extrapolate",
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
        enforce_max_density=enforce_max_density,
        max_density_enforcer_power=8,
        verbose=False,
    )


@persistent_cache
def tb_analysis(dataset: str = "breast_mibi"):
    return _tb_analysis(read_single_cell_df(dataset=dataset))


def _tb_analysis(single_cell_df):
    return Analysis(
        single_cell_df=single_cell_df,
        neighborhood_mode="extrapolate",
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL, T_CELL, B_CELL],  # all allowed
        cell_types_to_model=[T_CELL, B_CELL],
        polynomial_dataset_kwargs={
            "degree": 2,
            "log_transform": False,
            "scale_features": False,
        },
        enforce_max_density=True,
        max_density_enforcer_power=8,
        verbose=False,
    )


@persistent_cache
def get_3d_tb_ana():
    single_cell_df = read_single_cell_df_mibi(cell_type_definitions=CELL_TYPE_DEFINITIONS_CD4_CD8_B)

    return Analysis(
        single_cell_df=single_cell_df,
        neighborhood_mode="extrapolate",
        model_kwargs={
            "death_estimation": "mean",
            "truncate_division_rate": True,
        },
        # allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL, CD4_T_CELL, CD8_T_CELL, B_CELL],
        cell_types_to_model=[CD4_T_CELL, CD8_T_CELL, B_CELL],
        polynomial_dataset_kwargs={
            "degree": 2,
        },
        enforce_max_density=False,
        max_density_enforcer_power=8,
        verbose=False,
    )
