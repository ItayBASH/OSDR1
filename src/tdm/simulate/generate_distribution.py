import numpy as np
from multiprocessing import Pool

from tdm.tissue import Tissue, StubTissue, correction_factor
from tdm.dataset import (
    BootstrapDataset,
    ConcatDataset,
    NeighborsDataset,
)
from tdm.analysis import Analysis
from tdm.simulate import TissueStep
from tdm.utils import inv_log2_1p, microns
from tdm.model import Model

# default tissue width and height:
_1000_microns = microns(1000)


def get_random_positions_on_phase_portrait() -> tuple[float, float]:
    """
    Returns n random positions on the phase portrait.
    """
    return tuple((np.random.beta(2, 4, size=(1, 2)) * 7.0).squeeze().astype(float))


def get_tissue_from_analysis(analysis: Analysis, state: tuple[float, float]) -> Tissue:
    """Constructs a tissue from an analysis and a state.

    Args:
        analysis (Analysis): A two cell-type analysis object used for fetching:
            - cell types
            - neighborhood_size
            - model
        state (tuple[float,float]): initial densities of cell a and cell b transformed by log2_1p (see tdm.utils)

    Returns:
        Tissue: A StubTissue object with an initial density of cells a and b determined by the provided state.
    """
    # fetch required components from analysis:
    cell_a = analysis.cell_a
    cell_b = analysis.cell_b
    nbrhood_size = analysis.neighborhood_size

    # construct tissue:
    return get_tissue0(cell_a, cell_b, *state, nbrhood_size)


def get_tissue0(
    cell_a: str,
    cell_b: str,
    start_a: float,
    start_b: float,
    nbrhood_size: float,  # e.g 80*1e-6 for 80 microns
    tissue_width: float = _1000_microns,
    tissue_height: float = _1000_microns,
):
    """Returns a tissue with fibroblasts and macrophages uniformly distributed in space
    with a density corresponding to the provided starting positions on the phase portrait.

    Args:
        cell_a (str): _description_
        cell_b (str): _description_
        start_a (float): log cell count (per neighborhood)
        start_b (float): log cell count (per neighborhood)
        nbrhood_size (float): _description_
        tissue_height (float, optional): _description_. Defaults to _1000_microns.

    Returns:
        _type_: _description_
    """
    # convert to number of cells per-neighborhood:
    start_a, start_b = inv_log2_1p(start_a), inv_log2_1p(start_b)

    # convert to number of cells per-tissue:
    start_a = n_cells_per_tissue(start_a, tissue_width, tissue_height, nbrhood_size)
    start_b = n_cells_per_tissue(start_b, tissue_width, tissue_height, nbrhood_size)

    # sample fibroblasts uniformally in space:
    cell_a_vals = [
        (
            cell_a,
            np.random.uniform(0, tissue_width),
            np.random.uniform(0, tissue_height),
            0,  # division / death stub
        )
        for _ in range(start_a)
    ]

    # sample macrophages uniformally in space:
    macs = [
        (
            cell_b,
            np.random.uniform(0, tissue_width),
            np.random.uniform(0, tissue_height),
            0,  # division / death stub
        )
        for _ in range(start_b)
    ]

    # combine into a single list:
    tissue_cells = []
    tissue_cells.extend(cell_a_vals)
    tissue_cells.extend(macs)

    # construct tissue:
    tissue0 = StubTissue(
        cell_types=[cell_a, cell_b],
        cell_type_xy_tuples=tissue_cells,
        tissue_dimensions=(tissue_width, tissue_height),
    )

    return tissue0


def n_cells_per_tissue(n_cells_per_neighborhood, tissue_width, tissue_height, nbrhood_size):
    return int(
        n_cells_per_neighborhood * correction_factor(tissue_width, tissue_height, nbrhood_size, mode="to_tissue")
    )


def n_cells_per_neighborhood(n_cells_per_tissue, tissue_width, tissue_height, neighborhood_size, round_to_int=True):
    n = n_cells_per_tissue * correction_factor(tissue_width, tissue_height, neighborhood_size, mode="to_neighborhood")

    if round_to_int:
        return int(n)
    else:
        return n


"""
^^^^^^^^
"""


def simulate_one_tissue(
    cell_a: str,
    cell_b: str,
    model: Model,
    neighborhood_size: float,
    n_steps: int,
    tissue_width: float = _1000_microns,
    tissue_height: float = _1000_microns,
    state0: Tissue | tuple[float, float] | None = None,
    return_stepper: bool = False,
    verbose: bool = False,
    diffusion_coef: float = 1e-6,
):
    """
    Simulates a tissue for n_steps, starting from state0. Currently supports only two cells.

    Parameters:
        model (Model):
            The model to use for predicting division and death.
        nbrhood_size (float):
            The size of the neighborhood in which new cells are placed (e.g 100*1e-6 for 100 micron neighborhood).
    """
    # Construct a tissue with a random initial condition:
    if state0 is None:
        state0 = get_random_positions_on_phase_portrait()

    if isinstance(state0, Tissue):
        tissue0 = state0
    else:
        tissue0 = get_tissue0(cell_a, cell_b, *state0, neighborhood_size, tissue_width, tissue_height)

    # Simulate n_steps:
    stepper = TissueStep(
        tissue=tissue0, model=model, neighborhood_size=neighborhood_size, diffusion_coeff=diffusion_coef
    )
    stepper.step_n_times(n_steps, verbose=verbose)

    if return_stepper:
        return stepper
    else:  # return last tissue
        return stepper.tissues[-1]


def simulate_n_tissues(
    cell_a: str,
    cell_b: str,
    n: int,
    model: Model,
    neighborhood_size: float,
    n_steps: int,
    tissue_width: float = _1000_microns,
    tissue_height: float = _1000_microns,
):
    """
    Simulates n tissues for n_steps, starting each from a random initial condition.
    """
    with Pool(32) as p:
        tissues = p.starmap(
            simulate_one_tissue,
            [
                (  # tuple of arguments to simulate_one_tissue, order of arguments is important!
                    cell_a,
                    cell_b,
                    model,
                    neighborhood_size,
                    n_steps,
                    tissue_width,
                    tissue_height,
                )
                for _ in range(n)
            ],
        )
    return tissues


# uses simulate_n_tissues() which is tested.
def generate_cell_distribution_from_ground_truth_model(  # pragma: no cover
    cell_a: str,
    cell_b: str,
    model: Model,
    neighborhood_size: float,
    tissue_width: float = _1000_microns,
    tissue_height: float = _1000_microns,
    n_steps: int = 100,
    n_initial_conditions: int = 300,
):
    # 1. Simulate tissues from 300 random initial conditions and bias the model towards the steady points:
    n_steps = n_steps
    tissues = simulate_n_tissues(
        cell_a=cell_a,
        cell_b=cell_b,
        n=n_initial_conditions,
        model=model,
        neighborhood_size=neighborhood_size,
        n_steps=n_steps,
        tissue_width=tissue_width,
        tissue_height=tissue_height,
    )

    # 2. Create a neighbors dataset based on the last tissue in each simulation.
    ndss = [NeighborsDataset(t) for t in tissues if t.n_cells() > 0]
    ndss = [
        nds for nds in ndss if nds.n_cells() > 0
    ]  # filter again in case all cells were excluded because they were near tissue edge

    # 3. Construct the neighborhood distribution to sample from
    #    50k cells in total, equally from each dataset:
    n_cells_total = 50000
    n_cells_per_nds = int(n_cells_total / len(ndss))
    bdss = [BootstrapDataset(nds, seed=42, n_samples=n_cells_per_nds) for nds in ndss]
    # bdss = ndss
    ds = ConcatDataset(bdss)

    return ds, tissues
