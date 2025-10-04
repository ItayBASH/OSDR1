from copy import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from tdm.tissue import Tissue, StubTissue
from tdm.analysis import Analysis
from tdm.dataset import ExtrapolateNeighborsDataset
from tdm.model import Model


class TissueStep:
    def __init__(
        self,
        tissue: Tissue,
        ana: Analysis | None = None,
        model: Model | None = None,
        neighborhood_size: float | None = None,
        division_offset: float = 1.0,
        diffusion_coeff=0,
    ) -> None:
        """
        A class that computes the tissue at the next timestep by sampling
        division or death events for all cells.

        Note:
            Either ana or both model and neighborhood_size must be provided.

        Parameters:
            tissue (Tissue):
                The tissue at time t0.
            ana (Analysis, optional):
                The analysis object containing the model and neighborhood size.
            model (Model, optional):
                The model to use for predicting division and death. Required if ana not provided.
            neighborhood_size (float, optional):
                The size of the neighborhood in which new cells are placed (e.g 100*1e-6 for 100 micron neighborhood).
                Required if ana not provided.
            division_offset (float):
                Controls how far a cell appears after division, high values imply more chaotic motion.
            diffusion_coeff (float):
                The diffusion coefficient for random walk of cells.
        """
        self.tissue0 = tissue  # tissue at t0
        self.tissues = [tissue]  # tissue per time step
        self.tissue_width, self.tissue_height = self.tissue0.tissue_dimensions()
        self.division_offset = division_offset
        self.diffusion_coeff = diffusion_coeff

        if ana is not None:
            self.model = ana.model
            self.neighborhood_size = ana.neighborhood_size
        elif model is not None and neighborhood_size is not None:
            self.model = model
            self.neighborhood_size = neighborhood_size
        else:
            raise ValueError("Either ana or model and neighborhood_size must be provided")

    @property
    def last_tissue(self) -> Tissue:
        return self.tissues[-1]

    @last_tissue.setter
    def last_tissue(self, tissue: Tissue):
        self.tissues[-1] = tissue

    def step_n_times(self, n: int, verbose: bool = False) -> None:
        """
        Computes the tissue at the next n time steps and appends it to the list of tissues.
        """
        for _ in tqdm(range(n), disable=not verbose):
            self.step()
        return

    def step(self) -> None:
        """
        Computes the tissue at the next time step and appends it to the list of tissues.
        """
        # fetch tissue from last time step:
        tissue = self.tissues[-1]

        # fetch tissue_dimensions from original tissue:
        x_max, y_max = self.tissue0.tissue_dimensions()

        # construct a cell-counts Dataset based on tissue:
        ds = self.tissue_to_dataset(tissue)

        updated_cell_type_specific_dfs = []

        # we skip missing cell types, so get present types:
        cell_types_in_tissue = tissue.cell_df().cell_type.unique()

        # keep current tissue unchanged at next time step if it's empty
        if len(cell_types_in_tissue) == 0:
            self.tissues.append(copy(tissue))
            return

        for cell_type in cell_types_in_tissue:
            # fetch cell counts:
            cell_counts = ds.fetch(cell_type)[0]

            # sample observations:
            observations = self.model.sample_observations(cell_type, cell_counts, "both")

            # fetch all cells of current type:
            cells = tissue.cell_df()
            cells = cells.loc[cells.cell_type == cell_type]

            # new cell for each division:
            cells_from_divisions = cells.loc[observations == 1].copy().reset_index(drop=True)

            # place new cells uniformally in neighborhood of original cell:
            cells_from_divisions[["x", "y"]] += self.random_delta_in_neighborhood(
                cells_from_divisions.shape[0], division_offset=self.division_offset
            )

            cells_from_divisions = cells_from_divisions.loc[
                self.cells_inbound_mask(cells_from_divisions, x_max, y_max)
            ].reset_index(drop=True)

            # delete a cell for each '-1'
            cell_df_minus_deaths = cells.loc[observations != -1].copy().reset_index(drop=True)

            # concat all cells and append to list:
            cells = pd.concat([cell_df_minus_deaths, cells_from_divisions], axis=0)
            updated_cell_type_specific_dfs.append(cells)

        # concatenate cell_type-specific dfs
        cells = pd.concat(updated_cell_type_specific_dfs, axis=0)

        # add random walk for each cell
        cells[["x", "y"]] += np.random.normal(size=cells[["x", "y"]].shape) * self.diffusion_coeff

        # exclude new cells that are out of bounds:
        # this is unbiased if we're correcting cells at the boundary, example:
        # if 1/2 of a cell's neighborhood is out of bounds we add new cells
        # a chance of 1/2 and multiply the number of neighbors by 2 during correction.
        cells = cells.loc[self.cells_inbound_mask(cells, x_max, y_max)].reset_index(drop=True)

        # don't add empty tissues:
        if len(cells) == 0:
            return

        # construct a new Tissue
        tissue = StubTissue(
            cell_df=cells,
            cell_types=self.tissue0.cell_types(),
            tissue_dimensions=(x_max, y_max),
        )

        # append updated tissue
        self.tissues.append(tissue)

        return

    def random_delta_in_neighborhood(self, n_samples, division_offset=1):
        """
        Samples random locations in a circle centered at zero with
        radius equal to neighborhood size.
        """

        # neighborhood size:
        R = self.neighborhood_size

        # randomization:
        u_r = np.random.uniform(size=n_samples)
        u_theta = np.random.uniform(size=n_samples)

        # sample radii uniformally:
        # derived from: p(r<r') / p(r<R) = pi*r'^2 / pi*R^2
        r = R * np.sqrt(u_r)

        # sample angle:
        theta = 2 * np.pi * u_theta

        # compute location:
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return division_offset * np.array([x, y]).T

    def cells_inbound_mask(self, cell_df: pd.DataFrame, x_max: float, y_max: float) -> np.ndarray:
        x = cell_df.x.values
        y = cell_df.y.values

        x_is_in_bounds = (x <= x_max) & (x >= 0)
        y_is_in_bounds = (y <= y_max) & (y >= 0)

        return x_is_in_bounds & y_is_in_bounds

    def tissue_to_dataset(self, tissue: Tissue):
        """
        Returns a neighbor counts dataset based on tissue.
        """
        return ExtrapolateNeighborsDataset(tissue, neighborhood_size=self.neighborhood_size)
