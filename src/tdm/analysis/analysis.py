"""
Base class for all 2-cell analyses.
"""

from collections.abc import Iterable
import numpy as np

from joblib import dump, load
from tabulate import tabulate  # type: ignore
import pandas as pd
from typing import Literal, cast

from tdm.paths import CACHED_ANALYSES_DIR
from tdm.utils import cprint, verbosity, microns, log2_1p
from tdm.preprocess.single_cell_df import IMG_ID_COL, CELL_TYPE_COL, cell_types, n_cells_per_type
from tdm.tissue import Tissue
from tdm.dataset import (
    Dataset,
    NeighborsDataset,
    ExtrapolateNeighborsDataset,
    RestrictedNeighborsDataset,
    PolynomialDataset,
    ConcatDataset,
)
from tdm.model import Model, LogisticRegressionModel
from tdm.model.maximal_density_enforcer import MaximalDensityEnforcer, CellTypeSpecificDensityEnforcer

nds_classes = {"exclude": NeighborsDataset, "extrapolate": ExtrapolateNeighborsDataset}

# default neighborhood size:
_80_microns = microns(80)


def skip_phase(phase: int, start_phase: int, end_phase: int):
    return (phase < start_phase) or (phase > end_phase)


class Analysis:

    @verbosity
    def __init__(
        self,
        single_cell_df: pd.DataFrame,
        cell_types_to_model: list[str] | None = None,
        neighborhood_size: float = _80_microns,
        neighborhood_mode: Literal["exclude", "extrapolate"] = "exclude",
        nds_class_kwargs: dict | None = None,
        allowed_neighbor_types: list[str] | None = None,
        enforce_max_density: bool = True,
        max_density_enforcer_power: int = 8,
        polynomial_dataset_kwargs: dict | None = None,
        model_class: type[Model] = type[LogisticRegressionModel],  # type: ignore
        model_kwargs: dict | None = None,
        end_phase: int = 5,
        supported_cell_types: list[str] | None = None,
        verbose: bool = False,
    ):
        r"""An analysis organizes and performs all of the steps required to fit a model to the data in single_cell_df.

        Args:
            single_cell_df (pd.DataFrame): the single cell dataframe (see :ref:`Preprocess`)
            cell_types_to_model (list[str] | None, optional): defines the axes of the state-space of the dynamical model. defaults to all cell types.
            neighborhood_size (float, optional): radius for counting cell types. Defaults to _80_microns.
            neighborhood_mode (Literal[&quot;exclude&quot;, &quot;extrapolate&quot;], optional): exclude cells whose neighborhood exceeds tissue limits or correct for the unobserved fraction. Defaults to "exclude".
            nds_class_kwargs (dict | None, optional): keywords for the neighbors dataset class. Defaults to None.
            allowed_neighbor_types (list[str] | None, optional): exclude cells with neighbors outside this list. Defaults to None.
            xlim (tuple[float, float], optional): x limits for plots. Defaults to the maximal density of the first cell type.
            ylim (tuple[float, float], optional): y limits for plots. Defaults to the maximal density of the second cell type.
            enforce_max_density (bool, optional): whether to add a correction so that there is no net growth at maximal density. Defaults to False.
            max_density_enforcer_power (int, optional): high powers produce smaller corrections. Defaults to 4.
            polynomial_dataset_kwargs (dict | None, optional): parameters for the :class:`tdm.dataset.PolynomialDataset`. Defaults to None.
            model_class (type[Model], optional): type of model to fit. Defaults to type[LogisticRegressionModel].
            end_phase (int, optional): stop analysis at this phase (inclusive). Defaults to 5. For example, end_phase = 4 skips model fit.
            supported_cell_types (list[str] | None, optional): provide an explicit list of supported cell types when using a patient-level single_cell_df that might not have an instance of all types. Defaults to None.
            verbose (bool, optional): print stages of the analysis. Defaults to False.

        Examples:

            >>> ana = Analysis(
            >>>    single_cell_df=single_cell_df,
            >>>    cell_types_to_model=[FIBROBLAST, MACROPHAGE],
            >>>    allowed_neighbor_types=[FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL],
            >>>    polynomial_dataset_kwargs={"degree":2},
            >>>    xlim=(0,7.1),
            >>>    ylim=(0,6.1),
            >>>    neighborhood_mode='extrapolate',
            >>> )

        Warning:
            Analysis infers the cell types from the single_cell_df:

                >>> single_cell_df[CELL_TYPE_COL].unique()

            Provide an explicit list of ``supported_cell_types`` when using a small single_cell_df (e.g., one patient) that might not have an instance of every cell type.

        """
        # save arguments as instance properties:
        self._single_cell_df = single_cell_df
        self._cell_types_to_model = cell_types_to_model or cell_types(single_cell_df)
        self._neighborhood_size = neighborhood_size
        self._neighborhood_mode = neighborhood_mode
        self._nds_class_kwargs = nds_class_kwargs
        self._allowed_neighbor_types = allowed_neighbor_types
        self._enforce_max_density = enforce_max_density
        self._max_density_enforcer_power = max_density_enforcer_power
        self._polynomial_dataset_kwargs = polynomial_dataset_kwargs
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self._supported_cell_types = supported_cell_types

        # run analysis:
        self.run(end_phase=end_phase)

    @verbosity
    def run(
        self,
        start_phase: int = 1,
        end_phase: int = 5,
        verbose: bool = True,
    ):
        """Run the analysis.

            - 1: construct tissues
            - 2: count neighbors
            - 3: filter cell types
            - 4: transform features
            - 5: fit model

        Args:
            start_phase (int, optional): run phases from here (inclusive). Defaults to 1.
            end_phase (int, optional): stop at this phase (inclusive). Defaults to 5.
        """

        # Each phase has a tuple:
        # (message, method to run, attribute set)
        phases = [
            ("1/5 Constructing tissues", self._construct_tissues, "_tissues"),
            ("2/5 Counting cell-neighbors", self._count_neighbors, "_ndss"),
            ("3/5 Filtering cell-types", self._filter_cell_types, "_rnds"),
            ("4/5 Transforming features", self._transform_features, "_pds"),
            ("5/5 Fitting a model", self._fit_model, "_model"),
        ]

        # Loop through each phase and execute the corresponding method if not skipped
        for phase_num, (description, method, attr_name) in enumerate(phases, start=1):
            skip = skip_phase(phase_num, start_phase, end_phase)
            with analysis_phase(description, skip):
                if not skip:
                    setattr(self, attr_name, method())  # type: ignore

        # Some "post analysis" finishes:
        if (start_phase <= 2) and (end_phase >= 2):
            self._ax_lims = self._init_ax_lims()  # compute axis limits based on maximal neighbor counts

    @property
    def neighborhood_size(self) -> float:
        return self._neighborhood_size

    @property
    def tissues(self) -> list[Tissue]:
        return self._tissues

    @tissues.setter
    def tissues(self, value: list[Tissue]):
        self._tissues = value

    @property
    def ndss(self) -> list[NeighborsDataset]:
        return self._ndss

    @ndss.setter
    def ndss(self, value: list[NeighborsDataset]):
        self._ndss = value

    @property
    def nds(self) -> NeighborsDataset:
        if not hasattr(self, "_nds"):
            self._nds = cast(NeighborsDataset, ConcatDataset(self._ndss))
        return self._nds

    @property
    def rnds(self) -> RestrictedNeighborsDataset:
        return self._rnds

    @rnds.setter
    def rnds(self, value: RestrictedNeighborsDataset):
        self._rnds = value

    @property
    def pds(self) -> PolynomialDataset:
        return self._pds  # type: ignore

    @property
    def model(self) -> Model:
        return self._model  # type: ignore

    @property
    def cell_types(self) -> list[str]:
        return self._cell_types_to_model

    @property
    def cell_a(self) -> str:
        return self._cell_types_to_model[0]

    @property
    def cell_b(self) -> str:
        return self._cell_types_to_model[1]

    @property
    def cell_c(self) -> str:
        return self._cell_types_to_model[2]

    @property
    def xlim(self) -> tuple[float, float]:
        return self._ax_lims[0]

    @property
    def ylim(self) -> tuple[float, float]:
        return self._ax_lims[1]

    @property
    def zlim(self) -> tuple[float, float]:
        return self._ax_lims[2]

    @property
    def enforce_max_density(self) -> bool:
        return self._enforce_max_density

    def dump(self, filename: str):
        """Caches the analysis object.

        Warning:
            Overwrites existing files by default.

        Examples:
            >>> ana.dump("fibros_and_macs_15-05-2024.pkl")
            >>> ana = Analysis.load("fibros_and_macs_15-05-2024.pkl")

        Args:
            filename (str, optional): a descriptive name for the analysis, e.g "fibros_and_macs_15-05-2024.pkl".
        """
        dump(self, filename=CACHED_ANALYSES_DIR / filename)

    @staticmethod
    def load(filename: str):
        """Loads the analysis object.

        Examples:
            >>> ana.dump("fibros_and_macs_15-05-2024.pkl")
            >>> ana = Analysis.load("fibros_and_macs_15-05-2024.pkl")

        Args:
            filename (str): a descriptive name for the analysis, e.g "fibros_and_macs_15-05-2024.pkl"
        """
        return load(filename=CACHED_ANALYSES_DIR / filename)

    def tissue_states(self, scale_counts_to_common_radius: bool = True) -> pd.DataFrame:
        """Returns the number of cells of each type in each tissue in the analysis.

        Args:
            scale_counts_to_common_radius (bool, optional): scales the cell counts to a shared area of radius self.neighborhood_size. Defaults to True.

        Returns:
            pd.DataFrame: the cell counts
        """

        if scale_counts_to_common_radius:
            neighborhood_size = self.neighborhood_size
        else:
            neighborhood_size = None

        return pd.concat([t.n_cells_df(neighborhood_size=neighborhood_size) for t in self.tissues]).reset_index(
            drop=True
        )

    def get_tissues_by_ids(self, subject_ids: str | float | int | list | np.ndarray):

        if not isinstance(subject_ids, Iterable):
            subject_ids = [subject_ids]
        return [t for t in self.tissues if np.isin(t.subject_id, subject_ids)]

    def _construct_tissues(self) -> list[Tissue]:
        """
        Constructs a list of Tissue objects based on the given single cell DataFrame.

        Args:
            single_cell_df (pd.DataFrame): The single cell DataFrame containing the data.

        Returns:
            list[Tissue]: A list of Tissue objects.

        """
        # all supported cell_types:
        cell_types = self._supported_cell_types or list(self._single_cell_df[CELL_TYPE_COL].unique())

        tissues = []
        for _, img_df in iter(self._single_cell_df.groupby(IMG_ID_COL)):
            tissues.append(Tissue(img_df, cell_types=cell_types))

        return tissues

    def _count_neighbors(self) -> list[NeighborsDataset]:
        """
        Constructs and returns a list of NeighborsDataset objects for each tissue in the analysis.

        Returns:
            list[NeighborsDataset]: A list of NeighborsDataset objects.
        """
        nds_class = nds_classes[self._neighborhood_mode]
        ndss = []
        for tissue in self.tissues:
            nds = nds_class(tissue, self.neighborhood_size, **(self._nds_class_kwargs or {}))
            ndss.append(nds)
        return ndss

    def _init_ax_lims(self) -> list[tuple[float, float]]:
        all_counts = self.nds.fetch_all()[0]
        return [(0, log2_1p(all_counts[cell_type].max())) for cell_type in self.cell_types]

    def _filter_cell_types(self):
        """
        Constructs a RestrictedNeighborsDataset object based on the neighbors datasets of the analysis.

        Returns:
            RestrictedNeighborsDataset: The constructed RestrictedNeighborsDataset object.
        """
        return RestrictedNeighborsDataset(
            nds=self.nds,
            allowed_neighbor_types=self._allowed_neighbor_types,
            keep_types=self._cell_types_to_model,
        )

    def _transform_features(self):
        return PolynomialDataset(self.rnds, **(self._polynomial_dataset_kwargs or {}))

    def _fit_model(
        self,
        pds: Dataset | None = None,
        model_class: type[Model] = LogisticRegressionModel,
        model_kwargs: dict | None = None,
        enforce_max_density: bool | None = None,
        max_density_enforcer_power: int | None = None,
        max_density_enforcer_fixed_cell_counts: dict | None = None,
    ):

        # fit model:
        model = model_class(pds or self.pds, **(model_kwargs or self._model_kwargs or {}))

        if type(enforce_max_density) is bool:  # True or False
            enforce = enforce_max_density
        else:  # is None
            enforce = self.enforce_max_density

        # fit enforcer:
        if enforce:
            self.set_maximal_density_enforcer(
                model,
                max_density_enforcer_power=max_density_enforcer_power,
                max_density_enforcer_fixed_cell_counts=max_density_enforcer_fixed_cell_counts,
            )

        return model

    def set_maximal_density_enforcer(
        self,
        model: Model,
        max_density_enforcer_power: int | None = None,
        max_density_enforcer_fixed_cell_counts: dict | None = None,
    ):
        """_summary_

        Warning:
            modifies the passed model.

        Args:
            model (Model): a fitted model.
            max_density_enforcer_power (int | None): uses self._max_density_enforcer_power if None.
            max_density_enforcer_fixed_cell_counts (dict | None, optional): _description_. Defaults to None.
        """
        enforcer = self._maximal_density_enforcer(
            model,
            self.rnds,
            power=max_density_enforcer_power or self._max_density_enforcer_power,
            fixed_cell_counts=max_density_enforcer_fixed_cell_counts,  # required only for >2D analyses.
        )
        model.set_maximal_density_enforcement(enforcer=enforcer)

    def _maximal_density_enforcer(
        self, m: Model, nds: NeighborsDataset, fixed_cell_counts: dict[str, float] | None = None, power: int = 2
    ) -> MaximalDensityEnforcer:
        return CellTypeSpecificDensityEnforcer(
            model=m,
            nds=nds,
            cell_types=self._cell_types_to_model,
            power=power,
        )

    def __str__(self):
        cell_type_stats_df = tabulate(n_cells_per_type(self._single_cell_df), showindex="never")

        return f"""

Analysis:

Input: single_cell_df had the following cell-types TODO - add fraction of divisions per type
{cell_type_stats_df}

neighborhood_size = {int(self.neighborhood_size / 1e-6)} microns

Filters:
- cell-types used as model features: {self._cell_types_to_model}
- meighboring cell-types that exclude a cell from the analysis: {self._allowed_neighbor_types}


------------- TODO - COMPLETE REPORT ------------

"""


class analysis_phase:
    """Displays messages of progress through the phases of analysis."""

    def __init__(self, message: str, skip: bool):
        """Init the context manager.

        Args:
            msg (str): the message for the phase (e.g "1/5 Constructing Tissues")
        """
        self.message = message
        self.skip = skip

    def __enter__(self):
        """
        Print phase start message.
        """
        cprint(self.message, new_line=False, color="blue")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Print phase complete message.
        """
        if self.skip:
            cprint("Skipped..", color="yellow")
        else:
            cprint("[V]", color="green")
