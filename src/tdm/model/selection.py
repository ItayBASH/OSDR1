import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from copy import deepcopy

from tqdm import tqdm
import seaborn as sns

from tdm.analysis import Analysis


def cross_validation(
    single_cell_df: pd.DataFrame,
    model_dicts: dict,
    cell_types_to_model: list[str],
    allowed_neighbor_types: list[str],
    neighborhood_mode: str,
    n_splits: int = 10,
) -> pd.DataFrame:
    """Perform cross validation - partitioning data at the level of tissues.

    Args:
        single_cell_df (pd.DataFrame): the single cell dataframe.
        model_dicts (dict): a dictionary that maps a string describing a setting (e.g "degree 2, log-transform"), to a dictionary with kwargs.
        cell_types_to_model (list[str]): list of cell types to model. See: Analysis
        allowed_neighbor_types (list[str]): list of allowed neighbor types. See: Analysis
        neighborhood_mode (str): the neighborhood mode. See: Analysis
        n_splits (int, optional): number of folds for cross validation. Defaults to 10.

    Returns:
        results: dataframe with columns: setting, fold, loss and cell_type.


    Examples:
        >>>
    """

    ana = Analysis(
        single_cell_df=single_cell_df,
        cell_types_to_model=cell_types_to_model,
        neighborhood_mode=neighborhood_mode,
        allowed_neighbor_types=allowed_neighbor_types,
        end_phase=1,  # just split into tissues
        verbose=False,
    )

    res: dict[str, list[str | int | float]] = {"setting": [], "fold": [], "loss": [], "cell_type": []}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_i, (train_index, test_index) in tqdm(enumerate(kf.split(ana.tissues))):

        # partition tissues to train and test, and count neighbors separately for each:
        train_ana = deepcopy(ana)
        test_ana = deepcopy(ana)

        train_ana.tissues = [ana.tissues[i] for i in train_index]
        test_ana.tissues = [ana.tissues[i] for i in test_index]

        # phases 2,3 count neighbors and filter cells:
        train_ana.run(start_phase=2, end_phase=3, verbose=False)
        test_ana.run(start_phase=2, end_phase=3, verbose=False)

        # construct features and fit models to train and test sets:
        for model_name, model_dict in model_dicts.items():

            # overwrite original kwargs:
            if "model_kwargs" in model_dict.keys():
                train_ana._model_kwargs = model_dict["model_kwargs"]
                # test_ana._model_kwargs = model_dict['model_kwargs'] NOT REQUIRED, DON'T FIT A MODEL TO TEST SET

            if "polynomial_dataset_kwargs" in model_dict.keys():
                train_ana._polynomial_dataset_kwargs = model_dict["polynomial_dataset_kwargs"]
                # test_ana._polynomial_dataset_kwargs = model_dict['polynomial_dataset_kwargs'] USE TRAIN_ANA TO TRANSFORM TEST SET COUNTS INTO FEATURES, IN CASE OF SCALING

            # transform features and refit model:
            train_ana.run(start_phase=4, verbose=False)

            for cell_type in ana.cell_types:

                # transform (e.g normalize, log) counts from test dataset:
                test_counts, test_obs = test_ana.rnds.fetch(cell_type=cell_type)
                test_features = train_ana.pds.construct_features_from_counts(
                    test_counts, target_cell=cell_type
                )  # normalize according to training set
                train_model = train_ana.model

                # use _predict to get unprocessed probabilities
                test_probs = train_model._predict(cell_type=cell_type, obs="division", features=test_features)
                loss = log_loss(test_obs, test_probs)

                res["setting"].append(model_name)
                res["fold"].append(fold_i)
                res["loss"].append(loss)
                res["cell_type"].append(cell_type)

    return pd.DataFrame(res)


def cross_validation2(
    single_cell_df: pd.DataFrame,
    model_dicts: dict,
    cell_types_to_model: list[str],
    allowed_neighbor_types: list[str],
    neighborhood_mode: str,
    n_splits: int = 10,
) -> pd.DataFrame:
    """Perform cross validation - partitioning data at the level of tissues.

    Args:
        single_cell_df (pd.DataFrame): the single cell dataframe.
        model_dicts (dict): a dictionary that maps a string describing a setting (e.g "degree 2, log-transform"), to a dictionary with kwargs.
        cell_types_to_model (list[str]): list of cell types to model. See: Analysis
        allowed_neighbor_types (list[str]): list of allowed neighbor types. See: Analysis
        neighborhood_mode (str): the neighborhood mode. See: Analysis
        n_splits (int, optional): number of folds for cross validation. Defaults to 10.

    Returns:
        results: dataframe with columns: setting, fold, loss and cell_type.


    Examples:
        >>>
    """

    ana = Analysis(
        single_cell_df=single_cell_df,
        cell_types_to_model=cell_types_to_model,
        neighborhood_mode=neighborhood_mode,
        allowed_neighbor_types=allowed_neighbor_types,
        end_phase=3,  # count neighbors and filter cells
        verbose=False,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # init one fold generator per cell-type:
    fold_generators = {c: kf.split(*ana.rnds.fetch(c)) for c in ana.cell_types}

    res: dict[str, list[str | int | float]] = {"setting": [], "fold": [], "loss": [], "cell_type": []}

    for fold_i in tqdm(range(n_splits)):

        # partition tissues to train and test:
        train_ana = deepcopy(ana)
        test_ana = deepcopy(ana)

        # overwrite each cell's counts with the train/test subset:
        for cell_type in ana.rnds.cell_types():

            # non-partitioned data:
            features, obs = ana.rnds.fetch(cell_type)
            train_idxs, test_idxs = fold_generators[cell_type].__next__()

            train_ana.rnds.set_dataset(
                cell_type=cell_type,
                features=features.iloc[train_idxs].reset_index(drop=True),
                obs=obs.iloc[train_idxs].reset_index(drop=True),
            )

            test_ana.rnds.set_dataset(
                cell_type=cell_type,
                features=features.iloc[test_idxs].reset_index(drop=True),
                obs=obs.iloc[test_idxs].reset_index(drop=True),
            )

        # transform features and fit model according to each setting:
        for model_name, model_dict in model_dicts.items():

            # modify only training analysis, and use it to transform test data
            # this makes sure normalization is consistent

            if "model_kwargs" in model_dict.keys():
                train_ana._model_kwargs = model_dict["model_kwargs"]

            if "polynomial_dataset_kwargs" in model_dict.keys():
                train_ana._polynomial_dataset_kwargs = model_dict["polynomial_dataset_kwargs"]

            # transform features and refit model:
            train_ana.run(start_phase=4, verbose=False)

            # evaluate model:
            for cell_type in ana.cell_types:

                # transform (e.g normalize, log) counts from test dataset:
                test_counts, test_obs = test_ana.rnds.fetch(cell_type=cell_type)
                test_features = train_ana.pds.construct_features_from_counts(
                    test_counts, target_cell=cell_type
                )  # normalize according to training set
                train_model = train_ana.model

                # use _predict to get unprocessed probabilities
                test_probs = train_model._predict(cell_type=cell_type, obs="division", features=test_features)
                loss = log_loss(test_obs, test_probs)

                res["setting"].append(model_name)
                res["fold"].append(fold_i)
                res["loss"].append(loss)
                res["cell_type"].append(cell_type)

    return pd.DataFrame(res)


def plot_cross_validation_result(res: pd.DataFrame):
    g = sns.FacetGrid(res, col="cell_type", height=3, aspect=1, sharey=True, sharex=False, col_wrap=3)
    g.map(sns.boxplot, "loss", "setting", order=res.setting.unique())

    # rotate xticks 45 degrees for all axes
    # for ax in g.axes.flat:
    #     # set_ticks to avoid warning
    #     ax.set_xticks(ax.get_xticks())
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # g.tight_layout()
