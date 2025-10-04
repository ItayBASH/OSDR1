from tdm.model.example_models import model_with_one_stable_fixed_point
from tdm.publications.first.analyses import fm_analysis
from tdm.model.logistic_regression import LogisticRegressionModel
import pytest


def test_parameters():
    m = model_with_one_stable_fixed_point()

    # test setter and getter:
    m.set_parameter(cell_type="F", parameter_name="F", value=-30, obs="division")
    assert m.get_parameter(cell_type="F", parameter_name="F", obs="division") == -30

    # test parameter funcs return something:
    assert len(m.parameters(cell_type="F", obs="division")) > 0
    assert len(m.parameter_stds(cell_type="F", obs="division")) > 0
    assert len(m.parameter_pvalues(cell_type="F", obs="division")) > 0


def test_fit():
    """
    A relatively naive test, making sure all flows run and produce predictions.
    """
    ana = fm_analysis()

    pds = ana.pds
    features, obs = pds.fetch("F")

    # standard fit:
    m = LogisticRegressionModel(
        dataset=pds,
    )
    # test that the model predicts something:
    assert len(m.predict(cell_type="F", obs="division", features=features)) > 0

    # fit regularized:
    m.fit(features=features, obs=obs, cell_type="F", regularization_alpha=100)
    assert len(m.predict(cell_type="F", obs="division", features=features)) > 0

    # fit regularized, with dict of alphas:
    m.fit(features=features, obs=obs, cell_type="F", regularization_alpha={"F": 100})
    assert len(m.predict(cell_type="F", obs="division", features=features)) > 0

    # fit regularized with score:
    m.fit(features=features, obs=obs, cell_type="F", regularization_alpha="score", score_kwargs={"metric": "bic"})
    assert len(m.predict(cell_type="F", obs="division", features=features)) > 0

    # fit regularized with score:
    m.fit(features=features, obs=obs, cell_type="F", regularization_alpha="score", score_kwargs={"metric": "aic"})
    assert len(m.predict(cell_type="F", obs="division", features=features)) > 0

    # fit regularized with cv:
    m.fit(features=features, obs=obs, cell_type="F", regularization_alpha="cv", cv_kwargs={"k_folds": 2})
    assert len(m.predict(cell_type="F", obs="division", features=features)) > 0

    # raises error if alpha is invalid:
    with pytest.raises(ValueError):
        m.fit(features=features, obs=obs, cell_type="F", regularization_alpha="invalid")  # type: ignore


def test_functions_raise_bad_argument():

    ana = fm_analysis()

    # Test invalid obs type
    with pytest.raises(ValueError):
        ana.model._predict(cell_type="F", obs="invalid", features=ana.pds.fetch("F")[0])
