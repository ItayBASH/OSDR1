import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from tdm.dataset.polynomial import MyPolynomialFeatures
import pytest


@pytest.mark.parametrize(
    "degree, include_bias",
    [
        (1, True),
        (1, False),
        (2, True),
        (2, False),
        (3, True),
        (3, False),
    ],
)
def test_manual_vs_sklearn(degree, include_bias):

    X = np.random.uniform(size=(19, 5)) * 10
    X = pd.DataFrame(X, columns=[c for c in "abcde"])

    # features from sklearn
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    sklearn_result = poly.fit_transform(X)
    sklearn_feature_names_out = poly.get_feature_names_out()

    # features from manual function:
    mypoly = MyPolynomialFeatures(degree=degree, include_bias=include_bias)
    my_result = mypoly.fit_transform(X.values)
    my_feature_names_out = mypoly.feature_names_out(X.columns)

    # validate shapes:
    assert (
        sklearn_result.shape == my_result.shape
    ), f"Shape mismatch: sklearn={sklearn_result.shape}, manual={my_result.shape}"

    # validate numbers:
    assert np.allclose(sklearn_result, my_result, rtol=1e-6, atol=1e-9)

    # check same feature names out:
    assert np.all(sklearn_feature_names_out == my_feature_names_out)
