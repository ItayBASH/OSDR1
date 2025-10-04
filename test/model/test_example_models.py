import numpy as np
import pandas as pd
from typing import Iterable

from tdm.model.model import Model
from tdm.numerical.phase_portrait_analysis import get_classified_fixed_points_inbounds
from tdm.model.example_models import (
    model_with_one_stable_fixed_point,
    model_with_two_stable_fixed_points,
    model_with_two_stable_fixed_points_on_axes,
    model_with_three_stable_fixed_points,
    get_models,
    get_valid_stable_points,
)

from tdm.numerical.phase_portrait_analysis import (
    has_semi_stable_point_on_x_axis,
    has_semi_stable_point_on_y_axis,
    has_stable_point_on_x_axis,
    has_stable_point_on_y_axis,
    has_stable_point_off_axis,
)


def to_features(F, M):
    return pd.DataFrame([[1, F, M, F**2, F * M, M**2]], columns=["1", "F", "M", "F^2", "F M", "M^2"])


def point_is_in_list(point: tuple[float, float], _list: Iterable[tuple[float, float]], eps: float = 1e-7):
    """
    Parameters:
        p (tuple[float, float]):
            x,y values of a point

        l (Iterable[tuple[float, float]]):
            a list of points

    Returns:
        true if p is in l, otherwise false

    """
    return (np.linalg.norm(np.array(point) - _list, axis=1) < eps).sum()


def test_model_with_one_stable_fixed_point():
    m = model_with_one_stable_fixed_point()

    D = 0.02
    D_max = 0.05

    # Model division rate is equal death rate (D) and equal to maximal value (D_max) in the correct places:
    assert np.allclose(m.predict("F", "division", to_features(4, 0)).item(), D)
    assert np.allclose(m.predict("M", "division", to_features(0, 4)).item(), D)

    assert np.allclose(m.predict("F", "division", to_features(2, 0)).item(), D_max)
    assert np.allclose(m.predict("M", "division", to_features(0, 2)).item(), D_max)

    fps_dict = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))

    stable_fps = fps_dict["stable"]
    assert len(stable_fps) == 1
    assert point_is_in_list((4, 4), stable_fps)

    semistable_fps = fps_dict["semi-stable"]
    assert len(semistable_fps) == 3
    assert point_is_in_list((0, 0), semistable_fps)
    assert point_is_in_list((4, 0), semistable_fps)
    assert point_is_in_list((0, 4), semistable_fps)

    unstable_fps = fps_dict["unstable"]
    assert len(unstable_fps) == 0


def test_model_with_two_stable_fixed_points():
    m = model_with_two_stable_fixed_points()

    D = 0.02
    D_max = 0.05

    # Firoblast model division rate is equal to death rate (D) at (4,0) and (2,0):
    assert np.allclose(m.predict("F", "division", to_features(4, 0)).item(), D)
    assert np.allclose(m.predict("F", "division", to_features(2, 0)).item(), D)

    # Macrophage model division rate is equal to death rate (D) at (0,4) but not at (0,2):
    assert np.allclose(m.predict("M", "division", to_features(0, 4)).item(), D)
    assert not np.allclose(m.predict("M", "division", to_features(0, 2)).item(), D)

    # Fibroblast model reaches max value at (3,0):
    assert np.allclose(m.predict("F", "division", to_features(3, 0)).item(), D_max)

    # Macrophage model reaches max value at (0,2):
    assert np.allclose(m.predict("M", "division", to_features(0, 2)).item(), D_max)

    fps_dict = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))

    stable_fps = fps_dict["stable"]
    assert len(stable_fps) == 2
    assert point_is_in_list((4, 4), stable_fps)
    assert point_is_in_list((0, 4), stable_fps)

    semistable_fps = fps_dict["semi-stable"]
    assert len(semistable_fps) == 2
    assert point_is_in_list((0, 0), semistable_fps)
    assert point_is_in_list((4, 0), semistable_fps)

    unstable_fps = fps_dict["unstable"]
    assert len(unstable_fps) == 2
    assert point_is_in_list((2, 0), unstable_fps)
    assert point_is_in_list((2, 4), unstable_fps)


def test_model_with_two_stable_fixed_points_on_axes():
    m = model_with_two_stable_fixed_points_on_axes()

    D = 0.02
    D_max = 0.06

    # Firoblast model division rate is equal to death rate (D) at (6,0), (2,0) and (4,2):
    assert np.allclose(m.predict("F", "division", to_features(6, 0)).item(), D)
    assert np.allclose(m.predict("F", "division", to_features(2, 0)).item(), D)
    assert np.allclose(m.predict("F", "division", to_features(4.5, 2)).item(), D)

    # Macrophage model division rate is equal to death rate (D) at (0,6), (0,2) and (2,4):
    assert np.allclose(m.predict("M", "division", to_features(0, 6)).item(), D)
    assert np.allclose(m.predict("M", "division", to_features(0, 2)).item(), D)
    assert np.allclose(m.predict("M", "division", to_features(2, 4.5)).item(), D)

    # Fibroblast model reaches max value at (4,0):
    assert np.allclose(m.predict("F", "division", to_features(4.5, 0)).item(), D_max)

    # Macrophage model reaches max value at (0,4):
    assert np.allclose(m.predict("M", "division", to_features(0, 4.5)).item(), D_max)

    fps_dict = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))

    stable_fps = fps_dict["stable"]
    assert len(stable_fps) == 3
    assert point_is_in_list((6, 0), stable_fps)
    assert point_is_in_list((0, 6), stable_fps)
    assert point_is_in_list((0, 0), stable_fps)

    semistable_fps = fps_dict["semi-stable"]
    assert len(semistable_fps) == 0

    unstable_fps = fps_dict["unstable"]
    assert len(unstable_fps) == 2
    assert point_is_in_list((2, 0), unstable_fps)
    assert point_is_in_list((0, 2), unstable_fps)


def test_model_with_three_stable_fixed_points():
    m = model_with_three_stable_fixed_points()

    D = 0.02
    D_max = 0.05

    # Firoblast model division rate is equal to death rate (D) at (4,0), (2,0) and (4,2):
    assert np.allclose(m.predict("F", "division", to_features(4, 0)).item(), D)
    assert np.allclose(m.predict("F", "division", to_features(2, 0)).item(), D)

    # Macrophage model division rate is equal to death rate (D) at (0,6), (0,2) and (2,4):
    assert np.allclose(m.predict("M", "division", to_features(0, 4)).item(), D)
    assert np.allclose(m.predict("M", "division", to_features(0, 2)).item(), D)

    # Fibroblast model reaches max value at (2,0):
    assert np.allclose(m.predict("F", "division", to_features(3, 0)).item(), D_max)

    # Macrophage model reaches max value at (0,2):
    assert np.allclose(m.predict("M", "division", to_features(0, 3)).item(), D_max)

    fps_dict = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))

    stable_fps = fps_dict["stable"]
    assert len(stable_fps) == 4
    assert point_is_in_list((4, 0), stable_fps)
    assert point_is_in_list((0, 4), stable_fps)
    assert point_is_in_list((4, 4), stable_fps)
    assert point_is_in_list((0, 0), stable_fps)

    semistable_fps = fps_dict["semi-stable"]
    assert len(semistable_fps) == 0

    unstable_fps = fps_dict["unstable"]
    assert len(unstable_fps) == 5
    assert point_is_in_list((2, 0), unstable_fps)
    assert point_is_in_list((0, 2), unstable_fps)
    assert point_is_in_list((2, 2), unstable_fps)
    assert point_is_in_list((4, 2), unstable_fps)
    assert point_is_in_list((2, 4), unstable_fps)


def test_get_models():
    models = get_models()
    assert len(models) == 4
    assert all(isinstance(m, Model) for m in models)


def test_model_identification_funcs():

    # correctly identifies stable_point_off_axis:
    m = model_with_one_stable_fixed_point()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert has_stable_point_off_axis(fps)

    # doesn't identify stable_point_off_axis when it's not present:
    m = model_with_two_stable_fixed_points_on_axes()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert not has_stable_point_off_axis(fps)

    # correctly identifies stable_point_on_y_axis:
    m = model_with_two_stable_fixed_points_on_axes()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert has_stable_point_on_y_axis(fps)

    # doesn't identify stable_point_on_y_axis when it's not present:
    m = model_with_one_stable_fixed_point()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert not has_stable_point_on_y_axis(fps)

    # correctly identifies stable_point_on_x_axis:
    m = model_with_two_stable_fixed_points_on_axes()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert has_stable_point_on_x_axis(fps)

    # doesn't identify stable_point_on_x_axis when it's not present:
    m = model_with_two_stable_fixed_points()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert not has_stable_point_on_x_axis(fps)

    # correctly identifies semi_stable_point_on_y_axis:
    m = model_with_one_stable_fixed_point()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert has_semi_stable_point_on_y_axis(fps)

    # doesn't identify semi_stable_point_on_y_axis when it's not present:
    m = model_with_two_stable_fixed_points_on_axes()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert not has_semi_stable_point_on_y_axis(fps)

    # correctly identifies semi_stable_point_on_x_axis:
    m = model_with_one_stable_fixed_point()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert has_semi_stable_point_on_x_axis(fps)

    # doesn't identify semi_stable_point_on_x_axis when it's not present:
    m = model_with_two_stable_fixed_points_on_axes()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert not has_semi_stable_point_on_x_axis(fps)

    # counts the correct number of stable points:
    m = model_with_two_stable_fixed_points()
    fps = get_classified_fixed_points_inbounds(m, "F", "M", (0, 8), (0, 8))
    assert len(get_valid_stable_points(fps["stable"], fps["unstable"])) == 2

    # case where there is an unstable point within less than 1 cell from an axis:
    assert len(get_valid_stable_points([(3, 0)], [(3, 1)])) == 0
