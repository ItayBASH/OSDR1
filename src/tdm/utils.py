"""
Small functions used everywhere.
"""

import sys
import io
import numpy as np
import pandas as pd
from functools import wraps


def log2_1p(x):
    return np.log2(1 + x)


def inv_log2_1p(n):
    return 2**n - 1


def microns(n: float | pd.Series | np.ndarray) -> float:
    return n * 1e-6  # type: ignore


def dict_to_dataframe(d: dict | pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(d, pd.DataFrame):
        return d

    if np.all([np.isscalar(i) for i in d.values()]):
        d = {c: [i] for c, i in d.items()}
    return pd.DataFrame(d, columns=columns)


def exclude_idxs(_list: list, idxs: list[int]) -> list:
    return [e for i, e in enumerate(_list) if i not in idxs]


def subtract_from_list(_list: list, elements: list):
    return [e for e in _list if e not in elements]


def cprint(
    text,
    new_line: bool = True,
    bold: bool = False,
    underline: bool = False,
    color: str = "white",
):
    """
    Print text with color and style.

    Args:
        text (str): The text to print.
        new_line (bool): Whether to print a new line after the text.
        bold (bool): Whether to print the text in bold.
        underline (bool): Whether to print the text with an underline.
        color (str): The color of the text.
    """
    style_prefix = ""
    if bold:
        style_prefix += "\033[1m"
    if underline:
        style_prefix += "\033[4m"
    if color == "black":
        style_prefix += "\033[30m"
    if color == "red":
        style_prefix += "\033[91m"
    if color == "green":
        style_prefix += "\033[92m"
    if color == "blue":
        style_prefix += "\033[94m"
    if color == "yellow":
        style_prefix += "\033[93m"

    style_suffix = "\033[0m" if bold or underline else ""

    end = "\n" if new_line else " "

    if style_prefix == "" and style_suffix == "":
        print(text)
    else:
        print(style_prefix + text + style_suffix, end=end)


class Verbosity:
    def __init__(self, verbose=True):
        """Initialize Verbosity with a verbosity flag."""
        self.verbose = verbose
        self.null_output = io.StringIO()

    def __enter__(self):
        """Manage entry into a with-block; decide output destination based on verbosity."""
        self.original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = sys.stdout if self.verbose else self.null_output
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Manage exit from a with-block; restore original stdout."""
        sys.stdout = self.original_stdout  # Restore original stdout after leaving the block


def verbosity(func):
    """Decorator to control verbosity of function outputs dynamically."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        # Check for a 'verbose' keyword argument in the function call
        verbose = kwargs.pop("verbose", True)  # Default to True if not provided
        with Verbosity(verbose):
            return func(*args, **kwargs)

    return wrapped
