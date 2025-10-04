from tdm.cell_types import CELL_TYPE_TO_FULL_NAME


def log_density_axis(cell_type: str) -> str:
    """Constructs the axis label for a log2 cell density plot.

    Args:
        cell_type (str): one of the cell types in CELL_TYPES_ARRAY (e.g 'F')

    Returns:
        str: formatted axis label
    """
    if cell_type in CELL_TYPE_TO_FULL_NAME.keys():
        return f"{CELL_TYPE_TO_FULL_NAME[cell_type]} density (log2 # cells)"
    else:
        return f"{cell_type} density (log2 # cells)"
