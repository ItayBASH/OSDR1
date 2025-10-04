import pandas as pd
from tdm.dataset import Dataset
from tdm.utils import log2_1p


def max_density_per_cell_type(
    ds: Dataset, log_transform: bool = False, quantile: float | None = None
) -> dict[str, float]:
    """Computes the maximal density for each type.

    Args:
        nds (NeighborsDataset): _description_

    Returns:
        dict[str, float]: _description_
    """
    types = ds.cell_types()
    counts = pd.concat([ds.fetch(t)[0] for t in types])

    if quantile is None:
        counts = {t: counts[t].max() for t in types}
    else:
        counts = {t: counts[t].quantile(q=quantile) for t in types}

    if log_transform:
        counts = {k: log2_1p(v) for k, v in counts.items()}

    return counts
