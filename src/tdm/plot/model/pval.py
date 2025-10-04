from tdm.analysis import Analysis
import pandas as pd


def LLR_pval_per_cell_type(ana: Analysis) -> pd.DataFrame:
    """Get the log-likelihood-ratio pvalues for the model of each cell-type.

    Warning:
        Works only for analyses using :class:`~tdm.model.LogisticRegressionModel`
    """
    m = ana.model
    pvals = pd.DataFrame(
        {
            "cell_type": m.cell_types(),
            "LLR p-value": [m.models[c]["division"].llr_pvalue for c in m.cell_types()],
        }
    )
    return pvals


def fit_summary(ana: Analysis, cell_type: str) -> str:
    m = ana.model
    return m.models[cell_type]["division"].summary()
