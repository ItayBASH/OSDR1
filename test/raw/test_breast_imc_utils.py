from tdm.raw.breast_imc import read_single_cell_df, N_PATIENTS_WITH_PRIMARY_TUMORS, N_PATIENTS_WITH_MATCHED_LYMPH_METS
import pytest


@pytest.mark.skip("Can't currently download this dataset automatically, multiple R files, clustering etc.")
def test_read_single_cell_df():
    """Read the single cell dataframe and test basic statistics."""
    # test correct number of primary tumor patients:
    sc_df = read_single_cell_df(tissue_type="primary tumor")
    assert len(sc_df["PID"].unique()) == N_PATIENTS_WITH_PRIMARY_TUMORS

    # test correct number of matched lymph node mets patients:
    sc_df = read_single_cell_df(tissue_type="lymph node mts")
    assert len(sc_df["PID"].unique()) == N_PATIENTS_WITH_MATCHED_LYMPH_METS
