from tdm.tissue.tissue import Tissue
from tdm.publications.first.analyses import fm_analysis
from tdm.simulate.tissue_step import TissueStep
from tdm.simulate.generate_distribution import (
    get_tissue0,
    get_random_positions_on_phase_portrait,
    get_tissue_from_analysis,
    n_cells_per_neighborhood,
    simulate_n_tissues,
    simulate_one_tissue,
)


def test_tissue_step():
    ana = fm_analysis()
    start_a, start_b = get_random_positions_on_phase_portrait()

    tissue = get_tissue0(
        cell_a=ana.cell_a,
        cell_b=ana.cell_b,
        start_a=start_a,
        start_b=start_b,
        nbrhood_size=ana.neighborhood_size,
        tissue_width=ana.tissues[0].tissue_dimensions()[0],
        tissue_height=ana.tissues[0].tissue_dimensions()[1],
    )

    stepper = TissueStep(tissue, ana)
    stepper.step_n_times(3)

    assert len(stepper.tissues) == 4

    # test last_tissue getter:
    assert isinstance(stepper.last_tissue, Tissue)

    # test last_tissue setter:
    tissue_2 = get_tissue_from_analysis(ana, (start_a, start_b))  # testing another tissue constructor here
    last_tissue_len_before_assignment = len(stepper.last_tissue.cell_df())
    stepper.last_tissue = tissue_2 + stepper.last_tissue
    assert len(stepper.last_tissue.cell_df()) > last_tissue_len_before_assignment

    # test stepper doesn't fail when cells reach zero:
    ana.model.set_death_prob(cell_type="F", val=0.9999)  # near-certain death probability
    ana.model.set_death_prob(cell_type="M", val=0.9999)  # near-certain death probability

    stepper = TissueStep(tissue, ana)
    stepper.step_n_times(100)
    assert len(stepper.tissues) < 90  # we don't add empty tissues


def test_n_cell_conversion():

    n = n_cells_per_neighborhood(
        n_cells_per_tissue=100,
        tissue_width=10,
        tissue_height=10,
        neighborhood_size=1,
    )
    # tissue area = 100, neighborhood area = pi * 1^2 = 3.14
    assert n == int(100 * (3.14 / 100))


def test_simulate_n_tissues():

    ana = fm_analysis()
    n = 3  # number of tissues to simulate
    tissues = simulate_n_tissues(
        cell_a=ana.cell_a,
        cell_b=ana.cell_b,
        n=n,
        model=ana.model,
        neighborhood_size=ana.neighborhood_size,
        n_steps=10,
    )

    assert len(tissues) == n
    assert isinstance(tissues[0], Tissue)
    assert isinstance(tissues[-1], Tissue)


def test_simulate_one_tissue():
    ana = fm_analysis()
    tissue = simulate_one_tissue(
        cell_a=ana.cell_a,
        cell_b=ana.cell_b,
        model=ana.model,
        neighborhood_size=ana.neighborhood_size,
        n_steps=10,
    )
    assert isinstance(tissue, Tissue)
