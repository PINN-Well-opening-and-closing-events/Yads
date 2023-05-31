import pytest
import numpy as np  # type: ignore
from yads.physics.mobility import total_mobility, calculate_mobility, d_total_mobility_ds
from yads.mesh import load_mesh


def test_wrong_inputs():
    with pytest.raises(ValueError, match=f"Model not handled yet"):
        total_mobility(0.5, 0.5, 0.5, model="error")

    mu_wrong, mu_ok = -1.0, 1.0
    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        total_mobility(1.0, mu_w=mu_wrong, mu_o=mu_ok)
        total_mobility(1.0, mu_w=mu_wrong, mu_o=mu_ok, model="quadratic")

    grid = load_mesh("./meshes/2D/Tests/rod_10_1/rod_10_1.mesh")
    P = S = np.ones(grid.nb_cells)
    Pb = {"1": 1.0, "2": 2.0}
    Sb_d = {"1": 1.0, "2": 0.1}
    Sb_n = {"1": None, "2": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        calculate_mobility(grid, P, S, Pb, Sb_dict, mu_w=mu_wrong, mu_o=mu_ok)

    P_wrong = S_wrong = np.ones(grid.nb_faces)

    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        d_total_mobility_ds(sw=S, mu_w=mu_wrong, mu_o=mu_ok, model='cross')

    with pytest.raises(ValueError, match=r"Model not handled yet"):
        d_total_mobility_ds(sw=S, mu_w=mu_ok, mu_o=mu_ok, model='error')


def test_output():
    grid = load_mesh("./meshes/2D/Tests/rod_10_1/rod_10_1.mesh")
    mu_wrong, mu_ok = -1.0, 1.0
    P = S = np.ones(grid.nb_cells)
    Pb = {"1": 1.0, "2": 2.0}
    Sb_d = {"1": 1.0, "2": 0.1}
    Sb_n = {"1": None, "2": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    calculate_mobility(grid, P, S, Pb, Sb_dict, mu_w=mu_ok, mu_o=mu_ok)
