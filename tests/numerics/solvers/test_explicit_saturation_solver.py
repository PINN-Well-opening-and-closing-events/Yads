import pytest
import numpy as np  # type: ignore

from yads.mesh import load_mesh
from yads.numerics.solvers.explicit_saturation_solver import explicit_saturation_solver


def test_wrong_inputs():
    grid = load_mesh("./meshes/2D/Tests/rod_3_1/rod_3_1.mesh")

    P = np.zeros(grid.nb_cells)
    S = np.full(grid.nb_cells, 0.0)
    K = np.ones(grid.nb_cells)
    phi = np.ones(grid.nb_cells)

    T = np.ones(grid.nb_faces)

    Pb = {"1": 2.0, "2": 1.0}
    Sb_d = {"1": 1.0, "2": 0.1}
    Sb_n = {"1": None, "2": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    dt = 0.1
    mu_w = mu_o = 1.0

    K_wrong_1 = np.ones(grid.nb_faces)
    with pytest.raises(ValueError, match=r"K length must match grid.nb_cells"):
        explicit_saturation_solver(
            grid, P, S, K_wrong_1, T, phi, Pb, Sb_dict, dt, mu_w, mu_o
        )
    K_wrong_2 = np.full(grid.nb_cells, -1.0)
    with pytest.raises(
        ValueError, match=r"Permeability K must contain only positive values"
    ):
        explicit_saturation_solver(
            grid, P, S, K_wrong_2, T, phi, Pb, Sb_dict, dt, mu_w, mu_o
        )
    S_wrong_1 = np.full(grid.nb_cells, -1.0)
    with pytest.raises(
        ValueError, match=r"Saturation S must have all its values between 0 and 1"
    ):
        explicit_saturation_solver(
            grid, P, S_wrong_1, K, T, phi, Pb, Sb_dict, dt, mu_w, mu_o
        )
    phi_wrong_1 = np.full(grid.nb_cells, -1.0)
    with pytest.raises(
        ValueError, match=r"Porosity phi must contain only positive values"
    ):
        explicit_saturation_solver(
            grid, P, S, K, T, phi_wrong_1, Pb, Sb_dict, dt, mu_w, mu_o
        )
    with pytest.raises(ValueError, match=r"Time step dt must be positive"):
        explicit_saturation_solver(grid, P, S, K, T, phi, Pb, Sb_dict, -1, mu_w, mu_o)


def test_output():
    grid = load_mesh("./meshes/2D/Tests/rod_3_1/rod_3_1.mesh")

    P = np.zeros(grid.nb_cells)
    S = np.full(grid.nb_cells, 0.0)
    K = np.ones(grid.nb_cells)
    phi = np.ones(grid.nb_cells)

    T = np.ones(grid.nb_faces)
    M = np.ones(grid.nb_faces)

    Pb = {"1": 2.0, "2": 1.0}
    Sb_d = {"1": 1.0, "2": 0.1}
    Sb_n = {"1": None, "2": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    dt = 0.1
    mu_w, mu_o = 1.0, 1.0
