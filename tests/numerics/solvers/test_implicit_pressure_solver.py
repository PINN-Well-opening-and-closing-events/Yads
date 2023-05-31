import pytest
import numpy as np  # type: ignore

from yads.mesh import load_mesh
from yads.numerics.solvers.implicit_pressure_solver import implicit_pressure_solver


def test_wrong_inputs():
    grid = load_mesh("./meshes/2D/Tests/rod_3_1/rod_3_1.mesh")
    K = np.ones(grid.nb_cells)
    T = np.ones(grid.nb_faces)
    S = np.ones(grid.nb_cells)
    P = np.ones(grid.nb_cells)

    Pb = {"1": 2.0, "2": 1.0}

    Sb_d = {"1": 1.0, "2": 0.1}
    Sb_n = {"1": None, "2": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    mu_w, mu_o = 1.0, 1.0

    K_wrong_1 = np.full(grid.nb_cells, -1.0)
    with pytest.raises(
        ValueError, match=r"Permeability K must contain only positive values"
    ):
        implicit_pressure_solver(grid, K_wrong_1, T, P, S, Pb, Sb_dict, mu_w, mu_o)

    K_wrong_2 = np.ones(grid.nb_faces)
    with pytest.raises(ValueError, match=r"K length must match grid.nb_cells"):
        implicit_pressure_solver(grid, K_wrong_2, T, P, S, Pb, Sb_dict, mu_w, mu_o)
