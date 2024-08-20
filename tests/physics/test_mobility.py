import pytest
import numpy as np  # type: ignore
from yads.physics.mobility import (
    total_mobility,
    calculate_mobility,
    d_total_mobility_ds,
)
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian


def test_wrong_inputs():
    with pytest.raises(ValueError, match=f"Model not handled yet"):
        total_mobility(0.5, 0.5, 0.5, model="error")

    mu_wrong, mu_ok = -1.0, 1.0
    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        total_mobility(1.0, mu_w=mu_wrong, mu_o=mu_ok)
        total_mobility(1.0, mu_w=mu_wrong, mu_o=mu_ok, model="quadratic")

    grid = create_2d_cartesian(50 * 200, 1000, 10, 1)
    P = S = np.ones(grid.nb_cells)
    Pb = {"left": 1.0, "right": 2.0}
    Sb_d = {"left": 1.0, "right": 0.1}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        calculate_mobility(grid, P, S, Pb, Sb_dict, mu_w=mu_wrong, mu_o=mu_ok)

    P_wrong = S_wrong = np.ones(grid.nb_faces)

    with pytest.raises(ValueError, match=r"viscosity must be positive"):
        d_total_mobility_ds(sw=S, mu_w=mu_wrong, mu_o=mu_ok, model="cross")

    with pytest.raises(ValueError, match=r"Model not handled yet"):
        d_total_mobility_ds(sw=S, mu_w=mu_ok, mu_o=mu_ok, model="error")


def test_output():
    grid = create_2d_cartesian(50 * 200, 1000, 10, 1)
    mu_wrong, mu_ok = -1.0, 1.0
    P = S = np.ones(grid.nb_cells)
    Pb = {"left": 1.0, "right": 2.0}
    Sb_d = {"left": 1.0, "right": 0.1}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    calculate_mobility(grid, P, S, Pb, Sb_dict, mu_w=mu_ok, mu_o=mu_ok)
