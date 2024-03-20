import pytest
import numpy as np  # type: ignore

from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.mesh import load_mesh
from yads.numerics.solvers.implicit_saturation_solver import implicit_saturation_solver
from yads.numerics.solvers.implicit_pressure_solver import implicit_pressure_solver
from yads.numerics.physics.calculate_transmissivity import calculate_transmissivity
import yads.physics as yp


def test_wrong_inputs():
    grid = create_2d_cartesian(50 * 200, 1000, 10, 1)
    # Porosity
    phi = np.ones(grid.nb_cells)
    # Diffusion coefficient (i.e Permeability)
    K = np.ones(grid.nb_cells)
    # Water saturation initialization
    S = np.full(grid.nb_cells, 0.1)
    # Pressure initialization
    P = np.full(grid.nb_cells, 1.5)

    T = calculate_transmissivity(grid, K)

    mu_w = 1.0
    mu_o = 1.0

    M = yp.total_mobility(S, mu_w, mu_o)

    # BOUNDARY CONDITIONS #
    Pb = {"left": 2.0, "right": 1.0}
    Sb_d = {"left": 1.0, "right": 0.1}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 0.01
    P = implicit_pressure_solver(grid, K, T, M, P, Pb, Sb_dict, mu_w, mu_o)

    with pytest.raises(ValueError, match=r"phi length must match grid.nb_cells"):
        phi_wrong = np.ones(grid.nb_faces)
        implicit_saturation_solver(grid, P, S, T, phi_wrong, Pb, Sb_dict, dt, mu_w)

    with pytest.raises(ValueError, match=r"Time step dt must be positive"):
        dt_wrong = -1.0
        implicit_saturation_solver(grid, P, S, T, phi, Pb, Sb_dict, dt_wrong, mu_w)

    with pytest.raises(ValueError, match=r"newton stop criterion eps must be positive"):
        eps_wrong = -1.0
        implicit_saturation_solver(
            grid, P, S, T, phi, Pb, Sb_dict, dt, mu_w, eps=eps_wrong
        )
