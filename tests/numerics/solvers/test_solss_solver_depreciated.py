import pytest
import numpy as np  # type: ignore

from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.mesh import load_mesh
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step_depreciated
from yads.wells import Well


def test_wrong_inputs():
    grid = create_2d_cartesian(50 * 200, 1000, 10, 1)

    P = np.zeros(grid.nb_cells)
    S = np.full(grid.nb_cells, 0.0)
    K = np.ones(grid.nb_cells)
    phi = np.ones(grid.nb_cells)

    T = np.ones(grid.nb_faces)

    Pb = {"left": 2.0, "right": 1.0}
    Sb_d = {"left": 1.0, "right": 0.1}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    dt = 0.1
    mu_w = mu_g = 1.0


def test_output():
    grid = create_2d_cartesian(50 * 200, 1000, 10, 1)

    P = np.zeros(grid.nb_cells)
    S = np.full(grid.nb_cells, 0.0)
    K = np.ones(grid.nb_cells)
    phi = np.ones(grid.nb_cells)

    T = np.ones(grid.nb_faces)

    Pb = {"left": 2.0, "right": 1.0}
    Sb_d = {"left": 1.0, "right": 0.1}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
    dt = 0.1
    mu_w, mu_g = 1.0, 1.0

    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Neumann": -0.1},
        s_inj=1.0,
        schedule=[[0, 0.5]],
        mode="injector",
    )

    solss_newton_step_depreciated(
        grid=grid,
        P_i=P,
        S_i=S,
        K=K,
        T=T,
        phi=phi,
        Pb=Pb,
        Sb_dict=Sb_dict,
        dt_min=-1,
        mu_w=mu_w,
        mu_g=mu_g,
        dt_init=dt,
        S_guess=S,
        P_guess=P,
        wells=[well_test],
    )

    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Dirichlet": 2.0},
        s_inj=1.0,
        schedule=[[0, 0.5]],
        mode="injector",
    )

    solss_newton_step_depreciated(
        grid=grid,
        P_i=P,
        S_i=S,
        K=K,
        T=T,
        phi=phi,
        Pb=Pb,
        Sb_dict=Sb_dict,
        dt_min=-1,
        mu_w=mu_w,
        mu_g=mu_g,
        dt_init=dt,
        S_guess=S,
        P_guess=P,
        wells=[well_test],
    )
