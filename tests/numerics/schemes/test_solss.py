import pytest
import numpy as np

from yads.numerics.schemes.solss import solss
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.wells import Well


def test_wrong_inputs():
    pass


def test_outputs():
    # 2D example
    grid = create_2d_cartesian(Lx=3, Ly=3, Nx=3, Ny=3)

    # PHYSICS

    # Porosity
    phi = np.ones(grid.nb_cells)
    # Diffusion coefficient (i.e Permeability)
    K = np.ones(grid.nb_cells)
    # Water saturation initialization
    S = np.full(grid.nb_cells, 0.01)
    # Pressure initialization
    P = np.full(grid.nb_cells, 1.5)

    mu_w = 1.0
    mu_o = 1.0

    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 2.0, "right": 1.0}

    # Saturation
    Sb_d = {"left": 1.0, "right": 0.1}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 0.5
    total_sim_time = 0.5
    max_iter = 5

    well_test = Well(
        name="well test",
        cell_group=np.array([[0.5, 0.5]]),
        radius=0.1,
        control={"Neumann": -0.1},
        s_inj=1.0,
        schedule=[[0, 0.5]],
        mode="injector",
    )

    solss(
        grid,
        P,
        S,
        Pb,
        Sb_dict,
        phi,
        K,
        mu_w,
        mu_o,
        total_sim_time=total_sim_time,
        dt_init=dt,
        max_newton_iter=max_iter,
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

    solss(
        grid,
        P,
        S,
        Pb,
        Sb_dict,
        phi,
        K,
        mu_w,
        mu_o,
        total_sim_time=total_sim_time,
        dt_init=dt,
        max_newton_iter=max_iter,
        wells=[well_test],
    )
