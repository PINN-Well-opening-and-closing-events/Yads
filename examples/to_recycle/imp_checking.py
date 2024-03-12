import yads.mesh as ym
from yads.mesh import Mesh
from yads.wells import Well
from yads.numerics import calculate_transmissivity
import numpy as np
from yads.numerics.solvers.implicit_pressure_solver_with_wells import (
    implicit_pressure_solver,
)
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
import copy


def solss_model_test_1_iter(
    grid: Mesh,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_g,
    mu_w,
    dt_init,
    total_sim_time,
    kr_model: str = "cross",
    max_newton_iter=10,
    eps=1e-6,
    wells=None,
    P_guess=None,
    S_guess=None,
):
    """

    grid:
    P:
    S: initial gas saturation or injected phase
    Pb:
    Sb_dict:
    phi:
    K:
    mu_g: gas viscosity or injected phase viscosity
    mu_w:
    kr_model:
    dt_init:
    total_sim_time:
    auto_dt:
    eps:

    Returns

    """
    T = calculate_transmissivity(grid, K)
    step = 0
    dt = dt_init
    dt_list = []
    total_time = 0.0
    newton_list = []
    dt_min = 0.0

    simulation_state = {
        "metadata": {
            "kr_model": kr_model,
            "P0": P.tolist(),
            "S0": S.tolist(),
            "Pb": Pb,
            "Sb_dict": Sb_dict,
            "T": T.tolist(),
            "K": K.tolist(),
            "Phi": phi.tolist(),
            "dt": dt_init,
            "wells": [well.well_to_dict() for well in wells],
            "eps": eps,
            "max_newton_iter": max_newton_iter,
            "mu_w": mu_w,
            "mu_g": mu_g,
            "total_sim_time": total_sim_time,
        },
        "data": {},
    }

    simulation_state["metadata"]["grid data"] = {
        "Lx": max(grid.node_coordinates[:, 0]),
        "Ly": max(grid.node_coordinates[:, 1]),
        "nb_cells": grid.nb_cells,
        "nb_faces": grid.nb_faces,
        "nb_nodes": grid.nb_nodes,
        "cell_centers": grid.centers(item="cell").tolist(),
        "type": grid.type,
        "dimension": grid.dim,
    }
    if wells:
        for well in wells:
            grid.connect_well(well)

    dt_save = copy.deepcopy(dt)
    effective_wells = []

    if wells:
        for well in wells:
            for schedule in well.schedule:
                if schedule[0] <= total_time < schedule[1]:
                    effective_wells.append(well)

    S_i = S
    P_i = P

    if P_guess is None:
        P_guess = P_i
    if S_guess is None:
        S_guess = S_i

    P_i_plus_1, S_i_plus_1, dt, nb_newton = solss_newton_step(
        grid=grid,
        P_i=P_i,
        S_i=S_i,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        dt_min=dt_min,
        wells=effective_wells,
        max_newton_iter=max_newton_iter,
        eps=eps,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
    )

    # number of newton iterations
    newton_list.append(nb_newton)
    # number of newton fails
    i = 0
    if dt != dt_save:
        while dt_save / 2**i != dt:
            newton_list.append(-1)
            i += 1
    dt_list.append(dt)

    total_time += dt

    # update simulation state
    simulation_state["data"][str(total_time)] = {
        "step": step,
        "P": P_i_plus_1.tolist(),
        "S": S_i_plus_1.tolist(),
        "dt": dt,
        "nb_newton": max_newton_iter * i + nb_newton,
        "total_time": total_time,
    }

    # print("number of newton iterations: ", max_newton_iter * i + nb_newton)
    return simulation_state, max_newton_iter * i + nb_newton


def main():
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 100, 1)
    phi = 0.2

    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

    T = calculate_transmissivity(grid, K)
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 100.0e5, "right": 100.0e5}

    # Saturation

    Sb_d = {"left": None, "right": None}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years
    total_sim_time = 30 * (60 * 60 * 24 * 365.25)
    max_newton_iter = 20

    productor = Well(
        name="productor",
        cell_group=np.array([[7500.0, 500.0]]),
        radius=0.1,
        control={"Dirichlet": 100.0e5},
        s_inj=0.0,
        schedule=[[0.0, total_sim_time]],
        mode="productor",
    )

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[3000.0, 500.0]]),
        radius=0.1,
        control={"Neumann": -0.0001},
        s_inj=1.0,
        schedule=[
            [0.0, total_sim_time],
        ],
        mode="injector",
    )

    P_imp = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        # wells=[well_co2, productor],
        wells=None,
    )

    print(P_imp)

    _, classic_nb_newton = solss_model_test_1_iter(
        grid=grid,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        total_sim_time=dt,
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[],
        P_guess=P_imp,
        S_guess=S,
    )

    print(classic_nb_newton)

    P_imp = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=[well_co2],
    )

    print(P_imp / 10e4)
    _, classic_nb_newton = solss_model_test_1_iter(
        grid=grid,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        total_sim_time=dt,
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp,
        S_guess=S,
    )

    print(classic_nb_newton)

    P_imp = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=[well_co2, productor],
    )
    print(P_imp / 10e4)

    _, classic_nb_newton = solss_model_test_1_iter(
        grid=grid,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        total_sim_time=dt,
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2, productor],
        P_guess=P_imp,
        S_guess=S,
    )

    print(classic_nb_newton)


if __name__ == "__main__":
    main()
