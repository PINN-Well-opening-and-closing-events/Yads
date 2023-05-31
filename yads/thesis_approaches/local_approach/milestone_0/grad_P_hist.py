from typing import Union, List
from ast import literal_eval
import pandas as pd
import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.numerics import calculate_transmissivity, implicit_pressure_solver
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
from yads.wells import Well
from yads.mesh import Mesh
import yads.mesh as ym
from yads.mesh.utils import load_json


def hybrid_newton_inference(
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
    kr_model: str = "quadratic",
    max_newton_iter=200,
    eps=1e-6,
    wells: Union[List[Well], None] = None,
    P_guess=None,
    S_guess=None,
):
    dt = dt_init
    i = 0

    if wells:
        for well in wells:
            grid.connect_well(well)

    effective_wells = wells

    S_i = S
    P_i = P

    if P_guess is None:
        P_guess = P_i
    if S_guess is None:
        S_guess = S_i

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = solss_newton_step(
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
        dt_min=dt,
        wells=effective_wells,
        max_newton_iter=max_newton_iter,
        eps=eps,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
    )

    return P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms


def launch_inference(qt):
    grid_mid = load_json("shp_co2_1d.json")
    well_co2_mid = Well(
        name="well co2 mid",
        cell_group=np.array([[5000.0, 500]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, qt[1]],],
        mode="injector",
    )

    ################################
    P_imp_global_mid = implicit_pressure_solver(
        grid=grid_mid,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=[well_co2_mid],
    )

    well_co2_left = Well(
        name="well co2 left",
        cell_group=np.array([[2500.0, 500]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, qt[1]],],
        mode="injector",
    )

    grid_left = load_json("shp_co2_1d.json")
    ################################
    P_imp_global_left = implicit_pressure_solver(
        grid=grid_left,
        K=K,
        T=T,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=[well_co2_left],
    )

    well_cell_idx_left = -123456789
    for c in grid_left.cell_groups[well_co2_left.name]:
        well_cell_idx_left = c
    assert well_cell_idx_left >= 0

    well_cell_idx_mid = -123456789
    for c in grid_mid.cell_groups[well_co2_mid.name]:
        well_cell_idx_mid = c
    assert well_cell_idx_mid >= 0

    # mid well Grads P
    # centered grad
    Grad_P_global_mid = [
        (P_imp_global_mid[i + 1] - P_imp_global_mid[i - 1]) / (2 * 50)
        for i in range(1, len(P_imp_global_mid) - 1)
    ]
    left_Grad_mid = np.mean(Grad_P_global_mid[0 : well_cell_idx_mid - 1])
    right_Grad_mid = np.mean(Grad_P_global_mid[well_cell_idx_mid:-1])

    # left well Grads P
    # centered grad
    Grad_P_global_left = [
        (P_imp_global_left[i + 1] - P_imp_global_left[i - 1]) / (2 * 50)
        for i in range(1, len(P_imp_global_left) - 1)
    ]
    left_Grad_left = np.mean(Grad_P_global_left[0 : well_cell_idx_left - 1])
    right_Grad_left = np.mean(Grad_P_global_left[well_cell_idx_left:-1])

    return left_Grad_mid, right_Grad_mid, left_Grad_left, right_Grad_left


def main():
    qts = test[["q", "dt"]].to_numpy()
    left_Grads_mid, right_Grads_mid = [], []
    left_Grads_left, right_Grads_left = [], []
    for i in range(len(test)):
        (
            left_grad_mid,
            right_grad_mid,
            left_grad_left,
            right_grad_left,
        ) = launch_inference(qt=qts[i])
        left_Grads_mid.append(left_grad_mid)
        right_Grads_mid.append(right_grad_mid)
        left_Grads_left.append(left_grad_left)
        right_Grads_left.append(right_grad_left)

        if rank == 0:
            print(f"saving simulation number {i}")
    dict_df = {
        "left_Grads_mid": left_Grads_mid,
        "right_Grads_mid": right_Grads_mid,
        "left_Grads_left": left_Grads_left,
        "right_Grads_left": right_Grads_left,
    }

    df = pd.DataFrame([dict_df])
    df.to_csv(f"train_pressure_Grads.csv", sep="\t", index=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    max_range = max(
        [
            np.max(left_Grads_mid),
            np.max(right_Grads_mid),
            np.max(left_Grads_left),
            np.max(right_Grads_left),
        ]
    )
    min_range = min(
        [
            np.min(left_Grads_mid),
            np.min(right_Grads_mid),
            np.min(left_Grads_left),
            np.min(right_Grads_left),
        ]
    )

    ax1.hist(left_Grads_left, bins=range(0, int(max_range) + 1, 100), label="left well")
    ax1.hist(
        left_Grads_mid,
        bins=range(0, int(max_range) + 1, 100),
        label="mid well",
        alpha=0.75,
    )

    ax2.hist(
        right_Grads_left, bins=range(int(min_range) - 1, 0, 100), label="left well"
    )
    ax2.hist(
        right_Grads_mid,
        bins=range(int(min_range) - 1, 0, 100),
        label="mid well",
        alpha=0.75,
    )

    ax1.legend(loc="upper right", fontsize=20)
    ax2.legend(loc="upper left", fontsize=20)
    plt.show()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    if rank == 0:
        test_full = pd.read_csv(
            "sci_pres_results/data/train_well_extension_10.csv",
            converters={"S_local": literal_eval, "Res_local": literal_eval},
            sep="\t",
        )

        test_split = np.array_split(test_full, nb_proc)

        # define reservoir setup
        grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)
        grid.to_json("shp_co2_1d")
    else:
        grid = None
        test_split = None

    test = comm.scatter(test_split, root=0)
    grid = comm.bcast(grid, root=0)

    phi = 0.2
    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)
    T = calculate_transmissivity(grid, K)

    # viscosity
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 110.0e5, "right": 100.0e5}

    # Saturation
    Sb_d = {"left": 0.0, "right": 0.0}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years

    max_newton_iter = 200
    eps = 1e-6

    if rank == 0:
        print("launching main")
    del test_split
    main()
