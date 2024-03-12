import copy
from typing import Union, List
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import subprocess as sp
import torch
import numpy as np
import pickle
from models.FNO import FNO2d, UnitGaussianNormalizer

import sys

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")
sys.path.append("/work/lechevaa/PycharmProjects/yads")

from yads.mesh.utils import load_json
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import solss_newton_step, implicit_pressure_solver
from yads.wells import Well
from yads.mesh import Mesh
from yads.numerics.solvers.newton import res


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
    total_sim_time,
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


def launch_inference(qt, log_qt, i, test_P, test_S):
    print(f"launching step number {i}, with {qt[0], qt[1], qt[2][0]}")
    dict_save = {"q": qt[0], "total_sim_time": qt[1], "S0": qt[2][0]}
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1475.0, 2225]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[
            [0.0, qt[1]],
        ],
        mode="injector",
    )
    Sb_dict["Dirichlet"] = {
        "injector_one": qt[2][0],
        "injector_two": qt[2][0],
        "right": 0.0,
    }
    S = np.full(grid.nb_cells, qt[2][0])
    ################################
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

    dict_save["P_imp"] = P_imp.tolist()

    # Data prep for model
    well_x, well_y = 1475, 2225
    grid_dxy = 50
    d = 4
    cells_d = grid.find_cells_inside_square(
        (grid_dxy * (well_x / grid_dxy - d), grid_dxy * (well_y / grid_dxy + d)),
        (grid_dxy * (well_x / grid_dxy + d), grid_dxy * (well_y / grid_dxy - d)),
    )

    well_loc_idx = 40
    P_imp_local = P_imp[cells_d]
    S_n = S[cells_d]
    # shape prep
    local_shape = 2 * ext + 1
    q_flat_zeros = np.zeros((local_shape * local_shape))
    q_flat_zeros[well_loc_idx] = log_qt[0]
    log_q = torch.from_numpy(np.reshape(q_flat_zeros, (local_shape, local_shape, 1)))
    log_dt = torch.from_numpy(np.full((local_shape, local_shape, 1), log_qt[1]))
    S_n = torch.from_numpy(np.array(np.reshape(S_n, (local_shape, local_shape, 1))))
    P_imp_local_n = torch.from_numpy(
        np.array(np.reshape(P_imp_local, (local_shape, local_shape, 1)))
    )

    # normalizer prep
    log_q_n = q_normalizer.encode(log_q)
    log_dt_n = dt_normalizer.encode(log_dt)

    P_imp_local_n = P_imp_normalizer.encode(np.log10(P_imp_local_n))
    #
    x = torch.cat([log_q_n, log_dt_n, S_n, P_imp_local_n], 2).float()
    x = x.reshape(1, local_shape, local_shape, 4)
    S_pred = model(x)
    S_pred = S_pred.detach().numpy()

    S_pred = np.reshape(S_pred, (local_shape * local_shape))

    dict_save["S_predict_local"] = S_pred.tolist()

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=grid,
        P=P_imp,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        total_sim_time=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp,
        S_guess=S,
    )

    dict_save["P_i_plus_1_classic"] = P_i_plus_1.tolist()
    dict_save["S_i_plus_1_classic"] = S_i_plus_1.tolist()
    dict_save["nb_newton_classic"] = nb_newton
    dict_save["dt_sim_classic"] = dt_sim
    dict_save["norms_classic"] = norms

    S_pred_global = copy.deepcopy(S)
    S_pred_global[cells_d] = S_pred

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=grid,
        P=P_imp,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        total_sim_time=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp,
        S_guess=S_pred_global,
    )

    dict_save["S_i_plus_1_hybrid"] = S_i_plus_1.tolist()
    dict_save["P_i_plus_1_hybrid"] = P_i_plus_1.tolist()
    dict_save["nb_newton_hybrid"] = nb_newton
    dict_save["dt_sim_hybrid"] = dt_sim
    dict_save["norms_hybrid"] = norms
    print(
        f"step {i}: Newton classic {dict_save['nb_newton_classic']}, hybrid {dict_save['nb_newton_hybrid']}"
    )
    print(f"step number {i} finished")
    return dict_save


def main():
    test["log_q"] = -np.log10(-test["q"])
    test["log_dt"] = np.log(test["dt"])
    qts = test[["q", "dt", "S0_local"]].to_numpy()
    log_qts = test[["log_q", "log_dt"]].to_numpy()
    P_imps = test["P_imp_local"].to_numpy()
    for i in range(len(test)):
        result = launch_inference(
            qt=qts[i], log_qt=log_qts[i], i=i, test_P=P_imps[i], test_S=None
        )
        df = pd.DataFrame([result])
        df.to_csv(
            f"./results/quantification_{ext}_test_{rank}_{len(test)}_{i}.csv",
            sep="\t",
            index=False,
        )
        if rank == 0:
            print(f"saving simulation number {i}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    ext = 4
    if rank == 0:
        test_full = test = pd.read_csv(
            "data/train_q_5_3_dt_1_10_S_0_06_P_imp_extension_4.csv",
            converters={"P_imp_local": literal_eval, "S0_local": literal_eval},
            sep="\t",
        )

        save_dir = "results"
        test_split = np.array_split(test_full, nb_proc)

        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
        # define reservoir setup
        grid = load_json("../../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
        # Boundary groups creation
        grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
        grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

    else:
        grid = None
        test_split = None

    test = comm.scatter(test_split, root=0)

    grid = comm.bcast(grid, root=0)
    # define reservoir setup

    # Permeability barrier zone creation
    barrier_1 = grid.find_cells_inside_square((1000.0, 2000.0), (1250.0, 0))
    barrier_2 = grid.find_cells_inside_square((2250.0, 3000.0), (2500.0, 1000.0))
    barrier_3 = grid.find_cells_inside_square((3500.0, 3000.0), (3750.0, 500.0))

    phi = 0.2
    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)
    permeability_barrier = 1.0e-15
    K[barrier_1] = permeability_barrier
    K[barrier_2] = permeability_barrier
    K[barrier_3] = permeability_barrier
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)
    T = calculate_transmissivity(grid, K)

    # viscosity
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"injector_one": 110.0e5, "injector_two": 105.0e5, "right": 100.0e5}
    # Saturation
    Sb_d = {"injector_one": 0.0, "injector_two": 0.0, "right": 0.0}
    Sb_n = {"injector_one": None, "injector_two": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    max_newton_iter = 200
    eps = 1e-6

    S_model = model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    S_model.load_state_dict(
        torch.load(
            "models/checkpoint_best_model_4_local_2d_1500.pt",
            map_location=torch.device("cpu"),
        )
    )
    q_normalizer = pickle.load(open("models/q_normalizer.pkl", "rb"))
    P_imp_normalizer = pickle.load(open("models/P_imp_normalizer.pkl", "rb"))
    dt_normalizer = pickle.load(open("models/dt_normalizer.pkl", "rb"))
    main()
