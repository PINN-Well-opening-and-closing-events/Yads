import copy
from typing import Union, List
from ast import literal_eval
import pandas as pd
import sys
import torch
import numpy as np
import pickle
from NN.FNO1D import FNO1d
from NN.UnitGaussianNormalizer import UnitGaussianNormalizer
from mpi4py import MPI
import os
import subprocess as sp

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import implicit_pressure_solver
from yads.numerics.solvers import solss_newton_step
from yads.wells import Well
from yads.mesh import Mesh
import yads.mesh as ym


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


def launch_inference(qt, log_qt, well_loc, i):
    dict_save = {"q": qt[0], "total_sim_time": qt[1], "well_loc": well_loc}
    vanilla_grid = copy.deepcopy(grid)
    well_co2 = Well(
        name="well co2",
        cell_group=np.array(well_loc),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[
            [0.0, qt[1]],
        ],
        mode="injector",
    )

    ################################
    P_imp_global = implicit_pressure_solver(
        grid=vanilla_grid,
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
    well_cell_idx = -123456789
    for c in vanilla_grid.cell_groups[well_co2.name]:
        well_cell_idx = c
    assert well_cell_idx >= 0

    dist = 21
    P_imp_local = P_imp_global[
        well_cell_idx - int((dist - 1) / 2) : well_cell_idx + int((dist - 1) / 2) + 1
    ]
    dict_save["well_cell_idx"] = well_cell_idx
    # Data prep for model
    # shape prep
    # create features maps

    q_flat_zeros = np.zeros(dist)
    q_flat_zeros[int((dist - 1) / 2)] = log_qt[0]

    log_q = torch.from_numpy(np.array(np.reshape(q_flat_zeros, (dist, 1))))
    log_dt = torch.from_numpy(np.array(np.full((dist, 1), log_qt[1])))
    S0 = torch.from_numpy(np.array(np.zeros((dist, 1))))
    log_P_imp_local = torch.from_numpy(
        np.array(np.log(np.array(P_imp_local)).reshape(dist, 1))
    )

    q_normalizer, dt_normalizer, P_imp_normalizer = x_normalizer
    log_q = q_normalizer.encode(log_q).float()
    log_dt = dt_normalizer.encode(log_dt).float()
    log_P_imp_local = P_imp_normalizer.encode(log_P_imp_local).float()
    x = torch.cat([log_q, log_dt, log_P_imp_local], 1).float()

    # normalizer prep

    x = x.reshape(1, dist, 3)

    S_pred_local = S_model(x)
    S_pred_local = S_pred_local.detach()
    S_pred_local = S_pred_local.reshape(dist)

    S_pred_global = np.full(grid.nb_cells, 0.0)

    S_pred_global[
        well_cell_idx - int((dist - 1) / 2) : well_cell_idx + int((dist - 1) / 2) + 1
    ] = S_pred_local
    dict_save["S_predict_local"] = S_pred_local.tolist()
    dict_save["S_predict_global"] = S_pred_global.tolist()

    dict_save["P_imp_global"] = P_imp_global.tolist()
    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=vanilla_grid,
        P=P_imp_global,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp_global,
        S_guess=S,
    )
    dict_save["P_i_plus_1_classic"] = P_i_plus_1.tolist()
    dict_save["S_i_plus_1_classic"] = S_i_plus_1.tolist()
    dict_save["nb_newton_classic"] = nb_newton
    dict_save["dt_sim_classic"] = dt_sim
    dict_save["norms_classic"] = norms
    print("classic", nb_newton)
    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms = hybrid_newton_inference(
        grid=vanilla_grid,
        P=P_imp_global,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=qt[1],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp_global,
        S_guess=S_pred_global,
    )

    dict_save["S_i_plus_1_hybrid"] = S_i_plus_1.tolist()
    dict_save["P_i_plus_1_hybrid"] = P_i_plus_1.tolist()
    dict_save["nb_newton_hybrid"] = nb_newton
    dict_save["dt_sim_hybrid"] = dt_sim
    dict_save["norms_hybrid"] = norms
    print("hybrid ", nb_newton)
    return dict_save


def main():
    test["log_q"] = -np.log10(-test["q"])
    test["log_dt"] = np.log(test["dt"])

    qts = test[["q", "dt"]].to_numpy()
    log_qts = test[["log_q", "log_dt"]].to_numpy()
    well_locs = test["well_loc"]

    for i in range(len(test)):
        result = launch_inference(
            qt=qts[i], log_qt=log_qts[i], well_loc=well_locs[i], i=i
        )
        df = pd.DataFrame([result])
        df.to_csv(
            f"./results/quantification_train_{rank}_{len(test)}_{i}.csv",
            sep="\t",
            index=False,
        )
        if rank == 0:
            print(f"saving simulation number {i}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    if rank == 0:
        test_full = pd.read_csv(
            "data/test_well_extension_10.csv",
            converters={
                "S_local": literal_eval,
                "Res_local": literal_eval,
                "well_loc": literal_eval,
            },
            sep="\t",
        )

        save_dir = "results"
        test_split = np.array_split(test_full, nb_proc)

        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
        # define reservoir setup
        grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)
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

    S_model = FNO1d(modes=16, width=64, n_features=3)
    S_model.load_state_dict(
        torch.load(
            "models/final_model_well_extension_10.pt",
            map_location=torch.device("cpu"),
        )
    )
    x_normalizer = pickle.load(
        open("models/final_xy_normalizer_well_extension_10.pkl", "rb")
    )

    if rank == 0:
        print("launching main")
    del test_split
    main()
