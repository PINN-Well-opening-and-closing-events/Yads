import copy
from typing import Union, List
import pandas as pd
import sys
import torch
import numpy as np
import pickle
from models.FNO import FNO2d
from mpi4py import MPI
import os
import subprocess as sp

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.mesh.utils import load_json
from yads.numerics import calculate_transmissivity, implicit_pressure_solver
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
from yads.wells import Well
from yads.mesh import Mesh


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


def launch_inference(qt, log_qt, i):
    dict_save = {"q": qt[0], "total_sim_time": qt[1], "S0": S}
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[20 * 51 / 2.0, 20 * 51 / 2]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[
            [0.0, qt[1]],
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
        wells=[well_co2],
    )

    dict_save["P_imp"] = P_imp.tolist()

    # Data prep for model
    cells_d = grid.find_cells_inside_square(
        (20 * (25 - ext), 20 * (26 + ext)), (20 * (26 + ext), 20 * (25 - ext))
    )
    P_imp_local = P_imp[cells_d]
    well_loc_idx = P_imp_local.argmax()

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
    S_n = S_normalizer.encode(S_n)
    P_imp_local_n = P_imp_normalizer.encode(np.log10(P_imp_local_n))

    x = torch.cat([log_q_n, log_dt_n, S_n, P_imp_local_n], 2).float()
    x = x.reshape(1, local_shape, local_shape, 4)
    S_pred = model(x)
    S_pred = S_pred.detach().numpy()
    S_pred = np.reshape(S_pred, (local_shape * local_shape))
    dict_save["S_predict_local"] = S_pred.tolist()

    S_hybrid = copy.deepcopy(S)
    S_hybrid[cells_d] = S_pred
    dict_save["S_predict_global"] = S_hybrid.tolist()
    ################################
    # Debug plot
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 12))
    # q_plot = q_normalizer.decode(log_q_n)
    # dt_plot = dt_normalizer.decode(log_dt_n)
    # S_plot = S_normalizer.decode(S_n)
    # P_imp_plot = P_imp_normalizer.decode(P_imp_local_n)
    #
    # ax1.imshow(q_plot.reshape(local_shape, local_shape).T)
    # ax2.imshow(dt_plot.reshape(local_shape, local_shape).T)
    # ax3.imshow(S_plot.reshape(local_shape, local_shape).T)
    # ax4.imshow(np.power(P_imp_plot.reshape(local_shape, local_shape).T, 10))
    # ax5.imshow(S_pred.reshape(local_shape, local_shape).T)
    # fig.axes[1].invert_yaxis()
    # fig.axes[2].invert_yaxis()
    # fig.axes[0].invert_yaxis()
    # fig.axes[3].invert_yaxis()
    # fig.axes[4].invert_yaxis()
    # plt.show()
    ################################
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
    print("classic", nb_newton)
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
        S_guess=S_hybrid,
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

    for i in range(len(test)):
        result = launch_inference(qt=qts[i], log_qt=log_qts[i], i=i)
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
    ext = 5
    if rank == 0:
        test_full = pd.read_csv(f"test_well_extension_10.csv", sep="\t", nrows=10)

        save_dir = "results"
        test_split = np.array_split(test_full, nb_proc)

        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
        # define reservoir setup
        grid = load_json("meshes/51x51_20.json")
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
    T = calculate_transmissivity(grid, K)
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

    # viscosity
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Pressure
    Pb = {"left": 100.0e5, "upper": 100.0e5, "right": 100.0e5, "lower": 100.0e5}
    # Saturation
    Sb_d = {"left": 0.0, "upper": 0.0, "right": 0.0, "lower": 0.0}
    Sb_n = {"left": None, "upper": None, "right": None, "lower": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    max_newton_iter = 200
    eps = 1e-6

    S_model = model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
    S_model.load_state_dict(
        torch.load(
            f"models/well_extension_{ext}/checkpoint_best_model_5_local_2d_500.pt",
            map_location=torch.device("cpu"),
        )
    )
    q_normalizer = pickle.load(
        open(f"models/well_extension_{ext}/q_normalizer.pkl", "rb")
    )
    S_normalizer = pickle.load(
        open(f"models/well_extension_{ext}/S_normalizer.pkl", "rb")
    )
    dt_normalizer = pickle.load(
        open(f"models/well_extension_{ext}/dt_normalizer.pkl", "rb")
    )
    P_imp_normalizer = pickle.load(
        open(f"models/well_extension_{ext}/P_imp_normalizer.pkl", "rb")
    )
    if rank == 0:
        print("launching main")
    del test_split
    main()
