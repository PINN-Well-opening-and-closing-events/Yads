import copy
from typing import Union, List
from ast import literal_eval
import pandas as pd
import sys
import torch
import numpy as np
import pickle
from ml_tools.FNO1D import FNO1d
from ml_tools.UnitGaussianNormalizer import UnitGaussianNormalizer
import matplotlib.pyplot as plt

import os
import subprocess as sp

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
    save_path = 'autopath',
):
    dt = dt_init

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
        debug_newton_mode=True,
        debug_newton_path=save_path
    )

    return P_i_plus_1, S_i_plus_1, dt_sim, nb_newton, norms


def launch_inference(qt, log_qt, i):
    dict_save = {"q": qt[0], "total_sim_time": qt[1]}
    vanilla_grid = copy.deepcopy(grid)
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[5000.0, 500]]),
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

    # Data prep for model
    # shape prep
    # create features maps

    q_flat_zeros = np.zeros(nb_cells)
    q_flat_zeros[100] = log_qt[0]

    log_q = torch.from_numpy(np.array(np.reshape(q_flat_zeros, (nb_cells, 1))))
    log_dt = torch.from_numpy(np.array(np.full((nb_cells, 1), log_qt[1])))
    S0 = torch.from_numpy(np.array(np.zeros((nb_cells, 1))))
    log_P_imp = torch.from_numpy(
        np.array(np.log10(np.array(P_imp_global).reshape(nb_cells, 1)))
    )

    log_q = q_normalizer.encode(log_q).float()
    log_dt = dt_normalizer.encode(log_dt).float()
    log_P_imp = P_imp_normalizer.encode(log_P_imp).float()

    x = torch.cat([log_q, log_dt, S0, log_P_imp], 1).float()

    # normalizer prep

    x = x.reshape(1, nb_cells, 4)

    S_pred_global = S_model(x)
    S_pred_global = S_pred_global.detach()
    S_pred_global = S_pred_global.reshape(nb_cells)


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
        save_path=f"results/test_classic_{i}.json"
    )

    # visualize prediction
    #fig = plt.plot(S_pred_global)
    # plt.plot(S_i_plus_1)
    # plt.show()
 
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
        save_path=f"results/test_hybrid_{i}.json"
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
        result = launch_inference(
            qt=qts[i], log_qt=log_qts[i], i=i
        )
        df = pd.DataFrame([result])
        df.to_csv(
            f"./results/test_{i}.csv",
            sep="\t",
            index=False,
        )
        
        print(f"saving simulation number {i}")


if __name__ == "__main__":
   
    test = pd.read_csv(
        "hybrid_newton_data/test.csv",
        sep="\t",
    )

    save_dir = "results"

    if not os.path.isdir(save_dir):
        sp.call(f"mkdir {save_dir}", shell=True)
    # define reservoir setup
    nb_cells = 201
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, nb_cells, 1)

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

    S_model = FNO1d(modes=12, width=64, n_features=4)
    S_model.load_state_dict(
        torch.load(
            "hybrid_newton_data/best_FNO1D_PS.pt",
            map_location=torch.device("cpu"),
        )
    )

    q_normalizer = pickle.load(
        open("hybrid_newton_data/GWM_q_normalizer.pkl", "rb")
    )
    dt_normalizer = pickle.load(
        open("hybrid_newton_data/GWM_dt_normalizer.pkl", "rb")
    )
    P_imp_normalizer = pickle.load(
        open("hybrid_newton_data/GWM_P_imp_normalizer.pkl", "rb")
    )

    main()
