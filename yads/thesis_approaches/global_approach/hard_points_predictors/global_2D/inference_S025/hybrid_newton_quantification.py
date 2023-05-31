from typing import Union, List
from ast import literal_eval
from joblib import delayed, Parallel
import pandas as pd

import torch
import numpy as np
from model.NeuralNetwork import DeconvNet

from yads.mesh.utils import load_json
from yads.numerics import calculate_transmissivity, implicit_pressure_solver
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
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

    # residual check for init
    X_pred_res = res(
        grid=grid,
        S_i=S_i,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        wells=effective_wells,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
    )

    X_classic_res = res(
        grid=grid,
        S_i=S_i,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        T=T,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt,
        wells=effective_wells,
        kr_model=kr_model,
        P_guess=P_i,
        S_guess=S_i,
    )

    if np.linalg.norm(X_pred_res, ord=2) > np.linalg.norm(X_classic_res, ord=2):
        print("Not using hybrid method")
        P_guess = P
        S_guess = S

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton = solss_newton_step(
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

    return P_i_plus_1, S_i_plus_1, dt_sim, nb_newton


def launch_inference(qt, log_qt, i):
    print(f"launching step number {i}, with {qt}")
    dict_save = {"q": qt[0], "total_sim_time": qt[1]}

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1500.0, 2250]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, qt[1]],],
        mode="injector",
    )

    S_pred = S_model(torch.reshape(log_qt, (1, 2)))
    S_pred = S_pred.detach().numpy()
    S_pred = S_pred.reshape(S.shape)
    dict_save["S_predict"] = S_pred.tolist()
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

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton = hybrid_newton_inference(
        grid=grid,
        P=P,
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
        P_guess=P,
        S_guess=S,
    )
    dict_save["P_i_plus_1_classic"] = P_i_plus_1.tolist()
    dict_save["S_i_plus_1_classic"] = S_i_plus_1.tolist()
    dict_save["nb_newton_classic"] = nb_newton
    dict_save["dt_sim_classic"] = dt_sim

    P_i_plus_1, S_i_plus_1, dt_sim, nb_newton = hybrid_newton_inference(
        grid=grid,
        P=P,
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
        S_guess=S_pred,
    )

    dict_save["S_i_plus_1_hybrid"] = P_i_plus_1.tolist()
    dict_save["P_i_plus_1_hybrid"] = S_i_plus_1.tolist()
    dict_save["nb_newton_hybrid"] = nb_newton
    dict_save["dt_sim_hybrid"] = dt_sim
    print(
        f"step {i}: Newton classic {dict_save['nb_newton_classic']}, hybrid {dict_save['nb_newton_hybrid']}"
    )
    print(f"step number {i} finished")
    return dict_save


def main():
    test = pd.read_csv(
        "data/test_5004_q_5_3_dt_01_20_S_025.csv",
        converters={"S": literal_eval, "P": literal_eval},
        sep="\t",
    )
    test["log_q"] = -np.log10(-test["q"])
    test["log_dt"] = np.log(test["dt_init"])
    qts = test[["q", "dt_init"]].to_numpy()
    log_qts = test[["log_q", "log_dt"]]
    log_qts_tens = torch.from_numpy(log_qts.values).float()
    result = Parallel(n_jobs=4)(
        delayed(launch_inference)(qt=qts[i], log_qt=log_qts_tens[i], i=i)
        for i in range(40, 44)
    )
    df = pd.DataFrame(result)
    df.to_csv("./results/result_quantification_test_40_43.csv", sep="\t", index=False)
    return


if __name__ == "__main__":
    # define reservoir setup
    grid = load_json("../../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
    # Boundary groups creation
    grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
    grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

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
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.25)
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
    Sb_d = {"injector_one": 0.25, "injector_two": 0.25, "right": 0.25}
    Sb_n = {"injector_one": None, "injector_two": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    max_newton_iter = 200
    eps = 1e-6

    S_model = DeconvNet(10, 512)
    S_model.load_state_dict(torch.load("model/global_2d_S025.pt"))

    main()
