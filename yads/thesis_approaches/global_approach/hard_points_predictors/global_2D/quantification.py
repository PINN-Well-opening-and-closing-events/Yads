import pandas as pd
import numpy as np
import pickle
from ast import literal_eval
from joblib import delayed, Parallel
import copy
from typing import Union, List

from yads.numerics import calculate_transmissivity, implicit_pressure_solver
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
from yads.wells import Well
import yads.mesh as ym
from yads.mesh import Mesh
from yads.numerics.solvers.newton import res


def solss_model_test_n_iter(
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
    wells: Union[List[Well], None] = None,
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

    # correct simulation total time and timestep to avoid Newton > max iter
    # end of corrections
    T = calculate_transmissivity(grid, K)
    P = implicit_pressure_solver(
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
        wells=wells,
    )
    dt_list = []
    dt = dt_init
    total_time = 0.0
    newton_list = []
    dt_min = dt_init / 2 ** 5
    i = 0

    if wells:
        for well in wells:
            grid.connect_well(well)

    dt_save = copy.deepcopy(dt_init)
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
        P_guess=P_guess,
        S_guess=P_i,
    )

    if np.linalg.norm(X_pred_res) > np.linalg.norm(X_classic_res):
        P_guess = P
        S_guess = S

    while total_time < total_sim_time and dt != -1:

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
        P_guess, S_guess = P_i_plus_1, S_i_plus_1
        # number of newton iterations
        newton_list.append(nb_newton)
        dt_list.append(dt)

        total_time += dt
        # print(f"Simulation progress: {total_time/total_sim_time*100}%")
        # number of newton fails
        if dt != dt_save and total_time < total_sim_time and dt != -1:
            while dt_save / 2 ** i != dt:
                i += 1

        # check if we need to synch dt with total_sim_time
        if dt + total_time > total_sim_time:
            dt = total_sim_time - total_time
    assert total_time == total_sim_time or total_time == -1
    return dt_list, max_newton_iter * i + sum(newton_list), newton_list


def launch_inf(qt, log_qt, i):
    print(f"launching step number {i}, with {qt}")
    dict_save = {
        "q": qt[0],
        "total_sim_time": qt[1],
        "classification": "None",
        "dt_pred_list": None,
        "P_imp": None,
        "dt_classic": None,
        "nb_newton_classic": None,
        "dt_boosted": None,
        "nb_newton_boosted": None,
    }

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[545.0, 545.0]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, total_sim_time],],
        mode="injector",
    )

    ### 1: Classify well event #######
    dt_pred_list = [qt[1]]
    classif_pred = classifier.predict(
        classifier_scaler.transform(log_qt.reshape(1, -1))
    )[0]
    if classif_pred == 0:
        # Gentle or Hard case
        dict_save["classification"] = "gentle"

    elif classif_pred == 1:
        nb_try = 0
        dict_save["classification"] = "giga_hard"
        # Giga hard case
        # must change dt until it classifies as 0
        while classif_pred != 0:
            if nb_try == 5:
                break
            dt_pred = (
                10
                ** dt_predict.predict(
                    dt_predict_scaler.transform(log_qt.reshape(1, -1))
                )[0]
            )
            dt_pred_list.append(dt_pred)
            log_qt = np.array([log_qt[0], dt_pred])
            classif_pred = classifier.predict(
                classifier_scaler.transform(log_qt.reshape(1, -1))
            )[0]
            nb_try += 1

    dict_save["dt_pred_list"] = dt_pred_list
    S_pred = S_predict.predict(S_predict_scaler.transform(log_qt.reshape(1, -1)))[0]
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
    (
        dict_save["dt_classic"],
        dict_save["nb_newton_classic"],
        dict_save["newton_list_classic"],
    ) = solss_model_test_n_iter(
        grid=grid,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt_pred_list[0],
        total_sim_time=dt_pred_list[0],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P,
        S_guess=S,
    )

    (
        dict_save["dt_boosted"],
        dict_save["nb_newton_boosted"],
        dict_save["newton_list_boosted"],
    ) = solss_model_test_n_iter(
        grid=grid,
        P=P,
        S=S,
        Pb=Pb,
        Sb_dict=Sb_dict,
        phi=phi,
        K=K,
        mu_g=mu_g,
        mu_w=mu_w,
        dt_init=dt_pred_list[-1],
        total_sim_time=dt_pred_list[0],
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2],
        P_guess=P_imp,
        S_guess=S_pred,
    )
    print(f"step number {i} finished")
    return dict_save


def main():
    test = pd.read_csv(
        "data/1000_10_100_newtons_2D_global_test.csv",
        converters={"S": literal_eval, "P": literal_eval},
    )
    test["log_q"] = np.log10(-test["q"])
    qts = test[["q", "dt_init"]].to_numpy()
    log_qts = test[["log_q", "dt_init"]].to_numpy()
    result = Parallel(n_jobs=-1)(
        delayed(launch_inf)(qt=qts[i], log_qt=log_qts[i], i=i)
        for i in range(test.shape[0])
    )
    df = pd.DataFrame(result)
    df.to_csv(
        "./data/quantification/result_quantification_test.csv", sep="\t", index=False
    )
    return


if __name__ == "__main__":
    grid = ym.two_D.create_2d_cartesian(1089, 1089, 33, 33)

    # dz = 100  # 100 meters
    # real_volume = np.multiply(grid.measures(item="cell"), dz)
    # grid.change_measures("cell", real_volume)

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

    Sb_d = {"left": 0.0, "right": 0.0}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years
    total_sim_time = 1 * (60 * 60 * 24 * 365.25)
    max_newton_iter = 10
    eps = 1e-6

    classifier_scaler = pickle.load(
        open("models/case_classifier_scaler_xgboost.pkl", "rb")
    )
    classifier = pickle.load(open("models/case_classifier_model_xgboost.pkl", "rb"))

    dt_predict = pickle.load(open("models/dt_predict_model_xgboost.pkl", "rb"))
    dt_predict_scaler = pickle.load(open("models/dt_predict_scaler_xgboost.pkl", "rb"))

    S_predict = pickle.load(open("models/S_predict_model_hard_xgboost.pkl", "rb"))
    S_predict_scaler = pickle.load(open("models/S_predict_scaler_xgboost.pkl", "rb"))

    main()
