from ast import literal_eval
from tensorflow import keras
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
import yads.mesh as ym
from yads.mesh import Mesh
from yads.wells import Well
from yads.numerics import calculate_transmissivity
import copy
import pickle
import numpy as np
import pandas as pd
from yads.numerics.solvers.implicit_pressure_solver_with_wells import (
    implicit_pressure_solver,
)
from joblib import delayed, Parallel


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
        while dt_save / 2 ** i != dt:
            newton_list.append(-1)
            i += 1
    dt_list.append(dt)

    total_time += dt
    return dt, max_newton_iter * i + nb_newton


def launch_inf(qt, knn_model_pred, mlp_model_pred, S_sol, i):
    print(f"launching step number {i}")
    dict_save = {
        "classic_newtons": None,
        "imp_newtons": None,
        "perfect_newtons": None,
        "mlp_newtons": None,
        "knn_newtons": None,
        "S_true": S_sol.tolist(),
        "S_knn": knn_model_pred.tolist(),
        "S_mlp": mlp_model_pred.tolist(),
        "P_imp": None,
        "q": qt[0],
        "t": qt[1],
        "dt_classic": None,
        "dt_knn": None,
        "dt_mlp": None,
        "dt_imp": None,
        "dt_perfect": None,
    }
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[3000.0, 500.0]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, total_sim_time],],
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
        wells=[well_co2, productor],
    )

    dict_save["P_imp"] = P_imp.tolist()
    dict_save["dt_classic"], dict_save["classic_newtons"] = solss_model_test_1_iter(
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
        wells=[well_co2, productor],
        P_guess=P,
        S_guess=S,
    )

    dict_save["dt_imp"], dict_save["imp_newtons"] = solss_model_test_1_iter(
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
        wells=[well_co2, productor],
        P_guess=P_imp,
        S_guess=S,
    )

    dict_save["dt_knn"], dict_save["knn_newtons"] = solss_model_test_1_iter(
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
        wells=[well_co2, productor],
        P_guess=P_imp,
        S_guess=knn_model_pred,
    )

    dict_save["dt_mlp"], dict_save["mlp_newtons"] = solss_model_test_1_iter(
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
        wells=[well_co2, productor],
        P_guess=P_imp,
        S_guess=mlp_model_pred,
    )

    dict_save["dt_perfect"], dict_save["perfect_newtons"] = solss_model_test_1_iter(
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
        wells=[well_co2, productor],
        P_guess=P_imp,
        S_guess=S_sol,
    )
    print(f"step number {i} finished")
    return dict_save


def main():
    result = Parallel(n_jobs=-1)(
        delayed(launch_inf)(
            qt=qt[i],
            mlp_model_pred=mlp_model_pred[i],
            knn_model_pred=knn_model_pred[i],
            S_sol=testY[i],
            i=i,
        )
        for i in range(test.shape[0])
    )
    df = pd.DataFrame(result)
    df.to_csv("./data/result_train_1000.csv")
    return


if __name__ == "__main__":
    all_scaler_path = "models/scaler.pkl"
    knn_model_path = "models/model_knn_dist.pkl"
    test = pd.read_csv(
        "data/train_1000.csv", converters={"S": literal_eval, "P": literal_eval}
    )
    test["log_q"] = -np.log10(-test["q"])
    testX, testY = test[["log_q", "total_time"]].to_numpy(), np.array(list(test["S"]))
    qt = test[["q", "total_time"]].to_numpy()
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 256, 1)
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

    scaler = pickle.load(open(all_scaler_path, "rb"))
    testX_scaled = scaler.transform(testX)
    knn_model = pickle.load(open(knn_model_path, "rb"))
    mlp_model = keras.models.load_model("models/MLP")

    knn_model_pred = knn_model.predict(testX_scaled)
    mlp_model_pred = mlp_model.predict(testX_scaled.reshape((testX_scaled.shape[0], 2)))

    main()
