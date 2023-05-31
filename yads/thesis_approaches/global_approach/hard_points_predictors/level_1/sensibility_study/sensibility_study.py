from ast import literal_eval
from yads.numerics.solvers.solss_solver_depreciated import solss_newton_step
import yads.mesh as ym
from yads.mesh import Mesh
from yads.wells import Well
from yads.numerics import calculate_transmissivity
import numpy as np
import pandas as pd
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
    kr_model: str = "quadratic",
    max_newton_iter=10,
    eps=1e-6,
    wells=None,
    P_guess=None,
    S_guess=None,
    debug_path=None,
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
    total_time = 0.0
    newton_list = []
    dt_min = dt_init / 2 ** 10

    if wells:
        for well in wells:
            grid.connect_well(well)

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
        dt_init=dt_init,
        dt_min=dt_min,
        wells=effective_wells,
        max_newton_iter=max_newton_iter,
        eps=eps,
        kr_model=kr_model,
        P_guess=P_guess,
        S_guess=S_guess,
        debug_newton_mode=True,
        debug_newton_path=debug_path,
    )

    # number of newton iterations
    newton_list.append(nb_newton)
    # number of newton fails
    i = 0
    if dt != dt_init and dt != -1:
        while dt_init / 2 ** i != dt:
            i += 1

    return dt, max_newton_iter * i + nb_newton


def launch_sim(qt, S_sol, P_imp, debug_path, i):
    print(f"launching simulation {i}")
    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[3000.0, 500.0]]),
        radius=0.1,
        control={"Neumann": qt[0]},
        s_inj=1.0,
        schedule=[[0.0, total_sim_time],],
        mode="injector",
    )
    S_02 = clipping(S_sol + np.random.normal(0, 1e-2, len(S_sol)))
    S_03 = clipping(S_sol + np.random.normal(0, 1e-3, len(S_sol)))
    S_04 = clipping(S_sol + np.random.normal(0, 1e-4, len(S_sol)))
    S_05 = clipping(S_sol + np.random.normal(0, 1e-5, len(S_sol)))

    solss_model_test_1_iter(
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
        debug_path=debug_path + f"_S_true_{i}",
    )

    solss_model_test_1_iter(
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
        S_guess=S_02,
        debug_path=debug_path + f"_S_02_{i}",
    )

    solss_model_test_1_iter(
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
        S_guess=S_03,
        debug_path=debug_path + f"_S_03_{i}",
    )

    solss_model_test_1_iter(
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
        S_guess=S_04,
        debug_path=debug_path + f"_S_04_{i}",
    )

    solss_model_test_1_iter(
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
        S_guess=S_05,
        debug_path=debug_path + f"_S_05_{i}",
    )
    print(f"end of simulation {i}")
    return


def literal_switch(x):
    try:
        return literal_eval(str(x))
    except Exception as e:
        return []


def clipping(sat):
    if not all([1.0 >= sw >= 0.0 for sw in sat]):
        for i, sw in enumerate(sat):
            if sw < 0:
                sat[i] = 0
            elif sw > 1:
                sat[i] = 1
    return sat


def main():
    result_train = pd.read_csv(
        "../data/giga_hard/result_giga_hard_train.csv",
        converters={"S_true": literal_eval, "P_imp": literal_eval},
    )
    S_true = np.array(list(result_train["S_true"]))
    P_imp = np.array(list(result_train["P_imp"]))
    qt = result_train[["q", "t"]].to_numpy()

    Parallel(n_jobs=-1)(
        delayed(launch_sim)(
            qt=qt[i],
            S_sol=S_true[i],
            P_imp=P_imp[i],
            debug_path="debug_path/sensibility_study_train",
            i=i,
        )
        for i in range(result_train.shape[0])
    )
    columns = ["index", "P_i_plus_1", "S_i_plus_1", "Residual", "dt_init", "q", "i"]

    # test.shape[0]
    df_true = pd.read_csv(
        "debug_path/sensibility_study_train_S_true.csv", index_col=False
    )
    df_true.columns = columns
    df_true["literal_Residual"] = df_true["Residual"].apply(lambda x: literal_switch(x))
    df_true["literal_P"] = df_true["P_i_plus_1"].apply(lambda x: literal_switch(x))
    df_true["literal_S"] = df_true["S_i_plus_1"].apply(lambda x: literal_switch(x))
    df_true.to_csv("debug_path/sensibility_study_train_S_true.csv", index=False)
    del df_true

    df_02 = pd.read_csv("debug_path/sensibility_study_train_S_02.csv", index_col=False)
    df_02.columns = columns
    df_02["literal_Residual"] = df_02["Residual"].apply(lambda x: literal_switch(x))
    df_02["literal_P"] = df_02["P_i_plus_1"].apply(lambda x: literal_switch(x))
    df_02["literal_S"] = df_02["S_i_plus_1"].apply(lambda x: literal_switch(x))
    df_02.to_csv("debug_path/sensibility_study_train_S_02.csv", index=False)
    del df_02

    df_03 = pd.read_csv("debug_path/sensibility_study_train_S_03.csv", index_col=False)
    df_03.columns = columns
    df_03["literal_Residual"] = df_03["Residual"].apply(lambda x: literal_switch(x))
    df_03["literal_P"] = df_03["P_i_plus_1"].apply(lambda x: literal_switch(x))
    df_03["literal_S"] = df_03["S_i_plus_1"].apply(lambda x: literal_switch(x))
    df_03.to_csv("debug_path/sensibility_study_train_S_03.csv", index=False)
    del df_03

    df_04 = pd.read_csv("debug_path/sensibility_study_train_S_04.csv", index_col=False)
    df_04.columns = columns
    df_04["literal_Residual"] = df_04["Residual"].apply(lambda x: literal_switch(x))
    df_04["literal_P"] = df_04["P_i_plus_1"].apply(lambda x: literal_switch(x))
    df_04["literal_S"] = df_04["S_i_plus_1"].apply(lambda x: literal_switch(x))
    df_04.to_csv("debug_path/sensibility_study_train_S_04.csv", index=False)
    del df_04

    df_05 = pd.read_csv("debug_path/sensibility_study_train_S_05.csv", index_col=False)
    df_05.columns = columns
    df_05["literal_Residual"] = df_05["Residual"].apply(lambda x: literal_switch(x))
    df_05["literal_P"] = df_05["P_i_plus_1"].apply(lambda x: literal_switch(x))
    df_05["literal_S"] = df_05["S_i_plus_1"].apply(lambda x: literal_switch(x))
    df_05.to_csv("debug_path/sensibility_study_train_S_05.csv", index=False)
    del df_05
    return


if __name__ == "__main__":
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

    total_sim_time = 1 * (60 * 60 * 24 * 365.25)
    max_newton_iter = 100

    productor = Well(
        name="productor",
        cell_group=np.array([[7500.0, 500.0]]),
        radius=0.1,
        control={"Dirichlet": 100.0e5},
        s_inj=0.0,
        schedule=[[0.0, total_sim_time]],
        mode="productor",
    )
    main()
