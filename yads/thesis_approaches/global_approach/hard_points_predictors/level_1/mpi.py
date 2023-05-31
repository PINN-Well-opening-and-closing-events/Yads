from mpi4py import MPI
import numpy as np
from pyDOE import lhs
import time
import sys

sys.path.append("/")
sys.path.append("/")

from yads.wells import Well
import yads.mesh as ym
from yads.predictors.hard_points_predictors.level_1.utils import (
    dict_to_args,
    data_dict_to_combi,
    generate_wells_from_q,
    data_to_df,
    hard_data_to_df,
    giga_hard_data_to_df,
    args_to_dict,
    data_dict_to_combi_hard,
    dict_to_args_hard,
)
from yads.thesis_approaches.data_generation import (
    raw_solss_1_iter,
    raw_solss_n_iter,
    raw_solss_1_hard_iter,
)
import copy


def main():
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 256, 1)

    phi = 0.2

    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)

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
        control={"Neumann": -0.02},
        s_inj=1.0,
        schedule=[[0.0, total_sim_time],],
        mode="injector",
    )

    data_dict = args_to_dict(
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
        total_sim_time=total_sim_time,
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=1e-6,
        wells=[well_co2, productor],
    )

    lhd = lhs(2, samples=10, criterion="maximin")

    q = -np.power(10, -6 + lhd[:, 0] * 3)
    dt_init = 1e-3 * dt + lhd[:, 1] * (dt - 1e-3 * dt)

    var_dict = {"dt_init": dt_init, "q": q}
    mapping = {}
    var_list = []
    var_list_serializable = []
    for i, key in enumerate(var_dict.keys()):
        if key in data_dict.keys():
            mapping[key] = i
            var_list.append(var_dict[key])
            var_list_serializable.append(var_dict[key])
        # q -> well flux
        elif key in ["q"]:
            if key == "q":
                well_ref = None
                other_wells = []
                for well in data_dict["wells"]:
                    if well.name == "well co2":
                        well_ref = well
                    else:
                        other_wells.append(well)
                assert well_ref is not None
                mapping["wells"] = i
                non_ser_wells = generate_wells_from_q(
                    var_dict["q"], well_ref, other_wells
                )
                var_list.append(non_ser_wells)
                var_list_serializable.append(
                    [[well.well_to_dict() for well in combi] for combi in non_ser_wells]
                )
        else:
            print("key error")
            return

    combinations = [(var_list[0][i], var_list[1][i]) for i in range(len(var_dict["q"]))]

    combinations_serializable = [
        (var_list_serializable[0][i], var_list_serializable[1][i])
        for i in range(len(var_dict["q"]))
    ]

    ### Parallel computation ###
    from joblib import delayed, Parallel

    ############### GLOBAL SIM PART ##########################

    def launch_sim(data_dict2, combination, mapping2, combination_ser, step):
        print(f"launching simulation number {step + 1}/{len(combinations)}")
        data_dict2 = data_dict_to_combi(data_dict2, combination, mapping2)
        args = dict_to_args(data_dict2)

        sim_state = raw_solss_1_iter(*args)
        print(f"simulation number {step + 1}/{len(combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON ALL POINTS")
    simulation_dataset = Parallel(n_jobs=-1)(
        delayed(launch_sim)(
            data_dict, combinations[i], mapping, combinations_serializable[i], i
        )
        for i in range(len(combinations))
    )
    simulation_dataset = list(filter(lambda ele: ele is not None, simulation_dataset))
    print("END OF SIMULATION ON ALL POINTS")
    ############################# HARD SIM PART #######################

    hard_var_dict = {"dt_init": [], "q": []}
    hard_combinations = []
    hard_combinations_ser = []
    for i, (comb, state) in enumerate(simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton > state["metadata"]["max_newton_iter"]:
                hard_combinations.append(combinations[i])
                hard_combinations_ser.append(combinations_serializable[i])
                hard_var_dict["dt_init"].append(var_dict["dt_init"][i])
                hard_var_dict["q"].append(var_dict["q"][i])

    hard_data_dict = copy.deepcopy(data_dict)
    # launching sim on points that have newton > max iter (10) with max iter = 100 this time
    hard_data_dict["max_newton_iter"] = 100

    def launch_hard_sim(data_dict2, combination, mapping2, combination_ser, step):
        print(f"launching simulation number {step + 1}/{len(hard_combinations)}")
        data_dict2 = data_dict_to_combi(data_dict2, combination, mapping2)
        args = dict_to_args(data_dict2)

        sim_state = raw_solss_1_hard_iter(*args)
        print(f"simulation number {step + 1}/{len(hard_combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON HARD POINTS")
    hard_simulation_dataset = []
    if len(hard_combinations) > 0:
        hard_simulation_dataset = Parallel(n_jobs=-1)(
            delayed(launch_hard_sim)(
                hard_data_dict,
                hard_combinations[i],
                mapping,
                hard_combinations_ser[i],
                i,
            )
            for i in range(len(hard_combinations))
        )
    hard_simulation_dataset = list(
        filter(lambda ele: ele is not None, hard_simulation_dataset)
    )
    print("END OF SIMULATION ON HARD POINTS")

    ##### GIGA HARD SIM PART ###################
    # Newton > 100
    giga_hard_var_dict = {"dt_init": [], "q": []}
    giga_hard_combinations = []
    giga_hard_combinations_ser = []
    ens = []
    for i, (comb, state) in enumerate(hard_simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton == 100:
                # not optimal but secure: n = 10, let's try first n = 3
                # n = nb_newton // 10
                n = 3
                ens.append(n)
                giga_hard_combinations.append(hard_combinations[i])
                giga_hard_combinations_ser.append(hard_combinations_ser[i])
                giga_hard_var_dict["dt_init"].append(hard_var_dict["dt_init"][i])
                giga_hard_var_dict["q"].append(hard_var_dict["q"][i])

    giga_hard_data_dict = copy.deepcopy(hard_data_dict)
    giga_hard_data_dict["max_newton_iter"] = 10

    def launch_giga_hard_sim(
        data_dict2, combination, mapping2, combination_ser, n, step
    ):
        print(f"launching simulation number {step + 1}/{len(giga_hard_combinations)}")
        data_dict2 = data_dict_to_combi_hard(data_dict2, combination, mapping2, n)
        args = dict_to_args_hard(data_dict2)

        sim_state = raw_solss_n_iter(*args)
        print(f"simulation number {step + 1}/{len(giga_hard_combinations)} finished")
        return [combination_ser, sim_state]

    print("LAUNCHING SIMULATION ON GIGA HARD POINTS")
    giga_hard_simulation_dataset = []
    if len(ens) > 0:
        giga_hard_simulation_dataset = Parallel(n_jobs=-1)(
            delayed(launch_giga_hard_sim)(
                giga_hard_data_dict,
                giga_hard_combinations[i],
                mapping,
                giga_hard_combinations_ser[i],
                ens[i],
                i,
            )
            for i in range(len(giga_hard_combinations))
        )
    giga_hard_simulation_dataset = list(
        filter(lambda ele: ele is not None, giga_hard_simulation_dataset)
    )
    print("END OF SIMULATION ON GIGA HARD POINTS")
    #### UPDATE HARD ####
    idx_to_remove = []
    for i, (comb, state) in enumerate(hard_simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton == 100:
                hard_simulation_dataset[i] = -1
    hard_simulation_dataset = list(
        filter(lambda ele: ele != -1, hard_simulation_dataset)
    )
    return simulation_dataset, hard_simulation_dataset, giga_hard_simulation_dataset


if __name__ == "__main__":
    start_time = time.time()
    dataset = main()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    (
        simulation_dataset,
        hard_simulation_dataset,
        giga_hard_simulation_dataset,
    ) = comm.gather(dataset, root=0)
    flat_dataset = [item for sublist in dataset for item in sublist]
    print(len(flat_dataset))
    savepath = "data/test_idc"
    print(f"realised in {time.time() - start_time:3e} seconds")
    # save everything to json
    # save features of interest in csv
    df_sim = data_to_df(simulation_dataset)
    df_sim.to_csv(savepath + ".csv")
    df_hard = hard_data_to_df(hard_simulation_dataset)
    df_hard.to_csv(savepath + "_hard.csv")
    df_giga_hard = giga_hard_data_to_df(giga_hard_simulation_dataset)
    df_giga_hard.to_csv(savepath + "_giga_hard.csv")
