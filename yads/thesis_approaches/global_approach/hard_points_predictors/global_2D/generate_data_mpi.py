from mpi4py import MPI
import numpy as np
from pyDOE import lhs
import time
import sys

sys.path.append("/")

from yads.wells import Well
import yads.mesh as ym
from yads.predictors.hard_points_predictors.global_2D.utils import (
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
from yads.mesh.utils import load_json
import copy


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()

    if rank == 0:
        lhd = lhs(2, samples=4, criterion="maximin")
        grid = None
    elif rank == 1:
        lhd = None
        grid = load_json("../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")
    else:
        lhd = None
        grid = None

    lhd = comm.bcast(lhd, root=0)
    grid = comm.bcast(grid, root=1)

    grid.add_face_group_by_line("injector_one", (0.0, 0.0), (0.0, 1000.0))
    grid.add_face_group_by_line("injector_two", (2500.0, 3000.0), (3500.0, 3000.0))

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
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

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

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years
    total_sim_time = 1 * (60 * 60 * 24 * 365.25)
    max_newton_iter = 10
    eps = 1e-6

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1500.0, 2250]]),
        radius=0.1,
        control={"Neumann": -0.002},
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
        eps=eps,
        wells=[well_co2],
    )

    q_pow = -np.power(10, -5.0 + lhd[:, 0] * (-3 - (-5.0)))
    dt_init = 1e-3 * dt + lhd[:, 1] * (2 * dt - 1e-3 * dt)

    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)

    var_dict = {"dt_init": dt_init, "q": q_pow}
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

    simulation_dataset = []

    for i in range(
        int(len(combinations) / nb_proc * rank),
        int(len(combinations) / nb_proc * (rank + 1)),
    ):
        data_dict = data_dict_to_combi(data_dict, combinations[i], mapping)
        args = dict_to_args(data_dict)
        print(f"launching simulation {i + 1} on proc {rank}")
        sim_state = raw_solss_1_iter(*args)
        simulation_dataset.append([combinations_serializable[i], sim_state])
        print(f"simulation number {i + 1}/{len(combinations)} finished.")

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

    hard_simulation_dataset = []
    if len(hard_combinations) > 0:
        for i in range(
            int(len(hard_combinations) / nb_proc * rank),
            int(len(hard_combinations) / nb_proc * (rank + 1)),
        ):
            data_dict = data_dict_to_combi(
                hard_data_dict, hard_combinations[i], mapping
            )
            args = dict_to_args(data_dict)
            print(
                f"launching hard simulation {i + 1}/{len(hard_combinations)} on proc {rank}"
            )
            sim_state = raw_solss_1_hard_iter(*args)
            hard_simulation_dataset.append([hard_combinations_ser[i], sim_state])
            print(f"hard simulation number {i + 1}/{len(hard_combinations)} finished.")

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

    giga_hard_simulation_dataset = []
    for i in range(
        int(len(giga_hard_combinations) / nb_proc * rank),
        int(len(giga_hard_combinations) / nb_proc * (rank + 1)),
    ):
        data_dict = data_dict_to_combi_hard(
            giga_hard_data_dict, giga_hard_combinations[i], mapping, ens[i]
        )
        args = dict_to_args_hard(data_dict)
        print(
            f"launching giga hard simulation {i + 1}/{len(giga_hard_combinations)} on proc {rank}"
        )
        sim_state = raw_solss_n_iter(*args)
        giga_hard_simulation_dataset.append([giga_hard_combinations_ser[i], sim_state])
        print(
            f"giga hard simulation number {i + 1}/{len(giga_hard_combinations)} finished."
        )

    #### UPDATE HARD ####
    for i, (comb, state) in enumerate(hard_simulation_dataset):
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if nb_newton == 100:
                hard_simulation_dataset[i] = -1
    hard_simulation_dataset = list(
        filter(lambda ele: ele != -1, hard_simulation_dataset)
    )

    ##### UPDATE ALL POINTS FOR CLASSIFICATION #######
    # print(hard_combinations)
    for i, (comb, state) in enumerate(simulation_dataset):
        case = "undetermined"
        for tot_t in state["data"].keys():
            nb_newton = state["data"][tot_t]["nb_newton"]
            if (
                0 <= nb_newton <= state["metadata"]["max_newton_iter"]
                and state["data"][tot_t]["dt_init"] != -1
            ):
                case = "gentle"
            else:
                if comb in giga_hard_combinations_ser:
                    case = "giga_hard"
                elif comb in hard_combinations_ser:
                    case = "hard"
        simulation_dataset[i].append(case)

    simulation_dataset = comm.gather(simulation_dataset, root=0)
    hard_simulation_dataset = comm.gather(hard_simulation_dataset, root=0)
    giga_hard_simulation_dataset = comm.gather(giga_hard_simulation_dataset, root=0)
    if rank == 0:
        simulation_dataset = [
            item for sublist in simulation_dataset for item in sublist
        ]
        simulation_dataset = list(
            filter(lambda ele: ele is not None, simulation_dataset)
        )

        hard_simulation_dataset = [
            item for sublist in hard_simulation_dataset for item in sublist
        ]
        hard_simulation_dataset = list(
            filter(lambda ele: ele is not None, hard_simulation_dataset)
        )

        giga_hard_simulation_dataset = [
            item for sublist in giga_hard_simulation_dataset for item in sublist
        ]
        giga_hard_simulation_dataset = list(
            filter(lambda ele: ele is not None, giga_hard_simulation_dataset)
        )

        # save features of interest in csv
        print(
            len(simulation_dataset),
            len(hard_simulation_dataset),
            len(giga_hard_simulation_dataset),
        )
        savepath = f"data_mpi/{len(simulation_dataset)}_qt_10_100_newtons_2D_global"
        df_sim = data_to_df(simulation_dataset)
        df_sim.to_csv(savepath + ".csv")
        df_hard = hard_data_to_df(hard_simulation_dataset)
        df_hard.to_csv(savepath + "_hard.csv")
        df_giga_hard = giga_hard_data_to_df(giga_hard_simulation_dataset)
        df_giga_hard.to_csv(savepath + "_giga_hard.csv")


if __name__ == "__main__":
    print("launching generate data mpi")
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
