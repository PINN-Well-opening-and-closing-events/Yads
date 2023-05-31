import os

from mpi4py import MPI
import numpy as np
from pyDOE import lhs
import time
import sys
import subprocess as sp

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.wells import Well
from yads.thesis_approaches.global_approach.hard_points_predictors.global_2D_S_var.utils import (
    dict_to_args,
    data_dict_to_combi,
    generate_wells_from_q,
    args_to_dict,
    better_data_to_df,
)
from yads.thesis_approaches.data_generation import raw_solss_1_iter
from yads.mesh.utils import load_json


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    save_dir = "q_5_35_dt_01_10"
    nb_data = 2
    np.random.seed(42)
    lhd_1 = lhs(3, samples=nb_data, criterion="maximin")
    lhd_2 = lhs(1, samples=nb_data, criterion="maximin")
    lhd_3 = lhs(2, samples=nb_data, criterion="maximin")

    if rank == 0:
        grid = load_json("../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_XS.json")
        # create dirs
        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
    else:
        grid = None

    grid = comm.bcast(grid, root=0)

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
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)

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

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years

    max_newton_iter = 200
    eps = 1e-6

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1500.0, 2250]]),
        radius=0.1,
        control={"Neumann": -0.002},
        s_inj=1.0,
        schedule=[[0.0, dt],],
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
        total_sim_time=dt,
        kr_model=kr_model,
        max_newton_iter=max_newton_iter,
        eps=eps,
        wells=[well_co2],
    )

    ##########  PART ONE  ################
    # scaling lhs data
    q_pow = -np.power(10, -5.0 + lhd_1[:, 0] * (-3.0 - (-5.0)))
    dt_init = 1 * dt + lhd_1[:, 1] * (10 * dt - 1 * dt)
    S0 = lhd_1[:, 2] * 0.6
    # rounding to avoid saving issues
    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)

    var_dict_1 = {"dt_init": dt_init, "q": q_pow, "S": S0}
    mapping_1 = {}
    var_list = []
    var_list_serializable = []

    #### Generate simulation params combinations #####
    for i, key in enumerate(var_dict_1.keys()):
        if key in data_dict.keys():
            mapping_1[key] = i
            var_list.append(var_dict_1[key])
            var_list_serializable.append(var_dict_1[key])
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
                mapping_1["wells"] = i
                non_ser_wells = generate_wells_from_q(
                    var_dict_1["q"], well_ref, other_wells
                )
                var_list.append(non_ser_wells)
                var_list_serializable.append(
                    [[well.well_to_dict() for well in combi] for combi in non_ser_wells]
                )
        else:
            print("key error")
            return
    nb_cells = data_dict["grid"].nb_cells

    combinations_1 = [
        (var_list[0][i], var_list[1][i], np.array([var_list[2][i]] * nb_cells))
        for i in range(len(var_dict_1["q"]))
    ]

    combinations_serializable_1 = [
        (
            var_list_serializable[0][i],
            var_list_serializable[1][i],
            [var_list_serializable[2][i]] * nb_cells,
        )
        for i in range(len(var_dict_1["q"]))
    ]

    del var_dict_1
    del var_list
    del var_list_serializable

    ##########  PART TWO  ################

    # scaling lhs data
    dt_init = 1 * dt + lhd_2 * (10 * dt - 1 * dt)
    # rounding to avoid saving issues
    dt_init = np.round(dt_init, 1)
    q_zeros = np.zeros(len(dt_init))

    var_dict_2 = {"dt_init": dt_init, "q": q_zeros}
    mapping_2 = {}
    var_list = []
    var_list_serializable = []

    #### Generate simulation params combinations #####
    for i, key in enumerate(var_dict_2.keys()):
        if key in data_dict.keys():
            mapping_2[key] = i
            var_list.append(var_dict_2[key])
            var_list_serializable.append(var_dict_2[key])
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
                mapping_2["wells"] = i
                non_ser_wells = generate_wells_from_q(
                    var_dict_2["q"], well_ref, other_wells
                )
                var_list.append(non_ser_wells)
                var_list_serializable.append(
                    [[well.well_to_dict() for well in combi] for combi in non_ser_wells]
                )
        else:
            print("key error")
            return

    combinations_2 = [
        (var_list[0][i], var_list[1][i]) for i in range(len(var_dict_2["q"]))
    ]

    combinations_serializable_2 = [
        (var_list_serializable[0][i], var_list_serializable[1][i])
        for i in range(len(var_dict_2["q"]))
    ]

    del var_dict_2
    del var_list
    del var_list_serializable

    ##########  PART THREE  ################
    q_pow = -np.power(10, -5.0 + lhd_3[:, 0] * (-3.0 - (-5.0)))
    dt_init = 1 * dt + lhd_3[:, 1] * (10 * dt - 1 * dt)
    # rounding to avoid saving issues

    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)

    var_dict_3 = {"dt_init": dt_init, "q": q_pow}
    mapping_3 = {}
    var_list = []
    var_list_serializable = []

    #### Generate simulation params combinations #####
    for i, key in enumerate(var_dict_3.keys()):
        if key in data_dict.keys():
            mapping_3[key] = i
            var_list.append(var_dict_3[key])
            var_list_serializable.append(var_dict_3[key])
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
                mapping_3["wells"] = i
                non_ser_wells = generate_wells_from_q(
                    var_dict_3["q"], well_ref, other_wells
                )
                var_list.append(non_ser_wells)
                var_list_serializable.append(
                    [[well.well_to_dict() for well in combi] for combi in non_ser_wells]
                )
        else:
            print("key error")
            return

    combinations_3 = [
        (var_list[0][i], var_list[1][i]) for i in range(len(var_dict_3["q"]))
    ]

    combinations_serializable_3 = [
        (var_list_serializable[0][i], var_list_serializable[1][i])
        for i in range(len(var_dict_3["q"]))
    ]

    del var_dict_3
    del var_list
    del var_list_serializable

    for i in range(
        int(len(combinations_1) / nb_proc * rank),
        int(len(combinations_1) / nb_proc * (rank + 1)),
    ):
        ##################  PART ONE  ##########################
        data_dict = data_dict_to_combi(data_dict, combinations_1[i], mapping_1)
        args = dict_to_args(data_dict)
        if rank == 0:
            print(
                f"launching part one simulation {i + 1}/{int(len(combinations_1) / nb_proc)} on proc {rank}"
            )
        sim_state_1 = raw_solss_1_iter(*args)
        # convert sim to Pandas DataFrame
        df_sim = better_data_to_df(combinations_serializable_1[i], sim_state_1)
        # create filename
        num_sim = int(len(combinations_1) / nb_proc * (rank + 1))
        filename = f"all_sim_1_{nb_data}_{nb_proc}_{rank}_{num_sim}_{i}"
        save_path = save_dir + "/" + filename
        # save to csv
        df_sim.to_csv(save_path + ".csv", sep="\t", index=False)

        ##################  PART TWO  ##########################
        # No well, only dt
        for tot_t in sim_state_1["data"].keys():
            # print(f"1: {combinations_serializable_1[i][0]},total {sim_state_1['data'][tot_t]['total_time']}, "
            #       f"init {sim_state_1['data'][tot_t]['dt_init']}")
            data_dict["S"] = np.array(sim_state_1["data"][tot_t]["S"])
            data_dict["P"] = np.array(sim_state_1["data"][tot_t]["P"])

        data_dict = data_dict_to_combi(data_dict, combinations_2[i], mapping_2)
        args = dict_to_args(data_dict)
        if rank == 0:
            print(
                f"launching part two simulation {i + 1}/{int(len(combinations_2) / nb_proc)} on proc {rank}"
            )
        sim_state_2 = raw_solss_1_iter(*args)
        # convert sim to Pandas DataFrame
        df_sim = better_data_to_df(combinations_serializable_2[i], sim_state_2)
        # create filename
        num_sim = int(len(combinations_2) / nb_proc * (rank + 1))
        filename = f"all_sim_2_{nb_data}_{nb_proc}_{rank}_{num_sim}_{i}"
        save_path = save_dir + "/" + filename
        # save to csv
        df_sim.to_csv(save_path + ".csv", sep="\t", index=False)
        ##################  PART THREE  ##########################
        # Well + dt
        for tot_t in sim_state_2["data"].keys():
            # print(f"2: {combinations_serializable_2[i][0]},total {sim_state_2['data'][tot_t]['total_time']}, "
            #       f"init {sim_state_2['data'][tot_t]['dt_init']}")
            data_dict["S"] = np.array(sim_state_2["data"][tot_t]["S"])
            data_dict["P"] = np.array(sim_state_2["data"][tot_t]["P"])
        data_dict = data_dict_to_combi(data_dict, combinations_3[i], mapping_3)
        args = dict_to_args(data_dict)
        if rank == 0:
            print(
                f"launching part three simulation {i + 1}/{int(len(combinations_3) / nb_proc)} on proc {rank}"
            )
        sim_state_3 = raw_solss_1_iter(*args)
        # convert sim to Pandas DataFrame
        df_sim = better_data_to_df(combinations_serializable_3[i], sim_state_3)
        # create filename
        num_sim = int(len(combinations_3) / nb_proc * (rank + 1))
        filename = f"all_sim_3_{nb_data}_{nb_proc}_{rank}_{num_sim}_{i}"
        save_path = save_dir + "/" + filename
        # save to csv
        df_sim.to_csv(save_path + ".csv", sep="\t", index=False)
        # for tot_t in sim_state_3['data'].keys():
        #     print(f"3: {combinations_serializable_3[i][0]},total {sim_state_3['data'][tot_t]['total_time']}, "
        #           f"init {sim_state_3['data'][tot_t]['dt_init']}")


if __name__ == "__main__":
    print("launching generate data mpi")
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
