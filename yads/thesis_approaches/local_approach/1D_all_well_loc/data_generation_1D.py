import copy
import os

from mpi4py import MPI
import numpy as np
from pyDOE import lhs
import time
import sys
import subprocess as sp

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/yads")
sys.path.append("/work/lechevaa/PycharmProjects/yads")

from yads.wells import Well
from yads.thesis_approaches.local_approach.milestone_0.utils import (
    dict_to_args,
    data_dict_to_combi,
    args_to_dict,
)
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import yads.mesh as ym


def generate_wells_from_q(q_list, well_ref: Well, other_wells, well_loc_list):
    import copy
    well_list = []

    for i, q in enumerate(q_list):
        new_loc = np.array([[well_loc_list[i], 500.]])
        well_displaced = Well(
            name=well_ref.name,
            cell_group=new_loc,
            radius=well_ref.radius,
            control=well_ref.control,
            s_inj=well_ref.injected_saturation,
            schedule=well_ref.schedule,
            mode="injector")

        well_displaced.set_control(q)
        well_list.append([copy.deepcopy(well_displaced)] + other_wells)
    return well_list


def better_data_to_df(combination_ser, sim_state):
    import pandas as pd
    list_of_dict = []
    well_co2 = combination_ser[1][0]
    well_loc = well_co2['cell_group']
    q = well_co2["control"]["Neumann"]
    P_imp = sim_state["metadata"]["P_imp"]
    data_dict = sim_state["data"]
    for tot_t in data_dict.keys():
        step = data_dict[tot_t]["step"]
        S = data_dict[tot_t]["S"]
        P = data_dict[tot_t]["P"]
        total_time = data_dict[tot_t]["total_time"]
        dt_init = data_dict[tot_t]["dt_init"]
        dt = data_dict[tot_t]["dt"]
        nb_newton = data_dict[tot_t]["nb_newton"]
        S0 = data_dict[tot_t]["S0"]
        B = data_dict[tot_t]["Res"]
        future_df = {
            "well_loc": well_co2['cell_group'],
            "q": q,
            "total_time": total_time,
            "step": step,
            "S": S,
            "dt": dt,
            "P": P,
            "P_imp": P_imp,
            "S0": S0,
            "nb_newton": nb_newton,
            "dt_init": dt_init,
            "Res": B,
        }
        list_of_dict.append(future_df)

    df = pd.DataFrame(list_of_dict)
    return df


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    save_dir = "data/manuscrit"
    nb_data = 5000
    np.random.seed(42)
    lhd = lhs(3, samples=nb_data, criterion="maximin")

    if rank == 0:
        grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 200, 1)
        # create dirs
        if not os.path.isdir(save_dir):
            sp.call(f"mkdir {save_dir}", shell=True)
    else:
        grid = None

    grid = comm.bcast(grid, root=0)
    vanilla_grid = copy.deepcopy(grid)
    phi = 0.2
    # Porosity
    phi = np.full(grid.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid.nb_cells, 100.0e-15)
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
    Pb = {"left": 110.0e5, "right": 100.0e5}

    # Saturation
    Sb_d = {"left": 0.0, "right": 0.0}
    Sb_n = {"left": None, "right": None}
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years

    max_newton_iter = 200
    eps = 1e-6

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[0.0, 500]]),
        radius=0.1,
        control={"Neumann": -5e-4},
        s_inj=1.0,
        schedule=[[0.0, dt],],
        mode="injector",
    )

    data_dict = args_to_dict(
        grid=vanilla_grid,
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

    # scaling lhs data
    q_pow = -np.power(10, -4.5 + lhd[:, 0] * (-3 - (-4.5)))
    dt_init = 5e-1 * dt + lhd[:, 1] * (5 * dt - 5e-1 * dt)
    well_locs = 2500 + lhd[:, 2] * (7500 - 2500)
    # rounding to avoid saving issues
    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)

    var_dict = {"dt_init": dt_init, "q": q_pow, "well_locs": well_locs}
    mapping = {}
    var_list = []
    var_list_serializable = []

    #### Generate simulation params combinations #####
    for i, key in enumerate(var_dict.keys()):
        if key in data_dict.keys():
            mapping[key] = i
            var_list.append(var_dict[key])
            var_list_serializable.append(var_dict[key])
        # q -> well flux
        elif key in ["q", "S"]:
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
                    var_dict["q"], well_ref, other_wells, var_dict['well_locs']
                )
                var_list.append(non_ser_wells)
                var_list_serializable.append(
                    [[well.well_to_dict() for well in combi] for combi in non_ser_wells]
                )
        else:
            print(f"key error: {key}")
            pass

    combinations = [(var_list[0][i], var_list[1][i]) for i in range(len(var_dict["q"]))]

    combinations_serializable = [
        (var_list_serializable[0][i], var_list_serializable[1][i],)
        for i in range(len(var_dict["q"]))
    ]

    # First simulation dataset Nmax iter

    for i in range(
        int(len(combinations) / nb_proc * rank),
        int(len(combinations) / nb_proc * (rank + 1)),
    ):
        data_dict = data_dict_to_combi(data_dict, combinations[i], mapping)
        data_dict['grid'] = copy.deepcopy(grid)
        args = dict_to_args(data_dict)
        if rank == 0:
            print(
                f"launching simulation {i + 1}/{int(len(combinations) / nb_proc)} on proc {rank}"
            )

        sim_state = raw_solss_1_iter(*args)
        # convert sim to Pandas DataFrame
        df_sim = better_data_to_df(combinations_serializable[i], sim_state)
        # create filename
        num_sim = int(len(combinations) / nb_proc * (rank + 1))
        filename = f"all_sim_{nb_data}_{nb_proc}_{rank}_{num_sim}_{i}"
        save_path = save_dir + "/" + filename
        # save to csv
        df_sim.to_csv(save_path + ".csv", sep="\t", index=False)


if __name__ == "__main__":
    print("launching generate data mpi")
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
