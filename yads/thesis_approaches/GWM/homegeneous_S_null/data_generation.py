import copy
import os

import numpy as np
from pyDOE import lhs
import time
import sys
import subprocess as sp
import pickle

from yads.mesh.two_D import create_2d_cartesian

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads")

from yads.wells import Well
from yads.thesis_approaches.GWM.homegeneous_S_null.utils import (
    dict_to_args,
    data_dict_to_combi,
    generate_wells_from_q,
    args_to_dict,
    better_data_to_df,
)
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import yads.mesh as ym


def main():
    save_dir = "idc"

    np.random.seed(42)
    Lx, Ly = 555, 555
    Nx, Ny = 11, 11
    grid_temp = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    # create dirs
    if not os.path.isdir(save_dir):
        sp.call(f"mkdir {save_dir}", shell=True)

    phi = 0.2
    # Porosity
    phi = np.full(grid_temp.nb_cells, phi)
    # Diffusion coefficient (i.e Permeability)
    K = np.full(grid_temp.nb_cells, 100.0e-15)

    # gaz saturation initialization
    S = np.full(grid_temp.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid_temp.nb_cells, 100.0e5)

    # viscosity
    mu_w = 0.571e-3
    mu_g = 0.0285e-3

    kr_model = "quadratic"
    # BOUNDARY CONDITIONS #
    # Read P boundary files
    grid = copy.deepcopy(grid_temp)
    P_boundary_path = "data/nb_samples_100_nb_boundaries_2"
    subfolders = [f.path for f in os.scandir(P_boundary_path) if f.is_dir()]
    all_groups, all_Pb_dicts = [], []
    for folder in subfolders:
        files = [
            f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
        ]
        for file in files:
            groups, Pb_dict = pickle.load(open(folder + "/" + file, "rb"))
            all_groups.append(groups)
            all_Pb_dicts.append(Pb_dict)
    nb_data = len(all_groups)
    lhd = lhs(2, samples=nb_data, criterion="maximin")

    dt = 1 * (60 * 60 * 24 * 365.25)  # in years

    max_newton_iter = 200
    eps = 1e-6

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[Lx / 2, Ly / 2]]),
        radius=0.1,
        control={"Neumann": -0.002},
        s_inj=1.0,
        schedule=[[0.0, dt]],
        mode="injector",
    )

    Pb = {}
    Sb_dict = {}

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
    #
    # scaling lhs data
    q_pow = -np.power(10, -4.0 + lhd[:, 0] * (-3 - (-4.0)))
    dt_init = dt + lhd[:, 1] * (5 * dt - dt)
    # rounding to avoid saving issues
    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)

    var_dict = {
        "dt_init": dt_init,
        "q": q_pow,
        "Pb": all_Pb_dicts,
        "groups": all_groups,
    }
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
        elif key in ["groups"]:
            mapping["groups"] = i
            var_list.append(var_dict[key])
            var_list_serializable.append(var_dict[key])
        else:
            print("key error")
            return

    nb_cells = data_dict["grid"].nb_cells
    combinations = [
        (var_list[0][i], var_list[1][i], var_list[2][i], var_list[3][i])
        for i in range(len(var_dict["q"]))
    ]

    combinations_serializable = [
        (
            var_list_serializable[0][i],
            var_list_serializable[1][i],
            var_list_serializable[2][i],
            var_list_serializable[3][i],
        )
        for i in range(len(var_dict["q"]))
    ]

    # First simulation dataset Nmax iter
    for i in range(nb_data):
        data_dict_temp = data_dict_to_combi(data_dict, combinations[i], mapping)
        grid_temp = copy.deepcopy(data_dict["grid"])
        groups = combinations[i][3]
        for group in groups:
            grid_temp.add_face_group_by_line(*group)
        data_dict_temp["grid"] = grid_temp

        Sb_d = copy.deepcopy(combinations[i][2])
        Sb_n = copy.deepcopy(combinations[i][2])
        for group in Sb_d.keys():
            Sb_d[group] = 0.0
            Sb_n[group] = None
        Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

        data_dict_temp["Sb_dict"] = Sb_dict
        args = dict_to_args(data_dict_temp)
        sim_state = raw_solss_1_iter(*args)
        # convert sim to Pandas DataFrame
        df_sim = better_data_to_df(combinations_serializable[i], sim_state)
        filename = f"idc_{i}"
        # create filename
        save_path = save_dir + "/" + filename
        # # save to csv
        df_sim.to_csv(save_path + ".csv", sep="\t", index=False)


if __name__ == "__main__":
    print("launching generate data mpi")
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
