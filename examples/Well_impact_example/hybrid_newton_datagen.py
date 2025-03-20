import os

import numpy as np
from pyDOE import lhs
import time
import subprocess as sp

from yads.wells import Well
from yads.thesis_approaches.local_approach.milestone_0.utils import (
    dict_to_args,
    data_dict_to_combi,
    generate_wells_from_q,
    args_to_dict,
    better_data_to_df,
)
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import yads.mesh as ym


def main():
    save_dir = "hybrid_newton_data/all_data"
    os.makedirs(save_dir, exist_ok=True)
    nb_data = 4096
    np.random.seed(42)
    lhd = lhs(2, samples=nb_data, criterion="maximin")

  
    grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 201, 1)
    # create dirs
    if not os.path.isdir(save_dir):
        sp.call(f"mkdir {save_dir}", shell=True)

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
        cell_group=np.array([[5000.0, 500]]),
        radius=0.1,
        control={"Neumann": -5e-4},
        s_inj=1.0,
        schedule=[
            [0.0, dt],
        ],
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

    # scaling lhs data
    q_pow = -np.power(10, -5.0 + lhd[:, 0] * (-3 - (-5.0)))
    dt_init = 5e-1 * dt + lhd[:, 1] * (5 * dt - 5e-1 * dt)

    # rounding to avoid saving issues
    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)

    var_dict = {"dt_init": dt_init, "q": q_pow}
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
        (
            var_list_serializable[0][i],
            var_list_serializable[1][i],
        )
        for i in range(len(var_dict["q"]))
    ]

    # First simulation dataset Nmax iter

    for i in range(nb_data):
        data_dict = data_dict_to_combi(data_dict, combinations[i], mapping)
        args = dict_to_args(data_dict)
        print(
            f"launching simulation {i + 1}/{nb_data}"
        )
        sim_state = raw_solss_1_iter(*args)
        # convert sim to Pandas DataFrame
        df_sim = better_data_to_df(combinations_serializable[i], sim_state)
        # create filename
       
        filename = f"hybrid_newton_dataset_{nb_data}_{i}"
        save_path = save_dir + "/" + filename
        # save to csv
        df_sim.to_csv(save_path + ".csv", sep="\t", index=False)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"realised in {time.time() - start_time:3e} seconds")
