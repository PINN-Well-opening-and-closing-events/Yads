from mpi4py import MPI
import numpy as np
from pyDOE import lhs
import time
import sys

sys.path.append("/")
# sys.path.append("/")
print(sys.path)
from yads.predictors.easy_points_predictors.mpi_scripts.utils import args_to_dict
from yads.wells import Well
import yads.mesh as ym
from yads.predictors.easy_points_predictors.mpi_scripts.utils import (
    dict_to_args,
    data_dict_to_combi,
    generate_wells_from_q,
    json_to_df,
)
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import json


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
    simulation_dataset = []
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nb_proc = comm.Get_size()
    for i in range(
        int(len(combinations) / nb_proc * rank),
        int(len(combinations) / nb_proc * (rank + 1)),
    ):
        data_dict = data_dict_to_combi(data_dict, combinations[i], mapping)
        args = dict_to_args(data_dict)
        print(f"launching simulation {i + 1} on proc {rank}")
        sim_state = raw_solss_1_iter(*args)
        simulation_dataset.append([combinations_serializable[i], sim_state])
    return simulation_dataset


if __name__ == "__main__":
    start_time = time.time()
    dataset = main()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dataset = comm.gather(dataset, root=0)
    flat_dataset = [item for sublist in dataset for item in sublist]
    print(len(flat_dataset))
    savepath = "data/test_idc"
    print(f"realised in {time.time() - start_time:3e} seconds")
    # save everything to json
    with open(savepath + ".json", "w") as fp:
        json.dump(flat_dataset, fp)

    # save features of interest in csv
    df = json_to_df(flat_dataset)
    df.to_csv(savepath + ".csv")
