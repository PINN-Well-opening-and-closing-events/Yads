import copy
import os
import numpy as np

import yads.mesh as ym
import yads.numerics as yn
from yads.thesis_approaches.data_generation import raw_solss_1_iter
import pandas as pd


if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isdir("data/train"):
    os.mkdir("data/train")
if not os.path.isdir("data/test"):
    os.mkdir("data/test")
if not os.path.isdir("data/validation"):
    os.mkdir("data/validation")


def better_data_to_df(pb, state):
    list_of_dict = []
    for t in state["data"].keys():
        future_df = {
            "total_time": state["data"][t]["total_time"],
            "step": state["data"][t]["step"],
            "S": state["data"][t]["S"],
            "dt": state["data"][t]["dt"],
            "P": state["data"][t]["P"],
            "P_imp": state["metadata"]["P_imp"],
            "P_init": state["metadata"]["P_init"],
            "S0": state["data"][t]["S0"],
            "nb_newton": state["data"][t]["nb_newton"],
            "dt_init": state["data"][t]["dt_init"],
            "Res": state["data"][t]["Res"],
            "Pb": pb,
        }
        list_of_dict.append(future_df)

    df = pd.DataFrame(list_of_dict)
    return df


# create 1D cartesian mesh 10 000m x 1000m with 200 cells in x direction and 1 in y direction
grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 100, 1)

# define initial time step and total simulation time in seconds
dt = 1 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 101 * (60 * 60 * 24 * 365.25)  # in years

#### PHYSICS ####
# Porosity
phi = np.full(grid.nb_cells, 0.2)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 300.0e-15)
# gaz saturation initialization
S = np.full(grid.nb_cells, 0.0)
# Pressure initialization
P = np.full(grid.nb_cells, 100.0e5)
# viscosity
mu_w = 0.571e-3
mu_g = 0.0285e-3
# relative permeability model
kr_model = "quadratic"

# BOUNDARY CONDITIONS #
# Pressure
nb_data = 10
np.random.seed(42)
P_list = np.random.uniform(low=105e5, high=200e5, size=nb_data)

# Saturation
# only water left and right
Sb_d = {"left": 1.0, "right": 0.0}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

# Fully Implicit scheme
eps = 1e-6
max_newton_iter = 200

for i, Pl in enumerate(P_list):
    Pb = {"left": Pl, "right": 100.0e5}
    # gaz saturation initialization
    S = np.full(grid.nb_cells, 0.0)
    # Pressure initialization
    P = np.full(grid.nb_cells, 100.0e5)
    print(f"launching simulation {i} with P = {Pl}")
    if i < 0.6 * len(P_list):
        save_dir = "train"
    elif 0.6 * len(P_list) <= i < 0.8 * len(P_list):
        save_dir = "test"
    else:
        save_dir = "validation"

    for nb_t in range(int(total_sim_time / dt)):
        sim_state = raw_solss_1_iter(
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
            wells=[],
            total_sim_time=dt,
            kr_model=kr_model,
            max_newton_iter=max_newton_iter,
            eps=eps,
        )
        data_dict = sim_state["data"]
        df_sim = better_data_to_df(Pb, sim_state)
        save_path = f"data/{save_dir}/cemracs_data_{nb_data}_{i}_{nb_t}"
        # save to csv
        if nb_t != 0:
            df_sim.to_csv(save_path + ".csv", sep="\t", index=False)

        for tot_t in data_dict.keys():
            S = copy.deepcopy(np.array(data_dict[tot_t]["S"]))
            P = copy.deepcopy(np.array(data_dict[tot_t]["P"]))
