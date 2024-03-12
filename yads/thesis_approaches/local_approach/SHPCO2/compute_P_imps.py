import sys

sys.path.append("/")
sys.path.append("/home/irsrvhome1/R16/lechevaa/YADS/Yads/yads")
sys.path.append("/work/lechevaa/PycharmProjects/yads")

import pandas as pd
from ast import literal_eval
import numpy as np
import os
import yads.mesh as ym
import matplotlib.pyplot as plt
from yads.numerics.solvers import implicit_pressure_solver
from yads.wells import Well
from yads.numerics.physics import calculate_transmissivity


path_dir = "data/case_0/data/train_q_5_3_dt_1_10_S_0_06.csv"

grid = ym.utils.load_json("../../../../meshes/SHP_CO2/2D/SHP_CO2_2D_S.json")


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
T = calculate_transmissivity(grid=grid, K=K)
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

P_imp_list = []

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_proc = comm.Get_size()

df = pd.read_csv(path_dir, sep="\t", converters={"S0": literal_eval})

for i in range(
    int(len(df) / nb_proc * rank),
    int(len(df) / nb_proc * (rank + 1)),
):
    q = df["q"].loc[i]
    dt = df["dt"].loc[i]
    S0 = df["S0"].loc[i]
    Sb_dict["Dirichlet"] = {"injector_one": S0[0], "injector_two": S0[0], "right": 0.0}

    well_co2 = Well(
        name="well co2",
        cell_group=np.array([[1500.0, 2250]]),
        radius=0.1,
        control={"Neumann": q},
        s_inj=1.0,
        schedule=[
            [0.0, dt],
        ],
        mode="injector",
    )

    P_imp = implicit_pressure_solver(
        grid=grid,
        K=K,
        T=T,
        P=P,
        S=S0,
        Pb=Pb,
        Sb_dict=Sb_dict,
        mu_g=mu_g,
        mu_w=mu_w,
        kr_model=kr_model,
        wells=[well_co2],
    )
    P_imp_list.append(P_imp.tolist())

all_P_imp = comm.gather(P_imp_list, root=0)
if rank == 0:
    flat_P_imp = [arr for sublist in all_P_imp for arr in sublist]
    print(len(flat_P_imp))
    df["P_imp"] = flat_P_imp
    df.to_csv(f"data/train_q_5_3_dt_1_10_S_0_06_P_imp.csv", sep="\t", index=False)
