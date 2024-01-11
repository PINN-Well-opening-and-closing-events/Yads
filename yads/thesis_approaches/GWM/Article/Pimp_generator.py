import copy

import pandas as pd

from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
import numpy as np
import subprocess as sp
from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import implicit_pressure_solver
import os
import pickle
from pyDOE import lhs

from yads.thesis_approaches.data_generation import raw_solss_1_iter
from yads.wells import Well

Nx, Ny = 9, 9
nb_bound_faces = Nx * 2 + Ny * 2
dxy = 50
Lx, Ly = Nx * dxy, Ny * dxy

P_min = 10e6
P_max = 20e6

grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

assert grid.nb_cells == Nx*Ny

# Porosity
phi = 0.2
phi = np.full(grid.nb_cells, phi)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 100.0e-15)

T = calculate_transmissivity(grid, K)
# Pressure initialization
P0 = np.full(grid.nb_cells, 100.0e5)
mu_w = 0.571e-3
mu_g = 0.0285e-3

kr_model = "quadratic"

folder_path = "nb_samples_100_nb_boundaries_1_size_9_9/"
sample_dirs = os.listdir(folder_path)
# create dirs
save_dir = "test_" + folder_path
if not os.path.isdir(save_dir):
    sp.call(f"mkdir {save_dir}", shell=True)

for seed, sample_num in enumerate(sample_dirs):
    rota_files = os.listdir(folder_path + sample_num)
    if not os.path.isdir(save_dir + sample_num):
        sp.call(f"mkdir {save_dir + sample_num}", shell=True)
    # set new seed for each batch
    np.random.seed(seed)
    # scaling lhs data
    nb_data = len(rota_files)
    dt = 1 * (60 * 60 * 24 * 365.25)  # in years
    lhd = lhs(3, samples=nb_data, criterion="maximin")
    q_pow = -np.power(10, -5.0 + lhd[:, 0] * (-3.0 - (-5.0)))
    dt_init = 1 * dt + lhd[:, 1] * (10 * dt - 1 * dt)
    S0 = lhd[:, 2] * 0.6
    # rounding to avoid saving issues
    q_pow = np.round(q_pow, 6)
    dt_init = np.round(dt_init, 1)
    for i, rota in enumerate(rota_files):
        with open(folder_path + sample_num + "/" + rota, 'rb') as f:
            (groups, Pb_dict) = pickle.load(f)

        # prepare for save
        well_co2 = Well(
            name="well co2",
            cell_group=np.array([[Lx/2, Ly/2]]),
            radius=0.1,
            control={"Neumann": q_pow[i]},
            s_inj=1.0,
            schedule=[[0.0,  dt_init[i]],],
            mode="injector",
        )
        grid_temp = copy.deepcopy(grid)
        for group in groups:
            grid_temp.add_face_group_by_line(*group)
        # Saturation
        Sb_d = copy.deepcopy(Pb_dict)
        Sb_n = copy.deepcopy(Pb_dict)
        for group in Sb_d.keys():
            Sb_d[group] = S0[i]
            Sb_n[group] = None
        Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
        S = np.full(grid_temp.nb_cells, S0[i])
        P_imp = implicit_pressure_solver(
            grid=grid_temp,
            K=K,
            T=T,
            P=P0,
            S=S,
            Pb=Pb_dict,
            Sb_dict=Sb_dict,
            mu_g=mu_g,
            mu_w=mu_w,
            wells=[well_co2]
        )

        sim_state = raw_solss_1_iter(
            grid=grid_temp,
            P=P_imp,
            S=S,
            Pb=Pb_dict,
            Sb_dict=Sb_dict,
            phi=phi,
            K=K,
            mu_g=mu_g,
            mu_w=mu_w,
            dt_init=dt,
            total_sim_time=dt,
            kr_model=kr_model,
            max_newton_iter=200,
            eps=1e-6,
            wells=[well_co2],
        )

        data_dict = sim_state["data"]
        for tot_t in data_dict.keys():
            S_i_plus_1 = data_dict[tot_t]["S"]
            P_i_plus_1 = data_dict[tot_t]["P"]
            nb_newton = data_dict[tot_t]["nb_newton"]
            S0 = data_dict[tot_t]["S0"]
            data_dict = {
                "groups": groups,
                "Pb_dict": Pb_dict,
                "Pb_file": folder_path + sample_num + "/" + rota,
                "q": well_co2.control['Neumann'],
                "S": S_i_plus_1,
                "dt": dt_init[i],
                "P": P_i_plus_1,
                "P_imp": P_imp.tolist(),
                "S0": S0,
                "nb_newton": nb_newton,
            }
            # print(nb_newton)
        df = pd.DataFrame([data_dict])
        save_path = save_dir + sample_num + "/" + rota
        # save to csv
        df.to_csv(save_path + ".csv", sep="\t", index=False)
        break