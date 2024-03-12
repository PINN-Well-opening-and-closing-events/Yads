import copy

from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
import numpy as np

from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import implicit_pressure_solver
from yads.thesis_approaches.GWM.Pressure_generator.plots import (
    plot_P_imp,
)
import pickle
import matplotlib.pyplot as plt

Lx, Ly = 9, 9
Nx, Ny = 9, 9

nb_samples = 100
radius = Lx / 2
nb_boundaries = 3

P_min = 10e6
P_max = 20e6
grid = create_2d_cartesian(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

phi = 0.2

# Porosity
phi = np.full(grid.nb_cells, phi)
# Diffusion coefficient (i.e Permeability)
K = np.full(grid.nb_cells, 100.0e-15)

T = calculate_transmissivity(grid, K)
# gaz saturation initialization
S = np.full(grid.nb_cells, 0.0)
# Pressure initialization
P0 = np.full(grid.nb_cells, 100.0e5)
mu_w = 0.571e-3
mu_g = 0.0285e-3

kr_model = "quadratic"

folder_path = "nb_samples_100_nb_boundaries_3/1/"
fig, axs = plt.subplots(1, 5, figsize=(15, 8))

for i in range(5):
    with open(folder_path + f"1_rotation_{11 + i}" + ".pkl", "rb") as f:
        (groups, Pb_dict) = pickle.load(f)
    grid_temp = copy.deepcopy(grid)
    for group in groups:
        grid_temp.add_face_group_by_line(*group)
    # Saturation
    Sb_d = copy.deepcopy(Pb_dict)
    Sb_n = copy.deepcopy(Pb_dict)
    for group in Sb_d.keys():
        Sb_d[group] = 0.0
        Sb_n[group] = None
    Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}
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
    )

    plot_P_imp(grid, P_imp, axs[i], Pmax=P_max, Pmin=P_min)

plt.show()
