import numpy as np

from yads.numerics.physics import calculate_transmissivity
from yads.thesis_approaches.GWM.Pressure_generator.P_generator import (
    P_imp_generator,
    P_imp_brute_force,
)
from yads.thesis_approaches.GWM.Pressure_generator.cov_mats import cov_matrix_P_dist
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian


Lx, Ly = 555, 555
Nx, Ny = 11, 11

nb_samples = 2
nb_boundaries = 4

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
# BOUNDARY CONDITIONS #

cor_ds = [3.0 / grid.nb_boundary_faces, 3]
seed = None


savepath = f"data/nb_samples_{nb_samples}_nb_boundaries_{nb_boundaries}"
P_imp_generator(
    grid,
    nb_samples,
    nb_boundaries,
    P_min,
    P_max,
    cov_matrix_P_dist,
    cor_ds,
    seed,
    savepath,
)

savepath = f"data/raw_nb_samples_{nb_samples}"
P_imp_brute_force(grid, nb_samples, P_min, P_max, seed, savepath)
