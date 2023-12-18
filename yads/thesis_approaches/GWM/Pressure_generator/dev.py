import copy

import numpy as np
from matplotlib import pyplot as plt

from yads.numerics.physics import calculate_transmissivity
from yads.numerics.solvers import implicit_pressure_solver
from yads.thesis_approaches.GWM.Pressure_generator.P_generator import (
    P_generator_wrapper,
    P_imp_generator,
)
from yads.thesis_approaches.GWM.Pressure_generator.P_interp_to_P_imp import (
    create_Pb_groups,
)
from yads.thesis_approaches.GWM.Pressure_generator.cov_mats import (
    cov_matrix_P_dist,
    cov_matrix_Id,
)
from yads.mesh.two_D.create_2D_cartesian import create_2d_cartesian
from yads.thesis_approaches.GWM.Pressure_generator.plots import (
    plot_circle_P,
    plot_circle_interp_P,
    plot_P_imp,
)
from yads.thesis_approaches.GWM.Pressure_generator.P_interpolation import P_interp

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
# BOUNDARY CONDITIONS #

cor_ds = [3.0 / grid.nb_boundary_faces, 3]
seed = 2
savepath = f"nb_samples_{nb_samples}_nb_boundaries_{nb_boundaries}"

# Wrapper of everything
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

