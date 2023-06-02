import numpy as np  # type: ignore
import yads.mesh as ym
import yads.numerics as yn
from yads.wells import Well
import time

#### IMPES: IMPLICIT PRESSURE EXPLICIT SATURATION (SOLVER) ####

#### System initialization ####
load_time = time.time()

# 2D example
grid = ym.two_D.create_2d_cartesian(Lx=3, Ly=3, Nx=10, Ny=10)

# PHYSICS

# Porosity
phi = np.ones(grid.nb_cells)
# Diffusion coefficient (i.e Permeability)
K = np.ones(grid.nb_cells)
# Water saturation initialization
S = np.full(grid.nb_cells, 0.01)
# Pressure initialization
P = np.full(grid.nb_cells, 1.5)

mu_w = 1.0
mu_o = 1.0

# BOUNDARY CONDITIONS #
# Pressure
Pb = {"left": 2.0, "right": 1.0}

# Saturation
Sb_d = {"left": 1.0, "right": 0.1}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

dt = 0.5
total_sim_time = 0.5
max_iter = 5


# IMPES SOLVER
print("IMPES")
yn.schemes.impes_solver(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_w,
    mu_o,
    wells=[],
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_iter=max_iter,
)

# Water saturation initialization
S = np.full(grid.nb_cells, 0.01)
# Pressure initialization
P = np.full(grid.nb_cells, 1.5)

print("IMPIMS")
yn.schemes.impims_solver(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_w,
    mu_o,
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_newton_iter=max_iter,
    wells=[],
)

print("FULLY IMPLICIT")
yn.schemes.solss(
    grid,
    P,
    S,
    Pb,
    Sb_dict,
    phi,
    K,
    mu_w,
    mu_o,
    total_sim_time=total_sim_time,
    dt_init=dt,
    max_newton_iter=max_iter,
    wells=[],
)
