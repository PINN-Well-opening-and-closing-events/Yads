import numpy as np
import os

import yads.mesh as ym
import yads.numerics as yn

# create 1D cartesian mesh 10 000m x 1000m with 200 cells in x direction and 1 in y direction
grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 100, 1)

# define initial time step and total simulation time in seconds
dt = 1 * (60 * 60 * 24 * 365.25)  # in years
total_sim_time = 100 * (60 * 60 * 24 * 365.25)  # in years

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

Pb = {"left": 150.0e5, "right": 100.0e5}

# Saturation
# only water left and right
Sb_d = {"left": 1.0, "right": 0.0}
Sb_n = {"left": None, "right": None}
Sb_dict = {"Dirichlet": Sb_d, "Neumann": Sb_n}

# Fully Implicit scheme
eps = 1e-6
max_newton_iter = 200

save_dir = "save"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
save_path = save_dir + "/cemracs_"
newton_list, dt_list = yn.schemes.solss(
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
    eps=eps,
    save=True,
    save_step=1,
    save_path=save_path,
    save_states_to_json=True,
    json_savepath="./cemracs.json",
)

print(newton_list)
